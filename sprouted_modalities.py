"""
sprouted_modalities — runtime, data-driven modality detectors.

Background. The baked-in modality detectors live in `signals.py` as Python
functions registered in `MODALITY_REGISTRY`. Adding one normally means an
operator running `apply_proposal` to scaffold a stub, a human writing the
detector body, a test, and a restart — the deliberate "human in the loop"
path. This module is the *other* path: simple, pattern-based modalities the
agent can sprout at runtime from data, with no source-code change and no
restart. A sprouted modality is a name plus a few case-insensitive regex
patterns plus an activation rule; the analyzer composes it into a detector
that behaves like any baked-in one.

What this module deliberately is and is not:
  - It IS a *data* layer. A sprouted modality is a dict that lives in a JSON
    file in the data directory, not Python source. The agent (via Cambium,
    in a later change) writes that data; nothing here executes
    agent-authored code. The worst a malformed entry can do is fail to
    compile (and be skipped) or match nothing useful.
  - It is NOT a code-injection surface. There is no `eval`, no `exec`, no
    import of agent-authored modules. The only agent-controlled input is
    regex pattern strings and a few numeric knobs, all validated here.

Safety of the regex surface. Python's stdlib `re` has no per-match timeout,
and `signal.SIGALRM` only works on the main thread (the webapp analyzes on
worker threads), so we cannot bound matching at runtime portably. Instead we
bound it at *validation* time and *input* time:
  - patterns must compile;
  - patterns are rejected if they contain nested-quantifier shapes that are
    the classic catastrophic-backtracking risk (a quantified group that is
    itself quantified, e.g. `(a+)+`, `(a*)*`, `(a+)*`);
  - pattern length, pattern count per modality, and modality count are
    capped;
  - each pattern only ever sees input truncated to MATCH_INPUT_CAP chars.
A pattern that slips past these still runs on bounded input, so the worst
case is bounded work, not an unbounded hang.

Schema (one entry in the JSON list):

    {
      "name": "legal_document",        # modality name (stored in _meta)
      "patterns": ["\\bhereinafter\\b", "\\bwhereas\\b", ...],
      "threshold": 0.2,                # activation floor for "fired"
      "match_mode": "fraction_lines",  # or "any" / "count"
      "domain": true,                  # participates in retrieval anchoring
      "status": "active",              # or "tentative" (cooling-off)
      "weight_factor": 1.0,            # multiplies W_MODALITY (tentative=0.5)
      "origin": {                      # provenance, for audit
        "proposal_index": 412,
        "sprouted_at_ms": 1730000000000,
        "source_indices": [380, 388, 395, 401, 409]
      }
    }

This module knows nothing about the chain, retrieval, or the agent — like
`metadata.py` and `signals.py`, it is pure schema + logic. The chain still
records *that* a sprout happened (a proposal_status record); this file is the
derived, rebuildable activation data, analogous to how the embedding store is
derived from the chain.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Limits — the bounds that keep the regex surface safe and the registry sane.
# ---------------------------------------------------------------------------

MAX_SPROUTED_MODALITIES = 64       # registry size cap
MAX_PATTERNS_PER_MODALITY = 24     # patterns in one modality
MAX_PATTERN_LENGTH = 200           # chars in one pattern string
MATCH_INPUT_CAP = 20_000           # chars of input any pattern ever sees

# Status values.
STATUS_TENTATIVE = "tentative"     # in cooling-off; detected but damped
STATUS_ACTIVE = "active"           # graduated; full weight

# Match modes — how a modality's patterns combine into an activation in [0,1].
MATCH_FRACTION_LINES = "fraction_lines"  # fraction of lines hitting any pattern
MATCH_ANY = "any"                        # 1.0 if any pattern hits, else 0.0
MATCH_COUNT = "count"                     # min(1, total hits / 5)
VALID_MATCH_MODES = {MATCH_FRACTION_LINES, MATCH_ANY, MATCH_COUNT}

# Weight factor applied to a tentative (cooling-off) modality's retrieval
# contribution, so a not-yet-graduated sprout nudges retrieval at reduced
# strength rather than full strength. Graduated modalities use 1.0.
TENTATIVE_WEIGHT_FACTOR = 0.5


# ---------------------------------------------------------------------------
# Catastrophic-backtracking screen
# ---------------------------------------------------------------------------

# Shapes where a quantifier is applied to a group that itself contains an
# unbounded quantifier — the classic ReDoS trigger, e.g. (a+)+, (a*)*,
# (a+)*, (.*)+ . We reject these at validation time. This is a conservative
# structural screen, not a full ReDoS analyzer; combined with the input cap
# it makes pathological blow-up impractical rather than merely unlikely.
_NESTED_QUANTIFIER = re.compile(r"\([^)]*[+*][^)]*\)\s*[+*]")


def _is_backtracking_risky(pattern: str) -> bool:
    """True if a pattern has a nested-quantifier shape we refuse to compile."""
    return bool(_NESTED_QUANTIFIER.search(pattern))


# ---------------------------------------------------------------------------
# A single sprouted modality
# ---------------------------------------------------------------------------

@dataclass
class SproutedModality:
    """
    One data-driven modality. `compiled` holds the successfully-compiled
    patterns (a pattern that fails validation is dropped, with a reason in
    `skipped`). A modality with no compiled patterns is inert (always 0.0)
    but still listed, so an operator can see it was sprouted and why it does
    nothing.
    """
    name: str
    patterns: list[str]
    threshold: float = 0.2
    match_mode: str = MATCH_FRACTION_LINES
    domain: bool = True
    status: str = STATUS_ACTIVE
    weight_factor: float = 1.0
    origin: dict = field(default_factory=dict)
    compiled: list = field(default_factory=list)
    skipped: list = field(default_factory=list)  # (pattern, reason) pairs

    def activation(self, text: str) -> float:
        """
        Score this modality on `text`, in [0, 1]. Input is truncated to
        MATCH_INPUT_CAP before any pattern runs, bounding match cost.
        """
        if not self.compiled or not text:
            return 0.0
        sample = text[:MATCH_INPUT_CAP]
        if self.match_mode == MATCH_ANY:
            return 1.0 if any(p.search(sample) for p in self.compiled) else 0.0
        if self.match_mode == MATCH_COUNT:
            total = sum(len(p.findall(sample)) for p in self.compiled)
            return min(1.0, total / 5.0)
        # MATCH_FRACTION_LINES (default): fraction of non-empty lines that
        # match any pattern. Robust to length and the natural unit for
        # "this document is mostly <mode>".
        lines = [ln for ln in sample.splitlines() if ln.strip()]
        if not lines:
            return 0.0
        hit = sum(1 for ln in lines if any(p.search(ln) for p in self.compiled))
        return hit / len(lines)

    def fires(self, text: str) -> bool:
        """Whether activation clears this modality's threshold."""
        return self.activation(text) >= self.threshold

    def effective_weight_factor(self) -> float:
        """Tentative modalities are damped; active ones full strength."""
        if self.status == STATUS_TENTATIVE:
            return min(self.weight_factor, TENTATIVE_WEIGHT_FACTOR)
        return self.weight_factor

    def to_dict(self) -> dict:
        """Serializable form (drops compiled patterns, which are derived)."""
        return {
            "name": self.name,
            "patterns": self.patterns,
            "threshold": self.threshold,
            "match_mode": self.match_mode,
            "domain": self.domain,
            "status": self.status,
            "weight_factor": self.weight_factor,
            "origin": self.origin,
        }


def _compile_patterns(patterns: list) -> tuple[list, list]:
    """
    Compile a list of pattern strings, screening each. Returns
    (compiled, skipped) where skipped is a list of (pattern, reason).
    Case-insensitive; never raises — a bad pattern is skipped, not fatal.
    """
    compiled: list = []
    skipped: list = []
    for p in patterns[:MAX_PATTERNS_PER_MODALITY]:
        if not isinstance(p, str) or not p:
            skipped.append((repr(p), "not a non-empty string"))
            continue
        if len(p) > MAX_PATTERN_LENGTH:
            skipped.append((p[:40] + "...", f"exceeds {MAX_PATTERN_LENGTH} chars"))
            continue
        if _is_backtracking_risky(p):
            skipped.append((p, "nested-quantifier backtracking risk"))
            continue
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error as e:
            skipped.append((p, f"regex error: {e}"))
    if len(patterns) > MAX_PATTERNS_PER_MODALITY:
        skipped.append(("...", f"more than {MAX_PATTERNS_PER_MODALITY} patterns; extra dropped"))
    return compiled, skipped


_VALID_NAME = re.compile(r"^[a-z][a-z0-9_]{1,39}$")


def build_modality(spec: dict) -> Optional[SproutedModality]:
    """
    Build a SproutedModality from one spec dict, or None if the spec is
    structurally invalid (bad/missing name, no patterns at all). Individual
    bad patterns are skipped rather than rejecting the whole modality, so a
    mostly-good spec still sprouts a usable detector.

    Names are constrained to a conservative identifier shape so a sprouted
    name can never collide with control characters or shadow code paths, and
    so it reads cleanly in `_meta.modalities_activated`.
    """
    if not isinstance(spec, dict):
        return None
    name = spec.get("name")
    if not isinstance(name, str) or not _VALID_NAME.match(name):
        return None
    raw_patterns = spec.get("patterns")
    if not isinstance(raw_patterns, list) or not raw_patterns:
        return None

    compiled, skipped = _compile_patterns(raw_patterns)

    threshold = spec.get("threshold", 0.2)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = 0.2
    threshold = max(0.0, min(1.0, threshold))

    match_mode = spec.get("match_mode", MATCH_FRACTION_LINES)
    if match_mode not in VALID_MATCH_MODES:
        match_mode = MATCH_FRACTION_LINES

    status = spec.get("status", STATUS_ACTIVE)
    if status not in (STATUS_ACTIVE, STATUS_TENTATIVE):
        status = STATUS_ACTIVE

    weight_factor = spec.get("weight_factor", 1.0)
    try:
        weight_factor = float(weight_factor)
    except (TypeError, ValueError):
        weight_factor = 1.0
    weight_factor = max(0.0, min(1.0, weight_factor))

    return SproutedModality(
        name=name,
        patterns=list(raw_patterns),
        threshold=threshold,
        match_mode=match_mode,
        domain=bool(spec.get("domain", True)),
        status=status,
        weight_factor=weight_factor,
        origin=spec.get("origin", {}) if isinstance(spec.get("origin"), dict) else {},
        compiled=compiled,
        skipped=skipped,
    )


# ---------------------------------------------------------------------------
# The registry: load / save a set of sprouted modalities from a JSON file
# ---------------------------------------------------------------------------

class SproutRegistry:
    """
    The set of sprouted modalities loaded from a JSON file. Built by `load`;
    queried by signals.py (to run the detectors) and retrieval.py (to know
    which sprouted modalities are domain-relevant, and at what weight). A
    missing or unreadable file yields an empty registry — sprouting is purely
    additive, so its absence simply means "no sprouted modalities," never an
    error.
    """

    def __init__(self, modalities: Optional[list] = None, path: Optional[Path] = None):
        self.modalities: list = modalities or []
        self.path = path

    @classmethod
    def load(cls, path) -> "SproutRegistry":
        """
        Load from a JSON file. Returns an empty registry if the file is
        missing or malformed (logged-shaped, never raised) — the system must
        always start, just without sprouted modalities.

        Caps the registry at MAX_SPROUTED_MODALITIES, but does so AFTER
        building and de-duplicating, and warns to stderr if the cap actually
        bit. Capping the raw list before filtering (the previous behavior)
        was both silent and subtly wrong: it could discard valid entries
        beyond position N while keeping duplicates or malformed entries
        within it. Filtering first means the cap applies to real, distinct
        modalities, and the warning tells the operator their sprout file
        outgrew the limit instead of silently losing capability.
        """
        p = Path(path)
        if not p.exists():
            return cls(modalities=[], path=p)
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            # Corrupt or unreadable — degrade to empty, don't crash boot.
            return cls(modalities=[], path=p)
        if not isinstance(raw, list):
            return cls(modalities=[], path=p)
        built: list = []
        seen: set = set()
        for entry in raw:
            m = build_modality(entry)
            if m is None or m.name in seen:
                continue
            seen.add(m.name)
            built.append(m)
        if len(built) > MAX_SPROUTED_MODALITIES:
            dropped = len(built) - MAX_SPROUTED_MODALITIES
            # Loud, but non-fatal — consistent with this module's
            # always-start contract. Keep the highest-priority slice
            # (registry order is the file's order, which is stable).
            sys.stderr.write(
                f"[sprouted_modalities] WARNING: {len(built)} valid "
                f"modalities in {p.name} exceeds MAX_SPROUTED_MODALITIES "
                f"({MAX_SPROUTED_MODALITIES}); keeping the first "
                f"{MAX_SPROUTED_MODALITIES}, dropping {dropped}. Prune the "
                f"sprout file or raise the cap to stop losing modalities.\n"
            )
            sys.stderr.flush()
            built = built[:MAX_SPROUTED_MODALITIES]
        return cls(modalities=built, path=p)

    def save(self, path=None) -> None:
        """Write the registry back to disk as JSON (sorted by name)."""
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("no path to save to")
        data = [m.to_dict() for m in sorted(self.modalities, key=lambda m: m.name)]
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def names(self) -> list:
        return [m.name for m in self.modalities]

    def domain_names(self) -> set:
        """Sprouted modalities flagged domain-relevant (participate in anchoring)."""
        return {m.name for m in self.modalities if m.domain}

    def by_name(self, name: str) -> Optional[SproutedModality]:
        for m in self.modalities:
            if m.name == name:
                return m
        return None

    def weight_factors(self) -> dict:
        """name -> effective weight factor, for retrieval's per-modality damping."""
        return {m.name: m.effective_weight_factor() for m in self.modalities}

    def as_detectors(self):
        """
        Return detector callables in the shape signals.py expects: each takes
        a SignalInput and returns a SignalHit. Built here so signals.py can
        treat sprouted modalities exactly like baked-in ones.
        """
        # Imported lazily to avoid a hard dependency cycle: signals.py imports
        # nothing from here at module load, and this import only happens if a
        # caller actually wants detectors.
        from signals import SignalHit

        def make(m: SproutedModality):
            def detector(inp) -> "SignalHit":
                act = m.activation(inp.content or "")
                return SignalHit(
                    "modality", m.name, act,
                    f"sprouted modality {m.name}: activation {act:.2f}",
                    {"sprouted": True, "status": m.status,
                     "domain": m.domain, "threshold": m.threshold},
                )
            detector.__name__ = f"sprouted_{m.name}"
            return detector

        return [make(m) for m in self.modalities]
