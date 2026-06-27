"""
metadata — write-time metadata embedded in record content.

Records carry a small `_meta` block inside their `content` dict that
captures the architectural primitives:

    {
      "_meta": {
        "schema_version":  3,
        "source":          "user" | "assistant" | "system" | "tool",
        "salience":        float in [0, 1],
        "confidence":      float in [0, 1],
        "supersedes":      int | null,   # record index this one corrects
        "epistemic_class": "known" | "inferred" | "speculative"
                           | "disputed" | "user_context",
        "exposure":        "private" | "summary" | "shared"
                           | "public" | "quarantine",
        "poq":             { ...Proof-of-Quality block... } | absent
      },
      ...the rest of the content (text, filename, etc.)
    }

Why content rather than top-level Record fields? The chain in chain.py is
deliberately content-agnostic — it signs and links arbitrary JSON. Putting
metadata inside content means:
  - The cryptographic core in chain.py doesn't change.
  - Old records (without _meta) keep verifying — we add safe defaults
    on read instead of fabricating fields on disk.
  - Schema migrations are non-destructive: a v1 record stays a v1 record
    forever; a later reader supplies sensible defaults.

Source is the most important field here. "The user said it" and "I inferred
it" and "a reflection concluded it" are different epistemic objects, and
conflating them is the failure mode reflection-of-reflection produces.

Schema history:
  v1 — no `_meta` block at all (pre-metadata records).
  v2 — `_meta` with source/salience/confidence/supersedes.
  v3 — adds `epistemic_class`, `exposure`, and an optional `poq` block.
       `epistemic_class` records *how the writer knows* the content;
       `exposure` records *who may see it* (the protected-zone primitive);
       `poq` carries the Proof-of-Quality score assigned before commit.

The v3 additions follow the same non-destructive rule as v2: a v1 or v2
record read by v3 code gets type-appropriate defaults synthesized in
memory, and is never rewritten on disk. `read_meta` is the single place
that upgrade happens.
"""

from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

# v1 = no _meta; v2 = _meta with source/salience/confidence/supersedes;
# v3 = adds epistemic_class, exposure, and an optional poq block. When this
# changes, the reader in `read_meta()` is responsible for upgrading old
# records in memory — never on disk.
CURRENT_SCHEMA_VERSION = 3


# ---------------------------------------------------------------------------
# Source — the single most important metadata field
# ---------------------------------------------------------------------------

# Where this record's content originated. Determines how much weight it
# should carry as evidence about the world.
SOURCE_USER = "user"            # something the user said, captured verbatim
SOURCE_ASSISTANT = "assistant"  # something the agent said or concluded
SOURCE_SYSTEM = "system"        # operator config — system prompt, genesis
SOURCE_TOOL = "tool"            # output from a tool call (file ingest, etc.)
SOURCE_PEER_AGENT = "peer_agent"  # another agent's memory, imported via a
                                  # signed Experience Capsule (capsule.py).
                                  # Distinct from `tool`: a tool is something
                                  # THIS agent ran; a peer_agent record is a
                                  # claim ANOTHER agent authored and signed.
                                  # Build spec calls this provenance
                                  # `imported_capsule`; we use the broader
                                  # `peer_agent` so any future agent-to-agent
                                  # provenance shares one source value.

VALID_SOURCES = {SOURCE_USER, SOURCE_ASSISTANT, SOURCE_SYSTEM, SOURCE_TOOL,
                 SOURCE_PEER_AGENT}


# Default source by record type, used when a record doesn't declare one.
# "observation" is always the user; "response" and "reflection" are always
# the assistant; system_prompt/genesis are operator-set.
DEFAULT_SOURCE_BY_TYPE = {
    "observation":   SOURCE_USER,
    "response":      SOURCE_ASSISTANT,
    # A resolution records the USER approving/rejecting a pending
    # operation — their decision, their source.
    "resolution":    SOURCE_USER,
    "reflection":    SOURCE_ASSISTANT,
    "revision":      SOURCE_ASSISTANT,
    "system_prompt": SOURCE_SYSTEM,
    "genesis":       SOURCE_SYSTEM,
    "file":          SOURCE_TOOL,
    # v1.2 record types — both are the agent's own consolidated output.
    # A "principle" is an extracted durable rule; a "proposal" is a
    # Cambium suggestion (new skill / modality / sense) awaiting review.
    "principle":     SOURCE_ASSISTANT,
    "proposal":      SOURCE_ASSISTANT,
    # Recurrence and escalation records for proposals (v1.2 recurrence
    # tracking). Both are the agent's own bookkeeping output.
    "proposal_recurrence": SOURCE_ASSISTANT,
    "proposal_status":     SOURCE_ASSISTANT,
    "sprout_status":       SOURCE_ASSISTANT,
    # Imported Experience Capsule records (capsule.py). Provenance is another
    # agent's signed claims; the origin identity is preserved inside the
    # record content, and the source is `peer_agent` so retrieval and PoQ can
    # treat imported memory as third-party by source as well as by epistemic
    # class.
    "imported_capsule":    SOURCE_PEER_AGENT,
    # Immune self-defense records (immune.py). `recovery` re-anchors the clean
    # lineage after a compromise; `quarantine_marker` records a molted range.
    # Both are the agent's own defensive bookkeeping (system-authored).
    "recovery":            SOURCE_SYSTEM,
    "quarantine_marker":   SOURCE_SYSTEM,
    # Continuum long-horizon-task records (continuum.py). Tool-derived work
    # ledger: a `task_open` opens a task, each `continuum` block ingests one
    # data-height chunk carrying a full state refresh.
    "task_open":           SOURCE_TOOL,
    "continuum":           SOURCE_TOOL,
    # Chronosynaptic collapse (chronosynaptic.py): the agent's own synthesis of
    # the highest-truth perspective path.
    "synthesis":           SOURCE_ASSISTANT,
    # Cambium faculty growth (faculties.py): the agent's own endogenous upgrades.
    "faculty":             SOURCE_ASSISTANT,
    "faculty_recur":       SOURCE_ASSISTANT,
    "promotion":           SOURCE_ASSISTANT,
}


# ---------------------------------------------------------------------------
# Epistemic class — how the writer knows the content
# ---------------------------------------------------------------------------

# Distinct from `source` (who produced it). Source answers "who said this";
# epistemic_class answers "what kind of knowing is this." A user statement
# is `user_context`; a reflection's conclusion is `inferred`; a verified
# file is `known`. The retriever and PoQ both read this.
EPISTEMIC_KNOWN = "known"               # verified / ground-truth
EPISTEMIC_INFERRED = "inferred"         # the agent reasoned to it
EPISTEMIC_SPECULATIVE = "speculative"   # a guess, flagged as such
EPISTEMIC_DISPUTED = "disputed"         # known to conflict with another record
EPISTEMIC_USER_CONTEXT = "user_context" # the user asserted it

VALID_EPISTEMIC_CLASSES = {
    EPISTEMIC_KNOWN, EPISTEMIC_INFERRED, EPISTEMIC_SPECULATIVE,
    EPISTEMIC_DISPUTED, EPISTEMIC_USER_CONTEXT,
}

# Default epistemic class by record type. Files and tool output are `known`
# (sha256-verified bytes); observations are `user_context`; the agent's own
# responses/reflections are `inferred`; genesis/system_prompt are `known`
# (operator declared them).
DEFAULT_EPISTEMIC_BY_TYPE = {
    "observation":   EPISTEMIC_USER_CONTEXT,
    "response":      EPISTEMIC_INFERRED,
    "reflection":    EPISTEMIC_INFERRED,
    "revision":      EPISTEMIC_INFERRED,
    "principle":     EPISTEMIC_INFERRED,
    "proposal":      EPISTEMIC_SPECULATIVE,
    "proposal_recurrence": EPISTEMIC_INFERRED,
    "proposal_status":     EPISTEMIC_INFERRED,
    "sprout_status":       EPISTEMIC_INFERRED,
    "system_prompt": EPISTEMIC_KNOWN,
    "genesis":       EPISTEMIC_KNOWN,
    "file":          EPISTEMIC_KNOWN,
    # Imported claims are never the importer's ground truth — default to
    # inferred (capsule.py demotes stronger origin classes to this ceiling).
    "imported_capsule": EPISTEMIC_INFERRED,
    # Immune records describe a defensive event the agent reasoned to.
    "recovery":          EPISTEMIC_INFERRED,
    "quarantine_marker": EPISTEMIC_INFERRED,
    # Continuum blocks ingest verified source/data chunks (sha-anchored).
    "task_open":         EPISTEMIC_KNOWN,
    # A tool_use record is a fact about what the agent executed (sanitized
    # args + result summary) — known, not inferred. (No longer written as
    # of v1.4 — kept so chains sealed before the change still read.)
    "tool_use":          EPISTEMIC_KNOWN,
    # A resolution is a fact about a USER decision (a pending operation
    # approved or rejected) — known ground truth, never inference.
    "resolution":        EPISTEMIC_KNOWN,
    "continuum":         EPISTEMIC_KNOWN,
    # A synthesis is reasoned conclusion, not ground truth.
    "synthesis":         EPISTEMIC_INFERRED,
    # Faculty growth is the agent's own inferred self-upgrade.
    "faculty":           EPISTEMIC_INFERRED,
    "faculty_recur":     EPISTEMIC_INFERRED,
    "promotion":         EPISTEMIC_INFERRED,
}


# ---------------------------------------------------------------------------
# Exposure — who may see this record (the protected-zone primitive)
# ---------------------------------------------------------------------------

# Exposure is a *visibility* tag, not an access-control enforcement point on
# its own — protected_zones.py reads it to decide what the agent may surface
# and what ordinary input may overwrite. Kept here because it is pure schema.
EXPOSURE_PRIVATE = "private"        # ordinary memory, local only
EXPOSURE_SUMMARY = "summary"        # may be shared only in summarized form
EXPOSURE_SHARED = "shared"          # shareable with a paired peer
EXPOSURE_PUBLIC = "public"          # safe to expose publicly
EXPOSURE_QUARANTINE = "quarantine"  # untrusted; never feeds belief/retrieval

VALID_EXPOSURES = {
    EXPOSURE_PRIVATE, EXPOSURE_SUMMARY, EXPOSURE_SHARED,
    EXPOSURE_PUBLIC, EXPOSURE_QUARANTINE,
}

# Default exposure by record type. Genesis and system_prompt are identity
# records — `summary` (readable in summarized form, not silently editable).
# Everything else defaults to `private`. Quarantine is never a default; it
# is only ever set explicitly when input is judged untrusted.
DEFAULT_EXPOSURE_BY_TYPE = {
    "genesis":       EXPOSURE_SUMMARY,
    "system_prompt": EXPOSURE_SUMMARY,
    # A recovery record is part of the agent's identity history (private, but
    # readable); a quarantine_marker is, by definition, untrusted content.
    "recovery":          EXPOSURE_PRIVATE,
    "quarantine_marker": EXPOSURE_QUARANTINE,
}


# ---------------------------------------------------------------------------
# Salience — write-time importance
# ---------------------------------------------------------------------------

# Default salience by record type. The intent here matches the V1.1
# distinction: reflections and revisions represent the agent's own judgment
# about what mattered, so they get high default salience. Genesis and
# system_prompt are foundational identity records. Observations and responses
# are the baseline — they're evidence of what was said, but most of what's
# said is small talk, not signal. Files are typically deliberate uploads,
# so they sit above conversational baseline.
#
# These are *defaults*. A specific record can override at write time —
# e.g. an observation that's clearly a major life event ("I just got a
# new job") deserves higher salience than the default 0.4.
DEFAULT_SALIENCE_BY_TYPE = {
    "principle":     0.90,
    "proposal_status": 0.85,
    "sprout_status": 0.80,
    "reflection":    0.85,
    "revision":      0.80,
    "genesis":       0.75,
    "proposal":      0.65,
    "file":          0.60,
    "proposal_recurrence": 0.45,
    "system_prompt": 0.55,
    "observation":   0.40,
    "response":      0.40,
    "imported_capsule": 0.35,
    # A recovery is a significant self-event; a quarantine_marker is bookkeeping.
    "recovery":          0.85,
    "quarantine_marker": 0.50,
    # Continuum blocks are a derived work-ledger — mid baseline salience.
    "task_open":         0.55,
    "continuum":         0.55,
    # A resolution settles whether proposed work actually happened — the
    # very next turns need it (it kills the stale-"pending" confabulation),
    # so it sits above conversational baseline.
    "resolution":        0.60,
    # A synthesis is a deliberate, high-value reasoning artifact.
    "synthesis":         0.80,
    # Faculty growth records are durable capability upgrades — high salience.
    "faculty":           0.75,
    "faculty_recur":     0.60,
    "promotion":         0.85,
}


# ---------------------------------------------------------------------------
# Per-kind half-lives for retrieval recency scoring (in days).
#
# Replaces the old uniform recency_weight blend with kind-aware decay.
# A record's recency contribution is 0.5 ** (age_days / half_life_days).
#
# Calibration:
#   - genesis / system_prompt: effectively no decay — they define identity
#   - reflection: long memory, the agent's own consolidated judgment
#   - revision: long-lived corrections to prior records
#   - file: medium — uploaded artifacts stay relevant for a while
#   - observation / response: short — most conversation is episodic
#
# These are tunable. The point is that "user said their name" and "user
# mentioned the weather" should not decay at the same rate.
# ---------------------------------------------------------------------------

# 1e6 days ~= 2700 years. Effectively no decay without using inf.
NO_DECAY_DAYS = 1_000_000

DEFAULT_HALF_LIFE_DAYS_BY_TYPE = {
    "genesis":       NO_DECAY_DAYS,
    "system_prompt": NO_DECAY_DAYS,
    "principle":     NO_DECAY_DAYS,  # extracted rules are durable identity
    "reflection":    75.0,     # ~2.5 months — a reflection orients RECENT
    #                            work; a long half-life let old reflections
    #                            accumulate as persistent retrieval magnets.
    #                            Salience stays high (0.85) so the lone
    #                            orienting reflection still surfaces; the
    #                            shorter half-life keeps it a recent-context
    #                            minority rather than a standing one.
    "revision":      365.0,    # ~1 year — corrections should hold
    "proposal":      120.0,    # ~4 months — a pending suggestion goes stale
    "proposal_status": 365.0,  # ~1 year — an escalation should hold
    "sprout_status": 365.0,    # ~1 year — sprout audit trail should hold
    "proposal_recurrence": 120.0,
    "file":          90.0,     # ~3 months
    "observation":   14.0,     # ~2 weeks
    "response":      14.0,
    "imported_capsule": 120.0,  # ~4 months — attributed third-party memory
    # Immune records are durable identity — a recovery should never decay out.
    "recovery":          NO_DECAY_DAYS,
    "quarantine_marker": NO_DECAY_DAYS,
    # Continuum blocks are task-scoped work; a few months is plenty.
    "task_open":         120.0,
    "continuum":         120.0,
    # A synthesis is consolidated judgment — long memory, like a reflection.
    "synthesis":         180.0,
    # Faculty upgrades are part of identity — they should not decay out.
    "faculty":           NO_DECAY_DAYS,
    "faculty_recur":     NO_DECAY_DAYS,
    "promotion":         NO_DECAY_DAYS,
}


# ---------------------------------------------------------------------------
# Read path — extract metadata from a record's content with safe defaults
# ---------------------------------------------------------------------------

class RecordMeta:
    """
    Resolved metadata for a record. Constructed by `read_meta(record)`,
    which fills missing fields with defaults appropriate to the record's
    type. This is the non-destructive migration rule: old records on disk
    are unchanged; readers supply defaults at read time.
    """

    __slots__ = ("schema_version", "source", "salience", "confidence",
                 "supersedes", "epistemic_class", "exposure", "poq",
                 "truncated", "tool_budget_exhausted",
                 "modalities_activated", "senses_activated",
                 "is_default")

    def __init__(
        self,
        schema_version: int,
        source: str,
        salience: float,
        confidence: float,
        supersedes: Optional[int],
        epistemic_class: str,
        exposure: str,
        poq: Optional[dict],
        truncated: bool,
        modalities_activated: list,
        senses_activated: list,
        is_default: bool,
        tool_budget_exhausted: bool = False,
    ):
        self.schema_version = schema_version
        self.source = source
        self.salience = salience
        self.confidence = confidence
        self.supersedes = supersedes
        # v3 fields. `epistemic_class` says how the content is known;
        # `exposure` says who may see it; `poq` is the Proof-of-Quality
        # block (None for records written before PoQ existed, or for
        # record types PoQ doesn't gate).
        self.epistemic_class = epistemic_class
        self.exposure = exposure
        self.poq = poq
        # True when the response was cut off at the model's max_tokens
        # limit rather than ending naturally. Set on response records
        # only; always False for other record types. Drives the
        # continuation-detection logic in `Agent._format_prompt` so
        # "continue" against a truncated response is unambiguous.
        self.truncated = truncated
        # True when the turn's TOOL round budget ran out mid-task — the
        # text ended cleanly (often a progress checkpoint), but the work
        # did not. Distinct from `truncated` (text cut mid-sentence):
        # "continue" against a budget-exhausted response means "resume
        # the task with a fresh budget", not "finish the sentence".
        self.tool_budget_exhausted = tool_budget_exhausted
        # The modality detectors (signals.py) that fired with non-trivial
        # activation when this record's content was analyzed by PoQ. For a
        # response record, this is "which capabilities the agent actually
        # used to produce this answer" — the data layer retrieval will read
        # later. Absent on records written before this field existed, and on
        # record types PoQ doesn't gate; read_meta defaults it to [].
        self.modalities_activated = modalities_activated
        # The sense detectors (signals.py) that fired with non-trivial
        # activation when this record's content was analyzed. Where modalities
        # answer "what kind of work produced this," senses answer "how did
        # this feel" — uncertainty, insight markers, emotional contour,
        # cognitive weather. Recorded on a record's _meta so the agent reads
        # back felt qualities along with the content. Deliberately NOT a
        # retrieval input: matching feeling-to-feeling would surface memories
        # by mood, which is closer to rumination than recall. Absent on
        # records written before this field existed; read_meta defaults to [].
        self.senses_activated = senses_activated
        # True if this metadata was synthesized from defaults (i.e. the
        # record predates the current schema). Useful for diagnostics;
        # never an excuse to rewrite the record on disk.
        self.is_default = is_default

    def __repr__(self) -> str:
        flag = " (default)" if self.is_default else ""
        return (f"RecordMeta(v{self.schema_version}, source={self.source}, "
                f"sal={self.salience:.2f}, conf={self.confidence:.2f}, "
                f"epi={self.epistemic_class}, exp={self.exposure}{flag})")


def read_meta(record) -> RecordMeta:
    """
    Read the metadata block from a record's content, falling back to
    type-appropriate defaults for any missing fields. Works on records of
    every schema version: v1 (no `_meta`), v2 (`_meta` without the v3
    fields), and v3 (`_meta` with everything).

    `record` is duck-typed: anything with `.type` and `.content` works.

    The v3 fields (`epistemic_class`, `exposure`, `poq`) are filled from
    per-type defaults when absent, so a v1 or v2 record reads cleanly
    under v3 code without ever being modified on disk.
    """
    rec_type = record.type
    content = record.content if isinstance(record.content, dict) else {}
    meta = content.get("_meta") if isinstance(content, dict) else None

    if meta is None:
        # v1 record — synthesize every field from type alone.
        return RecordMeta(
            schema_version=1,
            source=DEFAULT_SOURCE_BY_TYPE.get(rec_type, SOURCE_ASSISTANT),
            salience=DEFAULT_SALIENCE_BY_TYPE.get(rec_type, 0.4),
            confidence=1.0 if rec_type in ("observation", "system_prompt", "genesis", "file") else 0.8,
            supersedes=_legacy_supersedes(content, rec_type),
            epistemic_class=DEFAULT_EPISTEMIC_BY_TYPE.get(rec_type, EPISTEMIC_INFERRED),
            exposure=DEFAULT_EXPOSURE_BY_TYPE.get(rec_type, EXPOSURE_PRIVATE),
            poq=None,
            truncated=False,  # v1 records predate the truncation flag
            tool_budget_exhausted=False,  # likewise
            modalities_activated=[],  # v1 records predate this field
            senses_activated=[],      # v1 records predate this field too
            is_default=True,
        )

    # v2 or v3 record — fill any individually-missing fields with defaults.
    # A v2 record simply won't have epistemic_class/exposure/poq keys; the
    # `.get(...)` defaults below upgrade it in memory.
    poq = meta.get("poq")
    if not isinstance(poq, dict):
        poq = None
    return RecordMeta(
        schema_version=int(meta.get("schema_version", CURRENT_SCHEMA_VERSION)),
        source=meta.get("source", DEFAULT_SOURCE_BY_TYPE.get(rec_type, SOURCE_ASSISTANT)),
        salience=float(meta.get("salience", DEFAULT_SALIENCE_BY_TYPE.get(rec_type, 0.4))),
        confidence=float(meta.get("confidence", 1.0)),
        supersedes=meta.get("supersedes", _legacy_supersedes(content, rec_type)),
        epistemic_class=meta.get(
            "epistemic_class",
            DEFAULT_EPISTEMIC_BY_TYPE.get(rec_type, EPISTEMIC_INFERRED),
        ),
        exposure=meta.get(
            "exposure",
            DEFAULT_EXPOSURE_BY_TYPE.get(rec_type, EXPOSURE_PRIVATE),
        ),
        poq=poq,
        truncated=bool(meta.get("truncated", False)),
        tool_budget_exhausted=bool(meta.get("tool_budget_exhausted", False)),
        modalities_activated=(
            list(meta["modalities_activated"])
            if isinstance(meta.get("modalities_activated"), list)
            else []
        ),
        senses_activated=(
            list(meta["senses_activated"])
            if isinstance(meta.get("senses_activated"), list)
            else []
        ),
        is_default=False,
    )


def _legacy_supersedes(content: Any, rec_type: str) -> Optional[int]:
    """Pull supersedes from the pre-v2 location for revision records."""
    if rec_type == "revision" and isinstance(content, dict):
        v = content.get("revises_index")
        if isinstance(v, int):
            return v
    return None


# ---------------------------------------------------------------------------
# Write path — build the _meta block at append time
# ---------------------------------------------------------------------------

def build_meta(
    rec_type: str,
    source: Optional[str] = None,
    salience: Optional[float] = None,
    confidence: Optional[float] = None,
    supersedes: Optional[int] = None,
    epistemic_class: Optional[str] = None,
    exposure: Optional[str] = None,
    poq: Optional[dict] = None,
    truncated: Optional[bool] = None,
    tool_budget_exhausted: Optional[bool] = None,
    modalities_activated: Optional[list] = None,
    senses_activated: Optional[list] = None,
) -> dict:
    """
    Build a _meta dict for a new record. Any field left as None is filled
    from type-based defaults. Caller passes this through to chain.append
    inside the content dict under the "_meta" key.

    Validation is intentionally light — the chain itself doesn't care
    about metadata semantics; this is a read-side convention. We do
    range-clamp salience/confidence so a buggy caller can't poison
    retrieval with negative or >1 values, and we reject unknown enum
    values for source/epistemic_class/exposure so typos fail loudly at
    write time rather than silently at read time.

    `poq`, when given, is stored verbatim — it is produced by poq.py's
    `PoQResult.to_meta()` and is opaque to this module.
    """
    resolved_source = source or DEFAULT_SOURCE_BY_TYPE.get(rec_type, SOURCE_ASSISTANT)
    if resolved_source not in VALID_SOURCES:
        raise ValueError(f"unknown source: {resolved_source!r}")

    resolved_epistemic = epistemic_class or DEFAULT_EPISTEMIC_BY_TYPE.get(
        rec_type, EPISTEMIC_INFERRED
    )
    if resolved_epistemic not in VALID_EPISTEMIC_CLASSES:
        raise ValueError(f"unknown epistemic_class: {resolved_epistemic!r}")

    resolved_exposure = exposure or DEFAULT_EXPOSURE_BY_TYPE.get(
        rec_type, EXPOSURE_PRIVATE
    )
    if resolved_exposure not in VALID_EXPOSURES:
        raise ValueError(f"unknown exposure: {resolved_exposure!r}")

    resolved_salience = (
        salience if salience is not None
        else DEFAULT_SALIENCE_BY_TYPE.get(rec_type, 0.4)
    )
    resolved_confidence = confidence if confidence is not None else 1.0

    out = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "source": resolved_source,
        "salience": _clamp(resolved_salience),
        "confidence": _clamp(resolved_confidence),
        "epistemic_class": resolved_epistemic,
        "exposure": resolved_exposure,
    }
    if supersedes is not None:
        out["supersedes"] = supersedes
    if poq is not None:
        out["poq"] = poq
    if truncated:
        # Only emit `truncated` when True. Defaulting to absent (rather
        # than `False`) keeps the canonical JSON of completed responses
        # identical to what earlier versions wrote, so existing chains
        # don't see spurious content-hash changes on rebuild.
        out["truncated"] = True
    if tool_budget_exhausted:
        # Same absent-unless-True rule as `truncated`, same reason.
        out["tool_budget_exhausted"] = True
    if modalities_activated:
        # Only emit when non-empty, for the same reason as `truncated`:
        # a record that activated no modalities (or a record type PoQ
        # doesn't scan) writes no field, so its canonical JSON matches
        # what earlier versions produced. read_meta defaults absence to
        # []. Store a copy of the names, sorted for stable on-disk JSON
        # (the analyzer's ordering is not guaranteed) — and so two
        # records that fired the same set hash identically.
        out["modalities_activated"] = sorted(str(m) for m in modalities_activated)
    if senses_activated:
        # Same emit-only-when-non-empty discipline as modalities_activated,
        # for the same canonical-JSON / content-hash reason. Sorted for
        # stable on-disk ordering — two records that fired the same set of
        # senses hash identically.
        out["senses_activated"] = sorted(str(s) for s in senses_activated)
    return out


def _clamp(x: float) -> float:
    """
    Clamp to [0.0, 1.0] AND round to 6 decimal places.

    Rounding is what makes `canonical_json` deterministic across
    Python versions: an unrounded computed float (e.g.
    `default_salience * 0.5` from `protected_zones.salience_for_commit`)
    could str() differently between runtimes, producing different
    content hashes for what should be the same record. 6 decimal
    places is well past any signal a salience or confidence value
    actually carries (the values are subjective judgments, not
    measurements), and matches the precision PoQ dimensions ship at.

    All floats reaching `_meta` go through here. If you add a new
    float field to _meta, route it through _clamp too.
    """
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return round(float(x), 6)


# ---------------------------------------------------------------------------
# Convenience: half-life lookup with safe fallback
# ---------------------------------------------------------------------------

def half_life_days(rec_type: str) -> float:
    """Per-type half-life used by retrieval recency scoring."""
    return DEFAULT_HALF_LIFE_DAYS_BY_TYPE.get(rec_type, 30.0)
