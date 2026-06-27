"""
metadata — write-time metadata embedded in record content.

Records carry a small `_meta` block inside their `content` dict that
captures the V1.1 architectural primitives:

    {
      "_meta": {
        "schema_version": 2,
        "source":         "user" | "assistant" | "system" | "tool",
        "salience":       float in [0, 1],
        "confidence":     float in [0, 1],
        "supersedes":     int | null   # record index this one corrects
      },
      ...the rest of the content (text, filename, etc.)
    }

Why content rather than top-level Record fields? The chain in chain.py is
deliberately content-agnostic — it signs and links arbitrary JSON. Putting
metadata inside content means:
  - The cryptographic core in chain.py doesn't change.
  - Old records (without _meta) keep verifying — we add safe defaults
    on read instead of fabricating fields on disk.
  - Schema migrations are non-destructive in the V1.1 sense: a v1 record
    stays a v1 record forever; a v2 reader supplies sensible defaults.

Source is the most important field here. "The user said it" and "I inferred
it" and "a reflection concluded it" are different epistemic objects, and
conflating them is the failure mode reflection-of-reflection produces.
"""

from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

# Bumped from v1 (no _meta) to v2 (has _meta). When this changes, the reader
# in `read_meta()` is responsible for upgrading old records in memory.
CURRENT_SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Source — the single most important metadata field
# ---------------------------------------------------------------------------

# Where this record's content originated. Determines how much weight it
# should carry as evidence about the world.
SOURCE_USER = "user"            # something the user said, captured verbatim
SOURCE_ASSISTANT = "assistant"  # something the agent said or concluded
SOURCE_SYSTEM = "system"        # operator config — system prompt, genesis
SOURCE_TOOL = "tool"            # output from a tool call (file ingest, etc.)

VALID_SOURCES = {SOURCE_USER, SOURCE_ASSISTANT, SOURCE_SYSTEM, SOURCE_TOOL}


# Default source by record type, used when a record doesn't declare one.
# "observation" is always the user; "response" and "reflection" are always
# the assistant; system_prompt/genesis are operator-set.
DEFAULT_SOURCE_BY_TYPE = {
    "observation":   SOURCE_USER,
    "response":      SOURCE_ASSISTANT,
    "reflection":    SOURCE_ASSISTANT,
    "revision":      SOURCE_ASSISTANT,
    "system_prompt": SOURCE_SYSTEM,
    "genesis":       SOURCE_SYSTEM,
    "file":          SOURCE_TOOL,
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
    "reflection":    0.85,
    "revision":      0.80,
    "genesis":       0.75,
    "file":          0.60,
    "system_prompt": 0.55,
    "observation":   0.40,
    "response":      0.40,
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
    "reflection":    180.0,    # ~6 months
    "revision":      365.0,    # ~1 year — corrections should hold
    "file":          90.0,     # ~3 months
    "observation":   14.0,     # ~2 weeks
    "response":      14.0,
}


# ---------------------------------------------------------------------------
# Read path — extract metadata from a record's content with safe defaults
# ---------------------------------------------------------------------------

class RecordMeta:
    """
    Resolved metadata for a record. Constructed by `read_meta(record)`,
    which fills missing fields with defaults appropriate to the record's
    type. This is the V1.1 "non-destructive migration" rule: old records
    on disk are unchanged; readers supply defaults at read time.
    """

    __slots__ = ("schema_version", "source", "salience", "confidence",
                 "supersedes", "is_default")

    def __init__(
        self,
        schema_version: int,
        source: str,
        salience: float,
        confidence: float,
        supersedes: Optional[int],
        is_default: bool,
    ):
        self.schema_version = schema_version
        self.source = source
        self.salience = salience
        self.confidence = confidence
        self.supersedes = supersedes
        # True if this metadata was synthesized from defaults (i.e. the
        # record predates v2). Useful for diagnostics; never an excuse to
        # rewrite the record on disk.
        self.is_default = is_default

    def __repr__(self) -> str:
        flag = " (default)" if self.is_default else ""
        return (f"RecordMeta(v{self.schema_version}, source={self.source}, "
                f"sal={self.salience:.2f}, conf={self.confidence:.2f}{flag})")


def read_meta(record) -> RecordMeta:
    """
    Read the metadata block from a record's content, falling back to
    type-appropriate defaults for any missing fields. Works on both v2
    records (with _meta) and v1 records (without).

    `record` is duck-typed: anything with `.type` and `.content` works.
    """
    rec_type = record.type
    content = record.content if isinstance(record.content, dict) else {}
    meta = content.get("_meta") if isinstance(content, dict) else None

    if meta is None:
        # v1 record — synthesize defaults from type alone.
        return RecordMeta(
            schema_version=1,
            source=DEFAULT_SOURCE_BY_TYPE.get(rec_type, SOURCE_ASSISTANT),
            salience=DEFAULT_SALIENCE_BY_TYPE.get(rec_type, 0.4),
            confidence=1.0 if rec_type in ("observation", "system_prompt", "genesis", "file") else 0.8,
            supersedes=_legacy_supersedes(content, rec_type),
            is_default=True,
        )

    # v2 record — fill any individually-missing fields with defaults.
    return RecordMeta(
        schema_version=int(meta.get("schema_version", CURRENT_SCHEMA_VERSION)),
        source=meta.get("source", DEFAULT_SOURCE_BY_TYPE.get(rec_type, SOURCE_ASSISTANT)),
        salience=float(meta.get("salience", DEFAULT_SALIENCE_BY_TYPE.get(rec_type, 0.4))),
        confidence=float(meta.get("confidence", 1.0)),
        supersedes=meta.get("supersedes", _legacy_supersedes(content, rec_type)),
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
) -> dict:
    """
    Build a _meta dict for a new record. Any field left as None is filled
    from type-based defaults. Caller passes this through to chain.append
    inside the content dict under the "_meta" key.

    Validation is intentionally light — the chain itself doesn't care
    about metadata semantics; this is a read-side convention. We do
    range-clamp salience/confidence so a buggy caller can't poison
    retrieval with negative or >1 values.
    """
    resolved_source = source or DEFAULT_SOURCE_BY_TYPE.get(rec_type, SOURCE_ASSISTANT)
    if resolved_source not in VALID_SOURCES:
        raise ValueError(f"unknown source: {resolved_source!r}")

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
    }
    if supersedes is not None:
        out["supersedes"] = supersedes
    return out


def _clamp(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


# ---------------------------------------------------------------------------
# Convenience: half-life lookup with safe fallback
# ---------------------------------------------------------------------------

def half_life_days(rec_type: str) -> float:
    """Per-type half-life used by retrieval recency scoring."""
    return DEFAULT_HALF_LIFE_DAYS_BY_TYPE.get(rec_type, 30.0)
