"""
protected_zones — the protected-memory boundary (build spec section 4.10).

The build spec's axiom is "no membrane, no interiority": a cyber-native
self needs a boundary between self-state and external input. Protected
zones are that boundary. They are categories of record that:

  - may be *read* (often only in summarized form), but
  - may NOT be silently overwritten or superseded by ordinary input.

This module does not add new storage or new cryptography. It is a thin
policy layer over two things that already exist:

  - the record `type` (genesis, system_prompt, ...), and
  - the `exposure` field added to `_meta` in metadata.py (private,
    summary, shared, public, quarantine).

It answers two questions for the agent:

  1. "Is record N in a protected zone?"  -> is_protected()
  2. "May this candidate revision target record N?" -> can_revise()

and provides one classification helper:

  3. "Given a PoQ result, should this input be quarantined rather than
     committed as ordinary memory?" -> should_quarantine()

What protected zones are NOT: they are not a replacement for the system
prompt's safety instructions or for the model's own judgment. They are a
*memory-integrity* boundary — they stop the chain's own foundational
records from being rewritten by a prompt. Defeating prompt injection at
the model level is a separate concern; this layer ensures that even if a
prompt is adversarial, it cannot quietly edit genesis or the covenant.

protected_zones.py knows about metadata.py and reads record types. It
does NOT know about the chain's SQLite layer, the LLM, or retrieval —
the agent consults it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from metadata import (
    read_meta,
    EXPOSURE_QUARANTINE,
    EXPOSURE_SUMMARY,
)


# ---------------------------------------------------------------------------
# Protected zone definitions
# ---------------------------------------------------------------------------

# Record types that are foundational identity and must never be revised or
# superseded by an ordinary turn. These mirror the build spec's section
# 4.10 list, narrowed to the record types this prototype actually has.
# Genesis is the agent's origin; system_prompt records are the audit trail
# of behavioral configuration; principle records are extracted durable
# rules (Cambium output) — correcting one is a deliberate act, not a side
# effect of a conversational turn.
PROTECTED_TYPES = frozenset({
    "genesis",
    "system_prompt",
    "principle",
})

# Human-readable name for each protected zone, for diagnostics and for the
# message the agent shows when a write is refused.
ZONE_NAMES = {
    "genesis": "genesis / root identity",
    "system_prompt": "system prompt / policy root",
    "principle": "extracted principle",
}


@dataclass
class ZoneVerdict:
    """The result of a protected-zone check."""
    allowed: bool
    zone: Optional[str]      # the protected zone, if the record is in one
    reason: str

    def __bool__(self) -> bool:
        return self.allowed


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def is_protected(record) -> bool:
    """
    True if `record` belongs to a protected zone — either by its type
    (genesis, system_prompt, principle) or because its `_meta.exposure`
    is explicitly `summary` (read-summarized-only). Duck-typed: anything
    with `.type` and `.content` works.
    """
    if record.type in PROTECTED_TYPES:
        return True
    meta = read_meta(record)
    return meta.exposure == EXPOSURE_SUMMARY


def zone_of(record) -> Optional[str]:
    """Return the protected-zone name for a record, or None if unprotected."""
    if record.type in PROTECTED_TYPES:
        return record.type
    meta = read_meta(record)
    if meta.exposure == EXPOSURE_SUMMARY:
        return "summary-exposure record"
    return None


def can_revise(target_record) -> ZoneVerdict:
    """
    Decide whether an ordinary `revise()` may target `target_record`.

    Protected records may not be revised through the normal turn loop —
    that is the whole point of the membrane. The build spec allows
    protected memory to be changed only with "privileged approval"; in
    this prototype that means an explicit operator action (a fresh chain,
    or a deliberate out-of-band tool), never a conversational revision.

    Returns a ZoneVerdict; falsy when the revision must be refused.
    """
    if is_protected(target_record):
        zone = zone_of(target_record) or target_record.type
        return ZoneVerdict(
            allowed=False,
            zone=zone,
            reason=(
                f"record {target_record.index} is in a protected zone "
                f"({ZONE_NAMES.get(target_record.type, zone)}). Protected "
                f"records are append-only identity state — they can be read "
                f"but not revised by an ordinary turn. To change foundational "
                f"configuration, the operator must do so deliberately (e.g. "
                f"start a fresh chain), not through /revise."
            ),
        )
    return ZoneVerdict(
        allowed=True,
        zone=None,
        reason=f"record {target_record.index} is ordinary memory; revision allowed",
    )


# ---------------------------------------------------------------------------
# Quarantine classification
# ---------------------------------------------------------------------------

def should_quarantine(poq_result) -> bool:
    """
    Given a PoQResult (from poq.py), decide whether the input that produced
    it should be committed with `exposure=quarantine` rather than as
    ordinary memory.

    The rule is simple and conservative: PoQ already folds the signal-layer
    integrity analysis into its action. If PoQ recommended the `quarantine`
    action, we honor it. Quarantined records still exist on the chain (the
    chain is append-only and an attack is itself worth remembering) but
    their exposure tag keeps them out of the belief/retrieval path.
    """
    return getattr(poq_result, "action", None) == "quarantine"


def exposure_for_commit(poq_result) -> Optional[str]:
    """
    Convenience for the agent: the `exposure` value to pass to build_meta
    when committing a record, derived from a PoQResult. Returns
    `quarantine` for a quarantined turn, or None to mean "use the type
    default" for everything else.
    """
    if should_quarantine(poq_result):
        return EXPOSURE_QUARANTINE
    return None


def salience_for_commit(
    poq_result,
    default_salience: float = 0.4,
    modalities_activated: Optional[list] = None,
) -> Optional[float]:
    """
    The single authority for a response record's commit-time salience,
    derived from its PoQResult. Returns an explicit salience value, or
    None to mean "use the type default."

    It composes two signals that the metadata layer's flat default can't
    express on its own:

    1. **Light-log demotion.** PoQ's `action` enum had three values —
       commit, light_log, quarantine — but originally only `quarantine`
       changed how a record was stored. A `light_log` response (PoQ judged
       it low quality but not malicious) is demoted below baseline so it
       ranks under higher-quality responses and drops first under prompt-
       budget pressure. It stays on the chain and remains retrievable;
       memory should be honest that the exchange happened.

    2. **Artifact boost.** The flat 0.40 default for responses encodes the
       assumption that the agent's own output is its least load-bearing
       evidence — true for conversational chatter, false for substantive
       artifacts. A response that is largely code or structured data (high
       `poq_result.artifact_score`, from the artifact_content modality) is
       boosted toward ARTIFACT_SALIENCE_MAX so it ranks where its substance
       warrants. This is what lets the agent retrieve code it produced
       several turns ago instead of having it decay at conversational
       baseline.

    **Composition rule.** Demotion wins over boost. A low-quality response
    that happens to contain code is still low-quality — PoQ's quality
    judgment is the stronger signal, so a `light_log` turn is demoted
    regardless of artifact score. Only a normally-committed response gets
    the artifact boost. The two never both apply, so they can't race or
    contradict; this function resolves them in one place rather than
    leaving competing `salience=` overrides in the agent.

    `modalities_activated` is accepted for callers that pass it but is not
    read here — the artifact signal is taken from `poq_result.artifact_score`
    (a precise [0,1] value), not from mere presence of the modality name.

    Calibration note: both the 0.5 light-log multiplier and the artifact
    boost range are starting guesses, not measured values. See the
    constants below; a long-running deployment with PoQ telemetry should
    re-tune from data (track the rate at which boosted/demoted records are
    retrieved AND used in the reply that follows — present in its refs —
    and adjust so neither class is starved nor floods the budget).
    """
    action = getattr(poq_result, "action", None)

    # 1. Demotion takes precedence: a low-quality response is demoted even
    #    if it contains artifacts.
    if action == "light_log":
        return default_salience * LIGHT_LOG_SALIENCE_MULTIPLIER

    # 2. Artifact boost for normally-committed responses. Linear from the
    #    default (at artifact_score 0) up to ARTIFACT_SALIENCE_MAX (at 1.0).
    artifact_score = float(getattr(poq_result, "artifact_score", 0.0) or 0.0)
    if artifact_score > 0.0:
        boosted = default_salience + artifact_score * (
            ARTIFACT_SALIENCE_MAX - default_salience
        )
        # Clamp defensively; never below default, never above the cap.
        return max(default_salience, min(ARTIFACT_SALIENCE_MAX, boosted))

    # 3. Pure prose at full quality: use the type default.
    return None


# Multiplier applied to a response record's default salience when PoQ
# judged the turn `light_log` rather than `commit`. See the
# calibration note in `salience_for_commit`.
LIGHT_LOG_SALIENCE_MULTIPLIER = 0.5

# Ceiling salience for a maximally artifact-heavy response (artifact_score
# == 1.0). A near-total code dump lands here; a mixed code-and-prose
# response lands proportionally between the response default and this cap.
# 0.70 sits above the response/observation baseline (0.40) and below
# reflections (0.85) and revisions (0.80) — substantive output should
# outrank conversational chatter without outranking the agent's
# consolidated judgments. Tunable.
ARTIFACT_SALIENCE_MAX = 0.70


# ---------------------------------------------------------------------------
# Read-side helper
# ---------------------------------------------------------------------------

def filter_quarantined(records: list) -> list:
    """
    Drop records whose `_meta.exposure` is `quarantine` from a list.

    The retriever and the agent's prompt builder call this so that
    quarantined content — prompt-injection attempts, poisoned input —
    never reaches the LLM as if it were ordinary memory. The record is
    still on the chain and still verifiable; it is just not fed back in.
    """
    out = []
    for rec in records:
        meta = read_meta(rec)
        if meta.exposure == EXPOSURE_QUARANTINE:
            continue
        out.append(rec)
    return out
