"""
cambium — the growth mechanism (build spec section 4.8).

Cambium is named for the layer of a tree that produces new growth. In the
build spec it is the part of the system that watches accumulated history
for *recurring gaps* — the same correction happening again and again, the
same kind of failure, a contradiction cluster — and proposes new structure
in response.

Scope of this module, deliberately narrow:

  - It does NOT replace `Agent.reflect()`. Reflection's auto-trigger (every
    N turns, see run.py's AUTO_REFLECT_EVERY) is unchanged. Reflection
    consolidates "what mattered" into a reflection record.
  - Cambium is the *additional* output the build spec asks for: it scans
    the chain for repeated patterns and emits **proposals** — a suggested
    new skill, a suggested new modality, a suggested new sense, or a
    suggested principle. A proposal is a record, type "proposal", written
    with low-ish salience and `epistemic_class=speculative`.
  - Crucially, Cambium *proposes*; it never *applies*. The build spec's
    safety axiom is "the model may propose; policy must decide." A
    proposal sitting on the chain is a suggestion for a human (or a future
    privileged process) to act on — adding a detector to signals.py,
    say. Nothing in Cambium edits code or changes behavior on its own.

Trigger thresholds come straight from the build spec (Appendix E):
    same correction appears   >= 3 times  -> principle / skill proposal
    same failure mode appears >= 3 times  -> skill / sense proposal
    contradiction cluster                 -> sense proposal
    repeated retrieval miss / confusion    -> modality proposal

cambium.py reads the chain and metadata; it does not know about the LLM,
retrieval internals, or the web UI. The agent invokes it (e.g. alongside
reflection) and commits whatever proposals it returns.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from metadata import read_meta
from signals import TextAnalyzer


# ---------------------------------------------------------------------------
# Trigger thresholds (build spec Appendix E)
# ---------------------------------------------------------------------------

CORRECTION_REPEAT_THRESHOLD = 3   # same correction topic >= 3 times
FAILURE_REPEAT_THRESHOLD = 3      # same failure mode >= 3 times
CONTRADICTION_CLUSTER_THRESHOLD = 3  # contradiction-flagged records >= 3
CONFUSION_REPEAT_THRESHOLD = 3    # repeated user-confusion signals >= 3

# ----- Recurring-output-mode detection (auto-sprout trigger) -----
#
# Unlike the other triggers, which fire on *problems* (corrections,
# failures, confusion), this one fires on a recurring *kind of agent
# output*: the agent keeps producing responses that share a vocabulary
# cluster which no existing domain modality already captures. That is the
# signal that a new pattern-based modality would sharpen retrieval — e.g.
# the agent drafts legal language every day and would benefit from a
# `legal_document` mode. See cambium._check_recurring_output_mode and the
# sprout pipeline in agent.py.
#
# The DIVERSITY GATE guards against minting a modality from a transient
# burst ("everything you just said"). A candidate vocabulary must clear all
# of:
#   - OUTPUT_MODE_MIN_TRIGGERS distinct response records exhibiting it;
#   - OUTPUT_MODE_MIN_SPREAD_MS between the earliest and latest, so it is
#     not all one sitting;
#   - at least one INTERLEAVING non-matching response between matches, so it
#     is a recurring mode and not one contiguous run.
OUTPUT_MODE_MIN_TRIGGERS = 5
OUTPUT_MODE_MIN_SPREAD_MS = 2 * 60 * 60 * 1000   # 2 hours
OUTPUT_MODE_REQUIRE_INTERLEAVING = True

# How many shared keywords define a candidate output-mode vocabulary, and
# how many become regex patterns in the sprouted spec. Kept small so a
# sprouted modality is a tight, auditable set of word-boundary patterns.
OUTPUT_MODE_VOCAB_SIZE = 4
OUTPUT_MODE_MIN_VOCAB_OVERLAP = 2   # a response "exhibits" the mode if it
                                    # shares >= this many of the vocab words

# Distinctiveness ceiling: a candidate word appearing in more than this
# fraction of ALL scanned responses is treated as generic filler and
# excluded from the mode vocabulary. Keeps the detector from sprouting a
# modality out of ubiquitous conversational words ("sure", "good") that
# carry no domain signal.
OUTPUT_MODE_DOC_FREQ_CEILING = 0.6

# Cooling-off graduation. A freshly-sprouted modality lands as `tentative`
# (half retrieval weight). Each later scan that re-detects its vocabulary is
# a confirmation; once confirmations reach this threshold the modality
# graduates to `active` (full weight). This is the time-based filter that
# lets a one-off that cleared the gate fade instead of distorting retrieval.
#
# Counting note: this is a TOTAL-sightings threshold, matching
# `recurrence_count` (which counts the originating detection as 1, then each
# recurrence). So with the default 3, a sprout graduates after the original
# detection plus 2 re-detections. Same convention as
# RECURRENCE_ESCALATION_THRESHOLD, kept consistent on purpose.
OUTPUT_MODE_GRADUATION_CONFIRMATIONS = 3

# How many top keywords define a "topic" when clustering corrections.
_TOPIC_KEYWORDS = 3

# Recurrence escalation. When the same topic is re-detected by Cambium
# after a proposal for it already exists, that is a *recurrence* — the
# pattern keeps happening even though a proposal is on record. A proposal
# whose recurrence count reaches this threshold is "escalated": its
# salience is raised and it is surfaced at the top of the proposal list.
#
# Escalation deliberately does NOT auto-apply the proposal. A recurrence
# count measures how *persistent* a pattern is; it says nothing about
# whether the suggested code is correct, safe, or tested. Persistence is
# used to direct human attention, never to bypass human review — the
# build spec's rule stands: the model proposes, policy decides.
RECURRENCE_ESCALATION_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Proposal kinds
# ---------------------------------------------------------------------------

PROPOSAL_SKILL = "skill"
PROPOSAL_MODALITY = "modality"
PROPOSAL_SENSE = "sense"
PROPOSAL_PRINCIPLE = "principle"

VALID_PROPOSAL_KINDS = {
    PROPOSAL_SKILL, PROPOSAL_MODALITY, PROPOSAL_SENSE, PROPOSAL_PRINCIPLE,
}


@dataclass
class Proposal:
    """
    A single Cambium suggestion. The agent turns each of these into a
    `proposal` record on the chain. `kind` says what is being proposed;
    `title` and `rationale` are human-readable; `evidence` lists the
    record indices that triggered the proposal so a reviewer can check it.

    `topic_signature` is the coarse topic key Cambium used to cluster the
    evidence. It is stored on the record explicitly so a later scan can
    deduplicate reliably — re-deriving a signature from the human-readable
    title would pick up boilerplate words ("recurring", "about") instead
    of the real topic.
    """
    kind: str
    title: str
    rationale: str
    evidence: list = field(default_factory=list)   # record indices
    suggested_target: str = ""                     # e.g. "signals.py"
    topic_signature: str = ""                      # dedup key
    # For an auto-sprout candidate (recurring output mode): the ready-to-write
    # sprouted-modality spec (name, patterns, threshold, match_mode, domain).
    # None for ordinary proposals that a human reviews via apply_proposal.
    # When present, the agent's sprout pipeline can stage this directly into
    # the runtime registry (as a tentative modality) rather than only filing
    # it for human review.
    sprout_spec: dict = None

    def to_record_content(self) -> dict:
        """Shape for chain.append('proposal', content=...). The agent adds _meta."""
        content = {
            "proposal_kind": self.kind,
            "title": self.title,
            "rationale": self.rationale,
            "evidence_indices": list(self.evidence),
            "suggested_target": self.suggested_target,
            "topic_signature": self.topic_signature,
            "status": "open",   # open | accepted | declined — set by a reviewer
            # Recurrence tracking. A proposal starts at count 1 (the scan
            # that created it). Each later scan that re-detects the same
            # topic does NOT create a new proposal — it commits a
            # `proposal_recurrence` record and the effective count rises.
            # `recurrence_count` here is the value at creation time;
            # cambium.proposal_recurrence_count() computes the live total.
            "recurrence_count": 1,
            "escalated": False,
            # v2: adds recurrence_count / escalated. v1 proposal records
            # (without these) are treated as count 1, not escalated.
            "schema_version": 2,
        }
        if self.sprout_spec is not None:
            content["sprout_spec"] = self.sprout_spec
        return content


@dataclass
class Recurrence:
    """
    A re-detection of a topic that already has an open proposal on the
    chain. The agent commits each Recurrence as a small
    `proposal_recurrence` record that points back at the original
    proposal. Recurrences are how a proposal's count grows over time
    without ever creating duplicate proposals.
    """
    proposal_index: int          # chain index of the original proposal
    topic_signature: str
    new_evidence: list = field(default_factory=list)  # fresh record indices

    def to_record_content(self) -> dict:
        """Shape for chain.append('proposal_recurrence', content=...)."""
        return {
            "recurs_proposal_index": self.proposal_index,
            "topic_signature": self.topic_signature,
            "new_evidence_indices": list(self.new_evidence),
            "schema_version": 1,
        }


@dataclass
class CambiumReport:
    """
    Everything one Cambium scan found.

    `proposals`   — genuinely new suggestions (topics with no open proposal).
    `recurrences` — re-detections of topics that already have an open
                    proposal. Each will become a `proposal_recurrence`
                    record and may push the original proposal over the
                    escalation threshold.
    """
    proposals: list = field(default_factory=list)
    recurrences: list = field(default_factory=list)
    triggers_checked: dict = field(default_factory=dict)  # name -> count seen

    @property
    def has_proposals(self) -> bool:
        return len(self.proposals) > 0

    @property
    def has_recurrences(self) -> bool:
        return len(self.recurrences) > 0


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------

class Cambium:
    """
    Scans a chain for recurring gaps and produces proposals. Stateless
    apart from the thresholds; one instance can be reused. The agent calls
    `scan(chain)` and commits the returned proposals.
    """

    def __init__(
        self,
        correction_threshold: int = CORRECTION_REPEAT_THRESHOLD,
        failure_threshold: int = FAILURE_REPEAT_THRESHOLD,
        contradiction_threshold: int = CONTRADICTION_CLUSTER_THRESHOLD,
        confusion_threshold: int = CONFUSION_REPEAT_THRESHOLD,
        output_mode_min_triggers: int = OUTPUT_MODE_MIN_TRIGGERS,
    ):
        self.correction_threshold = correction_threshold
        self.failure_threshold = failure_threshold
        self.contradiction_threshold = contradiction_threshold
        self.confusion_threshold = confusion_threshold
        self.output_mode_min_triggers = output_mode_min_triggers

    def scan(
        self,
        chain,
        max_records: int = 500,
        *,
        since_idx: Optional[int] = None,
        lookback: Optional[int] = None,
        known_vocabulary: Optional[set] = None,
    ) -> CambiumReport:
        """
        Examine records on `chain` and return a CambiumReport. Does not
        write anything — the agent decides what to commit.

        Each trigger returns both new proposals (topics with no existing
        open proposal) and recurrences (topics that already have one). A
        recurrence is not a duplicate to be discarded — it is evidence
        that the pattern persists, and it is committed as a
        `proposal_recurrence` record that raises the original proposal's
        count.

        Scan window (three modes):

          1. **Tail-only (default, legacy).** With `since_idx=None`,
             scans the last `max_records` records. This is the original
             behavior. A pattern that recurs after `max_records` records
             have passed in between sightings will not be re-detected;
             use mode 2 to avoid that gap.

          2. **Incremental, watermark-driven.** With `since_idx=N`,
             scans `[max(0, N - lookback), length)` — the new tail since
             last scan, with a `lookback` window of context so the
             detectors can spot patterns that span the watermark
             boundary. Each record is examined at least once (when it's
             fresh in the new tail) plus revisited only while it's
             within `lookback` of the moving frontier. This is the mode
             that makes Cambium correct across a long chain without
             repeatedly rescanning the whole history.

             `lookback` defaults to `max_records` so existing callers
             that only pass `since_idx` get sensible behavior.

          3. **Full-chain.** With `since_idx=0` and `lookback=length`,
             scans every record. Linear in chain length. Reserved for
             explicit deep scans (see `Agent.run_cambium_full`); not the
             default cadence.

        The window endpoints determine what's *examined*. Dedup of
        already-open proposals is handled separately by
        `_existing_proposals_by_topic`, which reads from the same
        records list — so a tighter window means missed dedup
        opportunities. Keep `lookback` ≥ the typical inter-recurrence
        gap; the default `max_records=500` is fine for a chain whose
        recurrences land within a few hundred records of each other.
        """
        length = chain.length()
        if since_idx is None:
            # Mode 1: legacy tail-only.
            start = max(0, length - max_records)
        else:
            # Mode 2/3: incremental with lookback context.
            lb = lookback if lookback is not None else max_records
            start = max(0, since_idx - lb)
        records = list(chain.iter_records(start=start, end=length))

        # Map each already-proposed topic signature to its original
        # proposal record. A trigger that re-detects one of these topics
        # emits a Recurrence instead of a new Proposal.
        existing = self._existing_proposals_by_topic(records)

        proposals: list[Proposal] = []
        recurrences: list[Recurrence] = []
        triggers: dict = {}

        for name, check in (
            ("repeated_corrections", self._check_repeated_corrections),
            ("repeated_failures", self._check_repeated_failures),
            ("contradiction_cluster", self._check_contradiction_cluster),
            ("repeated_confusion", self._check_repeated_confusion),
        ):
            props, recs, count = check(records, existing)
            triggers[name] = count
            proposals.extend(props)
            recurrences.extend(recs)

        # Recurring-output-mode trigger (auto-sprout candidate). Separate
        # call because it needs the set of already-known domain modalities so
        # it doesn't re-propose a mode that's already captured.
        om_props, om_recs, om_count = self._check_recurring_output_mode(
            records, existing, known_vocabulary=known_vocabulary
        )
        triggers["recurring_output_mode"] = om_count
        proposals.extend(om_props)
        recurrences.extend(om_recs)

        return CambiumReport(
            proposals=proposals,
            recurrences=recurrences,
            triggers_checked=triggers,
        )

    # ----- helpers -----

    @staticmethod
    def _record_text(rec) -> str:
        if isinstance(rec.content, dict):
            return str(rec.content.get("text", ""))
        return str(rec.content)

    @staticmethod
    def _topic_signature(text: str) -> str:
        """A coarse topic key: the top content keywords, sorted, joined."""
        kws = [w for w, _ in TextAnalyzer.extract_keywords(text, _TOPIC_KEYWORDS)]
        return "+".join(sorted(kws)) if kws else ""

    def _existing_proposals_by_topic(self, records: list) -> dict:
        """
        Map topic signature -> the original proposal record, for every
        proposal already on the chain. Reads the explicit `topic_signature`
        field; falls back to re-deriving from the title only for legacy
        proposal records that predate the stored field.

        Declined proposals are excluded: if a reviewer declined a proposal,
        a fresh recurrence of the same topic should be allowed to surface
        a new proposal rather than silently re-attaching to the dead one.

        Effective-status resolution: the chain is append-only, so a
        proposal's stored `status` field stays "open" forever even after
        `apply_proposal.py --decline` records a decision. The actual
        decline lives in a separate `proposal_status` record. This
        function resolves the effective status by checking
        `proposal_status` records that mark each proposal, and skipping
        the ones whose latest status is "declined". Without this scan,
        declining a proposal via the supported tool had no effect on
        dedup: a fresh recurrence still re-attached to the declined
        proposal silently.
        """
        # Resolve effective status for every proposal in one scan of
        # proposal_status records — same shape as apply_proposal's
        # _current_status, but bulk so we don't repeat the scan.
        latest_status: dict = {}   # proposal_idx -> (status_record_idx, status)
        for sr in records:
            if sr.type != "proposal_status":
                continue
            if not isinstance(sr.content, dict):
                continue
            marks = sr.content.get("marks_proposal_index")
            new_status = sr.content.get("new_status")
            if not isinstance(marks, int) or not isinstance(new_status, str):
                continue
            prev = latest_status.get(marks)
            if prev is None or sr.index > prev[0]:
                latest_status[marks] = (sr.index, new_status)

        out: dict = {}
        for rec in records:
            if rec.type != "proposal":
                continue
            if not isinstance(rec.content, dict):
                continue
            # Effective status: a proposal_status record beats the
            # proposal's own stored `status` field. Either source saying
            # "declined" excludes the proposal from dedup.
            stored_status = rec.content.get("status", "open")
            effective_status = stored_status
            ls = latest_status.get(rec.index)
            if ls is not None:
                effective_status = ls[1]
            if effective_status == "declined":
                continue
            sig = rec.content.get("topic_signature")
            if not sig:
                sig = self._topic_signature(rec.content.get("title", ""))
            # Keep the earliest proposal for a topic — that is the one
            # recurrences accumulate against.
            if sig and sig not in out:
                out[sig] = rec
        return out

    def _emit(self, sig: str, indices: list, existing: dict,
              make_proposal) -> tuple:
        """
        Shared proposal-vs-recurrence decision, used by every trigger.

        Given a topic signature that has crossed its threshold:
          - if no proposal exists for the topic, build one via the
            `make_proposal` callback and return it as a new proposal;
          - if a proposal already exists, return a Recurrence pointing at
            it instead.

        Returns (proposal_or_None, recurrence_or_None) — exactly one is set.
        `make_proposal` is a zero-arg callable so the (sometimes expensive)
        rationale string is only built when a proposal is actually needed.
        """
        if sig in existing:
            original = existing[sig]
            return None, Recurrence(
                proposal_index=original.index,
                topic_signature=sig,
                new_evidence=list(indices),
            )
        return make_proposal(), None

    # ----- trigger 1: repeated corrections -----

    def _check_repeated_corrections(
        self, records: list, existing: dict
    ) -> tuple:
        """
        If the same topic gets corrected (revision records) >= threshold
        times, the agent keeps getting the same thing wrong — that is a
        candidate for an extracted principle (so the correction becomes a
        durable rule) and possibly a skill.

        Returns (new_proposals, recurrences, max_topic_count).
        """
        revisions = [r for r in records if r.type == "revision"]
        topic_to_indices: dict = {}
        for rev in revisions:
            sig = self._topic_signature(self._record_text(rev))
            if not sig:
                continue
            topic_to_indices.setdefault(sig, []).append(rev.index)

        proposals: list = []
        recurrences: list = []
        max_count = 0
        for sig, indices in topic_to_indices.items():
            max_count = max(max_count, len(indices))
            if len(indices) < self.correction_threshold:
                continue

            def make(sig=sig, indices=indices):
                return Proposal(
                    kind=PROPOSAL_PRINCIPLE,
                    title=f"Recurring correction about: {sig.replace('+', ', ')}",
                    rationale=(
                        f"The same topic has been corrected {len(indices)} "
                        f"times (records {indices}). The build spec's "
                        f"Appendix E treats a correction recurring >= "
                        f"{self.correction_threshold} times as a principle-"
                        f"extraction trigger: the correction should become a "
                        f"durable rule rather than being re-learned each time."
                    ),
                    evidence=indices,
                    suggested_target="a 'principle' record + agent behavior",
                    topic_signature=sig,
                )

            prop, rec = self._emit(sig, indices, existing, make)
            if prop:
                proposals.append(prop)
            if rec:
                recurrences.append(rec)
        return proposals, recurrences, max_count

    # ----- trigger 2: repeated failures -----

    # Phrases that mark a turn as a failure / dead end, in either the
    # user's words or the agent's. Deliberately small and conservative.
    _FAILURE_MARKERS = (
        "that's wrong", "thats wrong", "not what i asked", "doesn't work",
        "does not work", "still broken", "still failing", "error again",
        "that's not right", "incorrect", "you misunderstood",
        "i can't", "i cannot", "i don't have", "unable to",
    )

    def _check_repeated_failures(
        self, records: list, existing: dict
    ) -> tuple:
        """
        If failure-marked turns cluster on the same topic, propose a skill
        — a documented, repeatable procedure for that recurring task.

        Returns (new_proposals, recurrences, max_topic_count).
        """
        failure_records = []
        for rec in records:
            if rec.type not in ("observation", "response"):
                continue
            text = self._record_text(rec).lower()
            if any(m in text for m in self._FAILURE_MARKERS):
                failure_records.append(rec)

        topic_to_indices: dict = {}
        for rec in failure_records:
            sig = self._topic_signature(self._record_text(rec))
            if not sig:
                continue
            topic_to_indices.setdefault(sig, []).append(rec.index)

        proposals: list = []
        recurrences: list = []
        max_count = 0
        for sig, indices in topic_to_indices.items():
            max_count = max(max_count, len(indices))
            if len(indices) < self.failure_threshold:
                continue

            def make(sig=sig, indices=indices):
                return Proposal(
                    kind=PROPOSAL_SKILL,
                    title=f"Recurring failure mode around: {sig.replace('+', ', ')}",
                    rationale=(
                        f"{len(indices)} turns on this topic were marked as "
                        f"failures or dead ends (records {indices}). A "
                        f"recurring failure mode is the build spec's signal "
                        f"to compile a skill: a documented, repeatable "
                        f"procedure so the same task stops failing."
                    ),
                    evidence=indices,
                    suggested_target="a 'skill' capsule / documented procedure",
                    topic_signature=sig,
                )

            prop, rec = self._emit(sig, indices, existing, make)
            if prop:
                proposals.append(prop)
            if rec:
                recurrences.append(rec)
        return proposals, recurrences, max_count

    # ----- trigger 3: contradiction cluster -----

    def _check_contradiction_cluster(
        self, records: list, existing: dict
    ) -> tuple:
        """
        If many records carry a stored PoQ contradiction signal, or simply
        read as internally contradictory, propose a sense — a sharper
        contradiction detector — since the current one is being stressed.

        Returns (new_proposals, recurrences, contradiction_count).
        """
        contradiction_indices = []
        for rec in records:
            meta = read_meta(rec)
            # A record whose PoQ block flagged a contradiction dimension.
            if meta.poq:
                dims = meta.poq.get("dimensions", {})
                if dims.get("contradiction", 0.0) >= 0.5:
                    contradiction_indices.append(rec.index)
                    continue
            # Or one flagged as epistemically disputed.
            if meta.epistemic_class == "disputed":
                contradiction_indices.append(rec.index)

        count = len(contradiction_indices)
        sig = "contradiction-handling"
        if count < self.contradiction_threshold:
            return [], [], count

        def make():
            return Proposal(
                kind=PROPOSAL_SENSE,
                title="Contradiction cluster — sharper contradiction sense needed",
                rationale=(
                    f"{count} records carry a contradiction or disputed "
                    f"signal (records {contradiction_indices[:15]}). A "
                    f"contradiction cluster is the build spec's trigger for "
                    f"a new sense: the current contradiction detector in "
                    f"signals.py is firing often enough that a sharper, "
                    f"more specific detector would help resolve rather than "
                    f"just flag the conflicts."
                ),
                evidence=contradiction_indices[:30],
                suggested_target="a new sense detector in signals.py",
                topic_signature=sig,
            )

        prop, rec = self._emit(sig, contradiction_indices[:30], existing, make)
        return ([prop] if prop else []), ([rec] if rec else []), count

    # ----- trigger 4: repeated user confusion -----

    _CONFUSION_MARKERS = (
        "i'm confused", "im confused", "i don't understand",
        "i dont understand", "what do you mean", "that's unclear",
        "thats unclear", "makes no sense", "lost me", "huh",
    )

    def _check_repeated_confusion(
        self, records: list, existing: dict
    ) -> tuple:
        """
        Repeated user confusion suggests the agent keeps failing to land an
        explanation in some domain — propose a modality (an interpretive
        frame) tuned for that domain.

        Returns (new_proposals, recurrences, max_topic_count).
        """
        confusion_records = []
        for rec in records:
            if rec.type != "observation":
                continue
            text = self._record_text(rec).lower()
            if any(m in text for m in self._CONFUSION_MARKERS):
                confusion_records.append(rec)

        topic_to_indices: dict = {}
        for rec in confusion_records:
            # Confusion is often about the *prior* turn's topic; use the
            # confused message's own keywords as a coarse proxy.
            sig = self._topic_signature(self._record_text(rec)) or "general"
            topic_to_indices.setdefault(sig, []).append(rec.index)

        proposals: list = []
        recurrences: list = []
        max_count = 0
        for sig, indices in topic_to_indices.items():
            max_count = max(max_count, len(indices))
            if len(indices) < self.confusion_threshold:
                continue
            title_sig = f"confusion:{sig}"

            def make(sig=sig, indices=indices, title_sig=title_sig):
                return Proposal(
                    kind=PROPOSAL_MODALITY,
                    title=f"Recurring user confusion around: {sig.replace('+', ', ')}",
                    rationale=(
                        f"The user signalled confusion {len(indices)} times "
                        f"(records {indices}). Recurring confusion is the "
                        f"build spec's trigger for a new modality: an "
                        f"interpretive frame tuned to explain this domain "
                        f"more clearly than the current detector set does."
                    ),
                    evidence=indices,
                    suggested_target="a new modality detector in signals.py",
                    topic_signature=title_sig,
                )

            prop, rec = self._emit(title_sig, indices, existing, make)
            if prop:
                proposals.append(prop)
            if rec:
                recurrences.append(rec)
        return proposals, recurrences, max_count

    # ----- trigger 5: recurring output mode (auto-sprout candidate) -----

    @staticmethod
    def _derive_patterns(vocab: list) -> list:
        """
        Turn a list of vocabulary words into case-insensitive word-boundary
        regex patterns. Deterministic and auditable — a sprouted pattern is
        exactly `\\bword\\b` for each shared keyword, nothing inferred. Words
        are regex-escaped so a keyword with regex metacharacters can't form a
        malformed or dangerous pattern.
        """
        return [rf"\b{re.escape(w)}\b" for w in vocab]

    def _check_recurring_output_mode(
        self, records: list, existing: dict,
        known_vocabulary: Optional[set] = None,
    ) -> tuple:
        """
        Detect a recurring *kind of agent output* that no existing domain
        modality captures, and (if it clears the diversity gate) propose a
        pattern-based modality to sprout for it.

        This is the auto-sprout trigger. Unlike the problem-driven triggers,
        it reads `response` records: it clusters them by the vocabulary they
        share, and a cluster that recurs widely enough — across enough
        distinct responses, spread over enough time, interleaved with
        non-matching responses — becomes a candidate. The candidate carries a
        ready-to-write `sprout_spec` (word-boundary patterns over the shared
        vocabulary). The agent stages it as a *tentative* sprout; it only
        reaches full retrieval weight after cooling-off graduation.

        Why vocabulary and not "a modality that kept firing": the modality
        does not exist yet (that is the whole point), so there is nothing
        firing to count. The detectable signal is a shared keyword cluster in
        the output that the current domain modalities don't already explain.

        `known_vocabulary` lets the caller pass the set of content words
        already covered by existing domain modalities (baked-in keywords and
        the words inside already-sprouted patterns) so we don't re-propose a
        mode whose vocabulary is already captured. These are *words*, not
        modality names — a sprouted modality's patterns are `\\bword\\b`, so
        the caller extracts the words and passes them here. Dedup of an
        identical candidate that already has a proposal on the chain is
        handled separately by `_emit` via the topic signature.

        Returns (new_proposals, recurrences, max_cluster_size).
        """
        known = known_vocabulary or set()

        # Gather response records with their text and timestamp, in order.
        responses = [
            (rec.index, self._record_text(rec), rec.timestamp)
            for rec in records
            if rec.type == "response" and self._record_text(rec).strip()
        ]
        if len(responses) < self.output_mode_min_triggers:
            return [], [], 0

        # Build, per response, its top keyword set. Cluster responses by a
        # shared-vocabulary signature: the most common keywords across all
        # responses that recur. We take a global keyword frequency first,
        # then form a candidate vocabulary from the top shared words.
        from collections import Counter
        global_kw: Counter = Counter()
        doc_freq: Counter = Counter()   # in how many responses each word appears
        per_response_kw: list = []
        for idx, text, ts in responses:
            kws = [w for w, _ in TextAnalyzer.extract_keywords(text, top_n=12)]
            kw_set = set(kws)
            per_response_kw.append((idx, kw_set, ts))
            global_kw.update(kws)
            doc_freq.update(kw_set)

        # Distinctiveness guard: a word that appears in too large a fraction
        # of ALL responses is generic conversational filler ("sure", "good",
        # "sounds"), not a domain mode — anchoring on it would be noise. Drop
        # any candidate word whose document frequency exceeds the ceiling, so
        # the mode is built from vocabulary that is concentrated in a subset
        # of output, not ubiquitous across it. Without this, suppressing one
        # real mode just surfaces the next-most-common filler vocabulary.
        n_responses = len(responses)
        df_ceiling = OUTPUT_MODE_DOC_FREQ_CEILING * n_responses
        too_generic = {w for w, df in doc_freq.items() if df > df_ceiling}

        # Candidate vocabulary: the most common content words across
        # responses, minus anything an existing domain modality already keys
        # on and minus generic high-document-frequency filler.
        candidate_vocab = [
            w for w, _ in global_kw.most_common(OUTPUT_MODE_VOCAB_SIZE * 4)
            if w not in known and w not in too_generic
        ][:OUTPUT_MODE_VOCAB_SIZE]
        if len(candidate_vocab) < OUTPUT_MODE_MIN_VOCAB_OVERLAP:
            return [], [], 0
        vocab_set = set(candidate_vocab)

        # Which responses "exhibit" this mode: share >= MIN_VOCAB_OVERLAP of
        # the candidate vocabulary.
        matches = [
            (idx, ts) for idx, kws, ts in per_response_kw
            if len(kws & vocab_set) >= OUTPUT_MODE_MIN_VOCAB_OVERLAP
        ]
        max_cluster = len(matches)
        if max_cluster < self.output_mode_min_triggers:
            return [], [], max_cluster

        # ----- diversity gate -----
        # (a) distinct triggers already checked (>= min_triggers).
        # (b) temporal spread: earliest..latest must span the minimum.
        timestamps = [ts for _, ts in matches if ts]
        if len(timestamps) >= 2:
            spread = max(timestamps) - min(timestamps)
        else:
            spread = 0
        if spread < OUTPUT_MODE_MIN_SPREAD_MS:
            return [], [], max_cluster
        # (c) interleaving: at least one non-matching response between the
        # first and last match, so this is a recurring mode and not one
        # contiguous block of output.
        if OUTPUT_MODE_REQUIRE_INTERLEAVING:
            match_idx = {idx for idx, _ in matches}
            ordered = [idx for idx, _, _ in per_response_kw]
            first = ordered.index(matches[0][0])
            last = ordered.index(matches[-1][0])
            interior = ordered[first:last + 1]
            if not any(i not in match_idx for i in interior):
                return [], [], max_cluster

        # Cleared the gate. Build a deterministic modality name and a
        # ready-to-stage sprout spec.
        sig_vocab = sorted(vocab_set)
        topic_sig = "outputmode:" + "+".join(sig_vocab)
        # Name: a safe identifier from the top two vocab words.
        name = "mode_" + "_".join(sig_vocab[:2])
        name = re.sub(r"[^a-z0-9_]", "", name.lower())[:40] or "mode_sprout"
        evidence = [idx for idx, _ in matches]

        def make(name=name, sig_vocab=sig_vocab, evidence=evidence,
                 topic_sig=topic_sig):
            return Proposal(
                kind=PROPOSAL_MODALITY,
                title=f"Recurring output mode: {', '.join(sig_vocab)}",
                rationale=(
                    f"The agent produced {len(evidence)} responses sharing the "
                    f"vocabulary {sig_vocab} (records {evidence}), spread over "
                    f"time and interleaved with other output. No existing "
                    f"domain modality captures this mode. Sprouting a "
                    f"pattern-based modality for it would let retrieval anchor "
                    f"on this kind of work."
                ),
                evidence=evidence,
                suggested_target="a sprouted modality in sprouted_modalities.json",
                topic_signature=topic_sig,
                sprout_spec={
                    "name": name,
                    "patterns": self._derive_patterns(sig_vocab),
                    "threshold": 0.2,
                    "match_mode": "fraction_lines",
                    "domain": True,
                    "status": "tentative",
                },
            )

        prop, rec = self._emit(topic_sig, evidence, existing, make)
        proposals = [prop] if prop else []
        recurrences = [rec] if rec else []
        return proposals, recurrences, max_cluster
# Recurrence accounting — module-level helpers
#
# These read the chain to compute the *live* recurrence count for a
# proposal: its base count (1, from creation) plus every
# `proposal_recurrence` record that points at it. They are pure reads —
# nothing here writes to the chain. The agent uses them to decide when to
# escalate, and `apply_proposal.py` / the REPL use them for display.
# ---------------------------------------------------------------------------

def recurrence_count(chain, proposal_index: int) -> int:
    """
    Live recurrence count for the proposal at `proposal_index`: 1 for the
    proposal itself, plus one for each `proposal_recurrence` record that
    references it. Returns 0 if the index is not a proposal record.

    Return semantics, since callers occasionally trip on them:
      0  — `proposal_index` does not point at a proposal record (e.g.
           it was deleted, or refers to a different record type).
      1  — the proposal exists but has had no recurrences yet (its own
           creation counts as the first sighting).
      ≥2 — the proposal has been re-detected at least once.

    So "count > 1" is the right check for "this proposal has actually
    recurred," and "count == 0" is the right check for "this is not a
    valid proposal index." If you find yourself wanting to distinguish
    "missing" from "zero recurrences" more explicitly, use the
    `proposal_exists` companion below.

    For listing many proposals, prefer `recurrence_counts()` (the bulk
    helper below) — it returns counts for every proposal in a single
    indexed query.
    """
    rec = chain.get(proposal_index)
    if rec is None or rec.type != "proposal":
        return 0
    # +1 for the proposal's own creation (the first sighting); the
    # materialized index only stores subsequent recurrences. Indexed
    # lookup against `proposal_recurrence_index` — no scan, no silent
    # cap (the old implementation walked `proposal_recurrence` with a
    # `limit=10_000` ceiling that would silently undercount on a chain
    # with more than 10k recurrences).
    return 1 + chain.recurrence_count_for(proposal_index)


def proposal_exists(chain, proposal_index: int) -> bool:
    """
    True iff `proposal_index` points at an actual proposal record. Use
    this when you need to distinguish "missing" from "zero recurrences"
    — `recurrence_count` returns 0 for both, since 0 is meaningless for
    a real proposal (the proposal's own creation is the first sighting).
    """
    rec = chain.get(proposal_index)
    return rec is not None and rec.type == "proposal"


def recurrence_counts(chain) -> dict:
    """
    Bulk version of `recurrence_count`. Returns a dict mapping every
    proposal record's index to its live recurrence count.

    Backed by `Chain.all_recurrence_counts()` — a single indexed
    aggregation against the materialized `proposal_recurrence_index`,
    no per-record scan. The proposal list still requires a
    `query_by_type` (only because proposals themselves aren't
    indexed by a separate table), but proposals are sparse — far
    fewer than recurrences — so this is the cheaper of the two
    scans.
    """
    proposals = chain.query_by_type("proposal", limit=10_000)
    counts: dict = {p.index: 1 for p in proposals}  # +1 for the proposal itself
    for proposal_idx, n in chain.all_recurrence_counts().items():
        if proposal_idx in counts:
            counts[proposal_idx] += n
    return counts


def escalated_indices(chain) -> set:
    """
    Bulk version of `is_escalated`. Returns the set of proposal indices
    whose live recurrence count has reached the escalation threshold OR
    that carry an explicit `escalated` flag on the proposal record itself.

    Uses the materialized recurrence index via `recurrence_counts`.
    """
    counts = recurrence_counts(chain)
    escalated = {
        idx for idx, c in counts.items()
        if c >= RECURRENCE_ESCALATION_THRESHOLD
    }
    # Also honor any explicit flag on a proposal record itself (e.g. set
    # by a future tool). v1 proposals never have this; v2 store it.
    for p in chain.query_by_type("proposal", limit=10_000):
        if isinstance(p.content, dict) and p.content.get("escalated"):
            escalated.add(p.index)
    return escalated


def is_escalated(chain, proposal_index: int,
                 threshold: int = RECURRENCE_ESCALATION_THRESHOLD) -> bool:
    """
    True if the proposal's live recurrence count has reached the
    escalation threshold. A proposal can also be escalated by having its
    stored `escalated` flag set (the agent sets that flag when it commits
    the recurrence that crosses the line — see Agent.run_cambium).

    For listing many proposals, prefer `escalated_indices()` — it computes
    the same answer for every proposal in a single scan.
    """
    rec = chain.get(proposal_index)
    if rec is not None and isinstance(rec.content, dict):
        if rec.content.get("escalated"):
            return True
    return recurrence_count(chain, proposal_index) >= threshold
