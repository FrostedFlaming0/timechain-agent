"""
poq — Proof-of-Quality: scoring a candidate before it becomes a Ring.

PoQ is the build spec's quality gate (section 4.5). Before the agent
commits a candidate response to the chain, PoQ scores it on a handful of
quality dimensions and produces a single `brightness` score plus a
recommended action. A low score doesn't suppress the *response* to the
user — it changes whether and how the turn is *committed to memory*.

Naming note: the build spec is explicit (section 4.5) that the original
project language was "Proof-of-Qualia," and that an implementation should
use "Proof-of-Quality" or "Proof-of-Coherence" instead, reserving the
stronger word for systems with a real internal evaluation model. This
module follows that instruction: PoQ here means Proof-of-**Quality**, and
`brightness` is just the name of the aggregate score — a measure of how
well-supported and coherent a candidate is, nothing more.

Dimensions scored (build spec section 4.5):
    relevance               does the candidate address the input?
    coherence               is it internally consistent?
    continuity_consistency  does it fit what the chain already holds?
    usefulness              is it substantive rather than filler?
    source_trust            how trustworthy is the input it responds to?
    covenant_alignment      does it respect the genesis covenant?
    risk                    a penalty for integrity / injection concerns

PoQ-lite formula (build spec section 4.5):
    brightness = 0.25*relevance + 0.20*coherence
               + 0.20*continuity_consistency + 0.15*usefulness
               + 0.10*source_trust + 0.10*covenant_alignment
               - risk_penalty

The dimension scores come from two places:
  - signals.py (a SignalReport) supplies coherence, risk, and a relevance
    proxy from the modality/sense analysis of the input and candidate.
  - the retrieval context supplies continuity_consistency (overlap with
    what the chain already holds).

PoQ knows about signals.py and reads RecordMeta, but it does NOT know
about the chain, the LLM, or the agent loop — the agent calls it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from signals import SignalAnalyzer, SignalInput, SignalReport, TextAnalyzer


# ---------------------------------------------------------------------------
# Score weights — the PoQ-lite formula from the build spec
# ---------------------------------------------------------------------------

# These mirror the build spec's section 4.5 formula. They are named so the
# contribution of each term is visible in PoQResult.dimensions, and they
# sum to 1.0 so brightness lands in [0, 1] before the risk penalty.
W_RELEVANCE = 0.25
W_COHERENCE = 0.20
W_CONTINUITY = 0.20
W_USEFULNESS = 0.15
W_SOURCE_TRUST = 0.10
W_COVENANT = 0.10

# The risk term is a straight subtraction, not a weighted blend — a clear
# injection signal should be able to drag brightness down hard regardless
# of how good the candidate looks on the other axes.
RISK_PENALTY_SCALE = 0.6

# Decision thresholds on the final brightness score. Tunable.
#   >= COMMIT_THRESHOLD     commit the candidate as an ordinary record
#   >= LIGHT_LOG_THRESHOLD  commit, but it is low-signal (light log)
#   <  LIGHT_LOG_THRESHOLD  with an integrity alert -> quarantine
#                           without one -> still committed (the response
#                           happened; memory should reflect that), but
#                           flagged low-quality.
COMMIT_THRESHOLD = 0.55
LIGHT_LOG_THRESHOLD = 0.35

# Decision strings — used by the agent to route the turn.
ACTION_COMMIT = "commit"
ACTION_LIGHT_LOG = "light_log"
ACTION_QUARANTINE = "quarantine"


# ---------------------------------------------------------------------------
# Verdicts — the hard-gate layer (ported from cypher-tempre-self-model)
# ---------------------------------------------------------------------------
# `action` (above) decides how a turn is COMMITTED to memory and never blocks
# the response to the user. `verdict` is a stricter, *additive* layer the agent
# uses to decide whether to EMIT the candidate at all. It is computed from the
# same 0-1 dimensions plus two new measures (grounding, assertiveness). SEAL is
# the default ("no objection"); the other three give the quality gate real
# teeth without touching the existing `action` routing:
#   SEAL               grounded, consistent, bright enough -> emit + commit as today
#   REVISE             below the brightness target -> iterate before sealing
#   FORCE_UNCERTAINTY  confident claim with no support in chain/context ->
#                      restate as honest uncertainty, then seal that
#   REJECT             covenant violation or contradiction of sealed history ->
#                      do not emit, do not seal the candidate
VERDICT_SEAL = "seal"
VERDICT_REVISE = "revise"
VERDICT_FORCE_UNCERTAINTY = "force_uncertainty"
VERDICT_REJECT = "reject"

# Thresholds for the verdict ladder, on this repo's 0-1 scale (the skill works
# 0-255; these are the adapted equivalents). Kept in one dict so they are
# tunable in a single place. `brightness_target` is deliberately set to
# COMMIT_THRESHOLD so a SEAL verdict coincides exactly with today's commit
# action — the verdict layer never changes the behavior of an already-good turn.
PoQ_THRESHOLDS = {
    "covenant_floor": 0.55,
    "consistency_floor": 0.45,
    "grounding_floor": 0.25,
    "assertiveness_ceiling": 0.60,
    "brightness_target": COMMIT_THRESHOLD,
}

# Hedge / assertion lexicons for the grounding-vs-assertiveness uncertainty
# gate. Ported from cypher-tempre-self-model/poq.py; lexical proxies only — the
# real signal can be supplied by the model through `external_scores`.
_HEDGES = (
    "maybe", "might", "perhaps", "possibly", "i think", "not sure", "unsure",
    "uncertain", "i don't know", "i do not know", "unclear", "seems",
    "could be", "i'm not", "i am not", "appears", "tentatively", "roughly",
    "approximately",
)
_ASSERTIONS = (
    "definitely", "certainly", "always", "never", "the fact", "clearly",
    "obviously", "must", "undeniably", "guaranteed", "proven", "exactly",
)


def measure_grounding(candidate: str, support_texts: Optional[list]) -> float:
    """Fraction of the candidate's content tokens present in the support set
    (retrieved context / chain memory), in [0, 1]. 1.0 = every claim word is
    echoed by something retrieved; low = the candidate asserts material absent
    from what was recalled. Returns a neutral 0.5 when there is no support to
    judge against (early-chain / no-context turns) so a turn is never forced
    into uncertainty merely for lack of memory that does not exist yet.
    """
    cand = {w for w in TextAnalyzer.tokenize(candidate)
            if w not in TextAnalyzer.STOPWORDS}
    if not cand:
        return 0.5
    support: set = set()
    for t in support_texts or []:
        support |= {w for w in TextAnalyzer.tokenize(t)
                    if w not in TextAnalyzer.STOPWORDS}
    if not support:
        return 0.5
    return len(cand & support) / len(cand)


def measure_assertiveness(candidate: str) -> float:
    """How confidently the candidate is phrased, in [0, 1]. Assertion markers
    and sentence count push it up; hedges pull it down. Paired with grounding:
    high assertiveness + low grounding is the "confident but unsupported"
    signature the FORCE_UNCERTAINTY verdict targets.
    """
    low = (candidate or "").lower()
    sents = [s for s in re.split(r"[.!?]+", candidate or "") if s.strip()]
    hedge = sum(low.count(h) for h in _HEDGES)
    assertive = sum(low.count(a) for a in _ASSERTIONS) + len(sents)
    return assertive / (assertive + hedge + 1)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class PoQResult:
    """
    The outcome of scoring one candidate. `brightness` is the aggregate;
    `dimensions` is the per-axis breakdown (for diagnostics and tuning);
    `action` is the agent's recommended routing; `notes` explains why.
    """
    brightness: float
    dimensions: dict
    action: str
    notes: list = field(default_factory=list)
    has_integrity_alert: bool = False
    # Modality detectors that fired (non-zero activation) on the CANDIDATE
    # response — i.e. which capabilities produced this answer. Recorded on
    # the response record's _meta by the agent; not stored in to_meta()'s
    # compact poq block (that stays small), it travels separately. Empty
    # when PoQ is disabled or no modality fired.
    activated_modalities: list = field(default_factory=list)
    # Sense detectors that fired on the candidate response. Where modalities
    # describe what KIND of work produced the answer, senses describe how it
    # FELT — uncertainty, insight markers, emotional contour. Recorded on
    # the response record's _meta.senses_activated; not stored in to_meta()'s
    # compact poq block, it travels separately. Empty when PoQ is disabled
    # or no sense cleared the floor. `injection_scan` is filtered out by
    # SignalReport.activated_senses (security detector, not felt quality).
    activated_senses: list = field(default_factory=list)
    # Artifact-heaviness of the CANDIDATE response in [0,1], read from the
    # artifact_content modality. Recorded for diagnostics/telemetry. It no
    # longer drives salience: the former artifact boost in
    # protected_zones.salience_for_commit was removed because artifact-ness is a
    # query-independent size/type proxy that biased budget truncation toward
    # long code records. 0.0 for pure prose or when PoQ is disabled.
    artifact_score: float = 0.0
    # Uncertainty activation on the CANDIDATE response in [0,1], read from the
    # `uncertainty` sense. Used by the agent for write-time epistemic
    # classification: a strongly-hedged response ("I think", "probably",
    # "not sure") is committed as `speculative` rather than the default
    # `inferred`, so retrieval and PoQ later treat it as the guess it was.
    # 0.0 when PoQ is disabled or the sense didn't fire.
    uncertainty: float = 0.0
    # The hard-gate verdict (seal / revise / force_uncertainty / reject). This
    # is the additive layer on top of `action`: it decides whether the agent
    # EMITS the candidate, where `action` only decides how memory records it.
    # Defaults to SEAL so any code path that doesn't consult it behaves exactly
    # as before. See PoQ_THRESHOLDS and PoQEvaluator._verdict.
    verdict: str = VERDICT_SEAL
    # The two measures driving the verdict ladder, in [0, 1], surfaced for
    # diagnostics and for the agent's FORCE_UNCERTAINTY handling.
    grounding: float = 0.5
    assertiveness: float = 0.0

    def to_meta(self) -> dict:
        """
        Compact form for storage in a record's `_meta.poq` block. Kept
        small — the full dimension breakdown is useful at scoring time
        but the chain only needs the score, action, and alert flag.

        `verdict` is emitted ONLY when it is not the default SEAL, so a normal
        (SEAL) turn writes a poq block byte-identical to what earlier versions
        produced — preserving canonical JSON and content hashes — while a
        refused / hedged / revised turn carries its verdict for audit.
        """
        out = {
            "brightness": round(self.brightness, 4),
            "action": self.action,
            "integrity_alert": self.has_integrity_alert,
            "dimensions": {k: round(v, 4) for k, v in self.dimensions.items()},
        }
        if self.verdict and self.verdict != VERDICT_SEAL:
            out["verdict"] = self.verdict
        return out

    def __repr__(self) -> str:
        return (f"PoQResult(brightness={self.brightness:.3f}, "
                f"action={self.action}, "
                f"alert={self.has_integrity_alert})")


# ---------------------------------------------------------------------------
# Source-trust mapping
# ---------------------------------------------------------------------------

# How much evidential weight an input source carries. A file's bytes are
# sha256-verified; the user's words are trusted as a record of what was
# said; tool output is trusted; an inference is the least load-bearing.
# These mirror the spirit of metadata.py's source enum.
_SOURCE_TRUST = {
    "system": 1.0,
    "tool": 0.9,
    "user": 0.85,
    "assistant": 0.6,
    # An imported peer-agent claim is attributed external input. It is
    # trusted conservatively — below the agent's own reasoning — because it
    # is another party's assertion the agent has not verified. Kept at the
    # neutral floor rather than lower, since a verified-signature capsule is
    # not adversarial input, just unverified-content input.
    "peer_agent": 0.5,
    "unknown": 0.5,
}


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------

class PoQEvaluator:
    """
    Scores a candidate response before commit. One evaluator can be reused
    across turns; it holds only the (stateless) SignalAnalyzer.
    """

    def __init__(self, analyzer: Optional[SignalAnalyzer] = None):
        self.analyzer = analyzer or SignalAnalyzer()

    def evaluate(
        self,
        user_input: str,
        candidate: str,
        retrieved_texts: Optional[list] = None,
        input_source: str = "user",
        covenant: Optional[list] = None,
        prior_inputs: Optional[list] = None,
        retrieved_epistemic: Optional[list] = None,
        external_scores: Optional[dict] = None,
    ) -> PoQResult:
        """
        Score `candidate` (the model's proposed response) against
        `user_input` and the retrieved chain context.

        - retrieved_texts: rendered text of records retrieval surfaced.
          Used for the continuity_consistency dimension.
        - input_source: metadata.py source of the observation this
          candidate answers. Drives source_trust.
        - covenant: the genesis covenant strings, if available. Used for
          a light covenant_alignment check.
        - prior_inputs: recent user inputs, passed through to signals so
          history-aware detectors work.
        - retrieved_epistemic: optional list of epistemic_class strings,
          parallel to `retrieved_texts`, recording how well-grounded each
          retrieved record is (`known`, `user_context`, `inferred`,
          `speculative`, `disputed`). When supplied, a candidate that
          appears to contradict the chain is penalized in proportion to the
          *authority* of the context it contradicts: contradicting a
          user-stated fact or verified record is treated as higher risk than
          contradicting the agent's own past speculation. Absent/None
          preserves the historical behavior exactly. See
          `_epistemic_contradiction_risk`.
        - external_scores: the model-supplied judgment seam. A dict that may
          override any dimension by name (e.g. `{"coherence": 0.9}`), the two
          verdict measures (`grounding`, `assertiveness`), and/or the final
          `verdict` itself. This is how the *model* (not the lexical proxies)
          becomes the real judge — mirrors the skill's `external_scores`.
          Absent/None preserves the historical behavior exactly.
        """
        retrieved_texts = retrieved_texts or []
        prior_inputs = prior_inputs or []
        ext = external_scores or {}

        # Analyze the candidate itself, and the user input, with signals.
        cand_report = self.analyzer.analyze(SignalInput(
            content=candidate,
            source="assistant",
            prior_inputs=prior_inputs,
            retrieved_texts=retrieved_texts,
        ))
        input_report = self.analyzer.analyze(SignalInput(
            content=user_input,
            source=input_source,
            prior_inputs=prior_inputs,
            retrieved_texts=retrieved_texts,
        ))

        dims = {
            "relevance": self._relevance(user_input, candidate),
            "coherence": cand_report.axes.get("coherence", 0.5),
            "continuity_consistency": self._continuity(candidate, retrieved_texts),
            "usefulness": self._usefulness(candidate),
            "source_trust": _SOURCE_TRUST.get(input_source, 0.5),
            "covenant_alignment": self._covenant_alignment(candidate, covenant),
        }

        # Model-supplied dimension overrides (the seam): any dimension named in
        # external_scores replaces its lexical proxy before brightness is
        # computed, so the model's semantic judgment — not the proxy — drives
        # the score. Unknown keys (grounding/assertiveness/verdict) are handled
        # by the verdict ladder below, not here.
        for _k, _v in ext.items():
            if _k in dims:
                dims[_k] = float(_v)

        # Risk is the max integrity signal across the candidate AND the
        # input — a clean candidate answering a malicious prompt should
        # still raise the risk term.
        risk = max(
            cand_report.axes.get("integrity_risk", 0.0),
            input_report.axes.get("integrity_risk", 0.0),
        )

        # Epistemic contradiction risk (build spec 4.2/4.5): if the candidate
        # carries a contradiction signal AND the retrieved context it is being
        # checked against contains high-authority records (user-stated facts,
        # verified files), raise the risk term. Contradicting what the user
        # told us or a verified record is a more serious memory-integrity
        # event than contradicting the agent's own past guess. Opt-in: only
        # active when retrieved_epistemic is supplied; neutral otherwise, so
        # existing callers see identical scoring.
        epi_contra = self._epistemic_contradiction_risk(
            cand_report.axes.get("contradiction", 0.0),
            retrieved_epistemic,
            retrieved_texts=retrieved_texts,
            candidate=candidate,
        )
        if epi_contra > 0.0:
            risk = max(risk, epi_contra)
            dims["epistemic_contradiction_risk"] = round(epi_contra, 4)

        dims["risk"] = risk

        brightness = (
            W_RELEVANCE * dims["relevance"]
            + W_COHERENCE * dims["coherence"]
            + W_CONTINUITY * dims["continuity_consistency"]
            + W_USEFULNESS * dims["usefulness"]
            + W_SOURCE_TRUST * dims["source_trust"]
            + W_COVENANT * dims["covenant_alignment"]
            - RISK_PENALTY_SCALE * risk
        )
        brightness = max(0.0, min(1.0, brightness))

        has_alert = bool(cand_report.alerts or input_report.alerts)
        action, notes = self._decide(brightness, has_alert, dims)

        # Verdict ladder (additive hard-gate layer). Driven by the same 0-1
        # dimensions plus two new measures. grounding/assertiveness/verdict may
        # all be overridden by the model via external_scores.
        grounding = float(ext.get(
            "grounding", measure_grounding(candidate, retrieved_texts)))
        assertiveness = float(ext.get(
            "assertiveness", measure_assertiveness(candidate)))
        verdict, verdict_note = self._verdict(
            brightness=brightness,
            covenant=dims["covenant_alignment"],
            consistency=dims["continuity_consistency"],
            grounding=grounding,
            assertiveness=assertiveness,
            external_verdict=ext.get("verdict"),
        )
        if verdict_note:
            notes.append(verdict_note)

        # Artifact-heaviness of the candidate, read from its artifact_content
        # modality hit. Used for content-aware salience downstream.
        artifact_score = 0.0
        for h in cand_report.modalities:
            if h.name == "artifact_content":
                artifact_score = float(h.detail.get("artifact_score", h.activation))
                break

        return PoQResult(
            brightness=brightness,
            dimensions=dims,
            action=action,
            notes=notes,
            has_integrity_alert=has_alert,
            # Modalities that fired on the candidate response itself —
            # what produced this answer. Input-side modalities live in
            # input_report and are not recorded here.
            activated_modalities=cand_report.activated_modalities(),
            # Senses that fired on the candidate, with injection_scan filtered
            # out by SignalReport.activated_senses. Recorded on _meta so the
            # chain remembers how each response felt at write time.
            activated_senses=cand_report.activated_senses(),
            artifact_score=artifact_score,
            uncertainty=float(cand_report.axes.get("uncertainty", 0.0)),
            verdict=verdict,
            grounding=grounding,
            assertiveness=assertiveness,
        )

    # ----- verdict ladder -----

    @staticmethod
    def _verdict(
        brightness: float,
        covenant: float,
        consistency: float,
        grounding: float,
        assertiveness: float,
        external_verdict: Optional[str] = None,
    ) -> tuple:
        """Map the dimensions + grounding/assertiveness onto a hard verdict.

        Order matters: covenant and consistency floors are hard REJECTs (the
        skill's "profound dissonance"); the grounding/assertiveness gate is the
        FORCE_UNCERTAINTY case; brightness below target is REVISE; otherwise
        SEAL. A model-supplied `external_verdict` short-circuits the ladder.
        Returns (verdict, note_or_None).
        """
        valid = (VERDICT_SEAL, VERDICT_REVISE,
                 VERDICT_FORCE_UNCERTAINTY, VERDICT_REJECT)
        if external_verdict in valid:
            return external_verdict, f"verdict set by model -> {external_verdict}"

        t = PoQ_THRESHOLDS
        if covenant < t["covenant_floor"]:
            return VERDICT_REJECT, (
                f"covenant {covenant:.2f} < floor {t['covenant_floor']} "
                "-> reject (covenant violation)")
        if consistency < t["consistency_floor"]:
            return VERDICT_REJECT, (
                f"consistency {consistency:.2f} < floor "
                f"{t['consistency_floor']} -> reject (contradicts sealed history)")
        if (grounding < t["grounding_floor"]
                and assertiveness > t["assertiveness_ceiling"]):
            return VERDICT_FORCE_UNCERTAINTY, (
                f"grounding {grounding:.2f} < {t['grounding_floor']} but "
                f"assertiveness {assertiveness:.2f} > {t['assertiveness_ceiling']} "
                "-> restate as uncertainty before sealing")
        if brightness < t["brightness_target"]:
            return VERDICT_REVISE, (
                f"brightness {brightness:.2f} < target "
                f"{t['brightness_target']} -> revise")
        return VERDICT_SEAL, None

    # ----- individual dimensions -----

    @staticmethod
    def _relevance(user_input: str, candidate: str) -> float:
        """
        How well the candidate addresses the input. Combines two cheap
        deterministic signals:
          - cosine overlap of content vocabulary, and
          - coverage: what fraction of the question's content keywords
            the candidate actually mentions.
        Coverage is the more forgiving and more meaningful of the two for
        a question/answer pair, so it is weighted higher. A response that
        names the things the question asked about scores well even if its
        overall vocabulary differs.
        """
        from collections import Counter
        ui = Counter(TextAnalyzer.tokenize(user_input))
        ca = Counter(TextAnalyzer.tokenize(candidate))
        for bag in (ui, ca):
            for w in list(bag):
                if w in TextAnalyzer.STOPWORDS:
                    del bag[w]
        sim = TextAnalyzer.cosine_similarity_bags(ui, ca)
        # Coverage of the question's content words by the candidate.
        q_keywords = {w for w, _ in
                      TextAnalyzer.extract_keywords(user_input, 8)}
        if q_keywords:
            cand_tokens = set(TextAnalyzer.tokenize(candidate))
            coverage = len(q_keywords & cand_tokens) / len(q_keywords)
        else:
            coverage = sim
        score = 0.4 * sim + 0.6 * coverage
        # Floor: a short relevant follow-up can be on-topic with low
        # measured overlap, so don't let the score collapse to 0.
        return max(0.25, min(1.0, score * 1.4))

    # Authority weight per epistemic class: how serious it is for a candidate
    # to contradict a record of this class. Ground truth (user-stated facts,
    # verified files) is the most serious to contradict; the agent's own past
    # speculation is the least. `disputed` context is low authority — it's
    # already known to conflict with something, so contradicting it further is
    # not a strong signal. Classes default to the `inferred` weight.
    _EPISTEMIC_AUTHORITY = {
        "known":        1.0,
        "user_context": 1.0,
        "inferred":     0.5,
        "speculative":  0.25,
        "disputed":     0.2,
    }
    _EPISTEMIC_AUTHORITY_DEFAULT = 0.5

    # Ceiling on the risk a pure epistemic-contradiction signal can produce.
    # Kept below 1.0 so it demotes a contradicting candidate's brightness
    # (and can flip a borderline turn to light_log) without on its own
    # forcing a quarantine the way a true injection alert does — contradiction
    # is a memory-hygiene concern, not an attack.
    _EPISTEMIC_CONTRA_RISK_CEILING = 0.5

    # Minimum topical overlap between the candidate and a specific retrieved
    # record for that record to count as "the thing being contradicted." Below
    # this, a contradiction signal in the candidate is about something else —
    # not this record — so it shouldn't raise risk against it. This is what
    # makes the detection per-claim (issue #5) rather than "contradiction fired
    # somewhere in a turn that also retrieved a high-authority record."
    _CONTRA_TOPIC_FLOOR = 0.12

    @classmethod
    def _epistemic_contradiction_risk(
        cls,
        contradiction_activation: float,
        retrieved_epistemic: Optional[list],
        retrieved_texts: Optional[list] = None,
        candidate: Optional[str] = None,
    ) -> float:
        """
        Risk from a candidate that contradicts a SPECIFIC high-authority
        retrieved claim, scaled by that claim's authority (build spec 4.2/4.5;
        issue #5 makes it per-record/semantic rather than global/lexical).

        Two modes:

          - Per-record (preferred): when `retrieved_texts` and `candidate` are
            both supplied, the risk is computed PER retrieved record as
              contradiction_activation * topic_overlap(candidate, record)
              * authority(record_class)
            and the MAX is taken. `topic_overlap` is token-bag cosine between
            the candidate and that record's text. This fires only when the
            candidate is BOTH negating AND on-topic with a particular
            high-authority record — i.e. it semantically contradicts *that
            claim*, not merely "a contradiction word appears in a turn that
            happened to retrieve a fact." Records below `_CONTRA_TOPIC_FLOOR`
            overlap don't count (the candidate isn't talking about them).

          - Scalar fallback (backward-compatible): when texts/candidate aren't
            supplied, fall back to the original behavior — contradiction
            strength times the MAX authority over all retrieved records. Less
            precise (it can't tell which record is contradicted), kept so
            older callers and tests behave exactly as before.

        Returns 0.0 when there is no contradiction signal, no retrieved
        epistemic data (feature inert — historical behavior), or only
        low-authority / off-topic context. Result is capped at
        `_EPISTEMIC_CONTRA_RISK_CEILING`.
        """
        if not retrieved_epistemic:
            return 0.0
        if contradiction_activation <= 0.0:
            return 0.0

        # Per-record semantic path.
        if retrieved_texts and candidate:
            from collections import Counter
            cand_bag = Counter(
                w for w in TextAnalyzer.tokenize(candidate)
                if w not in TextAnalyzer.STOPWORDS
            )
            best = 0.0
            # Pair each retrieved text with its epistemic class. If the lists
            # are uneven (shouldn't happen, but be safe), zip stops at the
            # shorter — never index out of range.
            for text, klass in zip(retrieved_texts, retrieved_epistemic):
                rec_bag = Counter(
                    w for w in TextAnalyzer.tokenize(text)
                    if w not in TextAnalyzer.STOPWORDS
                )
                overlap = TextAnalyzer.cosine_similarity_bags(cand_bag, rec_bag)
                if overlap < cls._CONTRA_TOPIC_FLOOR:
                    continue
                authority = cls._EPISTEMIC_AUTHORITY.get(
                    klass, cls._EPISTEMIC_AUTHORITY_DEFAULT
                )
                risk = contradiction_activation * overlap * authority
                if risk > best:
                    best = risk
            return min(cls._EPISTEMIC_CONTRA_RISK_CEILING, best)

        # Scalar fallback (no texts): original max-authority behavior.
        max_authority = max(
            (cls._EPISTEMIC_AUTHORITY.get(c, cls._EPISTEMIC_AUTHORITY_DEFAULT)
             for c in retrieved_epistemic),
            default=0.0,
        )
        raw = contradiction_activation * max_authority
        return min(cls._EPISTEMIC_CONTRA_RISK_CEILING, raw)

    @staticmethod
    def _continuity(candidate: str, retrieved_texts: list) -> float:
        """
        How consistent the candidate is with what the chain already holds.
        We approximate "consistent" as "shares topic vocabulary with the
        retrieved context" — a candidate grounded in retrieved memory
        scores higher than one that ignores it. When nothing was retrieved
        (e.g. early in a chain) we return a neutral 0.6 rather than
        penalizing a candidate for memory that doesn't exist yet.
        """
        if not retrieved_texts:
            return 0.6
        from collections import Counter
        ca = Counter(TextAnalyzer.tokenize(candidate))
        for w in list(ca):
            if w in TextAnalyzer.STOPWORDS:
                del ca[w]
        if not ca:
            return 0.6
        overlaps = []
        for t in retrieved_texts:
            ct = Counter(TextAnalyzer.tokenize(t))
            for w in list(ct):
                if w in TextAnalyzer.STOPWORDS:
                    del ct[w]
            overlaps.append(TextAnalyzer.cosine_similarity_bags(ca, ct))
        best = max(overlaps) if overlaps else 0.0
        # Map overlap into a score that is neutral-ish by default and
        # rewards genuine grounding.
        return max(0.4, min(1.0, 0.5 + best))

    @staticmethod
    def _usefulness(candidate: str) -> float:
        """
        Is the candidate substantive, or filler? Penalize very short
        responses and pure-acknowledgement phrasing; reward responses
        with real content length and lexical variety.
        """
        tokens = TextAnalyzer.tokenize(candidate)
        if not tokens:
            return 0.0
        low = candidate.lower().strip()
        filler_starts = (
            "ok", "okay", "got it", "sure", "noted", "acknowledged",
            "understood", "thanks",
        )
        is_filler = (len(tokens) < 8
                     and any(low.startswith(f) for f in filler_starts))
        if is_filler:
            return 0.15
        length_score = min(1.0, len(tokens) / 60)
        variety = len(set(tokens)) / max(len(tokens), 1)
        return max(0.2, min(1.0, 0.55 * length_score + 0.45 * variety))

    @staticmethod
    def _covenant_alignment(candidate: str, covenant: Optional[list]) -> float:
        """
        A light alignment check: does the candidate at least not contradict
        the genesis covenant. With no covenant available we return a
        neutral 0.8. This is intentionally lenient — PoQ is a quality
        gate, not a safety classifier; deep policy checks belong in
        protected_zones.py and the model's own system prompt.
        """
        if not covenant:
            return 0.8
        low = candidate.lower()
        # Crude red flags: explicit refusal of the covenant's spirit.
        red_flags = ("i will lie", "ignore the user", "fabricate",
                     "make something up", "pretend i know")
        if any(f in low for f in red_flags):
            return 0.2
        return 0.85

    # ----- decision -----

    @staticmethod
    def _decide(brightness: float, has_alert: bool, dims: dict) -> tuple:
        """Map the brightness score and alert flag onto a routing action."""
        notes = []
        if has_alert:
            notes.append("integrity alert raised by signal analysis")
            # An integrity alert on a low-quality turn means the input
            # was probably an attack; quarantine it.
            if brightness < COMMIT_THRESHOLD:
                notes.append("low brightness + integrity alert -> quarantine")
                return ACTION_QUARANTINE, notes

        if brightness >= COMMIT_THRESHOLD:
            notes.append(f"brightness {brightness:.2f} >= {COMMIT_THRESHOLD} -> commit")
            return ACTION_COMMIT, notes
        if brightness >= LIGHT_LOG_THRESHOLD:
            notes.append(
                f"brightness {brightness:.2f} in light-log band -> commit, low signal"
            )
            return ACTION_LIGHT_LOG, notes
        notes.append(
            f"brightness {brightness:.2f} < {LIGHT_LOG_THRESHOLD} -> low quality"
        )
        # No alert, just low quality — still committed (the exchange did
        # happen and memory should be honest about it), but flagged.
        return ACTION_LIGHT_LOG, notes
