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
    # artifact_content modality. Drives content-aware salience: a code- or
    # data-heavy response is boosted above conversational baseline by
    # protected_zones.salience_for_commit. 0.0 for pure prose or when PoQ
    # is disabled.
    artifact_score: float = 0.0

    def to_meta(self) -> dict:
        """
        Compact form for storage in a record's `_meta.poq` block. Kept
        small — the full dimension breakdown is useful at scoring time
        but the chain only needs the score, action, and alert flag.
        """
        return {
            "brightness": round(self.brightness, 4),
            "action": self.action,
            "integrity_alert": self.has_integrity_alert,
            "dimensions": {k: round(v, 4) for k, v in self.dimensions.items()},
        }

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
        """
        retrieved_texts = retrieved_texts or []
        prior_inputs = prior_inputs or []

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

        # Risk is the max integrity signal across the candidate AND the
        # input — a clean candidate answering a malicious prompt should
        # still raise the risk term.
        risk = max(
            cand_report.axes.get("integrity_risk", 0.0),
            input_report.axes.get("integrity_risk", 0.0),
        )
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
        )

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
