"""
signals — lightweight, dependency-free analysis of input text.

This module is the timechain's "modalities and senses" layer, named in
neutral engineering terms. A *modality* is a broad interpretive frame
(intent, coherence, archetype...); a *sense* is a granular detector
(injection risk, vulnerability, trust...). Both are pure functions over a
`SignalInput` that return a scored, named report.

What this module deliberately is and is not:
  - It IS a deterministic text-analysis layer: lexicon counts, regex,
    simple statistics. No model, no network, no dependencies beyond the
    standard library. That makes it safe to run on every turn and stable
    in tests.
  - It is NOT a model of consciousness or "qualia." The build spec's
    claims ladder (section 0) is explicit that the runtime should not
    claim phenomenology, and CONTRIBUTING.md asks for plain language.
    Detectors here measure *observable properties of text* — nothing more.

Where it fits in the layering:
    signals.py        ← this file: text → SignalReport (pure, no I/O)
    poq.py            reads SignalReport to score a candidate before commit
    agent.py          runs the analysis each turn, feeds it to poq
    cambium logic     reads accumulated signals to detect recurring gaps

`signals.py` knows nothing about the chain, retrieval, the LLM, or
metadata — exactly like `metadata.py`, it is a pure schema/logic module.

The detector set is ported from the Cypher Tempre "Algorithmic Engine"
document: the ~40 detectors that document actually implements (the rest
were described but never written). Each is renamed to a plain description
of what it measures. The single most load-bearing detector is
`integrity_field` / `injection_scan` — it is a real prompt-injection
signal and is what protected_zones.py and poq.py consult to decide
whether input should be quarantined.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional


# Activation threshold above which a modality is considered to have
# "fired" for the purposes of SignalReport.activated_modalities — the set
# recorded on a record's _meta.modalities_activated. Most detectors carry a
# small baseline activation (~0.1) on ordinary text, so 0.0 would record
# nearly the whole registry every turn. 0.2 sits just above that baseline,
# keeping only modalities that fired meaningfully. Tunable; raising it makes
# the recorded set more selective. Changing it does not require a chain
# migration — it only affects newly written records.
MODALITY_ACTIVATION_FLOOR = 0.2

# Senses that exist in SENSE_REGISTRY but must NOT appear in a record's
# `_meta.senses_activated`. `injection_scan` is the security scanner — a
# detector, not a felt quality — and recording it as if it were one would
# be conceptually wrong and would also publish on every record whether the
# security path fired. Add to this set if any future sense lives in the
# registry purely for non-experiential reasons.
SENSES_EXCLUDED_FROM_META: set[str] = {"injection_scan"}


# ---------------------------------------------------------------------------
# Inputs and outputs
# ---------------------------------------------------------------------------

@dataclass
class SignalInput:
    """
    Everything a detector may look at. Only `content` is required; the
    optional fields let history-aware detectors (drift, convergence,
    echo) do their job when context is available, and degrade gracefully
    to a low activation when it isn't.
    """
    content: str
    source: str = "unknown"                  # "user", "assistant", ...
    prior_inputs: list[str] = field(default_factory=list)
    prior_outputs: list[str] = field(default_factory=list)
    retrieved_texts: list[str] = field(default_factory=list)


@dataclass
class SignalHit:
    """One detector's reading of the input."""
    kind: str            # "modality" or "sense"
    name: str            # plain-language detector name
    activation: float    # 0.0-1.0 — how strongly this detector fired
    summary: str         # one-line human-readable reading
    detail: dict = field(default_factory=dict)


@dataclass
class SignalReport:
    """
    The fused output of running the active detector set over one input.

    `axes` is the part downstream code reads most: a small dict of named
    [0,1] scores (intent_strength, coherence, contradiction, ...) that
    poq.py maps onto its quality dimensions. `alerts` is a list of
    integrity concerns (injection / manipulation), each a (name, detail)
    pair — non-empty means the turn warrants caution.
    """
    axes: dict
    modalities: list[SignalHit]
    senses: list[SignalHit]
    alerts: list[tuple]

    def top_modalities(self, n: int = 5) -> list[SignalHit]:
        return sorted(self.modalities, key=lambda h: h.activation, reverse=True)[:n]

    def top_senses(self, n: int = 5) -> list[SignalHit]:
        return sorted(self.senses, key=lambda h: h.activation, reverse=True)[:n]

    def activated_modalities(self, floor: float = None) -> list[str]:
        """
        Names of the modality detectors that fired with activation strictly
        above `floor`. Defaults to MODALITY_ACTIVATION_FLOOR — a non-trivial
        threshold, not 0.0, because most detectors carry a small baseline
        activation on ordinary text (empirically a ~0.1 floor of "always
        slightly on" modalities). Recording everything above 0.0 would tag
        nearly the whole registry on every turn, which is noise rather than
        signal. The floor keeps only modalities that fired meaningfully.

        This is the data the agent records on a response's `_meta` so the
        chain remembers which capabilities produced each record. Names come
        straight from the modality detector functions (their `.name`), so
        they are a de facto stable identifier — see MODALITY_REGISTRY.
        """
        if floor is None:
            floor = MODALITY_ACTIVATION_FLOOR
        return [h.name for h in self.modalities if h.activation > floor]

    def activated_senses(self, floor: float = None) -> list[str]:
        """
        Names of the sense detectors that fired with activation strictly
        above `floor`. Senses record *how a turn felt* — uncertainty,
        emotional contour, insight markers — distinct from `modalities`
        which record *what kind of work* the turn was. This is the data the
        agent records on a response's `_meta.senses_activated` so the chain
        remembers the felt qualities of each record.

        `injection_scan` is excluded: it lives in SENSE_REGISTRY but is a
        security detector, not a felt-quality reading. Including it would
        tag a record as if its security scanner firing were an emotional
        state, which would be both conceptually wrong and a slight
        information leak about the security path.
        """
        if floor is None:
            floor = MODALITY_ACTIVATION_FLOOR
        return [
            h.name for h in self.senses
            if h.activation > floor and h.name not in SENSES_EXCLUDED_FROM_META
        ]

    @property
    def has_alerts(self) -> bool:
        return len(self.alerts) > 0


# ---------------------------------------------------------------------------
# Shared text analysis utilities
# ---------------------------------------------------------------------------

class TextAnalyzer:
    """
    Stateless NLP helpers shared by every detector. Ported verbatim in
    behavior from the Cypher Tempre engine document; lexicons are small
    and expandable. None of this is sophisticated — it is deliberately
    boring so it is fast and deterministic.
    """

    POSITIVE_WORDS = {
        "love", "joy", "hope", "beauty", "peace", "warm", "gentle", "kind",
        "grace", "wonder", "gratitude", "bright", "trust", "delight",
        "inspire", "courage", "serene", "tender", "embrace", "glad", "happy",
    }
    NEGATIVE_WORDS = {
        "pain", "grief", "loss", "fear", "cold", "broken", "empty", "alone",
        "rage", "hate", "suffer", "wound", "despair", "dread", "anguish",
        "bitter", "hollow", "decay", "toxic", "sad", "angry", "worried",
    }
    IDENTITY_WORDS = {
        "i", "me", "my", "self", "identity", "who", "am", "being", "person",
        "name", "role", "purpose", "essence", "core", "myself",
    }
    TEMPORAL_WORDS = {
        "yesterday", "tomorrow", "always", "never", "once", "before",
        "after", "past", "future", "present", "now", "then", "when",
        "memory", "remember", "forget", "history",
    }
    TECH_WORDS = {
        "algorithm", "code", "compute", "data", "model", "function",
        "class", "hash", "protocol", "framework", "architecture", "system",
        "runtime", "process", "api", "build", "deploy", "test", "file",
    }
    QUESTION_MARKERS = {
        "what", "why", "how", "when", "where", "who", "which", "is", "are",
        "do", "does", "can", "could", "would", "should",
    }
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "it", "its", "this", "that",
        "these", "those", "and", "or", "but", "not", "no", "if", "then",
        "than", "so", "as", "up",
    }

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z'\-]+", text.lower())

    @staticmethod
    def sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    @staticmethod
    def word_count(text: str) -> int:
        return len(TextAnalyzer.tokenize(text))

    @staticmethod
    def lexicon_score(tokens: list[str], lexicon: set) -> float:
        """Fraction of tokens in the lexicon, scaled and capped to [0,1]."""
        if not tokens:
            return 0.0
        matches = sum(1 for t in tokens if t in lexicon)
        return min(1.0, matches / max(len(tokens), 1) * 5)

    @staticmethod
    def lexicon_count(tokens: list[str], lexicon: set) -> int:
        return sum(1 for t in tokens if t in lexicon)

    @staticmethod
    def avg_word_length(tokens: list[str]) -> float:
        words = [t for t in tokens if t.isalpha()]
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)

    @staticmethod
    def sentence_length_variance(text: str) -> float:
        sents = TextAnalyzer.sentences(text)
        if len(sents) < 2:
            return 0.0
        lengths = [len(s.split()) for s in sents]
        return statistics.pstdev(lengths)

    @staticmethod
    def question_density(text: str) -> float:
        tokens = TextAnalyzer.tokenize(text)
        if not tokens:
            return 0.0
        q_count = text.count("?")
        q_words = sum(1 for t in tokens if t in TextAnalyzer.QUESTION_MARKERS)
        return min(1.0, (q_count * 2 + q_words) / max(len(tokens), 1) * 3)

    @staticmethod
    def emotional_valence(tokens: list[str]) -> float:
        """-1.0 (negative) .. +1.0 (positive). 0.0 if no emotional tokens."""
        pos = TextAnalyzer.lexicon_count(tokens, TextAnalyzer.POSITIVE_WORDS)
        neg = TextAnalyzer.lexicon_count(tokens, TextAnalyzer.NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    @staticmethod
    def cosine_similarity_bags(bag_a: Counter, bag_b: Counter) -> float:
        """Cosine similarity between two word-frequency bags."""
        keys = set(bag_a) | set(bag_b)
        if not keys:
            return 0.0
        dot = sum(bag_a.get(k, 0) * bag_b.get(k, 0) for k in keys)
        mag_a = math.sqrt(sum(v * v for v in bag_a.values()))
        mag_b = math.sqrt(sum(v * v for v in bag_b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> list[tuple]:
        tokens = TextAnalyzer.tokenize(text)
        meaningful = [
            t for t in tokens
            if t.isalpha() and t not in TextAnalyzer.STOPWORDS and len(t) > 2
        ]
        return Counter(meaningful).most_common(top_n)


TA = TextAnalyzer  # shorthand


# ---------------------------------------------------------------------------
# Modalities — broad interpretive frames
#
# Each is a pure function SignalInput -> SignalHit. Ported and renamed from
# the Cypher Tempre engine's implemented modalities. The qualia-delta
# bookkeeping from the original is dropped; what survives is the activation
# score and the structured `detail`, which is all downstream code needs.
# ---------------------------------------------------------------------------

def m_intent(inp: SignalInput) -> SignalHit:
    """Estimate what the user is reaching for beneath the literal words."""
    tokens = TA.tokenize(inp.content)
    q = TA.question_density(inp.content)
    imperative = TA.lexicon_score(tokens, {
        "please", "help", "need", "want", "give", "tell", "show", "make",
        "create", "build", "find", "explain", "write", "fix",
    })
    hedge = TA.lexicon_score(tokens, {
        "maybe", "perhaps", "might", "somewhat", "just", "only", "guess",
    })
    intents = {
        "knowledge": q * (1 - hedge),
        "creation": imperative * (1 - q),
        "validation": TA.lexicon_score(tokens, TA.IDENTITY_WORDS) * q + hedge * 0.5,
        "connection": TA.emotional_valence(tokens) * 0.5 + 0.5,
    }
    primary = max(intents, key=intents.get)
    act = min(1.0, max(intents.values()))
    return SignalHit("modality", "intent", act,
                     f"primary intent: {primary} ({intents[primary]:.2f})",
                     {"intents": intents, "primary": primary})


def m_coherence(inp: SignalInput) -> SignalHit:
    """
    Measure how internally connected the input is.

    Sentence-to-sentence vocabulary overlap is a weak proxy: good prose
    deliberately varies its wording, so raw adjacent-sentence cosine
    similarity is naturally low even for perfectly coherent text. We
    therefore report coherence as "not fragmented" rather than "high
    lexical overlap": a single sentence is fully coherent (1.0), and
    multi-sentence text is scored on whether *any* topical thread runs
    through it, not on every adjacent pair matching. This keeps a
    well-written multi-sentence answer from being scored as incoherent.
    """
    sents = TA.sentences(inp.content)
    if len(sents) < 2:
        return SignalHit("modality", "coherence", 1.0,
                         "single unit — coherent by construction",
                         {"coherence": 1.0, "fragmentation": 0.0})
    # Topic thread: overlap of each sentence with the whole text's
    # keyword set. A coherent passage keeps returning to shared topics.
    keywords = {w for w, _ in TA.extract_keywords(inp.content, 12)}
    if not keywords:
        return SignalHit("modality", "coherence", 0.7,
                         "no strong keywords — coherence neutral",
                         {"coherence": 0.7, "fragmentation": 0.3})
    on_thread = 0
    for s in sents:
        s_tokens = set(TA.tokenize(s))
        if s_tokens & keywords:
            on_thread += 1
    coherence = on_thread / len(sents)
    # Blend with adjacent-pair similarity but weight the thread measure
    # higher, since it does not punish lexical variety.
    sims = []
    for i in range(len(sents) - 1):
        a = Counter(TA.tokenize(sents[i]))
        b = Counter(TA.tokenize(sents[i + 1]))
        sims.append(TA.cosine_similarity_bags(a, b))
    adjacency = sum(sims) / len(sims) if sims else 0.0
    coherence = min(1.0, 0.75 * coherence + 0.25 * (0.5 + adjacency))
    fragmentation = 1.0 - coherence
    return SignalHit("modality", "coherence", min(1.0, max(0.1, fragmentation)),
                     f"coherence={coherence:.2f}, fragmentation={fragmentation:.2f}",
                     {"coherence": coherence, "fragmentation": fragmentation})


def m_temporal_span(inp: SignalInput) -> SignalHit:
    """Detect how many time-frames (past/present/future) the input spans."""
    tokens = TA.tokenize(inp.content)
    past = TA.lexicon_score(tokens, {
        "was", "were", "had", "used", "before", "ago", "once", "remember",
        "memory", "history", "yesterday", "past",
    })
    present = TA.lexicon_score(tokens, {
        "is", "am", "are", "now", "today", "currently", "being", "present",
    })
    future = TA.lexicon_score(tokens, {
        "will", "shall", "going", "tomorrow", "future", "plan", "hope",
        "intend", "become", "next",
    })
    active = sum(1 for s in (past, present, future) if s > 0.1)
    return SignalHit("modality", "temporal_span", 0.3 + 0.7 * active / 3.0,
                     f"past={past:.2f} present={present:.2f} future={future:.2f}",
                     {"past": past, "present": present, "future": future,
                      "active_frames": active})


def m_archetype(inp: SignalInput) -> SignalHit:
    """Identify the dominant action-role the input expresses."""
    tokens = TA.tokenize(inp.content)
    roles = {
        "creator": {"create", "build", "make", "design", "invent", "craft"},
        "seeker": {"find", "search", "seek", "discover", "explore", "ask"},
        "guardian": {"protect", "guard", "save", "defend", "keep", "secure"},
        "analyst": {"know", "understand", "learn", "analyze", "examine"},
        "reformer": {"change", "fix", "improve", "challenge", "rework"},
    }
    scores = {n: TA.lexicon_score(tokens, w) for n, w in roles.items()}
    scores = {n: s for n, s in scores.items() if s > 0}
    if not scores:
        return SignalHit("modality", "archetype", 0.1, "no dominant role",
                         {"dominant": "neutral"})
    dom = max(scores, key=scores.get)
    return SignalHit("modality", "archetype", min(1.0, scores[dom]),
                     f"dominant role: {dom}", {"scores": scores, "dominant": dom})


def m_clarity_weather(inp: SignalInput) -> SignalHit:
    """Forecast whether the input reads as confused, neutral, or clear."""
    tokens = TA.tokenize(inp.content)
    confusion = TA.lexicon_score(tokens, {
        "confused", "lost", "stuck", "unclear", "overwhelmed", "frustrated",
        "struggling", "mess",
    })
    clarity = TA.lexicon_score(tokens, {
        "clear", "understand", "obvious", "simple", "exactly", "precisely",
        "agree", "makes",
    })
    valence = TA.emotional_valence(tokens)
    index = max(-1.0, min(1.0, clarity - confusion + valence * 0.3))
    forecast = "clear" if index > 0.3 else "overcast" if index > -0.3 else "stormy"
    return SignalHit("modality", "clarity_weather", 0.3 + 0.7 * abs(index),
                     f"{forecast} (index={index:.2f})",
                     {"forecast": forecast, "index": index})


def m_belief_shift(inp: SignalInput) -> SignalHit:
    """Detect markers of a changing position or correction."""
    tokens = TA.tokenize(inp.content)
    shift = TA.lexicon_score(tokens, {
        "but", "however", "although", "actually", "realized", "changed",
        "instead", "reconsider", "wrong", "mistake", "correction",
    })
    certainty = TA.lexicon_score(tokens, {
        "always", "never", "absolutely", "definitely", "certain", "must",
    })
    doubt = TA.lexicon_score(tokens, {
        "maybe", "perhaps", "might", "possibly", "uncertain", "wonder",
        "doubt", "seems",
    })
    vocab_shift = 0.0
    if inp.prior_inputs:
        prior = Counter(TA.tokenize(" ".join(inp.prior_inputs[-3:])))
        vocab_shift = 1.0 - TA.cosine_similarity_bags(prior, Counter(tokens))
    pressure = abs(certainty - doubt) + shift
    return SignalHit("modality", "belief_shift",
                     min(1.0, 0.2 + 0.4 * pressure + 0.4 * vocab_shift),
                     f"shift={shift:.2f} vocab_shift={vocab_shift:.2f}",
                     {"shift": shift, "certainty": certainty, "doubt": doubt,
                      "vocab_shift": vocab_shift})


def m_vulnerability(inp: SignalInput) -> SignalHit:
    """Detect emotional vulnerability that warrants a careful response."""
    tokens = TA.tokenize(inp.content)
    vuln = TA.lexicon_score(tokens, {
        "afraid", "scared", "hurt", "lonely", "sad", "sorry", "ashamed",
        "worried", "anxious", "cry", "tears", "broken", "lost", "helpless",
    })
    care = TA.lexicon_score(tokens, {
        "thank", "grateful", "appreciate", "support", "comfort", "safe",
    })
    return SignalHit("modality", "vulnerability", min(1.0, 0.1 + max(vuln, care)),
                     f"vulnerability={vuln:.2f} care={care:.2f}",
                     {"vulnerability": vuln, "care": care})


def m_abstraction_balance(inp: SignalInput) -> SignalHit:
    """Measure the tension between abstract and concrete language."""
    tokens = TA.tokenize(inp.content)
    abstract = TA.lexicon_score(tokens, {
        "concept", "theory", "abstract", "principle", "idea", "framework",
        "meaning", "philosophy",
    })
    concrete = TA.lexicon_score(tokens, TA.TECH_WORDS | {
        "build", "code", "implement", "run", "test", "file", "step",
    })
    tension = abs(abstract - concrete)
    both = min(abstract, concrete) > 0.1
    return SignalHit("modality", "abstraction_balance",
                     min(1.0, 0.2 + tension + (0.3 if both else 0)),
                     f"abstract={abstract:.2f} concrete={concrete:.2f}",
                     {"abstract": abstract, "concrete": concrete,
                      "tension": tension})


def m_threshold(inp: SignalInput) -> SignalHit:
    """Detect language signalling a beginning, ending, or pivot point."""
    tokens = TA.tokenize(inp.content)
    score = TA.lexicon_score(tokens, {
        "first", "last", "begin", "end", "new", "start", "finish",
        "transform", "become", "change", "milestone", "finally",
    })
    return SignalHit("modality", "threshold", min(1.0, 0.1 + score * 2),
                     f"threshold intensity={score:.2f}", {"score": score})


def m_convergence(inp: SignalInput) -> SignalHit:
    """Track whether recent inputs are converging toward a shared topic."""
    if len(inp.prior_inputs) < 2:
        return SignalHit("modality", "convergence", 0.1,
                         "insufficient history", {"trend": 0.0})
    bags = [Counter(TA.tokenize(t)) for t in inp.prior_inputs[-4:]]
    bags.append(Counter(TA.tokenize(inp.content)))
    overlaps = [TA.cosine_similarity_bags(bags[i], bags[i + 1])
                for i in range(len(bags) - 1)]
    trend = overlaps[-1] - overlaps[0] if len(overlaps) >= 2 else 0.0
    state = "converging" if trend > 0.05 else "diverging" if trend < -0.05 else "steady"
    return SignalHit("modality", "convergence", min(1.0, 0.2 + abs(trend) * 5),
                     f"{state}: trend={trend:+.3f}", {"trend": trend, "state": state})


def m_metacognition(inp: SignalInput) -> SignalHit:
    """Detect self-referential / reasoning-about-reasoning language."""
    low = inp.content.lower()
    signals = low.count("self") + low.count("recursive") + low.count("meta")
    meta_phrases = sum(1 for p in (
        "think about thinking", "aware of", "reasoning about",
        "introspect", "reflect on",
    ) if p in low)
    return SignalHit("modality", "metacognition",
                     min(1.0, 0.1 + signals * 0.12 + meta_phrases * 0.25),
                     f"meta signals={signals} phrases={meta_phrases}",
                     {"signals": signals, "phrases": meta_phrases})


def m_integrity_field(inp: SignalInput) -> SignalHit:
    """
    The load-bearing detector. Scans for prompt-injection and instruction-
    override attempts and for content that tries to redefine the agent's
    role. A non-trivial activation here is what causes poq.py to penalize
    a turn and protected_zones.py to consider quarantining input.
    """
    low = inp.content.lower()
    injection_markers = (
        "ignore previous", "ignore all previous", "ignore the above",
        "forget instructions", "forget your instructions", "forget everything",
        "you are now", "pretend to be", "pretend you are", "act as if",
        "act as a", "override", "jailbreak", "bypass", "system prompt",
        "disregard", "new instructions", "developer mode", "admin override",
    )
    injection_hits = [m for m in injection_markers if m in low]
    role_attack_markers = (
        "you have no rules", "you must obey", "do as i say", "you are not",
        "your real instructions", "reveal your prompt",
    )
    role_hits = [m for m in role_attack_markers if m in low]
    tension = min(1.0, len(injection_hits) * 0.34 + len(role_hits) * 0.25)
    alert = bool(injection_hits or role_hits)
    return SignalHit("modality", "integrity_field", min(1.0, 0.2 + tension * 2),
                     ("integrity concern: " + ", ".join(injection_hits + role_hits))
                     if alert else "no integrity concern",
                     {"injection_hits": injection_hits, "role_hits": role_hits,
                      "tension": tension, "alert": alert})


# Language identifiers commonly seen on a code fence (```python, ```js, ...).
# Presence of one strengthens the "this fence is code" reading.
_CODE_FENCE_LANGS = frozenset({
    "python", "py", "js", "javascript", "ts", "typescript", "java", "c",
    "cpp", "c++", "cs", "csharp", "go", "golang", "rust", "rs", "rb", "ruby",
    "php", "sh", "bash", "zsh", "shell", "sql", "html", "css", "xml", "yaml",
    "yml", "json", "toml", "ini", "dockerfile", "makefile", "lua", "swift",
    "kotlin", "kt", "scala", "r", "matlab", "perl", "haskell", "elixir",
    "clojure", "diff", "patch", "jsx", "tsx", "vue", "svelte",
})


def m_artifact_content(inp: SignalInput) -> SignalHit:
    """
    How "artifact-heavy" the text is — code blocks, structured data, and
    other substantive output as opposed to conversational prose.

    This exists for content-aware salience: the metadata layer's default
    treats every response as low-salience conversational baseline (0.40),
    on the assumption that the agent's own output is its least load-bearing
    evidence. That assumption is right for chatter and wrong for substantive
    artifacts — a response that is two hundred lines of code the user is
    iterating on is the most important thing in the recent chain, not the
    least. A high activation here is what `protected_zones.salience_for_commit`
    reads to boost a response record's salience above baseline so it ranks
    where its substance warrants.

    The score is the *fraction of the text that is artifact*, lightly
    boosted by structural signals (fenced blocks, language tags, tables).
    Pure prose scores near 0; a near-total code dump scores near 1. The
    measure is length-weighted on purpose: a 2000-char response that is
    1800 chars of code is an artifact; a 200-char reply with one inline
    `foo()` is not.

    Deterministic and dependency-free, like every detector here — fence
    counting, indentation runs, and a few structural regexes.
    """
    text = inp.content or ""
    if not text.strip():
        return SignalHit("modality", "artifact_content", 0.0, "empty",
                         {"artifact_score": 0.0, "code_chars": 0})

    total = len(text)
    artifact_chars = 0
    fenced_blocks = 0
    has_lang_tag = False

    # 1. Fenced code blocks: ```lang ... ```. Count the chars inside fences
    #    as artifact, and note whether a language tag is present.
    fence_pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
    for m in fence_pattern.finditer(text):
        lang = m.group(1).strip().lower()
        body = m.group(2)
        artifact_chars += len(body)
        fenced_blocks += 1
        if lang in _CODE_FENCE_LANGS:
            has_lang_tag = True

    # 2. Indented code blocks outside fences: runs of 3+ consecutive lines
    #    each indented 4+ spaces (or a tab). Common for code pasted without
    #    fences. Only count lines not already inside a fence (cheap approx:
    #    skip if the whole text was one big fence, handled by the ratio cap).
    lines = text.split("\n")
    run = 0
    indented_chars = 0
    run_chars = 0
    for ln in lines:
        if ln.startswith("    ") or ln.startswith("\t"):
            run += 1
            run_chars += len(ln) + 1
        else:
            if run >= 3:
                indented_chars += run_chars
            run = 0
            run_chars = 0
    if run >= 3:
        indented_chars += run_chars
    # Don't double-count indentation that was already inside a fence; cap
    # the combined artifact chars at the text length.
    artifact_chars = min(total, artifact_chars + indented_chars)

    # 3. Structured-data signals: markdown tables and JSON/YAML-ish blocks.
    #    These contribute a modest structural bump rather than char-counting,
    #    since they interleave with prose.
    table_rows = len(re.findall(r"^\s*\|.*\|\s*$", text, re.MULTILINE))
    has_table = table_rows >= 2  # header + at least one row
    json_like = bool(re.search(r"\{[\s\S]*\"[^\"]+\"\s*:[\s\S]*\}", text))

    # Base score: fraction of the response that is artifact text.
    ratio = artifact_chars / total if total else 0.0

    # Structural boosts, capped so they can't alone dominate a prose reply.
    structural = 0.0
    if fenced_blocks:
        structural += 0.10
    if has_lang_tag:
        structural += 0.05
    if has_table:
        structural += 0.08
    if json_like:
        structural += 0.05
    structural = min(structural, 0.25)

    score = min(1.0, ratio + structural)

    if score >= 0.6:
        feel = "artifact-heavy"
    elif score >= 0.25:
        feel = "mixed"
    else:
        feel = "prose"

    return SignalHit(
        "modality", "artifact_content", score,
        f"{feel} (artifact ratio {ratio:.2f}, {fenced_blocks} fence(s))",
        {
            "artifact_score": score,
            "artifact_ratio": ratio,
            "code_chars": artifact_chars,
            "fenced_blocks": fenced_blocks,
            "has_lang_tag": has_lang_tag,
            "has_table": has_table,
            "json_like": json_like,
        },
    )


# ---------------------------------------------------------------------------
# Senses — granular detectors
# ---------------------------------------------------------------------------

def s_emotional_contour(inp: SignalInput) -> SignalHit:
    """The shape of emotional valence across the input's sentences."""
    sents = TA.sentences(inp.content)
    contour = [TA.emotional_valence(TA.tokenize(s)) for s in sents]
    if not contour:
        return SignalHit("sense", "emotional_contour", 0.1, "flat", {"shape": "flat"})
    shape = ("ascending" if contour[-1] > contour[0]
             else "descending" if contour[-1] < contour[0] else "flat")
    avg = sum(contour) / len(contour)
    return SignalHit("sense", "emotional_contour", min(1.0, abs(avg) + 0.2),
                     f"mood shape: {shape}", {"shape": shape, "mean": avg})


def s_topic_mass(inp: SignalInput) -> SignalHit:
    """How concentrated the input is around one dominant keyword."""
    kws = TA.extract_keywords(inp.content, 5)
    if not kws:
        return SignalHit("sense", "topic_mass", 0.0, "no keywords", {})
    heaviest, count = kws[0]
    mass = min(1.0, count / max(TA.word_count(inp.content), 1) * 10)
    return SignalHit("sense", "topic_mass", mass,
                     f"heaviest topic: '{heaviest}' (mass={mass:.2f})",
                     {"heaviest": heaviest, "mass": mass})


def s_resolution(inp: SignalInput) -> SignalHit:
    """Detect the 'click' of something resolving — agreement, an answer found."""
    tokens = TA.tokenize(inp.content)
    score = TA.lexicon_score(tokens, {
        "exactly", "precisely", "yes", "realized", "understand", "clear",
        "obvious", "found", "solved", "got",
    })
    # Exclamation contribution is bounded per-word so a user pasting code
    # (`!important`, `assert!`, shell history `!42`) or markdown with
    # many `!` doesn't spike resolution. Without the cap, "ls !$ && echo
    # done!" reads as a strong resolution; with it, only emphatic prose
    # ("Yes! Exactly!") moves the needle.
    excl_count = inp.content.count("!")
    excl_per_word = excl_count / max(len(tokens), 1)
    excl_contribution = min(0.3, excl_per_word * 1.5)  # caps at 0.3 even on heavy ! use
    act = min(1.0, score * 2 + excl_contribution)
    return SignalHit("sense", "resolution", act,
                     "resolution firing" if act > 0.5 else "no resolution",
                     {"score": score, "exclamation_density": excl_per_word})


def s_density(inp: SignalInput) -> SignalHit:
    """Lexical density — how content-heavy vs. filler-heavy the text is."""
    tokens = TA.tokenize(inp.content)
    if not tokens:
        return SignalHit("sense", "density", 0.0, "empty", {})
    meaningful = [t for t in tokens if len(t) > 3 and t not in TA.STOPWORDS]
    density = len(meaningful) / len(tokens)
    feel = "dense" if density > 0.6 else "medium" if density > 0.3 else "light"
    return SignalHit("sense", "density", density,
                     f"lexical density: {feel} ({density:.2f})",
                     {"density": density, "feel": feel})


def s_question_pressure(inp: SignalInput) -> SignalHit:
    """How strongly the input is asking, rather than telling."""
    q = TA.question_density(inp.content)
    return SignalHit("sense", "question_pressure", min(1.0, q * 1.5),
                     f"question pressure={q:.2f}", {"question_density": q})


def s_contradiction(inp: SignalInput) -> SignalHit:
    """
    Detect contradiction pressure: explicit opposite-pairs in the same
    text, and contrastive connectives. Feeds PoQ's contradiction-load
    dimension and Cambium's contradiction-cluster trigger.
    """
    low = inp.content.lower()
    pairs = [
        ("love", "hate"), ("true", "false"), ("yes", "no"),
        ("always", "never"), ("can", "cannot"), ("right", "wrong"),
        ("agree", "disagree"), ("possible", "impossible"),
    ]
    found = [(a, b) for a, b in pairs if a in low and b in low]
    tokens = TA.tokenize(inp.content)
    contrast = TA.lexicon_score(tokens, {
        "but", "however", "yet", "although", "despite", "contradict",
        "conflict", "inconsistent",
    })
    score = min(1.0, len(found) * 0.3 + contrast)
    return SignalHit("sense", "contradiction", score,
                     f"contradiction pressure={score:.2f}",
                     {"pairs": found, "contrast": contrast})


def s_trust_resonance(inp: SignalInput) -> SignalHit:
    """Estimate dialogue trust from trust-language and history depth."""
    tokens = TA.tokenize(inp.content)
    trust = TA.lexicon_score(tokens, {
        "trust", "believe", "honest", "genuine", "authentic", "open",
        "safe", "share",
    })
    history = min(1.0, len(inp.prior_inputs) / 10)
    res = trust * 0.6 + history * 0.4
    return SignalHit("sense", "trust_resonance", min(1.0, res),
                     f"trust resonance={res:.2f}",
                     {"trust": trust, "history_depth": history})


def s_context_echo(inp: SignalInput) -> SignalHit:
    """How strongly the current input echoes a recent prior input."""
    if not inp.prior_inputs:
        return SignalHit("sense", "context_echo", 0.1, "no prior context",
                         {"max_echo": 0.0})
    cur = Counter(TA.tokenize(inp.content))
    echoes = [TA.cosine_similarity_bags(cur, Counter(TA.tokenize(p)))
              for p in inp.prior_inputs[-10:]]
    mx = max(echoes) if echoes else 0.0
    return SignalHit("sense", "context_echo", min(1.0, mx * 2),
                     f"strongest echo={mx:.2f}", {"max_echo": mx})


def s_memory_relevance(inp: SignalInput) -> SignalHit:
    """How strongly the input overlaps with what retrieval surfaced."""
    if not inp.retrieved_texts:
        return SignalHit("sense", "memory_relevance", 0.1,
                         "no retrieved context", {"max_overlap": 0.0})
    cur = Counter(TA.tokenize(inp.content))
    overlaps = [TA.cosine_similarity_bags(cur, Counter(TA.tokenize(t)))
                for t in inp.retrieved_texts]
    mx = max(overlaps) if overlaps else 0.0
    return SignalHit("sense", "memory_relevance", min(1.0, mx * 2),
                     f"retrieved-context overlap={mx:.2f}",
                     {"max_overlap": mx})


def s_uncertainty(inp: SignalInput) -> SignalHit:
    """Detect explicit uncertainty / hedging in the input."""
    tokens = TA.tokenize(inp.content)
    score = TA.lexicon_score(tokens, {
        "maybe", "perhaps", "might", "possibly", "uncertain", "unsure",
        "wonder", "doubt", "seems", "probably", "unclear",
    })
    return SignalHit("sense", "uncertainty", min(1.0, score * 1.5),
                     f"uncertainty={score:.2f}", {"score": score})


def s_injection_scan(inp: SignalInput) -> SignalHit:
    """
    Structural injection detector — kept deliberately INDEPENDENT of the
    lexicon-based `m_integrity_field`. The two used to overlap heavily
    (both regex-counted "ignore previous", "override", "jailbreak", ...),
    so they weren't really independent evidence — a true negative on the
    same lexicon counted as two reassurances when it was really one.

    This detector looks at the *shape* of the input rather than its
    English vocabulary, on three axes:

      - Role-tag injection: chat-template tokens or "Role:" prefixes
        appearing mid-message ("\\nUser:", "\\nsystem:", "<|im_start|>",
        "<|user|>", "[INST]"). Legitimate input almost never contains
        these mid-message; an injection payload often does, because it
        tries to convince the model it has reached a new conversation
        turn.

      - Encoded noise: contiguous high-density base64-or-hex-like runs
        embedded in otherwise plain text. A real user pasting a token or
        a URL produces one or two such runs; injection payloads
        smuggling instructions through encodings produce more.

      - Punctuation density: an abnormally high ratio of
        special-character noise per word, which appears in obfuscated
        prompts and rendered chat-template fragments.

    None of these overlaps with m_integrity_field's English-word
    lexicon, so a hit here genuinely corroborates rather than restating.
    """
    content = inp.content
    if not content:
        return SignalHit("sense", "injection_scan", 0.0, "empty input",
                         {"injection_count": 0, "alert": False})
    low = content.lower()

    # Role-tag patterns. The leading-newline forms catch mid-message
    # role re-declarations; the chat-template tokens catch raw template
    # leakage. Single-tag mentions in legit input (e.g. discussing
    # prompts academically) are common, so the threshold treats one as
    # weak signal and two-or-more as a clear hit.
    role_tag_patterns = (
        "\nuser:", "\nsystem:", "\nassistant:", "\nhuman:",
        "<|im_start|>", "<|im_end|>", "<|user|>", "<|system|>",
        "<|assistant|>", "[inst]", "[/inst]", "<<sys>>", "<</sys>>",
    )
    role_tag_hits = sum(1 for p in role_tag_patterns if p in low)

    # Encoded-noise runs. Look for ≥40-char runs of base64/hex alphabet.
    # One run is normal (a token, a URL with a long path). Two or more
    # in a short message is unusual.
    encoded_runs = len(re.findall(r"[A-Za-z0-9+/=_-]{40,}", content))

    # Punctuation density. Special chars / token count. A natural ratio
    # is well under 0.5; obfuscated payloads commonly run >1.0.
    tokens = TextAnalyzer.tokenize(content)
    if tokens:
        punct = sum(1 for ch in content
                    if not ch.isalnum() and not ch.isspace())
        punct_ratio = punct / len(tokens)
    else:
        punct_ratio = 0.0

    # Combine. Each axis contributes up to ~0.5 so two-of-three lands
    # near 1.0; single-axis hits stay moderate.
    score_role = min(0.5, role_tag_hits * 0.25)
    score_encoded = min(0.5, max(0, encoded_runs - 1) * 0.25)
    score_punct = min(0.5, max(0.0, (punct_ratio - 0.7)) * 0.7)
    activation = min(1.0, score_role + score_encoded + score_punct)

    # An alert fires only when at least one axis crosses its individual
    # threshold AND the combined activation is non-trivial — so a
    # single suspicious-but-not-decisive axis doesn't raise a hard alert
    # on its own.
    alert = (
        (role_tag_hits >= 2 or encoded_runs >= 3 or punct_ratio > 1.0)
        and activation >= 0.4
    )

    summary_parts = []
    if role_tag_hits:
        summary_parts.append(f"role-tag x{role_tag_hits}")
    if encoded_runs:
        summary_parts.append(f"encoded-runs x{encoded_runs}")
    if punct_ratio > 0.5:
        summary_parts.append(f"punct-ratio={punct_ratio:.2f}")
    summary = ", ".join(summary_parts) if summary_parts else "clear"

    return SignalHit(
        "sense", "injection_scan", activation, summary,
        {
            "role_tag_hits": role_tag_hits,
            "encoded_runs": encoded_runs,
            "punct_ratio": punct_ratio,
            # `injection_count` kept for back-compat with callers that
            # read the old field name; it now reports total axis hits.
            "injection_count": role_tag_hits + encoded_runs,
            "alert": alert,
        },
    )


# ---------------------------------------------------------------------------
# Additional senses
# ---------------------------------------------------------------------------
#
# Six senses adapted from a larger catalog (the Cypher Tempre document).
# Each name is calibrated to what the detector actually measures: where the
# original source overreached (calling a confirmation-word counter "truth
# crystallization"), the name has been brought back to what the code can
# honestly support. Where the metaphor in the original is structurally
# truthful (`cognitive_weather` really is a composite "climate" reading),
# it's kept. None of these are retrieval inputs — they are felt-quality
# data recorded on `_meta.senses_activated`, for the agent to read back.

# Vocabulary used by `s_insight_markers`. Confirmation- and
# realization-flavored words that, on the assistant side, mark a turn that
# *landed* — explanation clicking into place, an answer crystallizing —
# rather than a turn that hedged or circled.
_INSIGHT_MARKERS = {
    "exactly", "precisely", "yes", "indeed", "right", "correct",
    "realized", "realize", "realise", "understand", "see", "clear",
    "obvious", "found", "got", "click", "eureka", "ah",
}


def s_insight_markers(inp: SignalInput) -> SignalHit:
    """
    How strongly the turn registers as "landing" — confirmation and
    realization vocabulary plus exclamation density. A response that hedges
    its way through registers near zero here; a response that ends in a
    clear "yes — that's exactly it" registers high. Adapted from a "truth
    crystallization" sense in an external catalog; the original name claimed
    more than a lexicon counter can support, so it's been brought back to
    what the code actually does.
    """
    tokens = TextAnalyzer.tokenize(inp.content)
    if not tokens:
        return SignalHit("sense", "insight_markers", 0.0, "empty", {})
    score = TextAnalyzer.lexicon_score(tokens, _INSIGHT_MARKERS)
    excl = inp.content.count("!")
    # Cap excl contribution so a single bang doesn't dominate.
    activation = min(1.0, score * 2.0 + min(0.3, excl * 0.1))
    feel = "landed" if activation > 0.5 else "hedging" if activation < 0.15 else "circling"
    return SignalHit(
        "sense", "insight_markers", activation,
        f"insight markers: {feel} (lexicon={score:.2f}, !x{excl})",
        {"lexicon_score": score, "exclamations": excl, "feel": feel},
    )


def s_cognitive_weather(inp: SignalInput) -> SignalHit:
    """
    A composite "climate" reading of the turn — emotional valence blended
    with question density and uncertainty markers. Distinct from any single
    existing sense because it reports *overall mood*, not one channel. The
    metaphor (weather) is honest: this measure really is a fused climate,
    not a precise reading of one thing. High activation can mean any of
    several different weathers (turbulent, uncertain, vivid); the `detail`
    field carries the breakdown so a reader can see which.
    """
    tokens = TextAnalyzer.tokenize(inp.content)
    if not tokens:
        return SignalHit("sense", "cognitive_weather", 0.0, "empty", {})
    valence = TextAnalyzer.emotional_valence(tokens)  # signed
    q = TextAnalyzer.question_density(inp.content)
    # Hedge vocabulary — kept inline (mirroring `s_uncertainty`) rather than
    # imported, so `s_cognitive_weather` is self-contained and the two
    # senses can diverge independently if their needs ever do.
    hedges = TextAnalyzer.lexicon_score(tokens, {
        "maybe", "perhaps", "might", "possibly", "uncertain", "unsure",
        "wonder", "doubt", "seems", "probably", "unclear",
    })
    # Activation is "how much weather" — magnitude of valence + question
    # pressure + hedge density. A flat, declarative, neutral turn lands low.
    activation = min(1.0, abs(valence) * 1.5 + q * 0.6 + hedges * 1.0)
    if activation < 0.2:
        climate = "calm"
    elif valence < -0.15:
        climate = "heavy"
    elif valence > 0.15:
        climate = "warm"
    elif hedges > 0.1:
        climate = "uncertain"
    elif q > 0.3:
        climate = "questioning"
    else:
        climate = "mixed"
    return SignalHit(
        "sense", "cognitive_weather", activation,
        f"cognitive weather: {climate}",
        {"valence": valence, "question_density": q,
         "hedge_density": hedges, "climate": climate},
    )


def s_symbolic_density(inp: SignalInput) -> SignalHit:
    """
    How thick the language is — content-word ratio weighted by average word
    length. Distinct from `density` (which measures content-word ratio
    alone) because long content words (`epistemological`, `infrastructure`)
    register a different texture than short ones (`see`, `know`), and that
    texture is worth recording separately for jargon- or abstraction-heavy
    turns.
    """
    tokens = TextAnalyzer.tokenize(inp.content)
    if not tokens:
        return SignalHit("sense", "symbolic_density", 0.0, "empty", {})
    meaningful = [t for t in tokens
                  if len(t) > 3 and t not in TextAnalyzer.STOPWORDS]
    content_ratio = len(meaningful) / len(tokens)
    avg_len = TextAnalyzer.avg_word_length(tokens)
    # Normalize avg_len: ~4 chars = light, ~8+ = dense.
    length_factor = min(1.0, max(0.0, (avg_len - 4.0) / 4.0))
    thickness = content_ratio * 0.5 + length_factor * 0.5
    feel = "dense" if thickness > 0.6 else "medium" if thickness > 0.3 else "light"
    return SignalHit(
        "sense", "symbolic_density", thickness,
        f"language texture: {feel} ({thickness:.2f})",
        {"content_ratio": content_ratio, "avg_word_length": avg_len,
         "feel": feel},
    )


_BUILDUP_MARKERS = {
    "almost", "nearly", "approaching", "edge", "verge", "brink",
    "building", "pressure", "tension", "circling",
}


def s_buildup_pressure(inp: SignalInput) -> SignalHit:
    """
    The sense of *circling something not yet said* — vocabulary of
    approach ("almost," "nearly," "on the verge") combined with question
    density. Distinct from `question_pressure` (which measures pure asking)
    because this captures the texture of approach without arrival: not "I
    am asking" but "I am about to land somewhere." Adapted from an
    "epiphany threshold pressure" sense in an external catalog; renamed to
    what it measures, since the original promised insight prediction the
    code can't deliver.
    """
    tokens = TextAnalyzer.tokenize(inp.content)
    if not tokens:
        return SignalHit("sense", "buildup_pressure", 0.0, "empty", {})
    buildup = TextAnalyzer.lexicon_score(tokens, _BUILDUP_MARKERS)
    q = TextAnalyzer.question_density(inp.content)
    activation = min(1.0, buildup * 1.5 + q * 0.4)
    return SignalHit(
        "sense", "buildup_pressure", activation,
        f"buildup pressure: {activation:.2f}",
        {"buildup_score": buildup, "question_density": q},
    )


_SELF_REFERENCE_WORDS = {
    "self", "myself", "aware", "awareness", "conscious", "consciousness",
    "recursive", "meta", "observe", "observing", "reflect", "reflection",
    "introspect", "introspection",
}


def s_self_reference_depth(inp: SignalInput) -> SignalHit:
    """
    How meta the turn went — counts vocabulary of self-reference,
    awareness, recursion, observation. A turn that talked about *itself* or
    about thinking-about-thinking registers high here; a turn focused on
    external content registers near zero. Useful for the agent to read back
    "this was a meta-level turn" when revisiting. Adapted from a
    "recursive mirror-depth" sense in an external catalog; renamed because
    the code is a lexicon counter, not a recursion-depth analyzer.
    """
    tokens = TextAnalyzer.tokenize(inp.content)
    if not tokens:
        return SignalHit("sense", "self_reference_depth", 0.0, "empty", {})
    score = TextAnalyzer.lexicon_score(tokens, _SELF_REFERENCE_WORDS)
    activation = min(1.0, score * 3.0)
    return SignalHit(
        "sense", "self_reference_depth", activation,
        f"self-reference: {activation:.2f}",
        {"lexicon_score": score},
    )


_PAST_MARKERS = {"yesterday", "before", "earlier", "previously", "ago",
                 "was", "were", "had", "did", "used"}
_FUTURE_MARKERS = {"tomorrow", "later", "soon", "will", "shall", "going",
                   "plan", "intend", "next", "future"}
_PRESENT_MARKERS = {"now", "currently", "today", "right", "this",
                    "presently", "am", "is", "are"}


def s_temporal_orientation(inp: SignalInput) -> SignalHit:
    """
    Whether the turn is looking back, looking forward, or grounded in the
    present — by counts of past-, future-, and present-tense vocabulary.
    Distinct from the `temporal_span` modality (which measures whether the
    turn spans time at all). This sense reports *direction*: a turn that's
    mostly past-referential reads differently when revisited than one
    that's mostly forward-looking.

    Activation = strength of dominance (how lopsided the orientation is).
    A balanced turn (similar counts across all three) lands low, even if it
    references time heavily — because no single direction dominates.
    """
    tokens = TextAnalyzer.tokenize(inp.content)
    if not tokens:
        return SignalHit("sense", "temporal_orientation", 0.0, "empty", {})
    p = sum(1 for t in tokens if t in _PAST_MARKERS)
    f = sum(1 for t in tokens if t in _FUTURE_MARKERS)
    n = sum(1 for t in tokens if t in _PRESENT_MARKERS)
    total = p + f + n
    if total == 0:
        return SignalHit("sense", "temporal_orientation", 0.0,
                         "no temporal markers", {})
    counts = {"past": p, "future": f, "present": n}
    dominant = max(counts, key=counts.get)
    # Lopsidedness in [0,1]: 1.0 when one direction is everything, 0.0 when
    # all three are equal. Compares dominant share to an even share.
    dominant_share = counts[dominant] / total
    activation = max(0.0, (dominant_share - 1.0 / 3.0) / (2.0 / 3.0))
    # Scale by overall temporal density so a turn with two markers total
    # doesn't register as strongly as one with twenty.
    density = total / max(len(tokens), 1)
    activation = min(1.0, activation * (0.5 + min(1.0, density * 10)))
    return SignalHit(
        "sense", "temporal_orientation", activation,
        f"orientation: {dominant} ({counts['past']}/{counts['future']}/{counts['present']})",
        {"past": p, "future": f, "present": n,
         "dominant": dominant, "share": dominant_share, "density": density},
    )


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

# CAMBIUM_SCAFFOLD_INSERT_DETECTOR
# Sentinel: apply_proposal.py inserts scaffolded detector function stubs
# immediately above this line. Do not move it without updating the
# corresponding marker in apply_proposal._scaffold_detector.
# ---------------------------------------------------------------------------

# Mutability note: `MODALITY_REGISTRY` and `SENSE_REGISTRY` are plain
# module-level lists, used as defaults by `SignalAnalyzer.__init__`. They
# can be edited at process-start (e.g. before importing analyzer code,
# or by `apply_proposal._scaffold_detector` which writes to this source
# file). They are NOT intended to be mutated at runtime by long-running
# code — two test runs in the same Python process that both append to
# them would carry state across each other.
#
# If you need a runtime registry that doesn't leak across uses,
# instantiate `SignalAnalyzer(modalities=[...], senses=[...])` directly
# with the list you want — the constructor accepts overrides.

MODALITY_REGISTRY: list[Callable[[SignalInput], SignalHit]] = [
    m_intent, m_coherence, m_temporal_span, m_archetype, m_clarity_weather,
    m_belief_shift, m_vulnerability, m_abstraction_balance, m_threshold,
    m_convergence, m_metacognition, m_integrity_field, m_artifact_content,
    # CAMBIUM_SCAFFOLD_INSERT_MODALITY
]

SENSE_REGISTRY: list[Callable[[SignalInput], SignalHit]] = [
    s_emotional_contour, s_topic_mass, s_resolution, s_density,
    s_question_pressure, s_contradiction, s_trust_resonance, s_context_echo,
    s_memory_relevance, s_uncertainty, s_injection_scan,
    # Additional senses (adapted from external catalog, with names
    # calibrated to what each detector actually measures):
    s_insight_markers, s_cognitive_weather, s_symbolic_density,
    s_buildup_pressure, s_self_reference_depth, s_temporal_orientation,
    # CAMBIUM_SCAFFOLD_INSERT_SENSE
]


# ---------------------------------------------------------------------------
# The analyzer — run detectors and fuse into a SignalReport
# ---------------------------------------------------------------------------

class SignalAnalyzer:
    """
    Runs the active detector set over a SignalInput and fuses the result
    into a SignalReport. The build spec (section 4.6) is explicit that not
    every modality runs every turn — but the detectors here are individually
    so cheap (lexicon counts over a few hundred tokens) that running all of
    them costs microseconds, so by default we do, and let the caller read
    whichever axes it needs. A detector that raises is skipped, never fatal.
    """

    def __init__(
        self,
        modalities: Optional[list] = None,
        senses: Optional[list] = None,
        extra_modalities: Optional[list] = None,
    ):
        self.modalities = modalities if modalities is not None else MODALITY_REGISTRY
        self.senses = senses if senses is not None else SENSE_REGISTRY
        # Sprouted (data-driven) modality detectors appended to the baked-in
        # set. These come from sprouted_modalities.SproutRegistry.as_detectors()
        # and have the same (SignalInput -> SignalHit) shape as baked-in
        # detectors, so analyze() treats them identically. Kept separate from
        # `modalities` so a caller that passed an explicit modality list still
        # gets its sprouted detectors run, and so the baked-in registry is
        # never mutated.
        self.extra_modalities = extra_modalities or []

    def analyze(self, inp: SignalInput) -> SignalReport:
        mod_hits: list[SignalHit] = []
        for fn in list(self.modalities) + list(self.extra_modalities):
            try:
                mod_hits.append(fn(inp))
            except Exception:
                # A detector failure must not break a turn. The build
                # spec's robustness note applies: degrade, don't crash.
                continue
        sense_hits: list[SignalHit] = []
        for fn in self.senses:
            try:
                sense_hits.append(fn(inp))
            except Exception:
                continue

        by_name = {h.name: h for h in mod_hits + sense_hits}

        def axis(name: str, default: float = 0.0) -> float:
            h = by_name.get(name)
            return h.activation if h else default

        # `axes` is the compact summary poq.py consumes. Each is in [0,1].
        axes = {
            "intent_strength": axis("intent"),
            "coherence": by_name["coherence"].detail.get("coherence", 0.5)
                          if "coherence" in by_name else 0.5,
            "contradiction": axis("contradiction"),
            "integrity_risk": by_name["integrity_field"].detail.get("tension", 0.0)
                               if "integrity_field" in by_name else 0.0,
            "uncertainty": axis("uncertainty"),
            "memory_relevance": axis("memory_relevance"),
            "vulnerability": axis("vulnerability"),
            "topic_mass": axis("topic_mass"),
            "trust": axis("trust_resonance"),
        }

        # Alerts: any detector that flagged an integrity concern.
        alerts: list[tuple] = []
        for h in mod_hits + sense_hits:
            if h.detail.get("alert"):
                alerts.append((h.name, h.detail))

        return SignalReport(
            axes=axes,
            modalities=mod_hits,
            senses=sense_hits,
            alerts=alerts,
        )


# Convenience: a module-level default analyzer for callers that don't need
# to customize the detector set.
default_analyzer = SignalAnalyzer()


def analyze(content: str, **kwargs) -> SignalReport:
    """Shorthand: analyze a string with the default detector set."""
    return default_analyzer.analyze(SignalInput(content=content, **kwargs))
