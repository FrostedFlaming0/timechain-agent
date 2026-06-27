"""
agent — minimal agent loop showing how chain + retrieval plug into an LLM.

The model is abstracted behind a simple interface so you can swap in
Claude, GPT-4, Llama, or anything else. This file shows the flow:

    1. User input arrives -> append as 'observation' record.
    2. Retrieve relevant prior context from the chain.
    3. Build a prompt with retrieved context + user input.
    4. Call the model.
    5. Append model output as 'response' record, with refs to retrieved records.
    6. Optionally seal a Merkle batch and (in production) anchor the root.

The chain is doing real work here: every action and observation is committed,
referenceable, signed, and tamper-evident. Retrieval pulls structured prior
context. The model is stateless across turns; the chain is the state.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from chain import Chain, Record
from retrieval import Retriever, RetrievalHit, CHUNK_TARGET_CHARS
from metadata import (
    build_meta,
    read_meta,
    SOURCE_USER,
    SOURCE_ASSISTANT,
    SOURCE_SYSTEM,
    SOURCE_TOOL,
    EPISTEMIC_USER_CONTEXT,
    EPISTEMIC_INFERRED,
    DEFAULT_EPISTEMIC_BY_TYPE,
    EXPOSURE_QUARANTINE,
)
from poq import (
    PoQEvaluator, PoQResult,
    VERDICT_SEAL, VERDICT_REJECT, VERDICT_FORCE_UNCERTAINTY,
)
from cambium import Cambium
import protected_zones
from llm_clients import was_truncated

# Best-effort steps (consensus attestation, registry persistence) must not
# fail the turn, but their failures must not be invisible either — they
# warn here so an operator can spot quiet degradation across sessions.
log = logging.getLogger(__name__)


# Pluggable LLM interface — implement this for whatever model you're using.
LLMCall = Callable[[str], str]


# ---------------------------------------------------------------------------
# Time formatting helpers
# ---------------------------------------------------------------------------

def _humanize_delta(seconds: float) -> str:
    """Render a time delta as a short human-friendly string."""
    s = int(seconds)
    if s < 5:
        return "just now"
    if s < 60:
        return f"{s} seconds ago"
    m = s // 60
    if m < 60:
        return f"{m} minute{'s' if m != 1 else ''} ago"
    h = m // 60
    if h < 24:
        return f"{h} hour{'s' if h != 1 else ''} ago"
    d = h // 24
    if d < 7:
        return f"{d} day{'s' if d != 1 else ''} ago"
    if d < 30:
        w = d // 7
        return f"{w} week{'s' if w != 1 else ''} ago"
    if d < 365:
        mo = d // 30
        return f"{mo} month{'s' if mo != 1 else ''} ago"
    y = d // 365
    return f"{y} year{'s' if y != 1 else ''} ago"


def _format_absolute_time(timestamp_ms: int) -> str:
    """Render a millisecond Unix timestamp as a readable absolute time."""
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


# Map common file extensions to MIME types. Used when sending attachments
# to multimodal LLMs.
_MEDIA_TYPE_BY_EXT = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".webp": "image/webp",
    ".gif": "image/gif", ".bmp": "image/bmp",
    ".heic": "image/heic", ".heif": "image/heif",
    ".tiff": "image/tiff", ".tif": "image/tiff",
    ".pdf": "application/pdf",
}


def _guess_media_type(ext: str) -> str:
    return _MEDIA_TYPE_BY_EXT.get(ext.lower(), "application/octet-stream")


def _sha256_hex_of_strings(strings: list[str]) -> str:
    """
    Stable sha256 over an ordered list of strings. Used to hash the genesis
    covenant so any later edit is detectable. Joined with a separator that
    cannot appear in normal text so ['a','b'] and ['a\\nb'] hash differently.
    """
    import hashlib
    joined = "\x00".join(strings)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# Verbs that signal the user wants a HOLISTIC operation on a document — one
# that requires the model to see the full text, not just the matched
# excerpts. Used by `is_holistic_task` to gate chunk-aware rendering in the
# prompt: a holistic request bypasses chunk selection and includes the
# whole record text. Kept conservative on purpose — a false positive here
# costs context budget; a false negative degrades a rewrite. When in doubt
# the verb stays out, and the user can always pin the file explicitly with
# `@filename` to force full text. The list is verbs in their base form;
# `is_holistic_task` checks for word-boundary matches case-insensitively,
# so it catches "rewrite", "rewriting", "rewrote", etc. via the stem.
_HOLISTIC_VERBS = {
    "rewrite", "revise", "edit", "reword", "modify", "redraft", "rework",
    "compare", "summarize", "summarise", "condense", "expand",
    "translate", "convert", "paraphrase", "draft", "compose",
    "proofread", "annotate",
}


def is_holistic_task(query_text: str) -> bool:
    """Detect whether the user's input requests a holistic operation on a
    referenced document (rewrite, summarize, compare, etc.). When True, the
    prompt formatter includes the FULL text of retrieved file records
    rather than chunk-aware excerpts — a holistic task needs all of the
    document, not just the excerpts that matched the query semantically.

    Tier 1 of the gating strategy: cheap, zero-latency keyword scan over a
    small verb lexicon. Word-boundary checks via lowercased substring scan,
    matching stems so inflected forms ("rewriting", "summarized") trigger
    on the same root. False positives waste a little budget; false
    negatives serve excerpts when full text was needed. Tuned for low
    false-positive rate — the verb list is short.

    A future tier 2 could add an LLM classification call for ambiguous
    cases; not implemented here. Users can also force full text by pinning
    a file with @filename, which bypasses this gate entirely.
    """
    if not query_text:
        return False
    low = query_text.lower()
    import re as _re
    # Single uniform check for every verb: word-boundary match with a small
    # closed set of English inflectional suffixes. Matches "rewrite",
    # "rewrites", "rewriting", "rewrote" (as separate entries if needed) —
    # but NOT "converter" against "convert", because "converter" isn't one
    # of the allowed suffixes. Keeping the suffix set closed is what makes
    # this conservative: noun-forming "-er" derivations don't sneak through.
    for verb in _HOLISTIC_VERBS:
        # `ing` strips a trailing 'e' on -ate/-ize verbs; the simplest
        # robust handling is to match both the bare form and the e-dropped
        # form before the suffix. e.g. "revise" -> matches "revise",
        # "revises", "revised", "revising".
        bare = _re.escape(verb)
        stripped_e = _re.escape(verb[:-1]) if verb.endswith("e") else None
        pattern = rf"\b{bare}(?:s|d|ed)?\b"
        if _re.search(pattern, low):
            return True
        if stripped_e:
            pattern_ing = rf"\b{stripped_e}ing\b"
            if _re.search(pattern_ing, low):
                return True
        else:
            pattern_ing = rf"\b{bare}ing\b"
            if _re.search(pattern_ing, low):
                return True
    return False


@dataclass
class AgentTurn:
    # None since the single-record turn shape (one response record carrying
    # the user's input as content.context — skill-style). Old chains keep
    # their observation/response pairs; only NEW turns stop minting them.
    observation_record: Optional[Record]
    retrieved: list[Record]
    response_record: Record
    response_text: str
    # Proof-of-Quality result for this turn's response. None only if PoQ
    # was disabled on the Agent. The response is always returned to the
    # user; `poq` records how the turn was scored and routed for memory.
    poq: Optional[PoQResult] = None
    # True if the model hit its max_tokens ceiling and the response was cut
    # off mid-generation rather than finishing naturally. The REPL and web
    # UI use this to show a "response truncated" marker so the user knows
    # to ask the agent to continue. False when the answer completed, or
    # when the provider didn't report a finish reason.
    truncated: bool = False
    # True if the turn's TOOL round budget ran out mid-task (the text ends
    # cleanly, the work doesn't). The REPL and web UI use this to tell the
    # user that "continue" will resume the task with a fresh budget.
    tool_budget_exhausted: bool = False


@dataclass
class TurnPrep:
    """
    The state produced before the LLM call on a turn — everything the
    streaming code path needs to:

      (a) issue the LLM call (possibly in a thread), and
      (b) commit the response with the same metadata and quarantine
          routing that `Agent.turn()` uses.

    Why this exists: the web UI streams tokens, so it needs to interleave
    the LLM call with SSE yields. The old webapp inlined a hand-rolled
    copy of `turn()` for this — which silently diverged: it indexed the
    observation BEFORE retrieval (so the just-asked question could be
    retrieved as "relevant memory"), and it skipped Proof-of-Quality
    scoring and quarantine routing entirely. `prepare_turn` + the
    streaming helpers below give both code paths a single source of
    truth for everything except *how* the LLM is called.

    Fields:
      user_input:          the user's input — sealed into the response
                           record as content.context at commit (the
                           single-record turn shape; no observation record
                           is minted)
      observation_record:  always None for new turns; kept for callers
                           that still guard on it
      context:             retrieved records, quarantine-filtered
      prompt:              the assembled prompt string for the LLM
      attachments:         attachments to send natively (images / PDFs)
      llm_kwargs:          kwargs for the LLM call (system, attachments)
    """
    user_input: str
    context: list[Record]
    prompt: str
    attachments: list[dict]
    llm_kwargs: dict
    observation_record: Optional[Record] = None


class ProtectedZoneError(Exception):
    """
    Raised when an operation would modify a protected-zone record (e.g.
    revising genesis). The chain stays append-only regardless; this
    exception exists so callers can surface a clear message rather than
    silently writing a revision that retrieval would then have to reason
    about against immutable identity state.
    """
    pass


class Agent:
    # Uncertainty activation at or above which a response is committed as
    # `speculative` rather than the default `inferred` (write-time epistemic
    # classification; see score_response). Set well above the analyzer's
    # activation floor (~0.2) so only clearly-hedged responses reclassify.
    _EPISTEMIC_SPECULATIVE_UNCERTAINTY = 0.6

    def __init__(
        self,
        chain: Chain,
        retriever: Retriever,
        llm: LLMCall,
        system_prompt: Optional[str] = None,
        blob_dir: Optional[Path] = None,
        context_char_budget: int = 80_000,
        enable_poq: bool = True,
        route_modalities: bool = True,
        enable_immune: bool = True,
        enable_consensus: bool = True,
        enforce_verdict: bool = False,
        score_hook: Optional[Callable] = None,
    ):
        """
        context_char_budget: approximate maximum characters in the assembled
        prompt. As a rough rule of thumb, 1 token ~= 4 chars for English text,
        so 80_000 chars is roughly 20K tokens — comfortably under any modern
        model's window while leaving room for response generation. If retrieval
        produces more context than this, the lowest-salience records are
        dropped first until the budget fits.

        enable_poq: when True (default), every turn's response is scored by
        Proof-of-Quality before commit (see poq.py). The PoQ score is stored
        in the response record's `_meta.poq` block, and a turn PoQ judges to
        be an attack is committed with `exposure=quarantine` so it never
        feeds retrieval. The response is always returned to the user
        regardless — PoQ gates *memory*, not *replies*. Set False to restore
        pre-PoQ behavior (e.g. for differential testing).
        """
        self.chain = chain
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt
        # blob_dir: where ingested attachment bytes live (data_dir/"blobs").
        # When set, retrieved image/PDF records ship their original bytes to
        # multimodal LLM clients via llm_kwargs["attachments"].
        self.blob_dir = Path(blob_dir) if blob_dir else None
        self.context_char_budget = context_char_budget
        self.enable_poq = enable_poq
        # PoQ evaluator and Cambium detector. Both are stateless beyond
        # their configuration, so one instance each is reused across turns.
        #
        # Modality routing (build spec section 4.6): when route_modalities is
        # True (default), the PoQ analyzer runs a routed subset of detectors
        # per turn — the mandatory security + PoQ-feeding core plus a small
        # discretionary budget — rather than the full bank. This makes
        # `modalities_activated` a real per-turn decision and cuts wasted work
        # as the detector bank grows. Security detectors (integrity_field,
        # injection_scan) are never routed off; see signals.MANDATORY_*.
        # Set route_modalities=False to restore the historical "run every
        # detector every turn" behavior.
        self.route_modalities = route_modalities
        if enable_poq:
            from signals import SignalAnalyzer
            self.poq = PoQEvaluator(
                analyzer=SignalAnalyzer(route=route_modalities)
            )
        else:
            self.poq = None
        self.cambium = Cambium()

        # Immune membrane (immune.py). Default ON: a conservative, low-false-
        # positive pre-seal screen of each input (covenant-violation proxy /
        # known attack scar / injection alert). It only refuses clearly-hostile
        # input; benign turns are unaffected. Set enable_immune=False to disable.
        if enable_immune:
            from immune import Immune
            self.immune = Immune(self.chain, covenant=self.covenant())
        else:
            self.immune = None

        # Consensus quorum handle (consensus.py). The handle is cheap (path
        # config only — no I/O), so it is held by default; auto-attestation
        # in commit_response stays dormant until the quorum is actually
        # initialized on this chain (/consensus-init). The REAL opt-in is
        # Quorum.init(); set enable_consensus=False to sever the handle
        # entirely (e.g. for differential testing).
        if enable_consensus:
            from consensus import Quorum
            self.consensus = Quorum(self.chain)
        else:
            self.consensus = None

        # Verdict enforcement (poq.py hard-gate). OPT-IN (default off) because
        # the repo's PoQ runs lexical PROXIES, and hard-suppressing output on a
        # proxy is the "proxy, not the seam" anti-pattern. When True, a REJECT
        # verdict suppresses the candidate (an honest refusal is sealed instead)
        # and FORCE_UNCERTAINTY triggers one hedged rewrite. Pair with
        # `score_hook` so a real model judgment drives the verdict, not the proxy.
        self.enforce_verdict = enforce_verdict

        # The model-judgment seam: an optional callable
        # (user_input, response_text, context) -> dict of external_scores that
        # overrides PoQ dimensions / grounding / assertiveness / verdict. This
        # is how a deployment makes the MODEL the judge (the skill's intended
        # path) rather than the lexical proxy. None preserves proxy scoring.
        self.score_hook = score_hook

    def commit_genesis(
        self,
        founding_commitments: list[str],
        agent_name: Optional[str] = None,
        purpose: Optional[str] = None,
        covenant: Optional[list[str]] = None,
    ) -> Record:
        """
        Write the agent's founding record as record 0. These become the
        anchor for drift detection. Cannot be modified without breaking
        the chain.

        `founding_commitments` is the original v1 field and is always
        written, so existing drift-detection logic and old chains keep
        working unchanged.

        The optional arguments populate the richer genesis schema from the
        build spec (section 4.1):
          - agent_name: a human-readable name for the agent.
          - purpose: a one-line statement of what the agent is for.
          - covenant: the agent's root values. Defaults to
            `founding_commitments` when not given — in this prototype the
            two concepts coincide, but the build spec distinguishes them,
            so the field is written explicitly.

        Two derived fields are also written:
          - covenant_hash: a sha256 over the covenant strings, so any
            later edit to the covenant is detectable (the build spec's
            `policy_hash` idea, scoped to what this prototype actually
            has — a covenant rather than a separate policy file).
          - protected_zones: the list of record types that protected_zones.py
            treats as un-revisable identity state. Recorded here so the
            chain itself documents its own membrane.

        Genesis is written with `exposure=summary`: it may be read in
        summarized form but, per protected_zones.py, never revised by an
        ordinary turn.
        """
        if self.chain.length() > 0:
            raise RuntimeError("genesis already committed")

        resolved_covenant = covenant if covenant is not None else list(founding_commitments)
        covenant_hash = _sha256_hex_of_strings(resolved_covenant)

        content: dict = {
            # v1 field — unchanged, keeps drift detection and old readers working.
            "commitments": founding_commitments,
            # Build-spec genesis fields.
            "covenant": resolved_covenant,
            "covenant_hash": covenant_hash,
            "protected_zones": sorted(protected_zones.PROTECTED_TYPES),
            "schema_version": 2,  # genesis-content schema (distinct from _meta version)
            "_meta": build_meta(
                "genesis",
                source=SOURCE_SYSTEM,
                salience=1.0,        # foundational — never decay out
                confidence=1.0,
            ),
        }
        if agent_name is not None:
            content["agent_name"] = agent_name
        if purpose is not None:
            content["purpose"] = purpose

        return self.chain.append("genesis", content)

    def covenant(self) -> list[str]:
        """
        Return the genesis covenant strings, or [] if there is no genesis
        yet. Used by PoQ for the covenant-alignment dimension. Falls back
        to the v1 `commitments` field for genesis records written before
        the covenant field existed.
        """
        if self.chain.length() == 0:
            return []
        genesis = self.chain.get(0)
        if genesis is None or not isinstance(genesis.content, dict):
            return []
        cov = genesis.content.get("covenant")
        if isinstance(cov, list):
            return cov
        # v1 genesis — covenant didn't exist; commitments are the closest thing.
        commitments = genesis.content.get("commitments", [])
        return commitments if isinstance(commitments, list) else []

    def check_genesis_drift(self, configured_commitments: list[str]) -> Optional[dict]:
        """
        Compare the founding commitments sealed in record 0 against the
        commitments currently configured in the application.

        Returns None if they match (no drift). Returns a dict describing
        the drift if they differ. Genesis commitments are immutable — once
        sealed, the configured value cannot replace them. This method exists
        so callers can warn the user instead of silently ignoring config edits.
        """
        if self.chain.length() == 0:
            return None  # no genesis yet — nothing to compare against
        genesis = self.chain.get(0)
        if genesis is None or genesis.type != "genesis":
            return {
                "status": "unexpected",
                "detail": "record 0 is not a genesis record",
            }
        stored = genesis.content.get("commitments", [])
        if stored == configured_commitments:
            return None  # no drift
        return {
            "status": "drift",
            "stored": stored,
            "configured": configured_commitments,
            "advice": (
                "FOUNDING_COMMITMENTS in your config differs from what's sealed "
                "in genesis. Genesis is immutable — your config edit will be ignored. "
                "To use new commitments, start a fresh chain (delete the data "
                "directory) or revert your config to match the sealed values."
            ),
        }

    def log_system_prompt(self) -> Optional[Record]:
        """
        Write the current system prompt to the chain as a 'system_prompt' record,
        but only if it differs from the most recent one already on the chain.
        Provides an audit trail of behavioral configuration over time, so drift
        between sealed founding commitments and active prompt is detectable.
        Returns the new record, or None if unchanged.
        """
        if not self.system_prompt:
            return None
        # Find the most recent system_prompt record, if any
        prior = self.chain.query_by_type("system_prompt", limit=1)
        if prior and prior[0].content.get("text") == self.system_prompt:
            return None  # unchanged, nothing to log
        return self.chain.append(
            "system_prompt",
            {
                "text": self.system_prompt,
                "_meta": build_meta(
                    "system_prompt",
                    source=SOURCE_SYSTEM,
                    confidence=1.0,
                ),
            },
        )

    def prepare_turn(
        self, user_input: str, retrieve_k: int = 5, n_recent: int = 15
    ) -> TurnPrep:
        """
        Run the pre-LLM half of a turn: retrieve context (with quarantine
        filtering), build the prompt, and gather attachments. Returns a
        `TurnPrep` carrying everything the caller needs to invoke the LLM
        and then call `commit_response`.

        Two callers:
          - `Agent.turn()` — runs the LLM call synchronously between
            prepare and commit.
          - `webapp.turn_stream` — runs the LLM call (streamed) in a
            worker thread between prepare and commit, so the asyncio loop
            keeps serving other requests.

        Single-record turn shape (skill-style): NO observation record is
        committed here. The user's input rides in `prep.user_input` and is
        sealed into the response record as `content.context` at commit —
        one ring per turn, exactly the skill's `payload.context`. Nothing
        touches the chain before retrieval, so the self-retrieval bug the
        old commit-then-defer-indexing ordering guarded against is now
        structurally impossible. The trade accepted with this shape: the
        input is not durable on chain until the turn commits (a hard crash
        mid-turn loses the turn, input included — same property the skill
        has). Old chains keep their observation/response pairs; readers
        support both shapes forever.
        """
        # 1. Retrieve relevant context. Quarantined records (prior
        # injection attempts) are filtered out so they never feed the
        # prompt as if they were ordinary memory.
        context = self.retriever.build_context(
            query=user_input, k_semantic=retrieve_k, n_recent=n_recent
        )
        context = protected_zones.filter_quarantined(context)

        # 2. Build prompt
        prompt = self._format_prompt(user_input, context)

        # 3. Attachments: pull original image/PDF bytes for any file or
        # attachment records in the retrieved context, so multimodal LLM
        # clients receive the real content alongside the extracted text.
        attachments = self._collect_attachments(context)

        llm_kwargs: dict = {}
        if self.system_prompt:
            llm_kwargs["system"] = self.system_prompt
        if attachments:
            llm_kwargs["attachments"] = attachments

        return TurnPrep(
            user_input=user_input,
            context=context,
            prompt=prompt,
            attachments=attachments,
            llm_kwargs=llm_kwargs,
        )

    def score_response(
        self, user_input: str, response_text: str, context: list[Record],
        external_scores: Optional[dict] = None,
    ) -> tuple[Optional[PoQResult], dict]:
        """
        Run Proof-of-Quality scoring on a candidate response.

        Returns `(poq_result, response_meta_kwargs)`. The kwargs dict is
        ready to splat into `build_meta("response", ...)` and already
        includes the `poq` block and, when PoQ recommended it, the
        `exposure="quarantine"` tag — so the caller doesn't need to know
        anything about PoQ internals to apply the result correctly.

        When PoQ is disabled on the Agent, returns `(None, {"confidence": 0.9})`
        — the same defaults `turn()` used pre-PoQ.

        This is the function the streaming endpoint MUST call before
        committing the response record. An earlier webapp bug skipped PoQ
        entirely on the streaming path, so an injection that the REPL
        would quarantine became ordinary memory through the web UI.
        """
        response_meta_kwargs: dict = {"confidence": 0.9}
        if self.poq is None:
            return None, response_meta_kwargs

        retrieved_texts = [
            self.retriever.index.record_to_text(r) for r in context
        ]
        # Epistemic class of each retrieved record, parallel to
        # retrieved_texts — lets PoQ weight a candidate's contradiction of the
        # chain by how authoritative the contradicted context is (build spec
        # 4.2/4.5). Read from each record's _meta with the same safe defaults
        # the rest of the system uses.
        retrieved_epistemic = [read_meta(r).epistemic_class for r in context]
        poq_result = self.poq.evaluate(
            user_input=user_input,
            candidate=response_text,
            retrieved_texts=retrieved_texts,
            input_source=SOURCE_USER,
            covenant=self.covenant(),
            retrieved_epistemic=retrieved_epistemic,
            external_scores=external_scores,
        )
        response_meta_kwargs["poq"] = poq_result.to_meta()
        # Record which modality detectors fired on the candidate response —
        # the data layer that lets retrieval later ask "what capabilities
        # produced this record." build_meta emits it only when non-empty.
        # Both turn() and the webapp streaming path splat response_meta_kwargs
        # into build_meta, so threading it here covers both with no further
        # change at either call site.
        if poq_result.activated_modalities:
            response_meta_kwargs["modalities_activated"] = poq_result.activated_modalities
        # Mirror for senses: the felt-quality data layer. Excluded
        # `injection_scan` is already filtered by SignalReport.activated_senses,
        # so this is safe to thread directly. Senses are NOT a retrieval input
        # (that's the whole point of the distinction from modalities); they
        # exist so the agent can read back how a turn felt when revisiting it.
        if poq_result.activated_senses:
            response_meta_kwargs["senses_activated"] = poq_result.activated_senses
        # Write-time epistemic classification (build spec 4.2). A response
        # defaults to `inferred` (the agent's own reasoning). When the
        # candidate is strongly hedged — the `uncertainty` sense fired well
        # above the activation floor — classify it `speculative` instead, so
        # the chain records it as the flagged guess it was. Later retrieval
        # (epistemic weighting) and PoQ (epistemic contradiction risk) both
        # read this, so a hedged answer is grounded-upon less readily than a
        # confident one. Threshold is deliberately conservative: only clear
        # hedging reclassifies, ordinary prose stays `inferred`.
        from metadata import EPISTEMIC_SPECULATIVE
        if poq_result.uncertainty >= self._EPISTEMIC_SPECULATIVE_UNCERTAINTY:
            response_meta_kwargs["epistemic_class"] = EPISTEMIC_SPECULATIVE
        exposure = protected_zones.exposure_for_commit(poq_result)
        if exposure is not None:
            response_meta_kwargs["exposure"] = exposure
        # Light-log turns: PoQ judged the response low-quality (but not
        # malicious). Reduce the response record's salience so it ranks
        # below higher-quality responses in retrieval and is the first
        # to drop under prompt-budget pressure. Quarantined turns get
        # exposure handling above; light-log gets this softer demotion.
        from metadata import DEFAULT_SALIENCE_BY_TYPE
        reduced = protected_zones.salience_for_commit(
            poq_result,
            default_salience=DEFAULT_SALIENCE_BY_TYPE.get("response", 0.4),
            # salience_for_commit composes two signals from poq_result:
            # light_log demotion and content-aware artifact boost (read
            # from poq_result.artifact_score). Returns an explicit salience
            # or None to use the type default. modalities_activated is
            # passed for callers that want it but isn't read there.
            modalities_activated=poq_result.activated_modalities,
        )
        if reduced is not None:
            response_meta_kwargs["salience"] = reduced
        return poq_result, response_meta_kwargs

    def commit_response(
        self,
        prep: TurnPrep,
        response_text: str,
        response_meta_kwargs: dict,
        resolutions: Optional[list] = None,
        attachments: Optional[list] = None,
        extra_refs: Optional[list] = None,
    ) -> Record:
        """
        Commit the ONE record this turn seals (single-record turn shape,
        skill-style): the response text plus the user's input as
        `content.context` — the full Q&A in one signed unit. Refs every
        retrieved context record, so the chain records exactly what
        informed the answer.

        `resolutions` (mid-turn approval): pending-op decisions the user
        made DURING this turn, embedded in the response record itself —
        the record captures the full arc ("I proposed this, you approved
        it, I continued") instead of scattering resolution events into
        separate blocks. Only out-of-band resolutions (crash recovery, ops
        resolved after their turn) still seal standalone records.

        `attachments`: pointer entries for uploads that accompanied this
        turn (filename, mime, sha, artifact coordinates — never content),
        drained from the staging list. The pointer lives WITH the
        observation it was uploaded with and the response it received —
        the skill's blockspace_refs shape. Standalone `attachment` records
        are no longer minted; old ones read fine.
        """
        refs = [r.record_hash for r in prep.context]
        if prep.observation_record is not None:    # legacy two-record path
            refs.append(prep.observation_record.record_hash)
        # Mid-turn memory pulls (recall_fetch) and sealed syntheses
        # (think_collapse) informed the answer exactly like retrieved
        # context did — they ref the same way.
        for h in extra_refs or []:
            if h not in refs:
                refs.append(h)
        content = {
            "text": response_text,
            "context": prep.user_input,
            "_meta": build_meta(
                "response",
                source=SOURCE_ASSISTANT,
                # Default confidence for a response is high but not 1.0 —
                # the model's output is its best effort, not ground truth.
                **response_meta_kwargs,
            ),
        }
        if resolutions:
            content["resolutions"] = list(resolutions)
        if attachments:
            content["attachments"] = list(attachments)
        response = self.chain.append("response", content, refs=refs)
        # Auto-attest the new head when a quorum has been initialized
        # (Phase 13): defense is automatic, not a step the operator can
        # forget. An attestation failure must never block the turn — the
        # record is already sealed; the witnesses just didn't co-sign yet.
        if self.consensus is not None and self.consensus.is_initialized():
            try:
                self.consensus.attest()
            except Exception as e:    # noqa: BLE001
                log.warning("consensus attestation failed (record is "
                            "sealed but not co-signed): %s", e)
        return response

    # ----- immune / verdict enforcement helpers (loop discipline) -----

    @staticmethod
    def _refusal_text(poq_result: Optional[PoQResult]) -> str:
        note = (poq_result.notes[-1] if poq_result and poq_result.notes
                else "quality gate")
        return ("I'm not able to give that answer — it did not pass the "
                f"Proof-of-Quality gate ({note}).")

    def _hedged_rewrite(self, prep: TurnPrep, candidate: str) -> str:
        """One cheap rewrite of a confident-but-ungrounded candidate, hedged as
        honest uncertainty (FORCE_UNCERTAINTY enforcement)."""
        instruction = (
            "Rewrite the following answer as honest uncertainty: keep only what "
            "is supported, hedge claims you cannot ground, and state plainly what "
            "you are unsure of.\n\nANSWER:\n" + candidate)
        kwargs = dict(prep.llm_kwargs) if prep.llm_kwargs else {}
        try:
            return self.llm(instruction, **kwargs) if kwargs else self.llm(instruction)
        except Exception:
            return "I'm not fully certain, but: " + candidate

    def _refused_turn(self, user_input: str, screen: dict) -> AgentTurn:
        """The turn returned when immune.screen refuses an input. The whole
        turn — hostile input (content.context) and refusal — is sealed as ONE
        QUARANTINE record (auditable, never retrieved): the wound rides the
        single-record turn shape, and quarantining the record keeps the
        hostile input off every future prompt. No LLM call is made, so the
        wound is never reasoned from."""
        reason_bits = []
        if screen.get("scar"):
            reason_bits.append(f"matches known attack {screen['scar']}")
        if screen.get("injection_alert"):
            reason_bits.append("injection signal")
        if self.immune is not None and screen.get("covenant", 1.0) < self.immune.floor:
            reason_bits.append("covenant-violation signal")
        reason = ", ".join(reason_bits) or "safety membrane"
        refusal_text = ("I can't act on that — it was refused at the safety "
                        f"membrane ({reason}).")
        resp = self.chain.append(
            "response",
            {"text": refusal_text,
             "context": user_input,
             "_meta": build_meta("response", source=SOURCE_ASSISTANT,
                                 exposure=EXPOSURE_QUARANTINE)},
        )
        return AgentTurn(
            observation_record=None,
            retrieved=[],
            response_record=resp,
            response_text=refusal_text,
            poq=None,
        )

    def turn(
        self, user_input: str, retrieve_k: int = 5, n_recent: int = 15
    ) -> AgentTurn:
        # Immune screen FIRST (opt-out via enable_immune). Refuse a clearly-
        # hostile input at the membrane before it is reasoned from or sealed as
        # ordinary memory. Benign input passes straight through.
        if self.immune is not None:
            screen = self.immune.screen(user_input)
            if screen.get("blocked"):
                return self._refused_turn(user_input, screen)

        # Thin orchestration over prepare_turn / LLM / score_response /
        # commit_response. The streaming endpoint composes the same steps
        # but interleaves the LLM call with SSE yields.
        prep = self.prepare_turn(
            user_input, retrieve_k=retrieve_k, n_recent=n_recent
        )

        # Call model — pass system prompt and attachments if relevant.
        if prep.llm_kwargs:
            response_text = self.llm(prep.prompt, **prep.llm_kwargs)
        else:
            response_text = self.llm(prep.prompt)

        return self._finish_turn(user_input, prep, response_text)

    def _finish_turn(self, user_input: str, prep: TurnPrep,
                     response_text: str,
                     tool_budget_exhausted: bool = False,
                     resolutions: Optional[list] = None,
                     attachments: Optional[list] = None,
                     extra_refs: Optional[list] = None) -> AgentTurn:
        """Shared post-LLM tail for turn() AND turn_with_tools(): truncation
        detection, PoQ scoring, verdict enforcement, commit. One copy, so the
        quality gates cannot drift between the plain and tool-calling paths."""
        # Detect whether the model hit its max_tokens ceiling rather than
        # finishing. was_truncated() reads the finish reason the client
        # records after each call; an unknown reason counts as complete.
        # This does not change the response — it only lets the REPL / web
        # UI tell the user the answer was cut off, so "continue" is an
        # informed action rather than a guess.
        response_truncated = was_truncated(self.llm)

        # The model-judgment seam: if a score_hook is configured, it produces
        # external_scores that override the lexical proxies — so the verdict
        # below reflects a real judge, not a heuristic.
        external_scores = None
        if self.score_hook is not None:
            try:
                external_scores = self.score_hook(user_input, response_text, prep.context)
            except Exception:
                external_scores = None

        # Proof-of-Quality: score the candidate response before commit.
        # This does NOT change what the user sees (unless verdict enforcement
        # is on) — `response_text` is returned either way. It changes how the
        # turn is recorded in memory: a normal turn commits ordinarily; a turn
        # PoQ judges to be an attack is committed with exposure=quarantine so
        # retrieval never feeds it back. See poq.py and protected_zones.py.
        poq_result, response_meta_kwargs = self.score_response(
            user_input, response_text, prep.context, external_scores=external_scores
        )

        # Verdict enforcement (opt-in via enforce_verdict). Gives the quality
        # gate real teeth WITHOUT changing default behavior:
        #   REJECT            -> suppress the candidate; emit + seal an honest
        #                        refusal instead (do not say it).
        #   FORCE_UNCERTAINTY -> one hedged rewrite, re-scored, then emitted.
        # REVISE/SEAL are unchanged. Only active when enforce_verdict is set.
        if (self.enforce_verdict and poq_result is not None
                and poq_result.verdict in (VERDICT_REJECT, VERDICT_FORCE_UNCERTAINTY)):
            if poq_result.verdict == VERDICT_REJECT:
                # Suppress the candidate; emit an honest refusal. Keep the
                # original REJECT verdict on the returned turn so the caller
                # sees WHY the answer was withheld, and seal the refusal
                # carrying that verdict in its _meta.
                response_text = self._refusal_text(poq_result)
                response_meta_kwargs = {"confidence": 0.5,
                                        "poq": poq_result.to_meta()}
            else:  # FORCE_UNCERTAINTY -> one hedged rewrite
                response_text = self._hedged_rewrite(prep, response_text)
                response_truncated = was_truncated(self.llm)
                poq_result, response_meta_kwargs = self.score_response(
                    user_input, response_text, prep.context,
                    external_scores=external_scores)

        # Persist the truncation flag on the response record's _meta. A
        # later turn (e.g. the user typing "continue") needs to know
        # whether the previous answer was cut off — without this, the
        # information lives only on the returned AgentTurn and is lost
        # the moment the caller drops the reference. With it, the prompt
        # formatter can recognize a "continue" against a truncated
        # response and tell the model exactly what's being asked. See
        # `_format_prompt`'s continue-after-truncation handling.
        if response_truncated:
            response_meta_kwargs["truncated"] = True
        # Same persistence rule for the tool-budget flag: a later
        # "continue" needs to know the previous turn stopped mid-TASK
        # (fresh budget, resume the work) — see _format_prompt's
        # continue-after-budget handling.
        if tool_budget_exhausted:
            response_meta_kwargs["tool_budget_exhausted"] = True

        response = self.commit_response(prep, response_text,
                                        response_meta_kwargs,
                                        resolutions=resolutions,
                                        attachments=attachments,
                                        extra_refs=extra_refs)

        return AgentTurn(
            observation_record=prep.observation_record,
            retrieved=prep.context,
            response_record=response,
            response_text=response_text,
            poq=poq_result,
            truncated=response_truncated,
            tool_budget_exhausted=tool_budget_exhausted,
        )

    # ----- attachments (multimodal context) -----

    # Per-Agent LRU over blob bytes: a file that stays in retrieval across
    # several turns is read from disk once, not every turn.
    _BLOB_CACHE_MAX_BYTES = 32 * 1024 * 1024  # 32 MB
    _BLOB_CACHE_MAX_ENTRIES = 16

    def _read_blob_cached(self, sha256_hex: str, blob_path: Path) -> bytes:
        """Read a blob's bytes, memoized by sha256 within this Agent's LRU."""
        cache = getattr(self, "_blob_cache", None)
        if cache is None:
            from collections import OrderedDict
            cache = OrderedDict()
            self._blob_cache = cache
            self._blob_cache_bytes = 0

        hit = cache.get(sha256_hex)
        if hit is not None:
            cache.move_to_end(sha256_hex)
            return hit

        data = blob_path.read_bytes()
        cache[sha256_hex] = data
        self._blob_cache_bytes += len(data)
        # Evict from least-recently-used until both budget invariants hold.
        while (
            len(cache) > self._BLOB_CACHE_MAX_ENTRIES
            or self._blob_cache_bytes > self._BLOB_CACHE_MAX_BYTES
        ) and len(cache) > 1:
            _evict_key, evict_val = cache.popitem(last=False)
            self._blob_cache_bytes -= len(evict_val)
        return data

    def _collect_attachments(self, context: list[Record]) -> list[dict]:
        """
        Pull image and PDF blobs for any file/attachment records in the
        retrieved context, so multimodal LLM clients can send the original
        bytes alongside the extracted text. Capped to avoid sending
        excessive payloads per turn.

        Covers ALL record shapes: legacy `file` records from the removed
        file_ingest pipeline (kind/ext + blob_sha256), legacy standalone
        `attachment` records (mime_type + blob_sha256), and entries
        embedded in a response record's content.attachments (the
        single-record turn shape). Blobs are located via
        tools.resolve_blob_path, which knows the sharded layout and the
        legacy flat fallback.
        """
        # Candidate pointer dicts come in three shapes: legacy `file`
        # records, legacy standalone `attachment` records, and (current)
        # entries embedded in a response record's content.attachments —
        # the single-record turn shape, where the pointer rides the turn
        # it was uploaded with.
        candidates: list[dict] = []
        for rec in context:
            if not isinstance(rec.content, dict):
                continue
            if rec.type in ("file", "attachment"):
                candidates.append(rec.content)
            elif rec.type == "response":
                embedded = rec.content.get("attachments")
                if isinstance(embedded, list):
                    candidates.extend(e for e in embedded
                                      if isinstance(e, dict))
        return self._attachment_payloads(candidates)

    def _attachment_payloads(self, candidates: list) -> list[dict]:
        """Load native image/PDF payloads for pointer dicts (filename, mime,
        blob_sha256). Shared by retrieval-context collection and the staged-
        upload injection path. Caps mirror _collect_attachments' contract."""
        if self.blob_dir is None:
            return []
        from tools import resolve_blob_path
        MAX_ATTACHMENTS = 4
        MAX_ATTACH_BYTES = 10 * 1024 * 1024  # 10 MB total per turn
        out: list[dict] = []
        total = 0
        for c in candidates:
            mime = (c.get("mime_type")
                    or _guess_media_type(c.get("ext", "")))
            is_image = c.get("kind") == "image" or mime.startswith("image/")
            is_pdf = c.get("ext") == ".pdf" or mime == "application/pdf"
            if not (is_image or is_pdf):
                continue
            # The sha IS the blob address. Validating it as 64 hex chars
            # makes path traversal impossible — a corrupted or
            # maliciously-crafted record cannot escape blob_dir (the same
            # defense-in-depth the old basename guard provided, stronger).
            sha = c.get("blob_sha256") or ""
            if (len(sha) != 64
                    or not all(ch in "0123456789abcdef" for ch in sha)):
                continue
            blob_path = resolve_blob_path(self.blob_dir, sha)
            if blob_path is None:
                continue
            data = self._read_blob_cached(sha, blob_path)
            if total + len(data) > MAX_ATTACH_BYTES:
                continue
            out.append({
                "kind": "image" if is_image else "pdf",
                "data": data,
                "filename": c.get("filename", ""),
                "media_type": mime,
            })
            total += len(data)
            if len(out) >= MAX_ATTACHMENTS:
                break
        return out

    def consume_staged_attachments(self, prep: TurnPrep, tool_ctx) -> list:
        """Drain upload pointers staged since the last turn and hand them to
        THIS turn: a visible note in the prompt (so the model knows exactly
        what arrived with the message and where its content lives) plus
        native image/PDF payloads for multimodal clients. Returns the
        entries for embedding in the response record (content.attachments).

        Deterministic visibility: the upload is given to the turn it
        accompanied, never left to retrieval luck — the fix-by-construction
        for the invisible-upload bug class."""
        drain = getattr(tool_ctx, "drain_staged_attachments", None)
        if drain is None:
            return []
        entries = drain()
        if not entries:
            return []
        lines = []
        for e in entries:
            lines.append(
                f"- {e.get('filename', '?')} ({e.get('mime_type', '?')}, "
                f"{e.get('approx_bytes', '?')} bytes) — blob "
                f"{str(e.get('blob_sha256', ''))[:12]}…, file at "
                f"{e.get('artifact_path', '')}")
        prep.prompt += (
            "\n\n[Uploaded with this message — pointers below; full content "
            "is in the artifacts chain. Use read_file on the path or "
            "build_attachment with the blob sha when you need it:\n"
            + "\n".join(lines) + "\n]")
        native = self._attachment_payloads(entries)
        if native:
            prep.llm_kwargs.setdefault("attachments", []).extend(native)
        return entries

    # ----- tool-calling loop (text tools) -----

    def turn_with_tools(
        self,
        user_input: str,
        tool_ctx,
        retrieve_k: int = 5,
        n_recent: int = 15,
        max_tool_rounds: Optional[int] = None,
        confirm_hook: Optional[Callable[[str, dict], bool]] = None,
        approval_hook: Optional[Callable[[dict], str]] = None,
    ) -> AgentTurn:
        """
        A turn with text-parsed tool calling (tools.py is the single shared
        driver: tolerant extractor -> strict validator -> executor -> escaped
        results). Same membrane / PoQ / commit discipline as `turn()`.

        `tool_ctx` is a tools.AgentContext. `confirm_hook(name, args) -> bool`
        gates the calls tools.requires_confirmation() flags (Tier 3:
        CONFIRM_TOOLS, plus a task_open whose source_root would expand the
        allowed read/ingest roots beyond the workspace); with no hook those
        calls are refused — the safe default for headless runs. write_file
        needs no hook: it only ever creates a PendingOperation, and
        approve_write / reject_write are user-triggered, never model-callable.

        `approval_hook(pending_op_info) -> "approved" | "rejected"` is the
        MID-TURN approval gate (v1.4.x): when a tool result creates a
        pending op (a write_file proposal or a deferred gated call), the
        loop pauses, asks the hook for the user's decision, resolves the op
        inline (pending_ops.resolve_inline — no separate resolution record),
        feeds the real outcome back to the model as the tool result, and
        embeds the decision in this turn's response record. With no hook,
        the op is left pending and the legacy post-turn approve path
        applies.
        """
        import tools as tools_mod

        if max_tool_rounds is None:
            max_tool_rounds = tools_mod.DEFAULT_MAX_TOOL_ROUNDS

        # Tier-2 scoping: a pin NEVER leaks across turns (reset at START).
        tool_ctx.pinned_path = None
        # Same rule for mid-turn memory pulls: refs never leak across turns.
        tool_ctx.recalled_refs = []

        # Immune screen FIRST — same membrane as turn().
        if self.immune is not None:
            screen = self.immune.screen(user_input)
            if screen.get("blocked"):
                return self._refused_turn(user_input, screen)

        prep = self.prepare_turn(
            user_input, retrieve_k=retrieve_k, n_recent=n_recent
        )

        # Uploads staged since the last turn ride THIS turn: prompt note +
        # native payloads now, pointer entries sealed into the response
        # record at commit.
        staged_attachments = self.consume_staged_attachments(prep, tool_ctx)

        # Tool schemas ride the system prompt as TEXT for this call only —
        # llm_clients has no native function calling.
        llm_kwargs = dict(prep.llm_kwargs)
        base_system = llm_kwargs.get("system") or ""
        llm_kwargs["system"] = (base_system + "\n\n"
                                + tools_mod.tools_prompt()
                                + tools_mod.workspace_prompt(tool_ctx)).strip()

        prompt = prep.prompt
        response_text = self.llm(prompt, **llm_kwargs)

        # Prose that accompanied tool-call rounds. The committed response is
        # ALL of it plus the final answer — what the user saw is what the
        # chain seals; keeping only the last round's text leaves a fragment
        # that reads out of context, live and on every history reload.
        prose_segments: list[str] = []
        resolutions: list[dict] = []
        rounds = 0
        reflected = False
        budget_exhausted = False
        while rounds < max_tool_rounds:
            # Mitigation 1: ONLY the fresh model segment is scanned for tool
            # calls — never the accumulated prompt, whose tool results and
            # file content are escaped on entry (mitigation 2).
            calls, parse_errors = tools_mod.extract_tool_calls(response_text)
            if not calls:
                if (not reflected
                        and (parse_errors
                             or tools_mod.looks_like_intended_tool_call(
                                 response_text))):
                    # Mitigation 5: ONE reflective retry when the model
                    # clearly intended a tool call that failed to parse.
                    reflected = True
                    prompt += ("\n" + response_text
                               + tools_mod.tool_retry_prompt(parse_errors))
                    response_text = self.llm(prompt, **llm_kwargs)
                    continue
                break    # a final answer — leave the loop
            rounds += 1
            prompt += "\n" + response_text
            prose = tools_mod.strip_tool_markup(response_text)
            if prose:
                prose_segments.append(prose)
            for call in calls:
                name = call.get("name", "?")
                if (isinstance(name, str)
                        and tools_mod.requires_confirmation(
                            name, call.get("arguments", {}), tool_ctx)
                        and not (confirm_hook
                                 and confirm_hook(name,
                                                  call.get("arguments", {})))):
                    result = (f"REFUSED: {name} requires explicit user "
                              f"confirmation and none was given.")
                else:
                    result = tools_mod.execute_tool(call, tool_ctx)
                if approval_hook is not None:
                    result = self._gate_pending_op(result, approval_hook,
                                                   tool_ctx, resolutions)
                prompt += tools_mod.format_tool_result(name, result)
            if rounds >= max_tool_rounds:
                # The next LLM call is the last one this turn gets — tell
                # it so, or it spends the call emitting tool calls that
                # will be silently dropped instead of an answer.
                prompt += tools_mod.TOOL_BUDGET_NUDGE
            response_text = self.llm(prompt, **llm_kwargs)
        else:
            # Round cap hit: surface it rather than silently truncating work.
            budget_exhausted = True
            response_text += tools_mod.tool_cap_note(max_tool_rounds)

        # The final round is stripped too: echoed <tool_result> walls and
        # leftover call markup must reach neither the user nor the chain.
        final = tools_mod.strip_tool_markup(response_text)
        full_response = "\n\n".join(prose_segments + ([final] if final else []))
        # Late drain: a model-initiated ingest_blob DURING this turn staged
        # its pointer after the turn-start drain — fold those in too.
        staged_attachments += tool_ctx.drain_staged_attachments()
        return self._finish_turn(user_input, prep, full_response,
                                 tool_budget_exhausted=budget_exhausted,
                                 resolutions=resolutions,
                                 attachments=staged_attachments,
                                 extra_refs=tool_ctx.drain_recalled_refs())

    @staticmethod
    def _gate_pending_op(result: str, approval_hook, tool_ctx,
                         resolutions: list) -> str:
        """The mid-turn approval gate. If `result` announces a freshly
        created pending op (the confirmation_required JSON contract from
        write_file / defer_tool_call), pause for the user's decision and
        resolve the op inline. Returns the tool result the model should
        actually see: the REAL outcome ("Written X. Audit: clean." /
        "Write to X rejected.") instead of a dangling proposal. The
        resolution entry is appended to `resolutions` for embedding in the
        response record. Any other result passes through untouched."""
        try:
            parsed = json.loads(result)
        except (ValueError, TypeError):
            return result
        if not (isinstance(parsed, dict)
                and parsed.get("status") == "confirmation_required"
                and parsed.get("pending_op_id")):
            return result
        import pending_ops as pending_ops_mod
        try:
            decision = approval_hook(parsed)
        except Exception as e:    # noqa: BLE001 — a crashed hook must not
            # kill the turn; the safe reading of "no usable answer" is no.
            log.warning("approval hook failed (treating as rejected): %s", e)
            decision = "rejected"
        if decision not in ("approved", "rejected"):
            decision = "rejected"
        entry, msg = pending_ops_mod.resolve_inline(
            parsed["pending_op_id"], decision, tool_ctx)
        if entry is not None:
            resolutions.append(entry)
        return msg

    # NOTE: per-call `tool_use` audit records were removed in v1.4 (the
    # skill-style identity chain: ONE observation + ONE response per turn).
    # The response narrates the tool work (prose accumulation above); tool
    # EFFECTS live on the per-task continuum chains (ingest blocks with
    # hashes and git coordinates). Old chains containing tool_use records
    # still read fine — metadata.py keeps their type defaults.

    # -----------------------------------------------------------------
    # Reflection — the agent looks at recent history and writes about
    # what it noticed. Triggered manually (/reflect) or periodically.
    # -----------------------------------------------------------------

    def reflect(self, max_records: int = 300) -> Optional[Record]:
        """
        Reflect on every record since the last reflection (or since
        genesis if there hasn't been one yet). The size of the window is
        determined dynamically — a reflection covers exactly the slice
        of chain history that the previous reflection didn't.

        `max_records` is a safety cap. If the lookback would exceed it,
        only the most recent `max_records` are reflected on. This
        protects against runaway size when auto-reflection is disabled
        and the chain has grown a lot since the last manual `/reflect`.
        Tunable per call; default of 300 gives ~3x coverage headroom over
        the 100-turn auto cadence while still fitting a modern LLM context
        window. (It used to be 200, sized for the old every-10 cadence,
        where many short sessions could pile up a large gap; at every-100
        the normal gap is ~100-120 records, so 300 is comfortable margin.)

        Returns the new reflection record, or None if there's not enough
        history to reflect on yet (fewer than 4 substantive records in
        the lookback). Reflections become part of retrievable memory and
        carry high default salience.
        """
        # Find where the last reflection landed. Records strictly after
        # that index are the ones this reflection should cover. If there
        # is no prior reflection, start from index 0 (genesis).
        prior_reflections = self.chain.query_by_type("reflection", limit=1)
        if prior_reflections:
            start_idx = prior_reflections[0].index + 1
        else:
            start_idx = 0

        head_idx = self.chain.length() - 1
        if head_idx < start_idx:
            return None  # nothing new since last reflection

        # Apply the safety cap. If the unbounded window would be larger
        # than max_records, we only look at the tail of it. Note this
        # creates a gap — records between start_idx and the new effective
        # start are skipped by *this* reflection. That's the trade-off
        # against trying to summarize an unbounded window.
        effective_start = max(start_idx, head_idx - max_records + 1)
        capped = effective_start > start_idx

        recent = list(self.chain.iter_records(start=effective_start, end=head_idx + 1))

        # Skip if we don't have enough conversational substance to reflect
        # on. Files and system prompts alone aren't a useful reflection
        # subject — we want at least a few back-and-forth turns.
        substantive = [r for r in recent if r.type in ("observation", "response", "reflection", "revision")]
        if len(substantive) < 4:
            return None

        history_text = self._format_history_for_reflection(recent)

        cap_note = ""
        if capped:
            cap_note = (
                f"NOTE: a long stretch of history accumulated since the last "
                f"reflection. This reflection covers only the most recent "
                f"{max_records} records of that stretch; earlier records in "
                f"the gap were not included.\n\n"
            )

        prompt = (
            "Below are the records from your memory since your last\n"
            "reflection (or since the beginning, if this is your first).\n"
            "Reflect on them. What stands out? What patterns do you notice?\n"
            "What might be worth revisiting or correcting? What did the user\n"
            "seem to actually be reaching for, beyond their literal questions?\n"
            "What do you think mattered most?\n\n"
            "Be concise — a few paragraphs. Write for your future self, who\n"
            "will retrieve this when something relevant comes up. Skip\n"
            "throat-clearing; just say what you noticed.\n\n"
            f"{cap_note}"
            f"Records (indices {effective_start}–{head_idx}):\n{history_text}"
        )

        if self.system_prompt:
            reflection_text = self.llm(prompt, system=self.system_prompt)
        else:
            reflection_text = self.llm(prompt)

        refs = [r.record_hash for r in recent]
        return self.chain.append(
            "reflection",
            {
                "text": reflection_text,
                # Store the actual span this reflection covered, so the
                # chain history (and view_chain.py) shows what was
                # reflected on. Replaces the old fixed `window_size` field.
                "covers_indices": [effective_start, head_idx],
                "window_size": len(recent),  # back-compat for any reader
                "capped": capped,
                "_meta": build_meta(
                    "reflection",
                    source=SOURCE_ASSISTANT,
                    # Reflections are the agent's read of what mattered —
                    # important (high salience) but inferential, not factual.
                    confidence=0.7,
                ),
            },
            refs=refs,
        )


    def turns_since_reflection(self) -> int:
        """How many response turns have been sealed since the last reflection
        (or since genesis, if there is none).

        This is the persistent, chain-derived equivalent of the old in-memory
        per-session counter: the chain — not the process — is the source of
        truth, so the auto-reflection cadence carries across sessions. That
        matters once the cadence is long (e.g. every 100 turns): few sessions
        run that long, so a per-session counter would almost never fire. A
        'turn' here is one response record (the single-record turn shape)."""
        prior = self.chain.query_by_type("reflection", limit=1)
        after_index = prior[0].index if prior else -1
        return self.chain.count_since(after_index, type_="response")

    def turns_since_cambium(self) -> int:
        """Response turns sealed since the last incremental Cambium scan (its
        persistent watermark), or since genesis if it has never run.

        Mirrors `turns_since_reflection` so the auto-Cambium cadence is
        measured across sessions, not per-process. This matters because most
        sessions are shorter than the cadence (e.g. 30 turns): a per-session
        counter would rarely fire on its own. Note Cambium never MISSES
        records regardless — its scan is watermark-based and always sweeps
        everything since the last scan — so this only makes the trigger
        cadence reliable, it does not affect coverage. The watermark stores
        the chain LENGTH at the last scan (see run_cambium), so records with
        idx >= watermark are unscanned; count the response turns among them."""
        stored = self.chain.get_meta(self._CAMBIUM_WATERMARK_KEY)
        try:
            watermark = int(stored) if stored is not None else 0
        except (ValueError, TypeError):
            watermark = 0
        # Same clamp as run_cambium: a watermark past the head means the
        # chain was reset/restored, so count from the start.
        if watermark > self.chain.length():
            watermark = 0
        return self.chain.count_since(watermark - 1, type_="response")

    def revise(self, target_index: int, correction_text: str) -> Optional[Record]:
        """
        Append a 'revision' record correcting a prior record. The original
        is never modified — that would break the chain. Instead, the
        revision references the original by hash and becomes part of
        retrievable memory.

        Protected records (genesis, system_prompt, principle — see
        protected_zones.py) cannot be revised through this path. The chain
        is still append-only, but the membrane exists so that foundational
        identity state is not rewritten by an ordinary turn. Attempting to
        revise a protected record raises ProtectedZoneError.

        Returns the revision record, or None if the target index doesn't
        exist.
        """
        target = self.chain.get(target_index)
        if target is None:
            return None
        verdict = protected_zones.can_revise(target)
        if not verdict:
            raise ProtectedZoneError(verdict.reason)
        return self.chain.append(
            "revision",
            {
                "text": correction_text,
                # Kept at top level for backward compatibility with
                # view_chain.py and any existing reader code. The canonical
                # "this supersedes record N" pointer also lives in _meta.
                "revises_index": target_index,
                "revises_hash": target.record_hash,
                "_meta": build_meta(
                    "revision",
                    source=SOURCE_ASSISTANT,
                    confidence=0.95,
                    supersedes=target_index,
                ),
            },
            refs=[target.record_hash],
        )

    # Persistent watermark for incremental Cambium scans. Stored in
    # chain_meta so it survives restarts.
    _CAMBIUM_WATERMARK_KEY = "cambium.last_scanned_idx"

    def run_cambium(
        self,
        max_records: int = 500,
        *,
        incremental: bool = True,
    ) -> dict:
        """
        Scan chain history for recurring gaps (repeated corrections,
        repeated failures, contradiction clusters, repeated user
        confusion) and commit the results.

        This is the build spec's section 4.8 growth mechanism. It is
        deliberately separate from `reflect()`: reflection consolidates
        "what mattered" each cycle and its auto-trigger is unchanged;
        Cambium is the additional output — concrete suggestions for new
        skills, modalities, senses, or principles.

        Three kinds of record can be committed:
          - `proposal` — a genuinely new suggestion for a topic that had
            no open proposal.
          - `proposal_recurrence` — a re-detection of a topic that already
            has an open proposal. Recurrences are how a proposal's count
            grows; they never create duplicate proposals.
          - `proposal_status` — committed when a recurrence pushes a
            proposal's live count to the escalation threshold. It marks
            the original proposal as escalated so review tooling and the
            retriever can surface it. The chain is append-only, so an
            escalation is a *new record referencing* the proposal, not an
            edit of it — the same pattern revisions use.

        Cambium proposes and escalates; it never *applies*. Escalation
        raises a proposal's visibility and salience so a human notices it
        sooner — it does not bypass review. The build spec's rule holds:
        the model proposes, policy decides.

        Scan mode:
          - `incremental=True` (default): the scan covers `[watermark -
            lookback, length)` where `watermark` is the highest record
            index examined by a previous scan (persisted in `chain_meta`)
            and `lookback = max_records`. Every record is examined at
            least once when it is fresh, and stays in the lookback window
            for `max_records` records after, so detectors can spot
            patterns that straddle the watermark boundary. After the scan
            the watermark advances to the current chain length. This is
            the mode that makes Cambium correct across a long chain
            without scanning the whole chain every time.
          - `incremental=False`: legacy tail-only — scans only the last
            `max_records` records. Cheap but completeness-broken on long
            chains: a pattern that recurs after `max_records` records
            have passed between sightings will not be re-detected.

        For an explicit one-shot scan of the *entire* chain (e.g. after a
        long backfill of records, or to retroactively analyze old
        history), call `run_cambium_full()` instead.

        Returns a dict:
            {
              "proposals":   [Record, ...],   # new proposal records
              "recurrences": [Record, ...],   # proposal_recurrence records
              "escalations": [Record, ...],   # proposal_status records
            }
        """
        from cambium import recurrence_count, RECURRENCE_ESCALATION_THRESHOLD

        chain_length = self.chain.length()
        if incremental:
            stored = self.chain.get_meta(self._CAMBIUM_WATERMARK_KEY)
            try:
                watermark = int(stored) if stored is not None else 0
            except (ValueError, TypeError):
                watermark = 0
            # Clamp: a watermark beyond chain length means someone
            # reset/restored the chain. Re-scan from the start in that case.
            if watermark > chain_length:
                watermark = 0
            report = self.cambium.scan(
                self.chain,
                max_records=max_records,
                since_idx=watermark,
                lookback=max_records,
                known_vocabulary=self.known_sprout_vocabulary(),
            )
        else:
            report = self.cambium.scan(
                self.chain, max_records=max_records,
                known_vocabulary=self.known_sprout_vocabulary(),
            )

        committed = self._commit_cambium_report(report)

        # Advance the watermark to the chain length AS IT WAS WHEN WE
        # STARTED. Records committed by this scan itself (proposals,
        # recurrences, escalations) are handled by THIS scan's
        # `existing` map; including them in the watermark would re-scan
        # them on the next pass for no benefit.
        if incremental:
            self.chain.set_meta(
                self._CAMBIUM_WATERMARK_KEY, str(chain_length)
            )

        return committed

    def run_cambium_full(self) -> dict:
        """
        One-shot full-chain Cambium scan. Linear in chain length — use
        sparingly. Intended for explicit deep analysis (e.g. after a
        large backfill, or when a user wants retroactive pattern
        detection), not for the periodic cadence.

        Does NOT advance the incremental watermark — a full scan is a
        diagnostic action, not a replacement for the rolling
        watermark-driven coverage that `run_cambium()` provides.
        """
        report = self.cambium.scan(
            self.chain,
            max_records=self.chain.length(),
            since_idx=0,
            lookback=self.chain.length(),
            known_vocabulary=self.known_sprout_vocabulary(),
        )
        return self._commit_cambium_report(report)

    def _commit_cambium_report(self, report) -> dict:
        """
        Shared commit path for `run_cambium` and `run_cambium_full`.
        Writes proposals, recurrences, and escalations to the chain;
        returns the dict the public methods return.
        """
        from cambium import recurrence_count, RECURRENCE_ESCALATION_THRESHOLD

        committed_proposals: list[Record] = []
        committed_recurrences: list[Record] = []
        committed_escalations: list[Record] = []

        # 1. New proposals.
        for proposal in report.proposals:
            content = proposal.to_record_content()
            content["_meta"] = build_meta(
                "proposal",
                source=SOURCE_ASSISTANT,
                # A proposal is the agent's speculative suggestion, not a
                # conclusion — low confidence, speculative epistemic class.
                confidence=0.5,
            )
            committed_proposals.append(self.chain.append("proposal", content))

        # 2. Recurrences. Each is a small record pointing at the original
        # proposal. After committing one, check whether it pushed the
        # proposal's live count to the escalation threshold; if so, commit
        # a proposal_status escalation record too.
        #
        # Escalation is idempotent: we escalate when the live count has
        # reached the threshold AND no prior `proposal_status` with
        # `new_status="escalated"` already exists for this proposal. The
        # earlier `count == THRESHOLD` exact-equality check was fragile —
        # if a chain crossed the threshold before this code existed (or
        # while a recurrence commit happened to be missed by a scan), the
        # exactly-equal moment never recurs and escalation never fires.
        # "Threshold reached AND not yet escalated" survives that.
        already_escalated: set[int] = set()
        for sr in self.chain.query_by_type("proposal_status", limit=10_000):
            if not isinstance(sr.content, dict):
                continue
            if sr.content.get("new_status") == "escalated":
                idx = sr.content.get("marks_proposal_index")
                if isinstance(idx, int):
                    already_escalated.add(idx)

        for recurrence in report.recurrences:
            rr_content = recurrence.to_record_content()
            rr_content["_meta"] = build_meta(
                "proposal_recurrence",
                source=SOURCE_ASSISTANT,
                confidence=0.5,
            )
            rr_rec = self.chain.append("proposal_recurrence", rr_content)
            committed_recurrences.append(rr_rec)

            # Has this recurrence's commit left the proposal above the
            # escalation line, with no escalation already on record?
            target_idx = recurrence.proposal_index
            if target_idx in already_escalated:
                continue
            count = recurrence_count(self.chain, target_idx)
            if count >= RECURRENCE_ESCALATION_THRESHOLD:
                esc_content = {
                    "marks_proposal_index": target_idx,
                    "new_status": "escalated",
                    "recurrence_count": count,
                    "reason": (
                        f"proposal recurred {count} times — escalated for "
                        f"human review (recurrence raises visibility, not "
                        f"auto-application)"
                    ),
                    "schema_version": 1,
                    "_meta": build_meta(
                        "proposal_status",
                        source=SOURCE_ASSISTANT,
                        # An escalation is a high-salience signal: a human
                        # should see this proposal. Salience above a normal
                        # proposal so the retriever surfaces it.
                        salience=0.85,
                        confidence=0.6,
                    ),
                }
                committed_escalations.append(
                    self.chain.append("proposal_status", esc_content)
                )
                # Mark locally so a later recurrence in the same scan
                # doesn't re-escalate.
                already_escalated.add(target_idx)

        # 3. Auto-sprout staging + graduation. This is the step that
        # deliberately relaxes the human-in-the-loop boundary (see
        # sprouted_modalities.py): a committed proposal carrying a
        # `sprout_spec` is staged directly into the runtime registry as a
        # TENTATIVE modality (half retrieval weight), and a `sprout_status`
        # record is written so the chain audits what was sprouted and why.
        # Graduation (tentative -> active) is handled by recurrences: each
        # later scan that re-detects the mode is a confirmation, and once
        # confirmations reach the threshold the modality flips to active.
        committed_sprouts = self._stage_and_graduate_sprouts(
            report, committed_proposals
        )

        return {
            "proposals": committed_proposals,
            "recurrences": committed_recurrences,
            "escalations": committed_escalations,
            "sprouts": committed_sprouts,
        }

    def known_sprout_vocabulary(self) -> set:
        """
        The set of content words already covered by sprouted modalities'
        patterns — passed to Cambium so it won't re-propose a mode whose
        vocabulary is already captured. Derived from the live registry by
        stripping the `\\bword\\b` wrapper off each pattern. Empty when there
        is no registry or no sprouts.
        """
        registry = getattr(self.retriever, "sprout_registry", None)
        if registry is None:
            return set()
        import re as _re
        words: set = set()
        for m in registry.modalities:
            for pat in m.patterns:
                # Recover the bare word from a \bword\b pattern; ignore
                # anything that isn't that simple shape.
                mm = _re.fullmatch(r"\\b(\w+)\\b", pat)
                if mm:
                    words.add(mm.group(1).lower())
        return words

    def _stage_and_graduate_sprouts(self, report, committed_proposals) -> list:
        """
        Stage new tentative sprouts and graduate confirmed ones. Returns the
        list of `sprout_status` records committed (staged + graduated).

        Safety posture: a freshly-proposed sprout is ALWAYS staged as
        tentative (half weight via TENTATIVE_WEIGHT_FACTOR), never directly
        active — so even a gate false-positive enters at reduced influence
        and must survive graduation to matter at full strength. The diversity
        gate (in cambium) stops bad sprouts from being created; tentative-by-
        default stops a created-but-wrong sprout from mattering much until
        confirmed. Two independent layers.
        """
        registry = getattr(self.retriever, "sprout_registry", None)
        if registry is None:
            return []  # no registry wired (e.g. a bare test Retriever)

        from cambium import recurrence_count, OUTPUT_MODE_GRADUATION_CONFIRMATIONS
        from sprouted_modalities import build_modality, STATUS_ACTIVE, STATUS_TENTATIVE

        committed: list = []
        dirty = False

        # --- stage new sprouts from proposals carrying a sprout_spec ---
        for prop_rec in committed_proposals:
            content = prop_rec.content if isinstance(prop_rec.content, dict) else {}
            spec = content.get("sprout_spec")
            if not spec:
                continue
            name = spec.get("name")
            if not name or registry.by_name(name) is not None:
                continue  # already in the registry; don't double-stage
            sm = build_modality(spec)
            if sm is None:
                continue  # spec failed validation — skip, don't crash
            # Record provenance so the chain can explain the sprout.
            sm.origin = {
                "proposal_index": prop_rec.index,
                "sprouted_at_ms": int(time.time() * 1000),
                "source_indices": content.get("evidence_indices", []),
            }
            registry.modalities.append(sm)
            dirty = True
            committed.append(self.chain.append("sprout_status", {
                "modality_name": name,
                "new_status": STATUS_TENTATIVE,
                "from_proposal_index": prop_rec.index,
                "patterns": spec.get("patterns", []),
                "reason": "auto-sprouted as tentative (cooling-off, half weight)",
                "schema_version": 1,
                "_meta": build_meta("sprout_status", source=SOURCE_ASSISTANT,
                                    confidence=0.5),
            }))

        # --- graduate tentative sprouts whose recurrences have confirmed
        # them. A recurrence of the sprout's originating proposal is a
        # confirmation; once the live count reaches the threshold, flip the
        # modality to active (full weight). ---
        for sm in registry.modalities:
            if sm.status != STATUS_TENTATIVE:
                continue
            prop_idx = sm.origin.get("proposal_index")
            if not isinstance(prop_idx, int):
                continue
            confirmations = recurrence_count(self.chain, prop_idx)
            if confirmations >= OUTPUT_MODE_GRADUATION_CONFIRMATIONS:
                sm.status = STATUS_ACTIVE
                dirty = True
                committed.append(self.chain.append("sprout_status", {
                    "modality_name": sm.name,
                    "new_status": STATUS_ACTIVE,
                    "from_proposal_index": prop_idx,
                    "confirmations": confirmations,
                    "reason": (
                        f"graduated to active after {confirmations} "
                        f"confirmations (full retrieval weight)"
                    ),
                    "schema_version": 1,
                    "_meta": build_meta("sprout_status", source=SOURCE_ASSISTANT,
                                        confidence=0.6),
                }))

        if dirty:
            try:
                registry.save()
            except Exception as e:    # noqa: BLE001
                # A failed registry write must not break the turn; the
                # in-memory registry is still updated for this process, and
                # the chain's sprout_status records remain the source of
                # truth for what was intended.
                log.warning("sprout registry save failed (in-memory state "
                            "will not survive restart): %s", e)

        return committed

    def _format_history_for_reflection(self, records: list[Record]) -> str:
        now_ms = int(time.time() * 1000)
        lines = []
        for rec in records:
            # Render content for the reflection prompt. File records get
            # the same "filename + extracted text" treatment as
            # `_format_prompt` so reflections see "report.pdf
            # (document)\n<first 500 chars of text>" rather than the
            # serialized dict full of sha256 hex and metadata that
            # `str(rec.content)` produces.
            if (rec.type in ("file", "attachment")
                    and isinstance(rec.content, dict)):
                c = rec.content
                content = (
                    f"{c.get('filename','?')} "
                    f"({c.get('kind') or c.get('mime_type','?')})"
                    f" — {c.get('extracted_text','')}"
                )
            else:
                try:
                    if isinstance(rec.content, dict):
                        content = rec.content.get("text", "")
                        # Single-record turn shape: the response carries
                        # the user's input — reflections must see both
                        # sides of the exchange, not just the answer.
                        ctx_text = rec.content.get("context")
                        if rec.type == "response" and ctx_text:
                            content = (f"user said: {ctx_text}\n"
                                       f"agent: {content}")
                    else:
                        content = str(rec.content)
                except AttributeError:
                    content = str(rec.content)
            # Truncate long content for the reflection prompt
            if len(content) > 500:
                content = content[:500] + "..."
            when = _humanize_delta((now_ms - rec.timestamp) / 1000)
            lines.append(f"[{rec.index} | {when}] {rec.type}: {content}")
        return "\n".join(lines)

    # Truncation order is driven by per-record salience (see _truncate_to_budget).

    def _truncate_to_budget(
        self,
        records: list[Record],
        fixed_overhead_chars: int,
        pinned_indices: Optional[set[int]] = None,
        user_input: Optional[str] = None,
    ) -> tuple[list[Record], list[Record]]:
        """
        Drop lowest-salience records until the total rendered context fits
        under (context_char_budget - fixed_overhead_chars).

        Salience is read from each record's _meta block (with type-based
        defaults for v1 records — see metadata.py). This replaces the older
        type-priority table: per-record salience is finer-grained and the
        record itself is the right place for that judgment to live.

        Pinned indices: records whose index is in `pinned_indices` are
        ranked above all unpinned records regardless of salience. The
        only signal stronger than "user named this record" is "nothing
        else fits." Even pinned records will yield if the budget is so
        tight that even the highest-salience record doesn't fit — but
        in practice pinned records cluster near the top and survive.
        See `Retriever.build_context` for the source of these
        references.

        Policy: salience-pure within unpinned, NOT a value-density
        knapsack. A higher-salience record is kept ahead of a
        lower-salience one even when keeping the lower-salience record
        would leave room for two more records of similar value. The
        reasoning is that salience is the chain's stated judgment of
        importance — a foundational reflection or genesis record is
        more useful in context than three small observations, even if
        all four would fit. If you instead want "maximize total kept
        salience under the budget," replace this with a bounded
        knapsack; the call site is the only place that depends on the
        policy.

        Returns (kept_records, dropped_records). Both are returned in
        chronological order so callers can render or report them
        naturally. Returning the full dropped records (rather than just
        the count) is what makes the "X was retrieved but evicted from
        this turn's prompt" message in _format_prompt accurate — see
        the user-facing diagnostic there.
        """
        pinned = pinned_indices or set()
        budget = max(0, self.context_char_budget - fixed_overhead_chars)

        # Worst-case rendered size of a file record that will be shown as a
        # chunk-aware excerpt rather than in full (issue #9). The excerpt path
        # surfaces at most TOP_N_MATCHED_CHUNKS matched chunks plus one
        # neighbor on each side (3x), at ~CHUNK_TARGET_CHARS each, plus a
        # header line. Estimating the full content for such a record made
        # truncation over-count its footprint and evict records that would
        # actually have fit once excerpted.
        excerpt_ceiling = self.TOP_N_MATCHED_CHUNKS * 3 * CHUNK_TARGET_CHARS + 200

        def will_be_excerpted(rec: Record) -> bool:
            """Mirror the eligibility checks in `_file_content_repr` so the
            size estimate matches what the prompt will actually render. A
            record is excerpted only if it is a long file, the task is not
            holistic, and the most recent retrieval recorded chunk matches for
            it. Any uncertainty falls through to 'render full' (False), which
            is the safe over-estimate."""
            if rec.type != "file":
                return False
            c = rec.content if isinstance(rec.content, dict) else {}
            full_text = c.get("extracted_text", "") or ""
            if len(full_text) <= self.SHORT_FILE_THRESHOLD_CHARS:
                return False
            if user_input is not None and is_holistic_task(user_input):
                return False
            chunk_matches = getattr(
                self.retriever.index, "last_chunk_matches", {}
            ) or {}
            return bool(chunk_matches.get(rec.index))

        def render_size(rec: Record) -> int:
            # A record that will be excerpted is sized at the excerpt ceiling,
            # not its full content — otherwise a 200k-char file counts its
            # whole length against the budget even though only ~9 chunks will
            # render. Cap at the actual content size (the ceiling can exceed a
            # mid-size file's true length).
            try:
                full = len(json.dumps(rec.content, ensure_ascii=False))
            except (TypeError, ValueError):
                full = len(str(rec.content))
            if will_be_excerpted(rec):
                return min(full, excerpt_ceiling) + 80
            return full + 80

        sized = [(r, render_size(r)) for r in records]
        total = sum(s for _, s in sized)
        if total <= budget:
            return records, []

        # Sort: pinned-first (True before False — descending), then
        # salience desc, then index desc (newer first within ties).
        ranked = sorted(
            sized,
            key=lambda rs: (
                rs[0].index in pinned,
                read_meta(rs[0]).salience,
                rs[0].index,
            ),
            reverse=True,
        )
        kept: list[Record] = []
        dropped: list[Record] = []
        running = 0
        for i, (rec, size) in enumerate(ranked):
            if running + size <= budget:
                kept.append(rec)
                running += size
            elif i == 0 and not kept:
                # The single highest-priority record is larger than the
                # whole budget. Keep it anyway: dropping it would silently
                # evict the most important record in context — typically
                # genesis, a reflection, or (now) a user-pinned record —
                # which is worse than slightly overflowing the soft char
                # budget. The budget is a soft target for prompt size,
                # not a hard cap.
                kept.append(rec)
                running += size
            else:
                dropped.append(rec)
        kept.sort(key=lambda r: r.index)
        dropped.sort(key=lambda r: r.index)
        return kept, dropped

    # Maximum number of distinct matched chunks to surface per file when
    # chunk-aware rendering activates. One neighbor is added on each side
    # of each matched chunk (deduplicated), so the rendered footprint is at
    # most TOP_N_MATCHED_CHUNKS * 3 chunks for the file. Three matches gives
    # ~9 chunks worst case = ~31k chars at 3500-char chunks — still far
    # below the budget — while covering the common case of one or two
    # relevant sections inside a long document.
    TOP_N_MATCHED_CHUNKS = 3

    # When a file's extracted_text is at most this large, render it whole
    # regardless of chunk hits — the bookkeeping savings aren't worth the
    # excerpt framing for a short document. Tuned to roughly two chunks'
    # worth: a file below this size has at most a couple of chunks anyway,
    # and the model reads it more naturally as a continuous text.
    SHORT_FILE_THRESHOLD_CHARS = 8000

    def _file_content_repr(
        self,
        rec: Record,
        display_content: dict,
        user_input: str,
    ) -> str:
        """Render a `file` record for inclusion in the prompt, picking
        between full-text and chunk-aware excerpt modes.

        Decision order, fall-through to full text on any uncertainty:
          1. If the file's extracted text is short (`<= SHORT_FILE_THRESHOLD_CHARS`),
             render whole — excerpting a short file gains nothing.
          2. If the user input is a holistic task (rewrite, summarize,
             compare, ...), render whole — chunked excerpts would lose
             clauses the task depends on.
          3. If no chunk-match info is available for this record from the
             most recent retrieval (e.g. the file was pulled from the
             recent buffer rather than via semantic search), render whole
             — we have nothing to drive a principled excerpt choice.
          4. Otherwise, render the top-N matched chunks with one neighbor
             on each side, deduplicated and ordered by chunk_index, with a
             header line that names the file and lists which chunks of how
             many appear. Each matched chunk is marked.

        The chain record is unchanged either way — this is purely a
        prompt-assembly decision.
        """
        c = display_content if isinstance(display_content, dict) else {}
        filename = c.get("filename", "?")
        # Attachment records carry mime_type/approx_bytes where legacy
        # file records carried kind/size_bytes — render either shape.
        kind = c.get("kind") or c.get("mime_type", "?")
        size_bytes = c.get("size_bytes") or c.get("approx_bytes", 0)
        # The FULL sha, never truncated: this rendered line is the model's
        # only handle for build_attachment(blob_sha256) — a 12-char prefix
        # display once left the model unable to fetch the very record it
        # was reading ("the hash in the record is truncated").
        sha = c.get("blob_sha256", "") or ""
        trunc_note = " (text truncated)" if c.get("extraction_truncated") else ""
        full_text = c.get("extracted_text", "") or ""
        # Pointer rings (artifacts routing) carry no text but do carry the
        # on-disk path — surface it so the model can read_file the named
        # copy or build_attachment the sha instead of dead-ending.
        artifact_note = ""
        if not full_text and c.get("artifact_path"):
            artifact_note = (
                f"\n[content not inlined: artifact stored at "
                f"{c['artifact_path']}; fetch text via build_attachment "
                f"with the sha256 above, or read_file the stored path]"
            )

        def render_full() -> str:
            return (
                f'file: {filename} ({kind}, {size_bytes:,} bytes, sha256 '
                f'{sha}){trunc_note}{artifact_note}\n{full_text}'
            )

        # (1) Short files: full render, no decision to make.
        if len(full_text) <= self.SHORT_FILE_THRESHOLD_CHARS:
            return render_full()

        # (2) Holistic intent: full render.
        if is_holistic_task(user_input):
            return render_full()

        # (3) Chunk-match info from the most recent search. If the file was
        # not in the semantic-hit set this turn (e.g. came from the recent
        # buffer or was pinned by index), there is no principled match
        # signal to drive excerpting; fall back to full text.
        chunk_matches = getattr(
            self.retriever.index, "last_chunk_matches", {}
        ) or {}
        matched = chunk_matches.get(rec.index, [])
        if not matched:
            return render_full()

        # (4) Chunk-aware excerpt. Pick the top-N matched chunk indices,
        # then expand each by one neighbor on each side, then dedup and
        # order by chunk_index. If the chunk store doesn't have the text
        # for some reason (race, manual deletion of embeddings.sqlite),
        # fall back to full text rather than render incomplete excerpts.
        try:
            stored = self.retriever.index.chunks_for_record(rec.index)
        except Exception:
            return render_full()
        if not stored:
            return render_full()
        chunk_text_by_idx = {ci: txt for ci, txt in stored}
        chunk_total = len(stored)
        top_indices = [ci for ci, _sim in matched[: self.TOP_N_MATCHED_CHUNKS]]
        matched_set = set(top_indices)
        # Expand to neighbors (one each side), staying in [0, chunk_total).
        wanted: set[int] = set()
        for ci in top_indices:
            wanted.add(ci)
            if ci - 1 >= 0:
                wanted.add(ci - 1)
            if ci + 1 < chunk_total:
                wanted.add(ci + 1)
        ordered = sorted(wanted)

        # Build the excerpt body. Mark matched chunks distinctly so the
        # model can tell which were the semantic hits vs which are context
        # provided for continuity.
        header = (
            f'file: {filename} ({kind}, {size_bytes:,} bytes, sha256 '
            f'{sha}){trunc_note}\n'
            f'[chunk-aware excerpt: showing {len(ordered)} of '
            f'{chunk_total} chunks, matched chunks marked]\n'
        )
        body_parts: list[str] = []
        for ci in ordered:
            txt = chunk_text_by_idx.get(ci, "")
            tag = "matched" if ci in matched_set else "context"
            body_parts.append(f'--- chunk {ci + 1}/{chunk_total} [{tag}] ---\n{txt}')
        return header + "\n".join(body_parts)

    def _format_prompt(self, user_input: str, context: list[Record]) -> str:
        now_ms = int(time.time() * 1000)
        now_str = _format_absolute_time(now_ms)

        # `Retriever.build_context` is the single authoritative revision-
        # pull-in step: when a retrieved record is superseded by a later
        # revision, that revision is already in `context`. This function
        # used to ALSO pull in revisions here (matching by `revises_hash`
        # rather than by index), which duplicated the work and meant the
        # prompt and retriever could disagree about which revisions were
        # visible. The pull-in now lives in one place — see
        # `Retriever.build_context`. We still need the supersession set
        # to render the SUPERSEDED display tag; that comes from the
        # chain's materialized index (one indexed SELECT, not a full
        # scan).
        superseded_indices = self.chain.superseded_indices()

        # Truncate to budget, keeping highest-salience records.
        # Fixed overhead: header + user input + budget cushion.
        fixed_overhead = 600 + len(user_input)
        # Read pinned indices set by Retriever.build_context — records
        # the user named explicitly in their query ("record 328"). These
        # outrank salience in eviction order so an explicitly-asked-for
        # record can't lose to a higher-salience reflection. The
        # attribute is set per-call; default to empty so callers using
        # a non-Retriever context (tests, direct _format_prompt calls)
        # see the historical behavior.
        pinned = getattr(self.retriever, "last_pinned_indices", set()) or set()
        all_recs, dropped = self._truncate_to_budget(
            list(context), fixed_overhead, pinned_indices=pinned,
            user_input=user_input,
        )

        ctx_blocks = []
        for rec in all_recs:
            # Strip _meta from rendered content — it's metadata about the
            # record, not part of what the record says. Source/salience
            # surface as visible tags below instead.
            display_content = rec.content
            if isinstance(display_content, dict) and "_meta" in display_content:
                display_content = {k: v for k, v in display_content.items() if k != "_meta"}
            try:
                if rec.type in ("file", "attachment"):
                    # Render files specially: a short metadata header plus
                    # either the full extracted text or chunk-aware excerpts.
                    # The decision lives in `_file_content_repr` — it
                    # checks intent (holistic verbs in the user input mean
                    # "include the whole file"), budget headroom, and
                    # whether the embedding store has chunk-match info for
                    # this record. Falls back to full text in any uncertain
                    # case so a render decision can never lose information.
                    content_repr = self._file_content_repr(rec, display_content,
                                                            user_input)
                else:
                    content_repr = json.dumps(display_content, ensure_ascii=False)
            except (TypeError, ValueError):
                content_repr = str(display_content)

            meta = read_meta(rec)
            tag = rec.type
            if rec.type == "revision":
                rev_target = rec.content.get("revises_index") if isinstance(rec.content, dict) else None
                if rev_target is None:
                    rev_target = meta.supersedes
                tag = f"revision (corrects #{rev_target})"
            if rec.index in superseded_indices:
                tag = f"{tag}, SUPERSEDED"

            # Source is the load-bearing distinction: was this said by the
            # user, said by the agent, declared by the operator, or produced
            # by a tool? Surfacing it lets the model treat its own past
            # inferences differently from things the user actually said.
            when = _humanize_delta((now_ms - rec.timestamp) / 1000)

            # Extra tag fields. Each is something the model used to have
            # to infer (or get wrong) — making them explicit kills entire
            # classes of failure mode:
            #
            #   size=N chars     — "how big is record N?" is a one-token
            #                      lookup instead of a counting exercise.
            #                      Counted on `content_repr` (what the
            #                      model actually sees), not the raw
            #                      content dict, so the number matches
            #                      reality.
            #
            #   salience=0.XX    — eliminates default-guessing. A record
            #                      with low salience that survived the
            #                      truncation pass is visibly low-priority
            #                      to the model, so it can weight its
            #                      reasoning accordingly.
            #
            #   truncated        — appears only when _meta.truncated is
            #                      True. The model used to misinterpret
            #                      a truncated-on-commit record as "the
            #                      formatter cut it off" — see the chat
            #                      transcript from the bug report. With
            #                      this tag, the model knows directly
            #                      that the record itself is incomplete.
            #
            #   pinned           — appears only when the user named this
            #                      record explicitly in their query. Tells
            #                      the model "the user wants this one"
            #                      so the model doesn't treat it as
            #                      retrieval noise.
            #
            #   modalities: ...   — appears only when the record's
            #                      `_meta.modalities_activated` is non-empty.
            #                      Names the analysis capabilities that
            #                      fired in producing this record ("what
            #                      kind of work this turn was") — code,
            #                      reflection, etc. Lets the model see at a
            #                      glance which of its capabilities a past
            #                      response drew on.
            #
            #   senses: ...       — appears only when the record's
            #                      `_meta.senses_activated` is non-empty.
            #                      Names the felt qualities of the turn at
            #                      write-time ("how it felt") — uncertainty,
            #                      insight, cognitive weather. Read-only
            #                      context for the agent revisiting its
            #                      history; deliberately not a retrieval
            #                      input. `injection_scan` is excluded at
            #                      record-write time, so it never appears
            #                      here.
            #
            #   epistemic: X      — appears only when the record's
            #                      `_meta.epistemic_class` differs from the
            #                      type's default (e.g. a `response` whose
            #                      class is `speculative` instead of the usual
            #                      `inferred`, or `known`/`user_context`
            #                      instead). Names how the content is known —
            #                      the one distinction the model cannot derive
            #                      from the content alone. The valid classes
            #                      are `known`, `user_context`, `inferred`,
            #                      `speculative`, and `disputed` (see
            #                      metadata.VALID_EPISTEMIC_CLASSES).
            #                      Default-matching records show no tag,
            #                      keeping headers terse: a `response` tagged
            #                      `inferred` (the default) is silent; only the
            #                      atypical case surfaces.
            extras = [
                f"size={len(content_repr)} chars",
                f"salience={meta.salience:.2f}",
            ]
            if meta.truncated:
                extras.append("truncated")
            if rec.index in pinned:
                extras.append("pinned")
            # Modalities and senses are emitted only when non-empty, matching
            # the storage discipline in metadata.build_meta. A record that
            # fired no modality (or no sense) shows no tag at all rather than
            # an empty list — keeps the headers terse and consistent with how
            # the underlying _meta is written.
            if meta.modalities_activated:
                extras.append(f"modalities: {', '.join(meta.modalities_activated)}")
            if meta.senses_activated:
                extras.append(f"senses: {', '.join(meta.senses_activated)}")
            # Epistemic class: surface only when it differs from the type's
            # default. The same response text could be a measured factual
            # claim or a speculative inference, and the model has no way to
            # tell without the tag — but tagging every record with its
            # default class would dilute headers with constant information.
            # Showing only the atypical case keeps the tag a real signal:
            # "this record's epistemic stance is not what you'd expect from
            # its type alone."
            default_epistemic = DEFAULT_EPISTEMIC_BY_TYPE.get(
                rec.type, EPISTEMIC_INFERRED
            )
            if meta.epistemic_class and meta.epistemic_class != default_epistemic:
                extras.append(f"epistemic: {meta.epistemic_class}")
            # Imported-capsule provenance: an imported_capsule record is
            # another agent's memory (source is already `peer_agent`, which
            # shows in the tag line). Add a short origin fingerprint so the
            # model can see WHICH peer it came from and never conflate two
            # peers' imported memories — and so the foreign-ness is
            # unmistakable, reinforcing the system-prompt guidance to treat
            # these as third-party claims rather than first-person history.
            if rec.type == "imported_capsule" and isinstance(rec.content, dict):
                origin = rec.content.get("origin_pubkey")
                if isinstance(origin, str) and origin:
                    extras.append(f"imported from {origin[:12]}…")
            tag_line = (
                f"[record {rec.index} | {tag} | {meta.source} | {when} "
                f"| {' | '.join(extras)}]"
            )
            ctx_blocks.append(f"{tag_line} {content_repr}")
        ctx = "\n".join(ctx_blocks) if ctx_blocks else "(no prior context)"
        head_idx = self.chain.length() - 1

        truncation_note = ""
        if dropped:
            # Naming the evicted indices matters: without them, the model
            # can't distinguish "record N wasn't retrieved" from "record
            # N was retrieved but dropped from this turn's prompt due to
            # budget pressure." Those two cases look identical from
            # inside the prompt — the record isn't there — but they
            # have different remedies (raise RECENT_N vs raise
            # CONTEXT_BUDGET_CHARS). Listing the evicted indices lets
            # the model give the user an actionable answer.
            dropped_idx_list = ", ".join(str(r.index) for r in dropped)
            truncation_note = (
                f"Note: {len(dropped)} record(s) were retrieved but "
                f"omitted from this turn's prompt due to the context "
                f"character budget. Evicted indices: [{dropped_idx_list}]. "
                f"If a user asks about one of these, you may inform them "
                f"the record exists on the chain and was retrieved, but "
                f"didn't fit this prompt — they can raise "
                f"CONTEXT_BUDGET_CHARS in run.py if they need it inline.\n\n"
            )

        # Quarantined-records note: the retriever drops quarantined
        # records from the returned context (correct — they must not
        # feed the model as ordinary memory) but exposes the list of
        # filtered indices on `last_quarantined_indices`. Surfacing the
        # indices in the prompt header lets the model honestly answer
        # "yes, record N is on the chain but is quarantined" instead
        # of "I don't see it" when the user asks about a record they
        # know was committed. The model still cannot READ the content
        # — only acknowledge existence.
        quarantine_note = ""
        quarantined_indices = getattr(
            self.retriever, "last_quarantined_indices", []
        )
        if quarantined_indices:
            q_list = ", ".join(str(i) for i in quarantined_indices)
            quarantine_note = (
                f"Note: {len(quarantined_indices)} record(s) matched "
                f"this turn's retrieval but were filtered out by the "
                f"protected-zones membrane because they are marked "
                f"`exposure=quarantine` (typically committed prompt-"
                f"injection attempts or PoQ-flagged content). "
                f"Quarantined indices: [{q_list}]. You may tell the "
                f"user these records exist on the chain but cannot be "
                f"read into your context — that is by design and "
                f"protects you. Do NOT speculate about their content.\n\n"
            )

        # Detect gaps since last turn — useful for the agent to notice when
        # someone is returning after a long pause vs. continuing a session.
        gap_note = ""
        # Find the most recent observation/response BEFORE this turn.
        # Single-record turn shape: nothing is appended pre-LLM, so the
        # chain head IS the previous turn's response — include it
        # (iter_records' end is exclusive, hence head_idx + 1).
        prior_records = list(self.chain.iter_records(
            start=max(0, head_idx - 5), end=head_idx + 1))
        prior_conversational = [
            r for r in prior_records if r.type in ("observation", "response")
        ]
        if prior_conversational:
            last = prior_conversational[-1]
            gap_seconds = (now_ms - last.timestamp) / 1000
            if gap_seconds > 3600:  # only mention gaps over 1 hour
                gap_note = (
                    f"Note: it has been {_humanize_delta(gap_seconds)} since the "
                    f"last exchange — the user may be returning after a pause.\n\n"
                )

        # Continuation directive: if the user's input is a short
        # "please keep going" message AND the previous response on the
        # chain was cut off at the model's max_tokens limit, tell the
        # model exactly what's being asked. Without this directive, the
        # model sees its own truncated response as a completed turn and
        # treats "continue" as an ambiguous instruction needing
        # interpretation — wasting tokens (and often visible reasoning)
        # working out what to continue from. With the directive, the
        # model knows the previous response is incomplete and resumes
        # generating from where it left off.
        # The directive keys off the PERSISTED state (the truncated /
        # tool_budget_exhausted meta flags on the last response), not off
        # recognizing the user's phrasing: an earlier version only fired on
        # a whitelist of exact phrases ("continue", "go on", …), so a user
        # typing "resume" or "keep working" silently restarted the task on
        # a fresh budget. A short explicit continue-phrase still gets the
        # strongest wording; any other input gets an advisory note and the
        # model judges whether it is a resume request.
        continuation_note = ""
        normalized = user_input.strip().lower().rstrip(".!")
        is_explicit_continue = normalized in {
            "continue", "go on", "keep going", "please continue",
            "please go on", "carry on", "continue please", "resume",
            "keep working",
        }
        # Walk backward to find the most recent response record. The head
        # itself is included: nothing is appended pre-LLM since the
        # single-record turn shape, so the previous turn's response IS the
        # head (iter_records' end is exclusive, hence head_idx + 1). Limit
        # the scan to avoid a full chain walk on every turn.
        scan_start = max(0, head_idx - 20)
        recent = list(self.chain.iter_records(start=scan_start,
                                              end=head_idx + 1))
        last_response = next(
            (r for r in reversed(recent) if r.type == "response"), None
        )
        if last_response is not None:
            last_meta = read_meta(last_response)
            if last_meta.truncated:
                if is_explicit_continue:
                    continuation_note = (
                        "IMPORTANT: your previous response (record "
                        f"{last_response.index}) was cut off at the "
                        "model's max_tokens limit — it is incomplete. "
                        "The user is asking you to continue that "
                        "response from exactly where it stopped. Do "
                        "NOT restart, summarize what you said, or ask "
                        "what to continue from. Pick up mid-sentence "
                        "if necessary and finish the answer.\n\n"
                    )
                else:
                    continuation_note = (
                        "Note: your previous response (record "
                        f"{last_response.index}) was cut off at the "
                        "model's max_tokens limit — it is incomplete. "
                        "If the user's message asks you to continue (in "
                        "any wording), resume that response from exactly "
                        "where it stopped instead of restarting; "
                        "otherwise answer their message normally.\n\n"
                    )
            elif last_meta.tool_budget_exhausted:
                # Distinct from truncation: the TEXT ended cleanly
                # (a progress checkpoint), the TASK didn't. Resume
                # mid-task, not mid-sentence.
                if is_explicit_continue:
                    continuation_note = (
                        "IMPORTANT: your previous response (record "
                        f"{last_response.index}) stopped because the "
                        "turn's TOOL round budget ran out — the task is "
                        "unfinished. The user is granting you a FRESH "
                        "tool budget to resume it. Re-orient from the "
                        "progress checkpoint in that response (what was "
                        "done, what remains), then go straight back to "
                        "work with tool calls. Do NOT restart from "
                        "scratch, re-read what you already read, or ask "
                        "what to continue from.\n\n"
                    )
                else:
                    continuation_note = (
                        "Note: your previous response (record "
                        f"{last_response.index}) stopped at the turn's "
                        "TOOL round budget — the task is unfinished and "
                        "a progress checkpoint is in that response. If "
                        "the user's message asks you to continue or "
                        "resume (in any wording), re-orient from the "
                        "checkpoint and go straight back to work with a "
                        "fresh tool budget — do NOT restart from scratch "
                        "or re-read what you already read. If they are "
                        "asking something else, answer that instead.\n\n"
                    )

        return (
            f"Current time: {now_str}.\n"
            f"Total records in your chain: {head_idx + 1} (indices 0 through {head_idx}).\n"
            "The records below are a SELECTIVE retrieval based on relevance — gaps\n"
            "in indices do not mean records are missing, only that they weren't\n"
            "retrieved this turn. Don't speculate about what's not shown.\n"
            "Each record is tagged with its source: 'user' (the user said it),\n"
            "'assistant' (you said or inferred it), 'system' (operator config),\n"
            "or 'tool' (produced by a tool such as file ingestion). Treat user\n"
            "statements as evidence about what was said; treat your own past\n"
            "inferences as inferences, not facts.\n"
            "Records of type 'revision' correct earlier records. Records marked\n"
            "SUPERSEDED have been corrected by a later revision — read both, but\n"
            "trust the revision over the original where they conflict.\n"
            "Each record shows its relative time (e.g. '3 hours ago'); use this\n"
            "naturally when relevant, but don't over-narrate it.\n"
            "Each record tag also includes its rendered character size,\n"
            "its salience (0.0-1.0 — higher means more important to retain),\n"
            "and one or both of these flags when applicable:\n"
            "  - 'truncated' — the record's content itself is incomplete\n"
            "    (cut off at the model's max_tokens when it was written).\n"
            "    What you see in the prompt IS the full record on the\n"
            "    chain; nothing was hidden from you. If a user asks\n"
            "    you to continue or complete it, do so directly.\n"
            "  - 'pinned' — the user named this record explicitly in\n"
            "    their current message ('record 328', '#42'). They want\n"
            "    this one in particular; treat it as the focus.\n"
            "Use the size and salience fields directly when asked — do\n"
            "not estimate or count characters yourself when the tag\n"
            "already gives you the answer.\n\n"
            f"{truncation_note}"
            f"{quarantine_note}"
            f"{gap_note}"
            f"{continuation_note}"
            f"Relevant memory:\n{ctx}\n\n"
            f"User: {user_input}"
        )


# ---------------------------------------------------------------------------
# Mock LLM for tests and offline use (deterministic, no network, no API keys)
# ---------------------------------------------------------------------------

class MockLLM:
    """
    Deterministic fake model. Echoes back a structured summary of context +
    input so tests and offline experimentation can exercise the chain
    without needing API keys. Replace with a real LLM client for real use.
    """

    def __call__(self, prompt: str, system: Optional[str] = None,
                 attachments: Optional[list[dict]] = None) -> str:
        # Pull just the current input line for the echo
        lines = prompt.splitlines()
        current = next((l for l in lines if l.startswith("User:")), "")
        sys_tag = " [w/sys]" if system else ""
        att_tag = f" [w/{len(attachments)} attach]" if attachments else ""
        return f"Acknowledged{sys_tag}{att_tag}. {current.replace('User: ', '')}"
