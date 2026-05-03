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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from chain import Chain, Record
from retrieval import Retriever, RetrievalHit


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


@dataclass
class AgentTurn:
    observation_record: Record
    retrieved: list[Record]
    response_record: Record
    response_text: str


class Agent:
    def __init__(
        self,
        chain: Chain,
        retriever: Retriever,
        llm: LLMCall,
        system_prompt: Optional[str] = None,
        context_char_budget: int = 80_000,
    ):
        """
        context_char_budget: approximate maximum characters in the assembled
        prompt. As a rough rule of thumb, 1 token ~= 4 chars for English text,
        so 80_000 chars is roughly 20K tokens — comfortably under any modern
        model's window while leaving room for response generation. If retrieval
        produces more context than this, the lowest-salience records are
        dropped first until the budget fits.
        """
        self.chain = chain
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt
        self.context_char_budget = context_char_budget

    def commit_genesis(self, founding_commitments: list[str]) -> Record:
        """
        Write the agent's founding commitments as record 0. These become the
        anchor for drift detection. Cannot be modified without breaking the chain.
        """
        if self.chain.length() > 0:
            raise RuntimeError("genesis already committed")
        return self.chain.append(
            "genesis",
            {
                "commitments": founding_commitments,
                "schema_version": 1,
            },
        )

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
            {"text": self.system_prompt, "schema_version": 1},
        )

    def turn(self, user_input: str, retrieve_k: int = 5) -> AgentTurn:
        # 1. Commit observation
        obs = self.chain.append("observation", {"text": user_input})

        # 2. Retrieve relevant context
        context = self.retriever.build_context(
            query=user_input, k_semantic=retrieve_k, n_recent=3
        )

        # 3. Build prompt
        prompt = self._format_prompt(user_input, context)

        # 4. Call model — pass system prompt if configured
        if self.system_prompt:
            response_text = self.llm(prompt, system=self.system_prompt)
        else:
            response_text = self.llm(prompt)

        # 5. Commit response with refs to what informed it
        refs = [r.record_hash for r in context] + [obs.record_hash]
        response = self.chain.append(
            "response", {"text": response_text}, refs=refs
        )

        return AgentTurn(
            observation_record=obs,
            retrieved=context,
            response_record=response,
            response_text=response_text,
        )

    # -----------------------------------------------------------------
    # Reflection — the agent looks at recent history and writes about
    # what it noticed. Triggered manually (/reflect) or periodically.
    # -----------------------------------------------------------------

    def reflect(self, window: int = 20) -> Optional[Record]:
        """
        Read the last `window` records, ask the LLM to reflect on them, and
        write the reflection as a new chain record. Reflections become part
        of retrievable memory and get a salience boost in retrieval.

        Returns the new reflection record, or None if there's not enough
        history to reflect on yet.
        """
        recent = self.chain.query_recent(limit=window)
        # Skip if we have nothing meaningful — genesis only, or just the
        # system prompt record.
        substantive = [r for r in recent if r.type in ("observation", "response", "reflection", "revision")]
        if len(substantive) < 4:
            return None

        recent_chronological = sorted(recent, key=lambda r: r.index)
        history_text = self._format_history_for_reflection(recent_chronological)

        prompt = (
            "Below are the most recent records from your memory, in order.\n"
            "Reflect on them. What stands out? What patterns do you notice?\n"
            "What might be worth revisiting or correcting? What did the user\n"
            "seem to actually be reaching for, beyond their literal questions?\n"
            "What do you think mattered most?\n\n"
            "Be concise — a few paragraphs. Write for your future self, who\n"
            "will retrieve this when something relevant comes up. Skip\n"
            "throat-clearing; just say what you noticed.\n\n"
            f"Recent records:\n{history_text}"
        )

        if self.system_prompt:
            reflection_text = self.llm(prompt, system=self.system_prompt)
        else:
            reflection_text = self.llm(prompt)

        refs = [r.record_hash for r in recent_chronological]
        return self.chain.append(
            "reflection",
            {"text": reflection_text, "window_size": window},
            refs=refs,
        )

    # -----------------------------------------------------------------
    # Revision — the agent explicitly corrects a prior record. The
    # original record stays (chain is append-only), but the revision
    # becomes retrievable context that surfaces alongside the original.
    # -----------------------------------------------------------------

    def revise(self, target_index: int, correction_text: str) -> Optional[Record]:
        """
        Append a 'revision' record correcting a prior record. The original
        is never modified — that would break the chain. Instead, the
        revision references the original by hash and becomes part of
        retrievable memory.

        Returns the revision record, or None if the target index doesn't exist.
        """
        target = self.chain.get(target_index)
        if target is None:
            return None
        return self.chain.append(
            "revision",
            {
                "text": correction_text,
                "revises_index": target_index,
                "revises_hash": target.record_hash,
            },
            refs=[target.record_hash],
        )

    def _format_history_for_reflection(self, records: list[Record]) -> str:
        now_ms = int(time.time() * 1000)
        lines = []
        for rec in records:
            try:
                content = rec.content.get("text", "") if isinstance(rec.content, dict) else str(rec.content)
            except AttributeError:
                content = str(rec.content)
            # Truncate long content for the reflection prompt
            if len(content) > 500:
                content = content[:500] + "..."
            when = _humanize_delta((now_ms - rec.timestamp) / 1000)
            lines.append(f"[{rec.index} | {when}] {rec.type}: {content}")
        return "\n".join(lines)

    # Priority order for keeping records under the char budget. Higher
    # priority records are kept first when truncation is needed. Ties are
    # broken by recency (newer wins) inside _truncate_to_budget.
    _RETENTION_PRIORITY = {
        "reflection": 5,
        "revision": 4,
        "genesis": 3,
        "system_prompt": 2,
        "response": 1,
        "observation": 0,
    }

    def _truncate_to_budget(
        self, records: list[Record], fixed_overhead_chars: int
    ) -> tuple[list[Record], int]:
        """
        Drop lowest-priority records until the total rendered context fits
        under (context_char_budget - fixed_overhead_chars).

        Returns (kept_records, dropped_count). Kept records are returned in
        chronological order regardless of priority.
        """
        budget = max(0, self.context_char_budget - fixed_overhead_chars)

        def render_size(rec: Record) -> int:
            try:
                return len(json.dumps(rec.content, ensure_ascii=False))
            except (TypeError, ValueError):
                return len(str(rec.content))

        sized = [(r, render_size(r) + 80) for r in records]  # +80 for label/wrapping
        total = sum(s for _, s in sized)
        if total <= budget:
            return records, 0

        # Sort by retention priority desc, then index desc (newer first within type).
        ranked = sorted(
            sized,
            key=lambda rs: (self._RETENTION_PRIORITY.get(rs[0].type, 0), rs[0].index),
            reverse=True,
        )
        kept: list[Record] = []
        running = 0
        dropped = 0
        for rec, size in ranked:
            if running + size <= budget:
                kept.append(rec)
                running += size
            else:
                dropped += 1
        kept.sort(key=lambda r: r.index)
        return kept, dropped

    def _format_prompt(self, user_input: str, context: list[Record]) -> str:
        now_ms = int(time.time() * 1000)
        now_str = _format_absolute_time(now_ms)

        # Find any revisions that target records currently in context, so
        # the model sees both "what was originally said" and "the correction."
        context_hashes = {r.record_hash for r in context}
        revisions = self.chain.query_by_type("revision", limit=200)
        relevant_revisions = [
            rev for rev in revisions
            if rev.content.get("revises_hash") in context_hashes
            and rev.record_hash not in context_hashes
        ]

        # Merge revisions into context, sorted chronologically
        all_recs = sorted(context + relevant_revisions, key=lambda r: r.index)

        # Truncate to budget, keeping highest-salience records.
        # Fixed overhead: header + user input + budget cushion.
        fixed_overhead = 600 + len(user_input)
        all_recs, dropped = self._truncate_to_budget(all_recs, fixed_overhead)

        ctx_blocks = []
        for rec in all_recs:
            try:
                content_repr = json.dumps(rec.content, ensure_ascii=False)
            except (TypeError, ValueError):
                content_repr = str(rec.content)
            tag = rec.type
            if rec.type == "revision":
                tag = f"revision (corrects #{rec.content.get('revises_index')})"
            when = _humanize_delta((now_ms - rec.timestamp) / 1000)
            ctx_blocks.append(
                f"[record {rec.index} | {tag} | {when}] {content_repr}"
            )
        ctx = "\n".join(ctx_blocks) if ctx_blocks else "(no prior context)"
        head_idx = self.chain.length() - 1

        truncation_note = ""
        if dropped > 0:
            truncation_note = (
                f"Note: {dropped} lower-salience record(s) were omitted from "
                f"this turn's context to stay within the prompt budget.\n\n"
            )

        # Detect gaps since last turn — useful for the agent to notice when
        # someone is returning after a long pause vs. continuing a session.
        gap_note = ""
        # Find the most recent observation/response BEFORE this turn
        # (the just-appended observation is the head — skip it)
        prior_records = list(self.chain.iter_records(start=max(0, head_idx - 5), end=head_idx))
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

        return (
            f"Current time: {now_str}.\n"
            f"Total records in your chain: {head_idx + 1} (indices 0 through {head_idx}).\n"
            "The records below are a SELECTIVE retrieval based on relevance — gaps\n"
            "in indices do not mean records are missing, only that they weren't\n"
            "retrieved this turn. Don't speculate about what's not shown.\n"
            "Records of type 'revision' correct earlier records — when both are\n"
            "shown, the revision supersedes the original.\n"
            "Each record shows its relative time (e.g. '3 hours ago'); use this\n"
            "naturally when relevant, but don't over-narrate it.\n\n"
            f"{truncation_note}"
            f"{gap_note}"
            f"Relevant memory:\n{ctx}\n\n"
            f"User: {user_input}"
        )


# ---------------------------------------------------------------------------
# Mock LLM for the demo (so the demo is reproducible and offline)
# ---------------------------------------------------------------------------

class MockLLM:
    """
    Deterministic fake model. Echoes back a structured summary of context +
    input so the demo shows the chain doing its job without needing API keys.
    Replace with a real LLM client.
    """

    def __call__(self, prompt: str, system: Optional[str] = None) -> str:
        # Pull just the current input line for the echo
        lines = prompt.splitlines()
        current = next((l for l in lines if l.startswith("User:")), "")
        sys_tag = " [w/sys]" if system else ""
        return f"Acknowledged{sys_tag}. {current.replace('User: ', '')}"
