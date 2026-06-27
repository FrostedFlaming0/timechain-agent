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
from pathlib import Path
from typing import Callable, Optional

from chain import Chain, Record
from retrieval import Retriever, RetrievalHit
from file_ingest import IngestResult, ingest_file as _ingest_file
from metadata import (
    build_meta,
    read_meta,
    SOURCE_USER,
    SOURCE_ASSISTANT,
    SOURCE_SYSTEM,
    SOURCE_TOOL,
)


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
        blob_dir: Optional[Path] = None,
    ):
        """
        context_char_budget: approximate maximum characters in the assembled
        prompt. As a rough rule of thumb, 1 token ~= 4 chars for English text,
        so 80_000 chars is roughly 20K tokens — comfortably under any modern
        model's window while leaving room for response generation. If retrieval
        produces more context than this, the lowest-salience records are
        dropped first until the budget fits.

        blob_dir: directory where ingested file bytes are stored
        (content-addressed by sha256). Required for file ingestion.
        """
        self.chain = chain
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt
        self.context_char_budget = context_char_budget
        self.blob_dir = Path(blob_dir) if blob_dir else None

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
                "_meta": build_meta(
                    "genesis",
                    source=SOURCE_SYSTEM,
                    salience=1.0,        # foundational — never decay out
                    confidence=1.0,
                ),
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
            {
                "text": self.system_prompt,
                "_meta": build_meta(
                    "system_prompt",
                    source=SOURCE_SYSTEM,
                    confidence=1.0,
                ),
            },
        )

    def turn(self, user_input: str, retrieve_k: int = 5) -> AgentTurn:
        # 1. Commit observation
        obs = self.chain.append(
            "observation",
            {
                "text": user_input,
                "_meta": build_meta(
                    "observation",
                    source=SOURCE_USER,
                    confidence=1.0,  # the user said it; that's a fact about what was said
                ),
            },
        )

        # 2. Retrieve relevant context
        context = self.retriever.build_context(
            query=user_input, k_semantic=retrieve_k, n_recent=3
        )

        # 3. Build prompt
        prompt = self._format_prompt(user_input, context)

        # 4. Gather any file attachments to send natively (images, PDFs).
        # Multimodal models will use them; text-only models ignore them
        # and rely on the extracted_text already in the prompt.
        attachments = self._collect_attachments(context)

        # 5. Call model — pass system prompt and attachments if relevant
        llm_kwargs = {}
        if self.system_prompt:
            llm_kwargs["system"] = self.system_prompt
        if attachments:
            llm_kwargs["attachments"] = attachments
        response_text = self.llm(prompt, **llm_kwargs) if llm_kwargs else self.llm(prompt)

        # 6. Commit response with refs to what informed it
        refs = [r.record_hash for r in context] + [obs.record_hash]
        response = self.chain.append(
            "response",
            {
                "text": response_text,
                "_meta": build_meta(
                    "response",
                    source=SOURCE_ASSISTANT,
                    # Default confidence for a response is high but not 1.0 —
                    # the model's output is its best effort, not ground truth.
                    confidence=0.9,
                ),
            },
            refs=refs,
        )

        return AgentTurn(
            observation_record=obs,
            retrieved=context,
            response_record=response,
            response_text=response_text,
        )

    def _collect_attachments(self, context: list[Record]) -> list[dict]:
        """
        Pull image and PDF blobs for any file records in the retrieved context,
        so multimodal LLM clients can send the original bytes alongside the
        extracted text. Capped to avoid sending excessive payloads per turn.
        """
        if self.blob_dir is None:
            return []
        MAX_ATTACHMENTS = 4
        MAX_ATTACH_BYTES = 10 * 1024 * 1024  # 10 MB total per turn
        out: list[dict] = []
        total = 0
        for rec in context:
            if rec.type != "file" or not isinstance(rec.content, dict):
                continue
            kind = rec.content.get("kind")
            if kind not in ("image",) and rec.content.get("ext") != ".pdf":
                continue
            blob_path = self.blob_dir / rec.content.get("blob_path", "")
            if not blob_path.exists():
                continue
            data = blob_path.read_bytes()
            if total + len(data) > MAX_ATTACH_BYTES:
                continue
            attach = {
                "kind": "image" if kind == "image" else "pdf",
                "data": data,
                "filename": rec.content.get("filename", ""),
                "media_type": _guess_media_type(rec.content.get("ext", "")),
            }
            out.append(attach)
            total += len(data)
            if len(out) >= MAX_ATTACHMENTS:
                break
        return out

    # -----------------------------------------------------------------
    # Reflection — the agent looks at recent history and writes about
    # what it noticed. Triggered manually (/reflect) or periodically.
    # -----------------------------------------------------------------

    def reflect(self, max_records: int = 200) -> Optional[Record]:
        """
        Reflect on every record since the last reflection (or since
        genesis if there hasn't been one yet). The size of the window is
        determined dynamically — a reflection covers exactly the slice
        of chain history that the previous reflection didn't.

        `max_records` is a safety cap. If the lookback would exceed it,
        only the most recent `max_records` are reflected on. This
        protects against runaway size when auto-reflection is disabled
        and the chain has grown a lot since the last manual `/reflect`.
        Tunable per call; default of 200 fits comfortably in any modern
        LLM context window.

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

    # -----------------------------------------------------------------
    # File ingestion — read a file from disk, store its bytes content-
    # addressed in the blob dir, and append a 'file' record carrying
    # extracted text plus metadata. The blob is recoverable from disk;
    # the chain record makes the file searchable and provenance-checked.
    # -----------------------------------------------------------------

    def ingest_file(self, path: str | Path) -> Optional[Record]:
        """
        Append a 'file' record for `path`. Stores the file's bytes as a
        blob (content-addressed by sha256) and records extracted text
        plus metadata on the chain.

        Returns the new record, or raises if the file is missing,
        unsupported, or too large. Returns None only if no blob_dir
        was configured on this Agent.
        """
        if self.blob_dir is None:
            raise RuntimeError(
                "Agent.ingest_file requires blob_dir to be configured "
                "in the Agent constructor"
            )
        result: IngestResult = _ingest_file(path, self.blob_dir)
        content = result.to_record_content()
        content["_meta"] = build_meta(
            "file",
            source=SOURCE_TOOL,
            confidence=1.0,  # the bytes are what they are; sha256 verifies it
        )
        return self.chain.append("file", content)

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

    # Truncation order is driven by per-record salience (see _truncate_to_budget).

    def _truncate_to_budget(
        self, records: list[Record], fixed_overhead_chars: int
    ) -> tuple[list[Record], int]:
        """
        Drop lowest-salience records until the total rendered context fits
        under (context_char_budget - fixed_overhead_chars).

        Salience is read from each record's _meta block (with type-based
        defaults for v1 records — see metadata.py). This replaces the older
        type-priority table: per-record salience is finer-grained and the
        record itself is the right place for that judgment to live.

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

        # Sort by salience desc, then index desc (newer first within ties).
        ranked = sorted(
            sized,
            key=lambda rs: (read_meta(rs[0]).salience, rs[0].index),
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

        # Compute supersession set so we can flag superseded originals.
        # Read from revision _meta (or legacy revises_index) — see read_meta.
        superseded_indices = set()
        for rev in revisions:
            rmeta = read_meta(rev)
            if rmeta.supersedes is not None:
                superseded_indices.add(rmeta.supersedes)

        ctx_blocks = []
        for rec in all_recs:
            # Strip _meta from rendered content — it's metadata about the
            # record, not part of what the record says. Source/salience
            # surface as visible tags below instead.
            display_content = rec.content
            if isinstance(display_content, dict) and "_meta" in display_content:
                display_content = {k: v for k, v in display_content.items() if k != "_meta"}
            try:
                if rec.type == "file":
                    # Render files specially: a short metadata header plus
                    # the extracted text. Keeps prompt compact and readable.
                    c = display_content if isinstance(display_content, dict) else {}
                    trunc = " (text truncated)" if c.get("extraction_truncated") else ""
                    content_repr = (
                        f'file: {c.get("filename", "?")} ({c.get("kind", "?")}, '
                        f'{c.get("size_bytes", 0):,} bytes, sha256 '
                        f'{c.get("blob_sha256", "")[:12]}...){trunc}\n'
                        f'{c.get("extracted_text", "")}'
                    )
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
            ctx_blocks.append(
                f"[record {rec.index} | {tag} | {meta.source} | {when}] {content_repr}"
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
            "Each record is tagged with its source: 'user' (the user said it),\n"
            "'assistant' (you said or inferred it), 'system' (operator config),\n"
            "or 'tool' (produced by a tool such as file ingestion). Treat user\n"
            "statements as evidence about what was said; treat your own past\n"
            "inferences as inferences, not facts.\n"
            "Records of type 'revision' correct earlier records. Records marked\n"
            "SUPERSEDED have been corrected by a later revision — read both, but\n"
            "trust the revision over the original where they conflict.\n"
            "Each record shows its relative time (e.g. '3 hours ago'); use this\n"
            "naturally when relevant, but don't over-narrate it.\n\n"
            f"{truncation_note}"
            f"{gap_note}"
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
