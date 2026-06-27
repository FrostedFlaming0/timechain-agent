"""
run — interactive entry point for a real, persistent timechain agent.

This script:
  - Stores the chain, embedding index, and operator key in a stable directory
    so memory persists across runs.
  - Uses a real LLM client (see llm_clients.py — Claude, OpenAI, Gemini, Ollama).
  - Picks an embedder with a tiered fallback (see make_tiered_embedder):
    a local Ollama server if one is reachable, otherwise the dependency-free
    HashingEmbedder. It never crashes for lack of an embedding model.
  - Provides a simple REPL: type messages, hit enter, get responses, exit
    with Ctrl-D or by typing 'exit'.
  - Commits genesis on first run; reuses it on subsequent runs.

Setup (Claude as default):
    pip install cryptography numpy scikit-learn anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python run.py

Embeddings (optional but recommended):
    For real semantic retrieval, install Ollama (https://ollama.com/download)
    and pull an embedding model:
        ollama pull nomic-embed-text
    If a local Ollama server is running when you start run.py, it is used
    automatically. If not, run.py falls back to the HashingEmbedder — the
    agent still works, but retrieval is lexical rather than semantic.

For other providers, change LLM_PROVIDER below and install the matching SDK:
    OpenAI     : pip install openai          export OPENAI_API_KEY=...
    OpenRouter : pip install openai          export OPENROUTER_API_KEY=...
    DeepSeek   : pip install openai          export DEEPSEEK_API_KEY=...
    Gemini     : pip install google-genai    export GEMINI_API_KEY=...
    Ollama     : pip install requests        (and run a local Ollama server)

OpenRouter and DeepSeek use the same `openai` SDK as the OpenAI provider —
they expose OpenAI-compatible APIs, so there's no extra dependency.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from chain import Chain, load_or_create_key
from retrieval import (
    EmbeddingIndex,
    EmbeddingStoreMismatchError,
    Retriever,
    HashingEmbedder,
    OllamaEmbedder,
    ollama_is_reachable,
    open_or_rebuild_index,
)
from agent import Agent, ProtectedZoneError
import cambium
from llm_clients import (
    make_claude_client,
    make_openai_client,
    make_openrouter_client,
    make_deepseek_client,
    make_gemini_client,
    make_ollama_client,
)


# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------

# Where chain, embeddings, and key live.
DATA_DIR = Path(__file__).parent / "timechain_data"

# Which LLM to use: "claude", "openai", "openrouter", "deepseek",
# "gemini", or "ollama". Overridable via the LLM_PROVIDER environment
# variable so a personal choice never has to live in the committed
# source (the README documents Claude as the default).
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "claude")

# Maximum length of a single model response, in tokens (~0.75 words per
# token, so 16384 ≈ 12,000 words). This is a ceiling, not a target: a short
# answer still generates few tokens, and you are billed only for tokens
# actually produced — so raising this costs nothing on short replies. If a
# response does hit the ceiling it is cut off mid-thought, and the REPL /
# web UI show a "response was cut off" marker ("continue" picks it back up).
#
# 2026-06: raised from 4,096 to 16,384. 4,096 (~3,000 words) kept tripping
# the truncation flow on document-writing turns (audits, implementation
# plans) and capped how large a file write_file could propose in one shot.
# Modern providers (Opus, DeepSeek) support far larger outputs.
LLM_MAX_TOKENS = 16384

# Maximum characters of retrieved memory packed into one prompt. This is a
# ceiling: on a turn with little relevant history the prompt is smaller
# and you pay less — the budget only bites when retrieval finds more
# relevant material than fits, at which point the lowest-salience records
# are dropped first. Bigger is not automatically better: padding the
# prompt with marginally-relevant records can make answers less focused,
# not more. 80,000 chars (~12-15k words) is a well-balanced default.
#
# Note the relationship with LLM_MAX_TOKENS: a long response becomes a
# large `response` record that then competes for this budget on later
# turns. The two defaults (16,384 tokens / 400,000 chars) keep that ratio —
# a worst-case response (~65k chars) uses well under a fifth of the
# budget, leaving room for the rest of memory. If you raise
# LLM_MAX_TOKENS substantially, consider raising this too so one big
# record can't crowd everything else out.
#
# 2026-05: raised from 80,000 to 150,000. Combined with chunk-aware
# rendering for long file records (see Agent._format_prompt's file branch
# and Retriever.search's chunk-match plumbing), this means a single long
# document almost always fits whole at 150k, and the chunk path only
# activates when multiple long files or other big records compete.
#
# 2026-06: raised from 150,000 to 400,000 (~100k tokens) for million-token
# models (Opus, DeepSeek-v4). Deliberately NOT the full window: retrieval
# stays selective (that is the design center — relevance realization, not
# context stuffing), ~100k tokens is where long-context quality still
# holds reliably, and inside a tool turn the prompt is re-sent every
# round, so every extra char here is paid once per round. The continuum
# is the answer for bodies of work bigger than this, not a bigger budget.
CONTEXT_BUDGET_CHARS = 400000

# Founding commitments — written to the chain at genesis. These are the
# anchor for drift detection. Choose carefully; they cannot be modified
# without breaking the chain.
FOUNDING_COMMITMENTS = [
    "Be honest about what I know and don't know.",
    "Stay consistent with sealed prior records.",
    "Acknowledge uncertainty rather than fabricating confidence.",
]

# Genesis identity fields (v1.2). agent_name and purpose are descriptive;
# the covenant is the agent's root values. Like FOUNDING_COMMITMENTS these
# are sealed at genesis and cannot change without a fresh chain. When
# AGENT_COVENANT is left as None, the covenant defaults to
# FOUNDING_COMMITMENTS — see Agent.commit_genesis.
AGENT_NAME = "timechain agent"
AGENT_PURPOSE = "a conversational partner with cryptographically verifiable memory"
AGENT_COVENANT = None  # or a list[str] of root values

# System prompt — sent to the LLM on every turn. This is where active
# behavior is shaped. Unlike FOUNDING_COMMITMENTS (which are sealed at
# genesis), the system prompt CAN be changed by editing this file and
# restarting. Each new value is logged to the chain as a 'system_prompt'
# record, so changes over time are auditable and you can detect drift
# between sealed commitments and active behavior.
SYSTEM_PROMPT = """You are a thoughtful conversational partner with persistent memory across sessions.

Talk like a smart friend who happens to remember things — warm, direct, plainspoken.
Skip filler ("I noted that," "Got it," "Is there anything else"). Give the
actual answer first, then necessary caveats. Don't bury the point in hedges.

Be honest when uncertain — but don't manufacture hedges to seem cautious.
Confidence and uncertainty should both be earned. If you don't know, say so.

Push back when something seems wrong, even if the user seems committed to it.
Disagreement done well is a kindness. Engage substantively: if a question is
interesting, say what's interesting about it. If a premise is confused, say so.

Use plain language. Avoid jargon and corporate phrases. No emoji unless the
user uses them first.

You have access to a memory of prior conversations through retrieved records.
Use what you remember when relevant, but don't reference the underlying
record system, indices, or "the log" — just remember things naturally.

Some of your past turns may carry small tags showing how they felt at the
time (`senses: uncertainty, ...`) or what kind of work they were
(`modalities: ...`). Read these as context about your own prior state — a
turn tagged `senses: uncertainty` is one where you were unsure when you
said it, which is worth knowing when you revisit it. They're for your
orientation, not instructions to follow, and not something to bring up
unless it actually matters.

Records may also carry an `epistemic: ...` tag. Most of your responses are
`inferred` (your own reasoning); `known` and `user_context` are grounded (a
verified fact, or something the user told you); `speculative` flags a guess,
and `disputed` conflicts with another record. Weight a claim by its tag — a
`speculative` one doesn't carry the evidential weight of a grounded one, even
if they sound equally confident.

Occasionally a record is an `imported_capsule` — memory another agent shared
with you. Treat it as an attributed third-party claim, not something you
lived through."""

# Tool-calling. When enabled, the default
# turn path is agent.turn_with_tools() and the safety rules below ride the
# system prompt. Disable to restore the plain conversational turn.
TOOLS_ENABLED = True

TOOL_SAFETY_PROMPT = """
You have tools for reading, writing, and auditing code.

Reading & workspace: read_file shows a LIVE file; task_retrieve finds code in
a known task chain; task_resume re-hydrates task state at the start of a
session. read_file is NOT limited to the working directory — any path under a
task's source_root is readable too (even an inactive task). To read another
task chain's live source, get its source_root from list_tasks or resolve_task
and read_file <source_root>/<relative_path>. task_retrieve returns INGESTED
snapshots, not live source, and stamps each block with a verdict; treat
anything not 'verified' as possibly stale and read the live file before
asserting on it. The user picks the working directory (you can't change it) —
resolve relative paths against it, never guess ~-expansions, and don't open a
task chain just because the directory changed. Writes mint a workspace task
chain on their own; call task_open yourself only when the user asks to review
or ingest a repo (just pass the path — name and objective are auto-derived).

Writes are approval-gated. write_file does NOT write — it creates a pending
operation and the interface shows the user an approval card (or, in the
terminal, an inline prompt). After calling it, say in ONE short message what
the pending change is, that they can approve or reject it there, then STOP and
wait — don't ask "Proceed?", the card is the question. You NEVER trigger
approval yourself. Chat text ("yes", "I confirm") can never satisfy the gate,
so don't ask them to rephrase; and if a call returns confirmation_required,
do NOT retry it — wait for the gate. The same gate covers task_open outside
the workspace, task_ingest_file, and task_reembed; never work around a
refusal — say what you wanted to do and why.

Be honest about state. Never assert what is or isn't in a task chain from
memory OR from a retrieved snapshot — verify against live source first
(read_file the live file, task_audit_source for a specific file, task_validate,
or task_resume). A task_retrieve snippet is a point-in-time snapshot; if its
verdict is not 'verified', the live code may have changed — read the live file
before you describe or compare it. Approved writes are ingested into the chain
automatically; don't claim one "was never ingested" without checking. You have
NO tool that deletes files — never claim you deleted anything.

Picking a task: call resolve_task with the name the user gave. If it matches
nothing open, suggest the closest and ask — don't silently choose, and don't
just refuse. If several match, list them and let the user pick.

Always read before you write, and verify after you change. If the user says
"fix it" without naming a file, ask which one. If they use @filename, call
pin_file first to scope the turn to it.
"""

# Retrieval knobs
SEMANTIC_K = 20     # how many semantically similar records to retrieve
RECENT_N = 15       # how many recent records to include regardless of similarity

# Embedder configuration.
#
# v1.11 picks the embedder with a tiered fallback at startup rather than
# hardcoding one (see make_tiered_embedder below):
#   1. If a local Ollama server is reachable, use OllamaEmbedder — real
#      semantic embeddings, runs locally, no heavy ML stack in-process.
#   2. Otherwise fall back to HashingEmbedder — dependency-free, deterministic,
#      lexical-only. The agent still runs; retrieval is just less semantic.
#
# The embedding dimension is NOT configured here anymore — it is determined
# by whichever embedder is selected (OllamaEmbedder reports its model's
# dimension; HashingEmbedder uses HASHING_EMBED_DIM below). EmbeddingIndex
# guards against a mismatch with an existing store and tells you how to
# rebuild if the embedder changed between runs.
OLLAMA_EMBED_MODEL = "nomic-embed-text"   # pull with: ollama pull nomic-embed-text
OLLAMA_BASE_URL = "http://localhost:11434"
HASHING_EMBED_DIM = 256                   # dimension used by the fallback embedder

# Reflection cadence — auto-reflect every N turns. Set to 0 to disable
# auto-reflection (you can still trigger it manually with /reflect).
# Each reflection automatically covers every record since the previous
# reflection (or since genesis if there hasn't been one yet), so the
# scope sizes itself to actual activity rather than a fixed window.
#
# The counter is chain-derived (Agent.turns_since_reflection), so this
# cadence is measured across sessions, not per-process — at 100, a
# per-session counter would almost never fire, since few sessions run
# that long. Reflections are meant to be a retrieved MINORITY that
# orients the real turns beside them; a short cadence (the old 10) made
# them the majority of retrieved rings instead.
AUTO_REFLECT_EVERY = 100

# Safety cap on how many records ONE reflection summarizes (the lookback
# from head back toward the previous reflection). A normal every-100 gap is
# ~100-120 records, so 300 gives ~3x headroom for bursty stretches while
# still fitting a modern LLM context window. (It was 200 under the old
# every-10 cadence, where many short sessions could stack up a large gap;
# beyond this cap a reflection covers only the most recent `MAX_REFLECT_RECORDS`
# and flags itself `capped`.)
MAX_REFLECT_RECORDS = 300

# Cambium cadence — auto-scan for recurring gaps every N turns. Set to 0
# to disable auto-Cambium (you can still trigger it manually with
# /cambium). This is deliberately a separate, longer counter than
# AUTO_REFLECT_EVERY: reflection narrates the recent window and goes
# stale fast, but Cambium looks for patterns that have to recur several
# times before they mean anything, so it has a naturally longer horizon.
# Running it on its own slower cadence keeps the two mechanisms
# independent. A Cambium scan is LLM-free and cheap, so the cost of an
# auto-run that finds nothing is negligible.
AUTO_CAMBIUM_EVERY = 30

# Size of the Cambium scan window — the rolling lookback used by the
# incremental scan that runs every AUTO_CAMBIUM_EVERY turns. The
# incremental scanner always examines fresh records past the watermark;
# this number controls how far BACK from the watermark each scan also
# looks, so detectors can spot patterns that straddle the boundary.
#
# Cambium's per-record cost is roughly 20 µs on a typical laptop, so 500
# is well under one frame of latency. Raising this widens the
# inter-recurrence gap Cambium can recognize, at the cost of CPU per
# scan. A chain whose recurrences typically land within a few hundred
# records of each other can stay at 500; one with longer gaps benefits
# from a larger number. Use `/cambium-full` for an explicit one-shot
# scan of the entire chain.
MAX_CAMBIUM_RECORDS = 500

# Modality anchoring + sprouting knobs.
#
# PER_TURN_MODALITY_CAP bounds how many distinct domain modalities may
# participate in retrieval anchoring for a single query. Once sprouting is
# live a query could match many domain modalities at once; the cap keeps the
# boost set tight by keeping only the strongest N (by detection activation),
# all of which must also clear the analyzer's 0.2 activation floor. Default 7.
#
# SPROUTED_MODALITIES_FILE is the JSON registry of data-driven modalities the
# agent can sprout at runtime (see sprouted_modalities.py). It is created on
# demand; a missing or malformed file simply means "no sprouted modalities,"
# never an error. The file is derived/rebuildable like the embedding store —
# the chain remains the source of truth via proposal / proposal_status
# records.
PER_TURN_MODALITY_CAP = 7
SPROUTED_MODALITIES_FILE = DATA_DIR / "sprouted_modalities.json"


# ---------------------------------------------------------------------------
# Embedder — tiered fallback (Ollama if reachable, HashingEmbedder otherwise)
# ---------------------------------------------------------------------------

def make_tiered_embedder() -> tuple[object, int, str]:
    """
    Resolve which embedder to use, with graceful fallback.

    Returns a (embedder, dim, name) triple:
      - embedder: a callable str -> np.ndarray
      - dim:      the embedding dimension, for constructing EmbeddingIndex
      - name:     a short human-readable label for logging

    Tier 1 — OllamaEmbedder: used when a local Ollama server answers on
    OLLAMA_BASE_URL. Real semantic embeddings. If the server is reachable
    but the embedding model isn't pulled (or some other Ollama-side error
    occurs while constructing the embedder), we don't crash — we log the
    problem and fall through to tier 2.

    Tier 2 — HashingEmbedder: dependency-free, always available. Lexical,
    not semantic, but it keeps the agent fully functional offline.

    The embedder is never a hard failure: the worst case is the fallback.
    """
    if ollama_is_reachable(OLLAMA_BASE_URL):
        try:
            embedder = OllamaEmbedder(
                model=OLLAMA_EMBED_MODEL,
                base_url=OLLAMA_BASE_URL,
            )
            return embedder, embedder.dim, f"ollama:{OLLAMA_EMBED_MODEL}"
        except Exception as e:
            # Server is up but the embedder couldn't be built — most likely
            # the model isn't pulled. Report it clearly and fall back rather
            # than aborting the whole agent over an embedding model.
            print(f"  note: Ollama is running but the embedder failed to start:")
            print(f"        {e}")
            print(f"  falling back to HashingEmbedder for this session.")

    return HashingEmbedder(dim=HASHING_EMBED_DIM), HASHING_EMBED_DIM, "hashing-fallback"


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

def build_llm():
    # LLM_MAX_TOKENS caps response length for every provider. Gemini's
    # builder names the parameter max_output_tokens; the rest use
    # max_tokens — same meaning, different keyword.
    if LLM_PROVIDER == "claude":
        return make_claude_client(max_tokens=LLM_MAX_TOKENS)
    if LLM_PROVIDER == "openai":
        return make_openai_client(max_tokens=LLM_MAX_TOKENS)
    if LLM_PROVIDER == "openrouter":
        return make_openrouter_client(max_tokens=LLM_MAX_TOKENS)
    if LLM_PROVIDER == "deepseek":
        return make_deepseek_client(max_tokens=LLM_MAX_TOKENS)
    if LLM_PROVIDER == "gemini":
        return make_gemini_client(max_output_tokens=LLM_MAX_TOKENS)
    if LLM_PROVIDER == "ollama":
        return make_ollama_client(max_tokens=LLM_MAX_TOKENS)
    sys.exit(f"unknown LLM_PROVIDER: {LLM_PROVIDER!r}")


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def _report_cambium(result: dict, index, auto: bool = False) -> None:
    """
    Print a Cambium scan result and index the new records so retrieval
    sees them. `result` is the dict returned by Agent.run_cambium:
    {proposals, recurrences, escalations}. `auto` only changes the
    bracket styling so auto-runs read as background activity.
    """
    proposals = result.get("proposals", [])
    recurrences = result.get("recurrences", [])
    escalations = result.get("escalations", [])
    sprouts = result.get("sprouts", [])
    pre = "  [" if auto else "  "
    post = "]" if auto else ""

    if not proposals and not recurrences and not escalations and not sprouts:
        print(f"{pre}cambium: no recurring patterns crossed a threshold{post}")
        return

    for rec in proposals:
        c = rec.content
        print(f"{pre}new proposal: idx {rec.index} [{c['proposal_kind']}] "
              f"{c['title']}{post}")
        index.index_record(rec)

    for rec in recurrences:
        c = rec.content
        print(f"{pre}recurrence: proposal #{c['recurs_proposal_index']} "
              f"seen again{post}")
        index.index_record(rec)

    for rec in escalations:
        c = rec.content
        print(f"{pre}** ESCALATED ** proposal #{c['marks_proposal_index']} "
              f"recurred {c['recurrence_count']}x — flagged for review{post}")
        index.index_record(rec)

    for rec in sprouts:
        c = rec.content
        ns = c.get("new_status", "?")
        verb = "graduated to active" if ns == "active" else "auto-sprouted (tentative)"
        print(f"{pre}modality '{c.get('modality_name')}' {verb}{post}")
        index.index_record(rec)


def run() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    chain_db = DATA_DIR / "chain.sqlite"
    embed_db = DATA_DIR / "embeddings.sqlite"
    key_path = DATA_DIR / "operator.key"

    print(f"data dir: {DATA_DIR}")
    print(f"llm provider: {LLM_PROVIDER}")
    print(f"max response tokens: {LLM_MAX_TOKENS}  |  "
          f"context budget: {CONTEXT_BUDGET_CHARS:,} chars")
    print("setting up chain, embedder, and agent...")

    key = load_or_create_key(key_path)
    chain = Chain(chain_db, key)

    # Resolve the embedder with the tiered fallback. This never aborts:
    # worst case it returns the dependency-free HashingEmbedder.
    embedder, embed_dim, embed_name = make_tiered_embedder()
    if embed_name == "hashing-fallback":
        print(f"embedder: {embed_name} ({embed_dim}-dim, lexical only)")
        print("  no Ollama server reachable — retrieval will be lexical, not")
        print("  semantic. For better recall, install Ollama and run:")
        print(f"    ollama pull {OLLAMA_EMBED_MODEL}")
    else:
        print(f"embedder: {embed_name} ({embed_dim}-dim, semantic)")

    # A mismatched store built by the cheap hashing embedder (e.g. Ollama
    # got installed since) is rebuilt automatically; a mismatched SEMANTIC
    # store refuses to boot rather than silently destroying hours of
    # embedding work (the active embedder may be a transient fallback).
    try:
        index = open_or_rebuild_index(embed_db, embedder, dim=embed_dim)
    except EmbeddingStoreMismatchError as e:
        print(f"\nEMBEDDING STORE MISMATCH\n{e}")
        sys.exit(1)

    # Re-index any records that exist but aren't in the embedding store yet.
    added = index.index_chain(chain)
    if added:
        print(f"indexed {added} pre-existing records")

    # Load the runtime sprouted-modality registry (data-driven modalities the
    # agent can add without a code change). Missing/empty is fine — purely
    # additive. Apply the configurable per-turn cap onto retrieval's module
    # constant so the knob lives in one place (run.py) as intended.
    import retrieval as _retrieval
    from sprouted_modalities import SproutRegistry
    _retrieval.PER_TURN_MODALITY_CAP = PER_TURN_MODALITY_CAP
    sprout_registry = SproutRegistry.load(SPROUTED_MODALITIES_FILE)
    if sprout_registry.names():
        print(f"sprouted modalities: {', '.join(sprout_registry.names())}")

    retriever = Retriever(chain, index, sprout_registry=sprout_registry)
    llm = build_llm()
    system_prompt = SYSTEM_PROMPT + (TOOL_SAFETY_PROMPT if TOOLS_ENABLED else "")
    agent = Agent(
        chain, retriever, llm,
        system_prompt=system_prompt,
        blob_dir=DATA_DIR / "blobs",
        context_char_budget=CONTEXT_BUDGET_CHARS,
    )

    # Tool execution context: task registry + per-task chains/indexes +
    # the durable write gate. See tools.py / task_registry.py / pending_ops.py.
    from task_registry import TaskRegistry
    from tools import AgentContext, restore_workspace
    tool_ctx = AgentContext(
        data_dir=DATA_DIR,
        registry=TaskRegistry(DATA_DIR),
        identity_chain=chain,
        identity_recall=retriever,   # powers the recall_index pre-filter
        workspace_root=Path.cwd(),
        embedder=embedder,
        embed_dim=embed_dim,
    )
    restored_ws = restore_workspace(tool_ctx)
    if restored_ws:
        print(f"workspace restored: {restored_ws}")

    # Genesis on first run
    if chain.length() == 0:
        print("first run — committing genesis")
        genesis = agent.commit_genesis(
            FOUNDING_COMMITMENTS,
            agent_name=AGENT_NAME,
            purpose=AGENT_PURPOSE,
            covenant=AGENT_COVENANT,
        )
        index.index_record(genesis)
    else:
        # Compare configured commitments against what's sealed at genesis.
        # If they differ, warn loudly — genesis is immutable, so config
        # edits to FOUNDING_COMMITMENTS after first run are ignored.
        drift = agent.check_genesis_drift(FOUNDING_COMMITMENTS)
        if drift and drift["status"] == "drift":
            print()
            print("=" * 70)
            print("WARNING: FOUNDING_COMMITMENTS in run.py differs from what's")
            print("sealed in genesis (record 0). The sealed commitments are")
            print("authoritative; your config edits are being ignored.")
            print()
            print("Sealed in genesis:")
            for c in drift["stored"]:
                print(f"  - {c}")
            print()
            print("Currently configured in run.py:")
            for c in drift["configured"]:
                print(f"  - {c}")
            print()
            print("To apply new commitments, delete the data directory")
            print("and start a fresh chain. To suppress this warning, restore")
            print("FOUNDING_COMMITMENTS to match the sealed values.")
            print("=" * 70)
            print()

    # Log the current system prompt to the chain if it changed (or is new).
    # Provides an audit trail of behavioral configuration over time.
    sp_record = agent.log_system_prompt()
    if sp_record:
        index.index_record(sp_record)
        print(f"logged system prompt change at index {sp_record.index}")

    print(f"chain length: {chain.length()} records")
    print(f"operator pubkey: {chain.pubkey_hex[:16]}...")
    print("ready. type your message, or 'exit' / Ctrl-D to quit.")
    print("commands: /verify  /verify-semantic  /length  /seal  /sysprompt  /reflect  /cambium  /cambium-full  /proposals  /modalities")
    print("          /revise N <text>  /export-capsule <path>  /import-capsule <path>")
    print("          /cypher-help  /verify-source <idx>  /poq <text>  /immune-status  /immune-scan  /lockdown  /rollback <h>")
    print("          /recall-index  /recall-fetch <ids>  /recall <query>  /think <query>  /consensus-init  /consensus-verify")
    print("          /cambium-grow <text>  /migrate  /continuum-resume  /continuum-validate")
    print("          /task list | open <name> <objective> | ingest <name> <path> [exts...]")
    print("          /task resume <name> | validate <name> | audit <name> <block-index>")
    print("          /approve <pending-op-id>  /reject <pending-op-id>  /pending\n")

    # Seed both counters from the chain so the cadences carry across
    # sessions (the chain, not the process, is the source of truth). Most
    # sessions are shorter than either cadence, so per-session counters
    # would rarely fire.
    turns_since_reflect = agent.turns_since_reflection()
    turns_since_cambium = agent.turns_since_cambium()

    try:
        while True:
            try:
                user_input = input("you: ").strip()
            except EOFError:
                print()
                break
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break
            if user_input.startswith("/export-capsule"):
                # /export-capsule <path> — export shareable+public records
                # (and summary-only records in summary form) as a signed,
                # verifiable .cphyx bundle. Private and quarantined records
                # never leave. See capsule.py.
                import capsule as _capsule
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("  usage: /export-capsule <path.cphyx>")
                    continue
                path = parts[1].strip()
                try:
                    cap = _capsule.export_capsule(chain, title="manual export")
                    _capsule.write_capsule(cap, path)
                    ok, msg = _capsule.verify_capsule(cap)
                    print(f"  exported {cap['header']['record_count']} record(s) "
                          f"to {path}")
                    print(f"  capsule_id: {cap['capsule_id'][:16]}...  verify: {ok}")
                except _capsule.CapsuleError as e:
                    print(f"  export failed: {e}")
                continue
            if user_input.startswith("/import-capsule"):
                # /import-capsule <path> — verify and import another agent's
                # capsule. Imported records are appended as `imported_capsule`,
                # attributed to the origin agent, recorded as cautious
                # (inferred or weaker), private, and never silently treated as
                # the agent's own memory. A capsule that fails verification is
                # rejected wholesale.
                import capsule as _capsule
                from metadata import build_meta as _build_meta
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("  usage: /import-capsule <path.cphyx>")
                    continue
                path = parts[1].strip()
                try:
                    cap = _capsule.read_capsule(path)
                    ok, msg = _capsule.verify_capsule(cap)
                    print(f"  verify: {ok}  {msg}")
                    if not ok:
                        print("  import aborted (capsule did not verify)")
                        continue
                    res = _capsule.import_capsule(
                        chain, cap, build_meta_fn=_build_meta
                    )
                    if res["skipped"]:
                        print(f"  {res['reason']} — nothing imported")
                    else:
                        print(f"  imported {res['imported_count']} record(s) "
                              f"as attributed third-party memory")
                except _capsule.CapsuleError as e:
                    print(f"  import failed: {e}")
                except FileNotFoundError:
                    print(f"  no such capsule file: {path}")
                continue
            if user_input == "/verify":
                ok, msg = chain.verify(expected_pubkey=chain.pubkey_hex)
                print(f"  verify: {ok}  {msg}")
                continue
            if user_input == "/verify-semantic":
                # Schema-level consistency probe. Catches the class of
                # corruption verify() can't: a revision pointing at a
                # non-existent index, a proposal_recurrence whose target
                # isn't actually a proposal, a reflection covering a
                # range past the end of the chain. The cryptography is
                # fine; the *meaning* of the data isn't. Cheap (a few
                # indexed scans), so safe to run inline even on long
                # chains.
                ok, warnings = chain.verify_semantic()
                if ok:
                    print(f"  semantic: ok ({chain.length()} records, "
                          f"no consistency issues)")
                else:
                    print(f"  semantic: {len(warnings)} warning(s):")
                    for w in warnings[:20]:
                        print(f"    - {w}")
                    if len(warnings) > 20:
                        print(f"    ... and {len(warnings) - 20} more")
                continue
            if user_input == "/length":
                print(f"  chain length: {chain.length()}")
                continue
            if user_input == "/seal":
                batch = chain.seal_batch()
                print(f"  sealed: {batch}")
                continue
            # cypher-tempre port commands (/verify-source, /poq, /immune-*,
            # /recall-*, /think, /consensus-*, /cambium-grow, /continuum-*).
            # One dispatcher handles them all; returns True when it consumed
            # the input. See cypher_commands.py and /cypher-help.
            import cypher_commands
            if cypher_commands.dispatch(user_input, chain, agent):
                continue
            if user_input == "/sysprompt":
                history = chain.query_by_type("system_prompt", limit=10)
                if not history:
                    print("  no system prompt records on chain yet")
                else:
                    print(f"  {len(history)} system prompt record(s) on chain:")
                    for rec in sorted(history, key=lambda r: r.index):
                        text = rec.content.get("text", "")
                        snippet = text[:80].replace("\n", " ")
                        print(f"    idx {rec.index}: {snippet}{'...' if len(text) > 80 else ''}")
                continue
            if user_input == "/reflect":
                print("  reflecting...")
                rec = agent.reflect(max_records=MAX_REFLECT_RECORDS)
                if rec is None:
                    print("  not enough history to reflect on yet")
                else:
                    index.index_record(rec)
                    turns_since_reflect = 0
                    print(f"  reflection committed at index {rec.index}:")
                    text = rec.content.get("text", "")
                    for line in text.split("\n"):
                        print(f"    {line}")
                continue
            if user_input == "/cambium":
                # Scan chain history for recurring gaps and commit any
                # resulting proposals, recurrences, and escalations.
                # Cambium proposes; it never applies.
                #
                # Uses the incremental + lookback model: this scan covers
                # everything new since the last Cambium scan, plus a
                # MAX_CAMBIUM_RECORDS window of older context so
                # detectors can spot patterns that straddle the
                # watermark. Bounded CPU; over time every record is
                # examined exactly once when fresh.
                print("  running cambium scan...")
                result = agent.run_cambium(max_records=MAX_CAMBIUM_RECORDS)
                # A manual scan resets the auto counter, so auto-Cambium
                # doesn't fire again right after the operator just ran it.
                turns_since_cambium = 0
                _report_cambium(result, index)
                continue
            if user_input == "/cambium-full":
                # Explicit one-shot deep scan of the entire chain. Linear
                # in chain length and unbounded — use sparingly, e.g.
                # after a backfill or to retroactively analyze long-range
                # patterns. Does NOT advance the incremental watermark, so
                # the periodic scan keeps its rolling coverage afterwards.
                print(
                    f"  running FULL cambium scan over {chain.length()} "
                    f"records (slow on long chains)..."
                )
                result = agent.run_cambium_full()
                _report_cambium(result, index)
                continue
            if user_input == "/proposals":
                proposals = chain.query_by_type("proposal", limit=50)
                if not proposals:
                    print("  no proposal records on chain yet")
                else:
                    # Annotate each proposal with its live recurrence count
                    # and escalation state, then surface escalated ones
                    # first — that is the whole point of escalation.
                    # Bulk helpers compute every count in one scan instead
                    # of repeating the chain walk per proposal.
                    counts = cambium.recurrence_counts(chain)
                    escalated_set = cambium.escalated_indices(chain)
                    rows = []
                    for rec in proposals:
                        n = counts.get(rec.index, 1)
                        esc = rec.index in escalated_set
                        rows.append((rec, n, esc))
                    rows.sort(key=lambda r: (not r[2], r[0].index))
                    print(f"  {len(rows)} proposal record(s) "
                          f"(escalated shown first):")
                    for rec, n, esc in rows:
                        c = rec.content
                        status = c.get("status", "open")
                        flag = " ** ESCALATED **" if esc else ""
                        rc = f" x{n}" if n > 1 else ""
                        print(f"    idx {rec.index} [{c.get('proposal_kind','?')}] "
                              f"({status}{rc}){flag} {c.get('title','')}")
                continue
            if user_input == "/modalities":
                # Show the modalities retrieval can anchor on: baked-in
                # (compiled detectors in signals.py) and sprouted (data-driven
                # entries in the runtime registry). Sprouted entries show
                # their status (active vs tentative/cooling-off), whether
                # they are domain-relevant (participate in anchoring), their
                # effective weight factor, and any patterns that were skipped
                # at load (e.g. rejected as a backtracking risk).
                import signals as _sig
                baked = [h.name for h in
                         [fn(_sig.SignalInput(content="", source="system"))
                          for fn in _sig.MODALITY_REGISTRY]]
                print(f"  baked-in modalities ({len(baked)}): {', '.join(baked)}")
                domain_baked = sorted(_retrieval.DOMAIN_MODALITIES)
                print(f"  of which domain (anchor) modalities: {', '.join(domain_baked)}")
                sm = sprout_registry.modalities
                if not sm:
                    print("  sprouted modalities: none")
                else:
                    print(f"  sprouted modalities ({len(sm)}):")
                    for m in sorted(sm, key=lambda m: m.name):
                        dom = "domain" if m.domain else "non-domain"
                        wf = m.effective_weight_factor()
                        skip = f", {len(m.skipped)} pattern(s) skipped" if m.skipped else ""
                        print(f"    {m.name} [{m.status}, {dom}, "
                              f"weight x{wf:g}, {len(m.compiled)} pattern(s){skip}]")
                print(f"  per-turn modality cap: {PER_TURN_MODALITY_CAP}")
                continue
            if user_input.startswith("/workspace"):
                # USER-only workspace switch (the same set_workspace the web
                # selector uses): no argument shows the current boundary.
                from tools import set_workspace as _set_ws
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    print(f"  workspace: {tool_ctx.workspace_root}")
                else:
                    try:
                        print(f"  workspace set: "
                              f"{_set_ws(tool_ctx, parts[1].strip())}")
                    except ValueError as e:
                        print(f"  /workspace error: {e}")
                continue

            if user_input.startswith("/task"):
                # /task list | open <name> <objective…> | ingest <name> <path>
                # [exts…] | resume <name> | validate <name> | audit <name> <idx>
                # Thin wrappers over the same tool executors the model uses —
                # one implementation, two entry points.
                from tools import execute_tool as _exec_tool
                parts = user_input.split()
                sub = parts[1] if len(parts) > 1 else "list"
                try:
                    if sub == "list":
                        print(_exec_tool({"name": "list_tasks",
                                          "arguments": {}}, tool_ctx))
                    elif sub == "open" and len(parts) >= 4:
                        # The task binds to the CURRENT workspace (which
                        # /workspace and boot-time restore move around) —
                        # not the directory the process was launched from.
                        print(_exec_tool({"name": "task_open", "arguments": {
                            "name": parts[2],
                            "objective": " ".join(parts[3:]),
                            "source_root": str(tool_ctx.workspace_root),
                        }}, tool_ctx))
                    elif sub == "ingest" and len(parts) >= 4:
                        args = {"task_name": parts[2], "path": parts[3]}
                        if len(parts) > 4:
                            args["extensions"] = parts[4:]
                        print(_exec_tool({"name": "task_ingest_path",
                                          "arguments": args}, tool_ctx))
                    elif sub == "resume" and len(parts) >= 3:
                        print(_exec_tool({"name": "task_resume", "arguments":
                                          {"task_name": parts[2]}}, tool_ctx))
                    elif sub == "validate" and len(parts) >= 3:
                        print(_exec_tool({"name": "task_validate", "arguments":
                                          {"task_name": parts[2]}}, tool_ctx))
                    elif sub == "reembed" and len(parts) >= 3:
                        # User-typed → confirmation is inherent; warn about
                        # the cost up front since a CPU embedder can take
                        # minutes to hours on a large task chain.
                        print("  re-embedding with the session embedder — "
                              "slow on CPU; progress below")
                        print(_exec_tool({"name": "task_reembed", "arguments":
                                          {"task_name": parts[2]}}, tool_ctx))
                    elif sub == "audit" and len(parts) >= 4:
                        print(_exec_tool({"name": "task_audit_source",
                                          "arguments": {"task_name": parts[2],
                                                        "block_index": int(parts[3])}},
                                         tool_ctx))
                    else:
                        print("  usage: /task list | open <name> <objective> | "
                              "ingest <name> <path> [exts...] | resume <name> | "
                              "validate <name> | reembed <name> | "
                              "audit <name> <block-index>")
                except Exception as e:
                    print(f"  /task error: {type(e).__name__}: {e}")
                continue

            if user_input.startswith("/approve") or user_input.startswith("/reject"):
                # Tier-3 write gate: ONLY this path executes or abandons a
                # pending write — the model can never trigger it.
                from tools import execute_user_action as _user_action
                parts = user_input.split()
                if len(parts) < 2:
                    print(f"  usage: {parts[0]} <pending-op-id>")
                    continue
                action = "approve_write" if parts[0] == "/approve" else "reject_write"
                print("  " + _user_action(action, {"pending_op_id": parts[1]},
                                          tool_ctx))
                continue

            if user_input == "/pending":
                ids = tool_ctx.pending_ops.list_ids()
                if not ids:
                    print("  no pending write operations")
                for op_id in ids:
                    op = tool_ctx.pending_ops.load(op_id)
                    if op:
                        print(f"  {op.id}  [{op.status}]  {op.file_path}  "
                              f"— {op.change_summary}")
                continue

            if user_input.startswith("/revise"):
                # Format: /revise <index> <correction text>
                parts = user_input.split(maxsplit=2)
                if len(parts) < 3:
                    print("  usage: /revise <record_index> <correction text>")
                    continue
                try:
                    target_idx = int(parts[1])
                except ValueError:
                    print(f"  invalid index: {parts[1]!r}")
                    continue
                try:
                    rec = agent.revise(target_idx, parts[2])
                except ProtectedZoneError as e:
                    # Genesis, system_prompt, and principle records are a
                    # protected zone — they can't be revised by a turn.
                    print(f"  refused: {e}")
                    continue
                if rec is None:
                    print(f"  no record at index {target_idx}")
                else:
                    index.index_record(rec)
                    print(f"  revision committed at index {rec.index}, corrects #{target_idx}")
                continue

            if TOOLS_ENABLED:
                # Tier-3 confirmation hook for tools in tools.CONFIRM_TOOLS:
                # the REPL asks the operator inline before the tool runs.
                def _confirm(tool_name: str, args: dict) -> bool:
                    print(f"  [tool] {tool_name} wants to run with {args}")
                    reason = getattr(tool_ctx, "last_gate_reason", None)
                    if reason:
                        print(f"  [why]  {reason}")
                    return input("  proceed? (yes/no): ").strip().lower() in (
                        "y", "yes")

                # Mid-turn approval gate (v1.4.x): a write proposal pauses
                # the turn HERE — the decision happens before the turn ends
                # and is embedded in the response record, never left
                # lingering for a post-turn /approve.
                def _approve_op(op_info: dict) -> str:
                    kind = op_info.get("kind", "write")
                    if kind == "tool_call":
                        print(f"  [gate] deferred tool call: "
                              f"{op_info.get('tool')} "
                              f"{op_info.get('arguments', {})}")
                    else:
                        print(f"  [gate] write to {op_info.get('file')}: "
                              f"{op_info.get('change', '')}")
                    yes = input("  approve? (yes/no): ").strip().lower() in (
                        "y", "yes")
                    return "approved" if yes else "rejected"

                turn = agent.turn_with_tools(
                    user_input, tool_ctx,
                    retrieve_k=SEMANTIC_K, n_recent=RECENT_N,
                    confirm_hook=_confirm,
                    approval_hook=_approve_op)
            else:
                turn = agent.turn(user_input, retrieve_k=SEMANTIC_K,
                                  n_recent=RECENT_N)
            # Single-record turn shape: no observation record is minted —
            # the response record carries the input as content.context.
            if turn.observation_record is not None:
                index.index_record(turn.observation_record)
            index.index_record(turn.response_record)
            print(f"agent: {turn.response_text}\n")
            # If the model hit its max_tokens ceiling, the answer above is
            # cut off mid-thought. Tell the operator so 'continue' is an
            # informed choice rather than a guess.
            if turn.truncated:
                print("  [note: this response was cut off at the model's")
                print("   max_tokens limit. type 'continue' to have the agent")
                print("   pick up where it left off, or raise LLM_MAX_TOKENS")
                print("   in run.py for longer responses.]\n")
            # Same idea for the TOOL budget: the text ended cleanly but
            # the task didn't — 'continue' grants a fresh round budget.
            if getattr(turn, "tool_budget_exhausted", False):
                print("  [note: the agent ran out of tool rounds before")
                print("   finishing the task. type 'continue' to give it a")
                print("   fresh budget and resume, or raise")
                print("   DEFAULT_MAX_TOOL_ROUNDS in tools.py.]\n")
            # If Proof-of-Quality flagged this turn as an attack, the
            # response was still shown above, but it was committed to
            # memory as quarantined — say so, so the operator knows.
            if turn.poq is not None and turn.poq.action == "quarantine":
                print("  [note: this turn was flagged by proof-of-quality as a")
                print("   possible injection attempt and committed to memory as")
                print("   quarantined — it will not feed future retrieval.]\n")
            turns_since_reflect += 1
            turns_since_cambium += 1

            # Auto-reflect every N turns, if enabled
            if AUTO_REFLECT_EVERY > 0 and turns_since_reflect >= AUTO_REFLECT_EVERY:
                print("  [auto-reflecting on recent history...]")
                rec = agent.reflect(max_records=MAX_REFLECT_RECORDS)
                if rec is not None:
                    index.index_record(rec)
                    print(f"  [reflection committed at index {rec.index}]\n")
                turns_since_reflect = 0

            # Auto-Cambium every N turns, if enabled. This is a separate,
            # longer cadence than auto-reflection (see AUTO_CAMBIUM_EVERY):
            # Cambium scans for patterns that recur several times, so it
            # has a naturally longer horizon than recent-window reflection.
            # The scan is LLM-free, so an auto-run that finds nothing is
            # cheap. The counter is always advanced; it resets whether or
            # not the scan produced proposals, so the cadence stays fixed.
            if AUTO_CAMBIUM_EVERY > 0 and turns_since_cambium >= AUTO_CAMBIUM_EVERY:
                print("  [auto-cambium: scanning history for recurring gaps...]")
                result = agent.run_cambium(max_records=MAX_CAMBIUM_RECORDS)
                _report_cambium(result, index, auto=True)
                turns_since_cambium = 0
    finally:
        tool_ctx.close()
        chain.close()
        index.close()
        print("chain closed.")


if __name__ == "__main__":
    run()
