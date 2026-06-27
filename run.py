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

import sys
from pathlib import Path

import numpy as np

from chain import Chain, load_or_create_key
from retrieval import (
    EmbeddingIndex,
    Retriever,
    HashingEmbedder,
    OllamaEmbedder,
    ollama_is_reachable,
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
# "gemini", or "ollama"
LLM_PROVIDER = "claude"

# Maximum length of a single model response, in tokens (~0.75 words per
# token, so 4096 ≈ 3000 words). This is a ceiling, not a target: a short
# answer still generates few tokens, and you are billed only for tokens
# actually produced — so raising this costs nothing on short replies. If a
# response does hit the ceiling it is cut off mid-thought, and the REPL /
# web UI show a "response was cut off" marker. 1024 is conservative;
# 4096 is a comfortable default for a chat agent.
LLM_MAX_TOKENS = 4096

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
# turns. The two defaults (4096 tokens / 150,000 chars) are matched to
# modern model contexts — a worst-case response uses well under a fifth of
# the budget, leaving room for the rest of memory. If you raise
# LLM_MAX_TOKENS substantially, consider raising this too so one big
# record can't crowd everything else out.
#
# 2026-05: raised from 80,000 to 150,000. Combined with chunk-aware
# rendering for long file records (see Agent._format_prompt's file branch
# and Retriever.search's chunk-match plumbing), this means a single long
# document almost always fits whole at 150k, and the chunk path only
# activates when multiple long files or other big records compete.
CONTEXT_BUDGET_CHARS = 150000

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

Records may also carry an `epistemic: ...` tag when the nature of the
claim differs from what you'd expect for its kind. A response tagged
`epistemic: factual` is one stating a measured fact rather than the usual
inference; `epistemic: speculative` is one flagging itself as a guess.
Weight these accordingly — a speculative claim about something doesn't
carry the same evidential weight as a factual one, even if they sound
equally confident."""

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
AUTO_REFLECT_EVERY = 10

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

    try:
        index = EmbeddingIndex(embed_db, embedder, dim=embed_dim)
    except ValueError as e:
        # The embedding store was built with a different embedder (different
        # dimension). The chain is fine — only the derived embedding index is
        # stale. Tell the user how to rebuild rather than dying obscurely.
        print()
        print("=" * 70)
        print("EMBEDDING STORE MISMATCH")
        print(str(e))
        print("=" * 70)
        chain.close()
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
    blob_dir = DATA_DIR / "blobs"
    blob_dir.mkdir(parents=True, exist_ok=True)
    agent = Agent(
        chain, retriever, llm,
        system_prompt=SYSTEM_PROMPT,
        blob_dir=blob_dir,
        context_char_budget=CONTEXT_BUDGET_CHARS,
    )

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
    print("          /proposals  /revise N <text>  /file <path>\n")

    turns_since_reflect = 0
    turns_since_cambium = 0

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
                rec = agent.reflect()
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
            if user_input.startswith("/file"):
                # Format: /file <path-to-file>
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("  usage: /file <path-to-file>")
                    continue
                file_path = parts[1].strip().strip('"').strip("'")
                try:
                    rec = agent.ingest_file(file_path)
                except FileNotFoundError:
                    print(f"  no such file: {file_path}")
                    continue
                except ValueError as e:
                    print(f"  {e}")
                    continue
                except OSError as e:
                    print(f"  {e}")
                    continue
                except Exception as e:
                    print(f"  ingestion failed: {type(e).__name__}: {e}")
                    continue
                index.index_record(rec)
                c = rec.content
                truncated = " (text truncated)" if c.get("extraction_truncated") else ""
                print(
                    f"  ingested {c['filename']} as record {rec.index} "
                    f"({c['kind']}, {c['size_bytes']:,} bytes, "
                    f"sha256 {c['blob_sha256'][:12]}...){truncated}"
                )
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

            turn = agent.turn(user_input, retrieve_k=SEMANTIC_K, n_recent=RECENT_N)
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
                rec = agent.reflect()
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
        chain.close()
        index.close()
        print("chain closed.")


if __name__ == "__main__":
    run()
