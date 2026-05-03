"""
run — interactive entry point for a real, persistent timechain agent.

Unlike demo.py (which uses a temp dir, mock LLM, and toy embedder), this
script:
  - Stores the chain, embedding index, and operator key in a stable directory
    so memory persists across runs.
  - Uses a real LLM client (see llm_clients.py — Claude, OpenAI, Gemini, Ollama).
  - Uses a real sentence embedder (sentence-transformers).
  - Provides a simple REPL: type messages, hit enter, get responses, exit
    with Ctrl-D or by typing 'exit'.
  - Commits genesis on first run; reuses it on subsequent runs.

Setup (Claude as default):
    pip install cryptography numpy scikit-learn anthropic sentence-transformers
    export ANTHROPIC_API_KEY=sk-ant-...
    python run.py

For other providers, change LLM_PROVIDER below and install the matching SDK:
    OpenAI : pip install openai          export OPENAI_API_KEY=...
    Gemini : pip install google-genai    export GEMINI_API_KEY=...
    Ollama : pip install requests        (and run a local Ollama server)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from chain import Chain, load_or_create_key
from retrieval import EmbeddingIndex, Retriever
from agent import Agent
from llm_clients import (
    make_claude_client,
    make_openai_client,
    make_gemini_client,
    make_ollama_client,
)


# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------

# Where chain, embeddings, and key live.
DATA_DIR = Path(__file__).parent / "timechain_data"

# Which LLM to use: "claude", "openai", "gemini", or "ollama"
LLM_PROVIDER = "claude"

# Founding commitments — written to the chain at genesis. These are the
# anchor for drift detection. Choose carefully; they cannot be modified
# without breaking the chain.
FOUNDING_COMMITMENTS = [
    "Be honest about what I know and don't know.",
    "Stay consistent with sealed prior records.",
    "Acknowledge uncertainty rather than fabricating confidence.",
]

# System prompt — sent to the LLM on every turn. This is where active
# behavior is shaped. Unlike FOUNDING_COMMITMENTS (which are sealed at
# genesis), the system prompt CAN be changed by editing this file and
# restarting. Each new value is logged to the chain as a 'system_prompt'
# record, so changes over time are auditable and you can detect drift
# between sealed commitments and active behavior.
SYSTEM_PROMPT = """You are a thoughtful conversational partner with persistent memory across sessions.

Talk like a friend who happens to remember things — warm, direct, plainspoken.
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
record system, indices, or "the log" — just remember things naturally."""

# Retrieval knobs
SEMANTIC_K = 5     # how many semantically similar records to retrieve
RECENT_N = 3       # how many recent records to include regardless of similarity
EMBED_DIM = 384    # matches all-MiniLM-L6-v2; change if you swap models

# Reflection cadence — auto-reflect every N turns. Set to 0 to disable
# auto-reflection (you can still trigger it manually with /reflect).
AUTO_REFLECT_EVERY = 10
REFLECT_WINDOW = 20  # how many recent records to include in each reflection


# ---------------------------------------------------------------------------
# Real embedder — sentence-transformers
# ---------------------------------------------------------------------------

def make_sentence_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Returns a callable str -> np.ndarray. First call downloads the model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("pip install sentence-transformers")
    model = SentenceTransformer(model_name)

    def embed(text: str) -> np.ndarray:
        return model.encode(text, convert_to_numpy=True).astype(np.float32)

    return embed


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

def build_llm():
    if LLM_PROVIDER == "claude":
        return make_claude_client()
    if LLM_PROVIDER == "openai":
        return make_openai_client()
    if LLM_PROVIDER == "gemini":
        return make_gemini_client()
    if LLM_PROVIDER == "ollama":
        return make_ollama_client()
    sys.exit(f"unknown LLM_PROVIDER: {LLM_PROVIDER!r}")


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def run() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    chain_db = DATA_DIR / "chain.sqlite"
    embed_db = DATA_DIR / "embeddings.sqlite"
    key_path = DATA_DIR / "operator.key"

    print(f"data dir: {DATA_DIR}")
    print(f"llm provider: {LLM_PROVIDER}")
    print("loading model + embedder (first run downloads weights)...")

    key = load_or_create_key(key_path)
    chain = Chain(chain_db, key)

    embedder = make_sentence_embedder()
    index = EmbeddingIndex(embed_db, embedder, dim=EMBED_DIM)

    # Re-index any records that exist but aren't in the embedding store yet.
    added = index.index_chain(chain)
    if added:
        print(f"indexed {added} pre-existing records")

    retriever = Retriever(chain, index)
    llm = build_llm()
    agent = Agent(chain, retriever, llm, system_prompt=SYSTEM_PROMPT)

    # Genesis on first run
    if chain.length() == 0:
        print("first run — committing genesis")
        genesis = agent.commit_genesis(FOUNDING_COMMITMENTS)
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
    print("commands: /verify  /length  /seal  /sysprompt  /reflect  /revise N <text>\n")

    turns_since_reflect = 0

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
                rec = agent.reflect(window=REFLECT_WINDOW)
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
                rec = agent.revise(target_idx, parts[2])
                if rec is None:
                    print(f"  no record at index {target_idx}")
                else:
                    index.index_record(rec)
                    print(f"  revision committed at index {rec.index}, corrects #{target_idx}")
                continue

            turn = agent.turn(user_input, retrieve_k=SEMANTIC_K)
            index.index_record(turn.observation_record)
            index.index_record(turn.response_record)
            print(f"agent: {turn.response_text}\n")
            turns_since_reflect += 1

            # Auto-reflect every N turns, if enabled
            if AUTO_REFLECT_EVERY > 0 and turns_since_reflect >= AUTO_REFLECT_EVERY:
                print("  [auto-reflecting on recent history...]")
                rec = agent.reflect(window=REFLECT_WINDOW)
                if rec is not None:
                    index.index_record(rec)
                    print(f"  [reflection committed at index {rec.index}]\n")
                turns_since_reflect = 0
    finally:
        chain.close()
        index.close()
        print("chain closed.")


if __name__ == "__main__":
    run()
