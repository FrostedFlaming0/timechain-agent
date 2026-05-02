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

# Retrieval knobs
SEMANTIC_K = 5     # how many semantically similar records to retrieve
RECENT_N = 3       # how many recent records to include regardless of similarity
EMBED_DIM = 384    # matches all-MiniLM-L6-v2; change if you swap models


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
    agent = Agent(chain, retriever, llm)

    # Genesis on first run
    if chain.length() == 0:
        print("first run — committing genesis")
        genesis = agent.commit_genesis(FOUNDING_COMMITMENTS)
        index.index_record(genesis)

    print(f"chain length: {chain.length()} records")
    print(f"operator pubkey: {chain.pubkey_hex[:16]}...")
    print("ready. type your message, or 'exit' / Ctrl-D to quit.")
    print("commands: /verify  /length  /seal\n")

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

            turn = agent.turn(user_input, retrieve_k=SEMANTIC_K)
            index.index_record(turn.observation_record)
            index.index_record(turn.response_record)
            print(f"agent: {turn.response_text}\n")
    finally:
        chain.close()
        index.close()
        print("chain closed.")


if __name__ == "__main__":
    run()
