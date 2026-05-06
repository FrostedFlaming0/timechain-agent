# Timechain Agent

A persistent-memory AI agent built on a hash-chained, cryptographically signed,
append-only memory substrate. Memory survives across sessions, is tamper-evident,
and can be cryptographically verified at any time. Provider-agnostic: works with
Claude, GPT, Gemini, or local models via Ollama.

## What this is

A working Python implementation of an AI agent whose memory lives in a
signed append-only log (a "timechain") rather than in conventional chat
history or vector-database-only RAG. Every interaction is signed, hash-linked
to the previous record, and verifiable. The model is whatever you plug in;
the chain is the source of truth.

The agent reflects on its own history periodically, can revise prior records
when it realizes something was wrong, weights retrieval toward records it has
deemed important, and is grounded in time — it knows when "now" is, when each
remembered exchange happened, and when there's been a long gap between sessions.

## Architecture

- **SQLite** for storage
- **Ed25519** signatures on every record
- **SHA-256** content hashing and prior-record linking
- **Merkle batching** for periodic integrity summaries
- **OpenTimestamps anchoring to Bitcoin** (optional, for long-horizon
  third-party-verifiable integrity)
- **Hybrid retrieval** (semantic + recency + per-type salience) using
  sentence-transformer embeddings, with structural query patterns
  (ancestry walks, type filters, drift detection)
- **Sealed founding commitments** at genesis (immutable identity record)
  plus a **mutable system prompt** (active behavioral driver, also logged
  to the chain on every change for an auditable configuration history).
  Drift between configured and sealed commitments is detected at startup
  and surfaced loudly.
- **Reflection loop** — the agent periodically reviews recent history and
  writes its observations back to the chain
- **Revision records** — corrections to prior records that surface
  alongside the originals at retrieval time, without modifying history
- **Temporal awareness** — every prompt includes the current time, and
  retrieved records carry human-readable relative-time labels
- **Context budget** — retrieved records are truncated to fit a configurable
  character budget before being sent to the LLM, with lowest-salience
  records (observations, responses) dropped first to preserve high-value
  records (reflections, revisions, genesis)
- **Test suite** — pytest coverage of chain integrity, tamper detection,
  Merkle proofs, retrieval, agent workflows, drift detection, and budget
  truncation
- **File ingestion** — read documents, spreadsheets, presentations, images,
  and code files into the chain. Bytes are stored content-addressed under
  `blobs/`; extracted text plus metadata go on the chain. Multimodal LLMs
  receive image/PDF bytes natively when those records are in retrieval
  context.
- **Web UI (optional)** — webapp.py runs a local FastAPI server with a
  single-page browser frontend. Streaming responses, drag-and-drop file
  ingestion, inline image rendering, and a sidebar showing recent
  reflections and revisions. Same agent, same chain, just a different
  I/O layer.

See `ARCHITECTURE.md` for a detailed walkthrough of each component.

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python run.py
```

The first run commits a genesis record with your founding commitments,
logs your system prompt, and drops you into a REPL. Subsequent runs
reuse the same chain.

Slash commands inside the REPL:
- `/verify` — cryptographically validate the entire chain
- `/length` — current record count
- `/seal` — create a Merkle batch over recent records
- `/sysprompt` — show the system prompt history on chain
- `/reflect` — trigger a reflection over recent history
- `/revise N <text>` — append a correction record targeting record N
- `/file <path>` — ingest a file (document, image, spreadsheet, code, etc.)

To inspect the chain offline:
```bash
python view_chain.py --all
python view_chain.py --record 5
python view_chain.py --type reflection
python view_chain.py --verify
```

## Switching LLM providers

The default is Claude. To switch to OpenAI, Gemini, or a local Ollama model:

**1. Install the SDK:**

```bash
pip install openai           # for GPT
pip install google-genai     # for Gemini
pip install requests         # for Ollama
```

**2. Set the API key** (skip for Ollama — it runs locally):

```bash
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
```

**3. Edit `LLM_PROVIDER` in `run.py`:**

```python
LLM_PROVIDER = "openai"   # or "gemini", "ollama", "claude"
```

That's it. The chain, retrieval, and web UI all carry over — your existing
memory works with any provider.

For Ollama, install the [Ollama app](https://ollama.com/download), pull a
model (`ollama pull llama3.1:8b`), and make sure the local server is
running before you start `run.py`.

To override the default model for a provider, edit `build_llm()` in
`run.py`:

```python
def build_llm():
    if LLM_PROVIDER == "ollama":
        return make_ollama_client(model="qwen3:8b")
    # ...
```

Or hoist it to a config constant if you'll be switching models often.

A few practical notes:

- You can switch providers mid-chain — your existing memory carries over.
  The chain stores observations and responses, not which model produced
  them.
- Different providers will give different answers to the same prompt.
  Same memory, different reasoner.
- Streaming works on all four providers in the web UI.
- Default model names in `llm_clients.py` may go stale as providers
  release new versions. If you get a "model not found" error, look up
  the current name and pass it explicitly via `make_X_client(model=...)`.
  
## Web UI

The optional `timechain_web/webapp.py` server provides a browser-based chat
interface as an alternative to the REPL. It wraps the same agent stack —
same chain, same signing key, same configuration — and adds streaming
responses, drag-and-drop file ingestion, an image renderer for ingested
files, and a sidebar showing recent reflections and revisions.

Install the extra dependencies:

```bash
pip install fastapi uvicorn sse-starlette python-multipart
```

Run it:

```bash
python timechain_web/webapp.py
```

Then open `http://127.0.0.1:8765` in your browser.

The web UI uses the same configuration as `run.py` (it imports
`DATA_DIR`, `LLM_PROVIDER`, `SYSTEM_PROMPT`, etc. directly from that
module). To change the model, system prompt, or founding commitments,
edit `run.py` — both interfaces pick up the change.

All slash commands from the REPL work the same way:

- `/verify` `/length` `/seal` `/sysprompt`
- `/reflect`
- `/revise N <text>`
- `/file <path>`

Drag and drop a file anywhere on the page to ingest it without needing
`/file`. Images are rendered inline in chat; PDFs and other documents
appear as metadata cards. The chain record is identical to what `/file`
produces — same content-addressed blob, same provenance.

A few things worth knowing:

- The server binds to `127.0.0.1` only. The operator signing key lives
  in this process; don't expose it on a network. If you want to use the
  UI from another device, an SSH tunnel is the right answer rather than
  adding auth to the app.
- Only one browser tab is "active" at a time. Opening a second tab takes
  over the session — the first tab's next action will fail with a 409.
  This protects the chain's single-writer guarantee. Concurrent requests
  to the same chain are also serialized internally regardless of session
  state.
- Responses stream token-by-token via Server-Sent Events. All four
  providers in llm_clients.py (Claude, OpenAI, Gemini, Ollama) expose
  a .stream() method; the UI uses it automatically. Custom clients
  without a .stream() method fall back to non-streaming gracefully —
  the response just appears all at once instead of progressively.
- The web server doesn't append records that the REPL wouldn't append.
  Same record types, same retrieval, same reflection cadence
  (`AUTO_REFLECT_EVERY` from `run.py`). It's an I/O layer, not a
  different agent.

You can run `run.py` and `webapp.py` against the same chain at different
times, but not simultaneously — both want exclusive access to the SQLite
database and the signing key. Pick one interface per session.

## Tests

Run the full test suite with pytest:

```bash
pip install pytest
pytest test_timechain.py -v
```

If you can't install pytest (sandboxed environments, etc.), a standalone
runner is included:

```bash
python run_tests.py
```

Coverage includes chain signing and verification, tamper detection across
content/signature/prior-hash/deletion vectors, Merkle proof correctness,
retrieval semantics, full agent workflow integrity, genesis drift detection,
context-budget truncation, and time formatting.

## How it differs from "chatbot with memory"

Conventional persistent-memory chatbots store history in a database or vector
store and retrieve from it. This works but has three structural gaps:

1. **No tamper-evidence.** A conventional store can be modified by anyone
   with write access. This system signs every record and links them
   cryptographically; modification is detectable.
2. **No identity continuity guarantees.** Conventional system prompts can be
   silently changed. This system seals founding commitments at genesis and
   logs every system prompt change, so drift between commitment and
   configuration is auditable.
3. **No active memory.** Conventional retrieval is passive — records sit
   there until a query pulls them up. This system adds reflection (the agent
   writes its own summaries of what mattered) and revision (the agent can
   correct prior records without erasing them), with salience-weighted
   retrieval that surfaces those records preferentially.

The result is closer to a personal logbook the AI co-authors and cannot
unilaterally edit, rather than a chat history.

## Credit and lineage

The conceptual idea of using a Bitcoin-style timechain as a substrate for
AI memory and self-modeling originates with **Michael Joseph (Cyberphysics AI)**
and his Cypher Tempre architecture, released under the Cypher Tempre Open
Intelligence License. Their work motivated this project.

This implementation is an independent, simplified take on one specific piece
of that broader architecture: the hash-chained memory substrate itself, plus
the retrieval, reflection, revision, and agent-loop layers needed to make it
useful in practice. It deliberately omits components of the original (e.g. 
qualia scoring) in favor of a minimal, testable foundation. The goal here is
engineering tractability, not theoretical completeness.

The technical building blocks (Ed25519, SHA-256, Merkle trees, append-only
hash-linked logs, retrieval-augmented generation) are all standard and
well-documented in the cryptography and ML literature; nothing in this
implementation is novel cryptographically. The contribution is in
*combining* these primitives into a working personal-memory substrate
for LLM agents that anyone can run, inspect, and extend.

## License

MIT — see LICENSE.

The Cypher Tempre Open Intelligence License (which covers the original
conceptual architecture) permits free implementation and derivative works
on any computational substrate, with attribution to Michael Joseph as the
original architect. This project preserves that attribution above.

## Status

Prototype. Works end-to-end. Not production-hardened. Issues and PRs welcome.
