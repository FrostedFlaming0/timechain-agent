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

See `ARCHITECTURE.md` for a detailed walkthrough of each component.

## Quick start

```bash
pip install cryptography numpy scikit-learn anthropic sentence-transformers
# Optional, for file ingestion (per file type):
pip install pypdf python-docx openpyxl python-pptx Pillow chardet
export ANTHROPIC_API_KEY=sk-ant-...    # or use OpenAI / Gemini / Ollama
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
