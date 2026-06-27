# Timechain Agent

A persistent-memory AI agent built on a hash-chained, cryptographically signed,
append-only memory substrate. Memory survives across sessions, is tamper-evident,
and can be cryptographically verified at any time. Provider-agnostic: works with
Claude, GPT, Gemini, or local models via Ollama.

**Version: 1.1.** See [What changed in v1.1](#what-changed-in-v11) below if
you're upgrading from v1 (https://github.com/frostedflaming0/timechain-agent).

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
- **Per-record metadata** (v1.1) — every record carries a `_meta` block
  with `source` (user / assistant / system / tool), `salience`,
  `confidence`, and `schema_version`. Source is the load-bearing
  distinction: the LLM can see "this is what the user said" vs "this is
  what I inferred" rather than treating all records as equivalent.
- **Hybrid retrieval** with explicit, named score components: semantic
  similarity, per-record salience, and per-kind half-life recency decay.
  Observations decay over weeks; reflections over months; genesis and
  system prompts effectively never decay.
- **Revision-aware retrieval** (v1.1) — when a record has been corrected
  by a later revision, retrieval demotes the original and pulls in the
  correction so the model sees both together.
- **Sealed founding commitments** at genesis (immutable identity record)
  plus a **mutable system prompt** (active behavioral driver, also logged
  to the chain on every change for an auditable configuration history).
  Drift between configured and sealed commitments is detected at startup
  and surfaced loudly.
- **Reflection loop** — the agent periodically reviews recent history and
  writes its observations back to the chain.
- **Revision records** — corrections to prior records that surface
  alongside the originals at retrieval time, without modifying history.
- **Temporal awareness** — every prompt includes the current time, and
  retrieved records carry human-readable relative-time labels.
- **Context budget** — retrieved records are truncated to fit a configurable
  character budget before being sent to the LLM, with lowest-salience
  records dropped first. As of v1.1 truncation is driven by per-record
  salience read from the `_meta` block, not a hardcoded type table.
- **Test suite** — pytest coverage of chain integrity, tamper detection,
  Merkle proofs, retrieval, agent workflows, drift detection, and budget
  truncation.
- **File ingestion** — read documents, spreadsheets, presentations, images,
  and code files into the chain. Bytes are stored content-addressed under
  `blobs/`; extracted text plus metadata go on the chain. Multimodal LLMs
  receive image/PDF bytes natively when those records are in retrieval
  context.
- **Web UI (optional)** — `webapp.py` runs a local FastAPI server with a
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

## What changed in v1.1

v1.1 sharpens the "records are evidence, beliefs are derived" principle that
v1 already implemented in spirit. The cryptographic core is unchanged — the
chain, signing, hashing, Merkle batching, and verification all work
identically. What changed is how records are written and how retrieval
ranks them.

### Summary of changes

| Area | v1 | v1.1 |
|------|----|------|
| Record metadata | None on most records; `schema_version: 1` on a few | Every record carries a `_meta` block with `schema_version`, `source`, `salience`, `confidence`, `supersedes` |
| Salience | Per-type, hardcoded in `Retriever.DEFAULT_SALIENCE` | Per-record, written at append time; type-based defaults still kick in for v1 records |
| Recency | Linear blend `1 - (head - idx) / total` | Per-kind half-life decay: `0.5 ** (age_days / half_life_days)` |
| Source tracking | Not represented | Every record tagged `user` / `assistant` / `system` / `tool` |
| Superseded records | Surfaced equally with their corrections | Demoted with a `-0.30` penalty so corrections rank above them |
| Revision pull-in | Done in prompt rendering | Done in retrieval — superseded targets always pull their corrections |
| Truncation order | Hardcoded `_RETENTION_PRIORITY` table | Per-record salience |
| Reflection window | Fixed lookback (`REFLECT_WINDOW = 20`) | Dynamic: every record since the previous reflection, with a `max_records=200` safety cap |
| Files added | — | `metadata.py` (new) |
| Files changed | — | `agent.py`, `retrieval.py`, `run.py`, `test_timechain.py`, `timechain_web/webapp.py` |
| Files unchanged | — | `chain.py`, `file_ingest.py`, `llm_clients.py`, `view_chain.py`, `run_tests.py`, `timechain_web/static/index.html` |

### Why source matters

In v1, retrieval surfaced records but didn't distinguish "the user told me X"
from "I inferred X" from "a reflection concluded X." The LLM had to figure
that out from context. In v1.1, every record carries an explicit source tag
that the agent renders in the prompt:

```
[record 7 | observation       | user      | 3 hours ago] {"text": "I work at Acme"}
[record 8 | response          | assistant | 3 hours ago] {"text": "Got it — Acme."}
[record 12 | reflection       | assistant | 2 hours ago] {"text": "User mentioned working at Acme..."}
```

This isn't cosmetic. Reflection-of-reflection drift — the doc-cited failure
mode where summaries of summaries become "telephone with yourself" — is
much harder to fall into when the model can see at a glance that a claim
came from its own past inference rather than from the user.

### Per-kind half-lives

v1 treated all records as decaying at the same rate. v1.1 acknowledges that
"user told me their name" and "user mentioned the weather" should not decay
identically. Defaults (in `metadata.py`):

| Type | Half-life |
|------|-----------|
| `genesis` | effectively never |
| `system_prompt` | effectively never |
| `revision` | 1 year |
| `reflection` | 6 months |
| `file` | 3 months |
| `observation`, `response` | 2 weeks |

Tunable in `metadata.py`. The point is that defaults respect the kind of
information rather than applying a single uniform decay.

### Revision-aware retrieval

When you correct an earlier record with `/revise`, v1 wrote the revision
and called it done — both records would surface together at retrieval, but
with no preference. v1.1 demotes the superseded original (`-0.30` to its
score, applied after the weighted sum) and `build_context` automatically
pulls in any revision whose target appears in the result set. The model
always sees the original *and* the correction, with the correction
ranking higher.

### Migration: drop in the new files

If you're upgrading an existing chain:

1. Add `metadata.py` (new file).
2. Replace `agent.py`, `retrieval.py`, `run.py`, `test_timechain.py`, and
   `timechain_web/webapp.py`.
3. Leave `chain.py`, `file_ingest.py`, `llm_clients.py`, `view_chain.py`,
   `run_tests.py`, and `timechain_web/static/index.html` alone.

If you had customized `REFLECT_WINDOW` in `run.py`, that constant no longer
exists in v1.1 — the dynamic window in `agent.py`'s `reflect()` makes it
unnecessary. If you want a tighter cap on the maximum reflection size,
the new control is the `max_records` parameter on `Agent.reflect()`
(default 200, plenty for any normal cadence). `AUTO_REFLECT_EVERY` still
works the same way. 

Your existing chain works without modification. v1 records (no `_meta`
block) get sensible defaults at read time via `metadata.read_meta()`,
which inspects record type and synthesizes appropriate values without
ever rewriting the record on disk. New records appended after the upgrade
carry the v1.1 `_meta` block. The two coexist in the same chain
indefinitely; `/verify` still passes.

This is deliberate: append-only means append-only, including for schema
migrations. Old records stay as they were.

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
  providers in `llm_clients.py` (Claude, OpenAI, Gemini, Ollama) expose
  a `.stream()` method; the UI uses it automatically. Custom clients
  without a `.stream()` method fall back to non-streaming gracefully —
  the response just appears all at once instead of progressively.
- The web server doesn't append records that the REPL wouldn't append.
  Same record types, same retrieval, same reflection cadence
  (`AUTO_REFLECT_EVERY` from `run.py`). It's an I/O layer, not a
  different agent. As of v1.1 the streaming endpoint also writes the
  same `_meta` block as `agent.turn()` — there's no path through the
  app that produces a v1 record.

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
context-budget truncation, and time formatting. The v1 test suite passes
unchanged on v1.1 — the upgrade is non-breaking by design.

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

Prototype, version 1.1. Works end-to-end. Not production-hardened. Issues
and PRs welcome.
