# Timechain Architecture Overview

A guide to the files that make up the timechain agent, what each is responsible for, and how they fit together.

**Version: 1.1.** See [What changed in v1.1](#what-changed-in-v11) at the bottom for the diff against v1 (https://github.com/frostedflaming0/timechain-agent).

## The mental model

Think of this as a four-layer system:

```
┌─────────────────────────────────────────────────────────┐
│  run.py            ENTRY POINT — REPL, configuration,   │
│                    founding commitments, system prompt  │
├─────────────────────────────────────────────────────────┤
│  agent.py          AGENT LOOP — turns observation into  │
│                    LLM call into committed response,    │
│                    reflection, revision, time awareness │
├─────────────────────────────────────────────────────────┤
│  retrieval.py      RETRIEVAL — finds relevant prior     │
│                    records: semantic + per-record       │
│                    salience + per-kind half-life recency│
├─────────────────────────────────────────────────────────┤
│  metadata.py       METADATA SCHEMA — _meta block,       │
│                    source tags, salience defaults,      │
│                    half-life table, v1-fallback reader  │
├─────────────────────────────────────────────────────────┤
│  chain.py          STORAGE — append-only signed log,    │
│                    the source of truth                  │
└─────────────────────────────────────────────────────────┘

  llm_clients.py    PLUGGABLE — LLM provider clients
                    (Claude, OpenAI, Gemini, Ollama)
                    with optional system prompt and attachment support

  file_ingest.py    INGEST — read documents/images/spreadsheets/code
                    files into chain records + content-addressed blobs

  timechain_web/    OPTIONAL UI — FastAPI server + browser frontend
    webapp.py       Same agent, same chain, alternate I/O layer
    static/index.html

  test_timechain.py    TESTS — pytest suite covering all layers
  run_tests.py         standalone runner for environments without pytest
  view_chain.py        CLI inspector for chain contents
```

Each layer only knows about the layer below it. The chain doesn't know there's an LLM. The LLM client doesn't know there's a chain. This separation is what lets you swap providers without touching memory, and swap retrieval strategies without touching either. **`metadata.py` is a pure schema/convention module — no I/O, no dependencies on the chain or the agent — read by both retrieval and agent without coupling them.**

---

## chain.py — The source of truth

**Responsibility:** persist records as a hash-linked, signed, append-only log. Verify integrity. Batch records into Merkle trees for external anchoring.

**Knows about:** SQLite, cryptography, data structures.
**Does not know about:** LLMs, retrieval, prompts, agents, *or metadata schema*. The chain treats `content` as opaque JSON; metadata conventions live above it.

**Key components:**

The `Record` dataclass is the unit of state. Every record has an index (monotonic), a prior_hash (the previous record's hash), a timestamp (millisecond Unix epoch), a type (application-defined string), arbitrary JSON content, optional refs to other records, the operator's public key, a content hash, a record hash, and an Ed25519 signature over the record hash.

The `Chain` class wraps the SQLite database. Its core methods:
- `append(type_, content, refs)` — write a new record. Signs it, links to prior, commits to disk.
- `get(index)` / `get_by_hash(hash)` / `head()` — read records.
- `iter_records(start, end)` — walk a range.
- `query_by_type(type_, limit)` / `query_recent(limit)` — basic structured queries.
- `follow_refs(hash, depth)` — walk reference graph backward (ancestry).
- `verify(expected_pubkey)` — walk the entire chain, recompute every hash, verify every signature. Returns `(ok, message)`. This is what `/verify` calls.
- `seal_batch(batch_size)` — group recent records into a Merkle tree, store the root for later anchoring (e.g. to Bitcoin via OpenTimestamps).
- `inclusion_proof(index)` — given a record, generate a Merkle proof you can hand to a third party.

Helper functions: `canonical_json` for stable serialization, `sha256` for hashing, `merkle_root` and `merkle_proof` for tree operations, `load_or_create_key` for Ed25519 key management.

**Record types in use:**

| Type | Written when | Purpose |
|------|--------------|---------|
| `genesis` | First run only, record 0 | Sealed founding commitments |
| `system_prompt` | On startup if prompt changed | Audit trail of behavioral configuration |
| `observation` | Every user input | What the user said |
| `response` | Every model response | What the agent said |
| `reflection` | `/reflect` or auto-cadence | Agent's own summary of what mattered |
| `revision` | `/revise N <text>` | Correction to a prior record (original is preserved) |
| `file` | `/file <path>` | Ingested file: metadata + extracted text on chain, raw bytes in `blobs/` |

As of v1.1, every record's `content` dict carries a `_meta` block (see `metadata.py`). The chain itself is unaware of this — `_meta` is just JSON inside `content` from the chain's perspective — but reader code uses it to distinguish source, salience, and supersession.

**When you'd touch this file:** rarely. The schema is intentionally minimal so it doesn't need to change as your application evolves. You'd touch it if you wanted to add a new query pattern at the storage level, change the cryptography (e.g. add post-quantum signatures), or alter how Merkle batches are anchored.

**Operational note:** the SQLite connection uses WAL (write-ahead logging) mode with `synchronous=NORMAL`. This means concurrent reads don't block on writes, the database is faster, and a future background process (e.g. a Merkle-batching daemon) can run without `database is locked` errors. Durability remains intact for an append-only log; in the worst-case crash, only an unfinished tail write could be lost, and chain integrity holds.

---

## metadata.py — The record metadata convention

**Responsibility:** define the schema and defaults for the per-record `_meta` block, plus the v1-record-fallback reader.

**Knows about:** record types and their semantic meaning. Salience and decay defaults.
**Does not know about:** SQLite, embeddings, LLMs, prompts, the chain, the retriever, the agent. Pure schema.

This module is the centerpiece of the v1.1 upgrade. It exists to make a single architectural distinction explicit: **records are evidence, beliefs are derived.** A record's `_meta` block tags it with the metadata needed for retrieval and the agent to treat different kinds of evidence appropriately.

**The `_meta` block:**

Every v1.1 record carries a `_meta` dict inside its `content`:

```python
{
    "_meta": {
        "schema_version": 2,
        "source":         "user" | "assistant" | "system" | "tool",
        "salience":       0.0..1.0,    # write-time importance estimate
        "confidence":     0.0..1.0,    # how sure the writer was
        "supersedes":     int | absent  # record index this one corrects
    },
    # ...rest of the content (text, filename, etc.)
}
```

`schema_version` is `1` for old records (no `_meta` present) and `2` for v1.1 records.

**Sources** (the load-bearing distinction):

- `user` — captured verbatim from user input
- `assistant` — said or inferred by the agent (responses, reflections, revisions)
- `system` — operator-set (genesis, system prompts)
- `tool` — produced by a tool (file ingestion)

The point is to never collapse "the user said X" with "I inferred X." When the prompt formatter renders records to the LLM, it surfaces source as a visible tag — the model can see at a glance whether a given claim came from the user or from its own past output, and weight it accordingly.

**Default salience by type** (written at append time, overridable per record):

| Type | Default salience |
|------|------------------|
| `reflection` | 0.85 |
| `revision` | 0.80 |
| `genesis` | 0.75 (overridden to 1.0 in `commit_genesis`) |
| `file` | 0.60 |
| `system_prompt` | 0.55 |
| `observation`, `response` | 0.40 |

Reflections and revisions are written with high salience because they represent the agent's own consolidated judgment about what mattered. Observations and responses sit at conversational baseline. These are *defaults* — a specific record can override at write time (e.g. a clearly-significant user statement could be tagged with higher salience by future code).

**Per-kind half-lives** (in days, used by retrieval recency scoring):

| Type | Half-life |
|------|-----------|
| `genesis` | effectively never (1e6 days) |
| `system_prompt` | effectively never |
| `revision` | 365 days |
| `reflection` | 180 days |
| `file` | 90 days |
| `observation`, `response` | 14 days |

The recency contribution to a hit's score is `0.5 ** (age_days / half_life_for_type)`. This replaces v1's uniform linear decay with a model that respects the kind of information: identity records don't decay, conversational records do.

**Public functions:**
- `read_meta(record)` — extract metadata from any record. v1 records (no `_meta`) get type-based defaults synthesized in memory; v2 records get their stored values, with safe fallbacks for any individually-missing fields. The result is a `RecordMeta` dataclass with an `is_default` flag indicating whether values were synthesized.
- `build_meta(rec_type, source=, salience=, confidence=, supersedes=)` — build a `_meta` dict for a new record. Fills any unspecified field with type-appropriate defaults. Used by `agent.py` and the streaming path in `webapp.py` on every chain append.
- `half_life_days(rec_type)` — per-type half-life lookup, used by retrieval.

**The non-destructive migration rule:** `read_meta` synthesizes defaults *in memory* for v1 records. It never rewrites a record on disk. This is the point of append-only — even the schema's history is preserved. A v1 chain reads cleanly through v1.1 code, and any new appends carry the v1.1 `_meta` block.

**When you'd touch this file:** to tune salience defaults, adjust half-lives, add a new source enum value, or extend the `_meta` schema with new fields. Adding fields is safe — `read_meta`'s fallback handles missing fields by giving them defaults, so v1.1 records remain readable under future schemas.

---

## retrieval.py — Finding the relevant past

**Responsibility:** given a query, return the prior records most worth feeding to the LLM.

**Knows about:** the chain (read-only), embeddings, similarity scoring, `metadata.py` for salience/half-lives.
**Does not know about:** LLMs, prompts, agents.

**Key components:**

`EmbeddingIndex` is the vector store. It maintains a parallel SQLite database mapping each chain record to a vector embedding. Methods:
- `index_record(rec)` — embed and store one record.
- `index_chain(chain)` — embed every record not yet indexed (catch-up after restart).
- `search(query_text, k)` — return top-k records by cosine similarity.

`Retriever` is the query interface. It combines vector search with structural access patterns, per-record salience, per-kind recency decay, and revision-aware demotion. Methods:
- `hybrid(query, k, type_filter, recency_weight, salience_weights)` — semantic search with the v1.1 score formula. The optional kwargs are kept for backward compatibility but reinterpreted; salience now comes from the record's `_meta` block, not a global override.
- `ancestry(record_hash, depth)` — walk reference graph backward from a record.
- `recent(n, type_filter)` — pure temporal query, no embedding.
- `build_context(query, k_semantic, n_recent)` — the workhorse. Blends semantic + recent, dedupes, auto-pulls in revisions whose targets appear in the result set, and returns chronologically ordered records. This is what the agent loop calls every turn.
- `drift_against(anchor_query, recent_query)` — compare semantic neighborhood of two queries to detect drift over time.

**The score formula** (v1.1):

```
base    = W_SEMANTIC * cosine_similarity         # default 0.55
        + W_SALIENCE * record.salience           # default 0.25, from _meta
        + W_RECENCY  * recency_for_type          # default 0.20

score   = base − SUPERSEDED_PENALTY              # default 0.30
                 if a later revision supersedes this record
                 else 0
```

Where `recency_for_type = 0.5 ** (age_days / half_life_days_for_type)` with half-lives from `metadata.py`.

The weights are exposed as class constants on `Retriever` (`W_SEMANTIC`, `W_SALIENCE`, `W_RECENCY`, `SUPERSEDED_PENALTY`) so they're tunable in one place. Each `RetrievalHit` carries a `components` dict showing the contribution of each term — useful for debugging or for the agent surfacing "why this record" downstream.

**Revision-aware retrieval:**

When a record has been corrected by a later revision, two things happen at retrieval time:

1. The original gets a `−0.30` penalty applied after the weighted sum. This usually flips the ordering: the revision now ranks above its predecessor.
2. `build_context` automatically pulls any revision whose target appears in the result set, even if the revision itself wasn't a top-k semantic match. The model always sees the original *and* its correction together.

The original is never dropped — that would erase the conflict, and the conflict itself is informative ("you used to think X; you now think Y"). It's just demoted relative to the corrective record.

**`HashingEmbedder`** is a fallback dependency-free embedder for tests and offline use (bag-of-trigrams). In real use, `run.py` replaces it with a sentence-transformer.

**When you'd touch this file:** when you want different retrieval behavior. Change `W_*` constants to rebalance the score components. Add a method for time-window queries. Replace `EmbeddingIndex` with FAISS or pgvector while keeping the `Retriever` interface stable.

---

## agent.py — The conversation loop

**Responsibility:** orchestrate a turn. Take user input, retrieve relevant context, format a prompt with temporal grounding, call the LLM, write everything back to the chain (with metadata). Periodically reflect on history. Optionally revise prior records.

**Knows about:** the chain, the retriever, the LLM (as a callable), `metadata.py` for writing `_meta` blocks, the system prompt, time formatting.
**Does not know about:** which LLM provider, embedding internals, storage details.

**Key components:**

`Agent` class with these methods:
- `commit_genesis(commitments)` — write record 0 with the agent's founding commitments. Called once, on first run. Tagged `source=system, salience=1.0`.
- `check_genesis_drift(configured_commitments)` — compare currently-configured commitments against what's sealed at record 0. Returns `None` if they match, or a structured drift report if they differ. Used by `run.py` at startup to warn when configuration edits would silently be ignored.
- `log_system_prompt()` — write the current system prompt to the chain as a `system_prompt` record, but only if it differs from the last logged prompt. Provides an audit trail of behavioral configuration over time.
- `turn(user_input, retrieve_k)` — the heart of the loop:
  1. Append the user input as an `observation` record (`source=user, confidence=1.0`).
  2. Build context via the retriever.
  3. Format a prompt combining current time, retrieved context (with relative-time labels and source/SUPERSEDED tags), any revisions targeting retrieved records, and the new input. Truncate to fit the context budget if needed.
  4. Call the LLM with the system prompt.
  5. Append the response as a `response` record (`source=assistant, confidence=0.9`), with refs pointing to the records that informed it.
  6. Return an `AgentTurn` containing all three.
- `reflect(max_records=200)` — reflect on every record since the last reflection (or since genesis, if there hasn't been one). Asks the LLM "what stands out, what patterns, what's worth revisiting" and writes the result as a `reflection` record (`source=assistant, confidence=0.7` — reflections are inferential, not factual). The window sizes itself dynamically: each reflection covers exactly the slice the previous one didn't, so there are no gaps and no overlaps. `max_records` is a safety cap for the unusual case where auto-reflection has been disabled and a long stretch of history has accumulated; if the lookback would exceed it, only the most recent `max_records` are reflected on and the resulting record is flagged `capped: True`. Reflections become retrievable memory and have high default salience. Returns `None` if there are fewer than 4 substantive records since the last reflection (i.e. nothing meaningful new to reflect on).
- `revise(target_index, correction_text)` — append a `revision` record correcting a prior record (`source=assistant, supersedes=target_index`). The original is never modified. Both the legacy `revises_index`/`revises_hash` fields (for backward compatibility with `view_chain.py` and the web UI) and the canonical `_meta.supersedes` pointer are written.
- `_truncate_to_budget(records, fixed_overhead_chars)` — when retrieved context would exceed `context_char_budget`, drop lowest-salience records first. As of v1.1, ranking uses **per-record** salience read from each record's `_meta` block (with type-based defaults for v1 records via `read_meta`). Returns kept records (chronologically ordered) and a count of dropped records, which the prompt formatter surfaces as a note to the model.

`_format_prompt` is the method that builds the actual prompt string. **This is where memory context, time, source tagging, and revisions all come together.** Behavioral instructions live in the system prompt (separate channel); this method handles memory and grounding.

The prompt every turn includes:
- The current absolute time (so the model knows when "now" is).
- Total chain length and an explanation that retrieval is selective.
- Each retrieved record rendered with type, source tag (`user` / `assistant` / `system` / `tool`), and relative time.
- A `SUPERSEDED` marker on any record that has been corrected by a later revision.
- Revisions whose targets are in context (so the model sees both original and correction).
- A header explaining the source tags and the SUPERSEDED marker so the model knows how to interpret them.
- A gap note if it's been more than an hour since the last exchange ("the user may be returning after a pause").

The `_meta` block itself is **stripped** from rendered content — it's metadata about the record, not part of what the record says. Source and SUPERSEDED status surface as visible tags on the record header line instead.

Helper functions `_humanize_delta` and `_format_absolute_time` handle time formatting.

`MockLLM` is a deterministic fake LLM used in tests and for offline experimentation. Real use plugs in a function from `llm_clients.py`.

**When you'd touch this file:** when you want to change how turns work. Change `_format_prompt` to alter how memory is presented to the model. Modify `reflect()` to change reflection prompting. Add new agent capabilities (tool use, multi-step planning, scheduled actions).

---

## llm_clients.py — Provider abstraction

**Responsibility:** provide a consistent callable interface to any LLM provider, with retries, error handling, system prompt support, and sensible defaults.

**Knows about:** Anthropic, OpenAI, Google, and Ollama SDKs. Retry logic. Authentication.
**Does not know about:** the chain, the agent, retrieval, prompts, metadata.

**Key components:**

Four builder functions, all returning a callable with shape
`(prompt, system=None, attachments=None) -> str`. Each callable also
exposes `.stream(prompt, system=None, attachments=None)` — a generator
yielding text chunks, used by the web UI for streaming responses.
- `make_claude_client(model, max_tokens, temperature, timeout_s)` — Anthropic Claude. Default: `claude-opus-4-7`.
- `make_openai_client(model, ...)` — OpenAI. Default: `gpt-5.5`.
- `make_gemini_client(model, ...)` — Google Gemini. Default: `gemini-3.1-pro`.
- `make_ollama_client(model, base_url, ...)` — local Ollama. Default: `llama3.1:8b`.

Each builder:
- Imports its SDK lazily so you only need the libraries for the provider you use.
- Checks the relevant API key at startup, fails fast if missing.
- Returns a callable that takes a prompt string (and optionally a system prompt) and returns response text.
- Routes the system prompt to the provider's correct API field (Anthropic's `system` parameter, OpenAI's system message role, Gemini's `system_instruction`, Ollama's `system` field).
- Retries on transient errors (rate limits, timeouts, server errors) with exponential backoff. Does NOT retry on client errors (bad request, invalid model).

`_retry_with_backoff` is the shared retry helper.

**When you'd touch this file:** when you want to add a new provider, change default model versions as new ones release, adjust retry behavior, or expose more provider-specific options (tool use, structured output, multimodal beyond images and PDFs). Each builder is independent — changing one doesn't affect the others.

---

## file_ingest.py — Reading user files into the chain

**Responsibility:** convert a user file on disk into (a) a content-addressed blob and (b) a normalized record payload ready to go on the chain.

**Knows about:** file extensions, format-specific extractors (pypdf, python-docx, openpyxl, python-pptx, Pillow, chardet).
**Does not know about:** the chain, retrieval, LLMs, the agent, metadata. (The `_meta` block is added by `agent.ingest_file` after this module returns its `IngestResult`.)

**Supported file types:**

| Category | Extensions |
|----------|-----------|
| Documents | .pdf, .doc, .docx, .dot, .dotx, .txt, .rtf, .md, .hwp, .hwpx, .odt |
| Spreadsheets | .xlsx, .xls, .csv, .tsv, .ods |
| Presentations | .pptx, .ppt, .odp |
| Images | .jpg, .jpeg, .png, .webp, .heic, .heif, .gif, .bmp, .tiff |
| Code | .json, .yaml, .xml, .html, .css, .js, .ts, .py, .sh, .c, .cpp, .h, .java, .rs, .go, .rb, .php, .sql, .toml, .ini, plus more |

For each file: bytes are stored under `<DATA_DIR>/blobs/<sha256>` (content-addressed, deduped). A `file` record is appended to the chain with metadata + extracted text. The chain record is signed and tamper-evident; the blob is verifiable by hash.

**Key functions:**
- `ingest_file(path, blob_dir, max_bytes)` — main entry point. Reads bytes, hashes them, writes blob, extracts text, returns an `IngestResult`.
- `verify_blob(content, blob_dir)` — confirm an on-disk blob still matches the sha256 recorded on the chain.
- `is_supported(path)` / `classify_kind(ext)` — file-type helpers.

**Image handling:** images don't get OCR or captioning here — only metadata extraction. The actual visual content is sent to multimodal LLMs at retrieval time via the `attachments` parameter on the LLM clients (see `agent._collect_attachments`). Text-only models simply ignore the attachments and rely on the metadata in the prompt.

**When you'd touch this file:** to add a new file type (add the extension to the right `*_EXTS` set, write an `_extract_*` function), to change extraction limits (`MAX_EXTRACTED_CHARS`, `DEFAULT_MAX_FILE_BYTES`), or to swap an extractor (e.g. replace `pypdf` with `pdfplumber`).

---

## timechain_web/ — Optional browser UI

**Responsibility:** provide a browser-based chat interface as an alternative to the REPL. Same agent stack, same chain, same configuration — just a different I/O layer.

**Knows about:** FastAPI, Server-Sent Events, the same agent/chain/retriever that `run.py` uses (imported directly from `run.py` to share configuration). As of v1.1, also imports `metadata.build_meta` so the streaming-path inline appends carry the same `_meta` block as `agent.turn()` does.
**Does not know about:** anything `run.py` doesn't already know about. It's strictly an interface layer.

**Key components:**

`webapp.py` is a FastAPI server that:
- Boots the same Agent / Chain / Retriever stack as `run.py`, reusing `DATA_DIR`, `SYSTEM_PROMPT`, `FOUNDING_COMMITMENTS`, etc. directly.
- Exposes endpoints for chain inspection (`/api/chain/status`, `/api/chain/recent`, `/api/chain/verify`), file upload (`/api/upload`), slash commands (`/api/command`), and chat turns (`/api/turn`, `/api/turn/stream`).
- Streams responses via Server-Sent Events when the LLM client supports it (all four built-in providers do, via `llm.stream()`).
- Serves blob bytes by sha256 (`/blobs/<sha>`) so ingested images render inline in chat. Cross-checks the chain to refuse arbitrary file reads.
- Holds a single-session lock — only one browser tab is "active" at a time, protecting the chain's single-writer guarantee. All requests that touch the chain serialize through an asyncio lock regardless of session.

`static/index.html` is a single-file frontend (vanilla JS, no build step). A typing indicator shows during the latency before tokens arrive; drag-and-drop ingestion works anywhere on the page; a sidebar surfaces recent reflections and revisions. The frontend reads `content.text`, `content.filename`, `content.kind`, `content.revises_index`, and other top-level content fields exactly as in v1 — the new `_meta` block sits next to them and is ignored by the UI. No changes needed.

**The streaming endpoint** at `/api/turn/stream` deliberately inlines its chain writes (rather than calling `agent.turn()`) so it can split the LLM call into a thread. As of v1.1 these inline writes use `metadata.build_meta()` to attach the same `_meta` block that `agent.turn()` produces — there's no path through the app that produces a v1 record. If you ever extend this endpoint with new chain writes, do the same.

**What this layer does NOT add to the chain:**

The web UI never appends record types the REPL wouldn't append. Same observations, same responses, same reflections. If you can't tell from inspecting the chain whether a session was REPL or web, that's by design.

**When you'd touch these files:** to change UI behavior (frontend), to add or modify endpoints (backend), or to expose new features. The underlying agent contract is shared with `run.py` and shouldn't be modified here — changes to how turns work belong in `agent.py`.

---

## run.py — Configuration and entry point

**Responsibility:** wire everything together, manage the persistent storage location, define identity (founding commitments) and behavior (system prompt), run the REPL.

**Knows about:** all the other modules. Where data lives. Which provider to use. The founding commitments. The system prompt. Reflection cadence.

**Key components:**

Top-of-file configuration block:
- `DATA_DIR` — where chain, embeddings, and key live.
- `LLM_PROVIDER` — which client to use ("claude", "openai", "gemini", "ollama").
- `FOUNDING_COMMITMENTS` — committed at genesis, immutable thereafter. Define the agent's values.
- `SYSTEM_PROMPT` — sent to the LLM on every turn. Defines the agent's active behavior. Mutable; each version is logged to chain.
- `SEMANTIC_K` / `RECENT_N` — retrieval knobs.
- `EMBED_DIM` — must match the embedding model.
- `AUTO_REFLECT_EVERY` — how often the agent automatically reflects (in turns). Set to 0 to disable. Each reflection automatically covers every record since the previous reflection — there's no separate window setting in v1.1; the scope sizes itself to actual activity.

Functions:
- `make_sentence_embedder(model_name)` — wraps `sentence-transformers` for real semantic embeddings.
- `build_llm()` — returns the configured LLM client based on `LLM_PROVIDER`.
- `run()` — the REPL. Loads everything, commits genesis if needed, checks for drift between configured and sealed founding commitments and warns if they differ, logs system prompt if changed, loops on input, dispatches commands or turns, auto-reflects on cadence.

Slash commands:
- `/verify` — cryptographically validate the entire chain.
- `/length` — current record count (ground truth, no LLM in the loop).
- `/seal` — create a Merkle batch over recent records.
- `/sysprompt` — show all system prompt versions logged on chain.
- `/reflect` — manually trigger a reflection.
- `/revise N <text>` — append a correction to record N.
- `/file <path>` — ingest a file into the chain (document, image, etc.).

**When you'd touch this file:** all the time. This is your knob panel. Change `LLM_PROVIDER` to swap models. Change `SYSTEM_PROMPT` to adjust personality (auto-logged to chain). Change `SEMANTIC_K` and `RECENT_N` to tune retrieval. Change `FOUNDING_COMMITMENTS` *only on first run* — they're sealed at genesis and can't be changed without starting a new chain.

---

## How a turn flows through the system

```
You type: "What did I tell you about apples?"
           │  (in either run.py REPL or the web UI — same flow below)
           ▼
       run.py REPL  -or-  webapp.py /api/turn
           │ calls agent.turn(user_input)
           ▼
       agent.py
           │ 1. chain.append("observation",
           │      {"text": user_input,
           │       "_meta": build_meta("observation", source="user", ...)})
           │       │
           │       └─► chain.py: SQLite write, signed
           │
           │ 2. retriever.build_context(user_input)
           │       │
           │       └─► retrieval.py: semantic search + per-record salience
           │           + per-kind half-life recency. Demote superseded
           │           records, auto-pull revisions for any retrieved target.
           │
           │ 3. _format_prompt(user_input, context)
           │       │
           │       └─► strip _meta from rendered content; surface source
           │           ("user"/"assistant"/"system"/"tool") and SUPERSEDED
           │           tags on each record header. Add current time.
           │           Add gap-since-last-turn note if applicable.
           │
           │ 4. llm(prompt, system=system_prompt)
           │       │
           │       └─► llm_clients.py: API call to Claude/OpenAI/etc.
           │           with system prompt routed to provider's correct field
           │
           │ 5. chain.append("response",
           │      {"text": response,
           │       "_meta": build_meta("response", source="assistant", ...)},
           │      refs=[...])
           │       │
           │       └─► chain.py: SQLite write, signed, with refs
           │           pointing to what informed the response
           ▼
       run.py prints response
           │
           │ Every AUTO_REFLECT_EVERY turns:
           ▼
       agent.reflect()
           │
           └─► reads every record since the last reflection (or since
               genesis if first), asks LLM to summarize what mattered,
               writes a 'reflection' record (source=assistant, confidence=0.7)
               with refs to everything considered
```

Every turn writes two records: one for what you said (tagged `source=user`), one for what the agent said (tagged `source=assistant`). The agent's record references the records that informed it, so you have an auditable provenance trail. Periodic reflections add a third kind of record — the agent's evolving sense of what mattered — which then influences future retrieval through salience weighting.

---

## What's where: a quick reference

| If you want to change... | Edit this file |
|--------------------------|----------------|
| The LLM provider or model | `run.py` — `LLM_PROVIDER`, or `llm_clients.py` defaults |
| The agent's personality / tone | `run.py` — `SYSTEM_PROMPT` (auto-logged on change) |
| How often the agent reflects | `run.py` — `AUTO_REFLECT_EVERY` |
| How much context the agent sees | `run.py` — `SEMANTIC_K` and `RECENT_N` |
| Maximum prompt size before truncation | `agent.py` — `Agent` constructor `context_char_budget` (default 80,000) |
| Default salience by record type | `metadata.py` — `DEFAULT_SALIENCE_BY_TYPE` |
| Per-kind decay half-lives | `metadata.py` — `DEFAULT_HALF_LIFE_DAYS_BY_TYPE` |
| Source tag enum | `metadata.py` — `SOURCE_*` constants |
| Score-formula weights (semantic / salience / recency) | `retrieval.py` — `Retriever.W_SEMANTIC`, `W_SALIENCE`, `W_RECENCY` |
| How aggressively superseded records are demoted | `retrieval.py` — `Retriever.SUPERSEDED_PENALTY` |
| How records are dropped under budget pressure | `agent.py` — `_truncate_to_budget` (now driven by per-record salience) |
| How retrieval works generally | `retrieval.py` — `Retriever` methods |
| Where data is stored | `run.py` — `DATA_DIR` |
| What the agent commits to at genesis | `run.py` — `FOUNDING_COMMITMENTS` (first run only; later edits trigger a startup warning) |
| The record schema | `chain.py` — `Record` dataclass + SCHEMA |
| The metadata schema | `metadata.py` — `_meta` block structure |
| Cryptographic primitives | `chain.py` — top of file |
| How memory is presented to the model | `agent.py` — `_format_prompt` method |
| How the agent reflects | `agent.py` — `reflect` method and reflection prompt |
| Adding a new LLM provider | `llm_clients.py` — add a new `make_X_client()` |
| Adding a new file type | `file_ingest.py` — add to `*_EXTS` set and add an `_extract_*` function |
| Web UI behavior or appearance | `timechain_web/webapp.py` (server) and `timechain_web/static/index.html` (frontend) |
| Streaming responses to the browser | `llm_clients.py` — each client's `.stream()` method |
| Adding tests | `test_timechain.py` — pytest classes by concern |

---

## Three things worth understanding deeply

**The chain is the source of truth, the LLM is the reasoner.** When the LLM says something about your past ("you mentioned apples earlier"), it's reasoning over what was retrieved. When you want certainty, ask the chain directly via `/length`, `/verify`, or `view_chain.py`. The model's view is partial by design; the chain's view is total. This separation is what makes the architecture trustworthy — the model can hallucinate, but it can't fake a signed record.

**Genesis commitments are permanent; the system prompt is mutable but auditable.** The founding commitments sealed at record 0 are part of the chain's identity and cannot be changed. The system prompt can be edited freely, but every change is logged to the chain as a `system_prompt` record. This gives you the right combination: stable identity values plus iterable behavioral configuration, with full history of how behavior has been shaped over time. You can always check whether the current configuration honors the sealed commitments.

**Memory is active, not passive.** Standard RAG retrieves records and stops there. This system also reflects (writes its own summaries of what mattered), revises (corrects prior records without erasing them), and weights retrieval by salience (the agent's own reflections surface preferentially). The result is closer to how human memory actually works — consolidating, revising, letting unimportant things fade — implemented as additional record types on the same append-only chain rather than as new mechanisms outside it.

**(v1.1 addition.) Records are evidence; beliefs are derived.** Every record carries an explicit `source` tag — user, assistant, system, or tool — so the LLM can see at a glance where a claim originated. "The user said this" and "I inferred this in a reflection two months ago" are different epistemic objects, and treating them as equivalent is the failure mode that makes long-running agents drift. Source tagging plus per-kind decay plus revision-aware retrieval together make this honest: the agent's past inferences fade faster than user statements, get demoted when corrected, and never masquerade as ground truth.

---

## What changed in v1.1

v1.1 is a "Tier 1 sharpening" of v1's architecture, not a redesign. v1 already implemented append-only memory with cryptographic verification, reflection, revision, and salience-weighted retrieval. v1.1 makes the metadata that drives those mechanisms explicit and per-record rather than hardcoded and per-type.

### New file

`metadata.py` — defines the `_meta` block schema, source enum, salience defaults, half-life table, and the `read_meta` fallback reader for v1 records.

### Changed files

| File | Change |
|------|--------|
| `agent.py` | Every `chain.append` writes a `_meta` block via `build_meta`. Truncation switched from `_RETENTION_PRIORITY` (now removed) to per-record salience. Prompt rendering strips `_meta`, surfaces `source` and `SUPERSEDED` as visible tags. `reflect()` now sizes its window dynamically (every record since the previous reflection, with a `max_records` safety cap) — no more fixed lookback. |
| `retrieval.py` | New score formula: `W_SEMANTIC*sim + W_SALIENCE*salience + W_RECENCY*recency − supersession_penalty`. Per-kind half-life recency. Per-record salience read from `_meta`. `build_context` auto-pulls revisions for superseded targets. `RetrievalHit.components` exposes the breakdown. |
| `run.py` | `REFLECT_WINDOW` constant removed (the dynamic reflection in `agent.py` no longer needs it). Both `reflect()` call sites (manual `/reflect` and auto-reflect) drop the `window=` keyword arg. |
| `timechain_web/webapp.py` | Streaming endpoint's inline chain writes now use `build_meta` so streamed turns produce v1.1 records, not v1 records. `REFLECT_WINDOW` import dropped; all three `reflect()` call sites updated to the no-arg form. |
| `test_timechain.py` | Four call sites that passed `window=10` or `window=20` to `reflect()` updated to the no-arg form. No coverage change — the tests didn't assert on window-specific behavior. |

### Unchanged files

`chain.py`, `file_ingest.py`, `llm_clients.py`, `view_chain.py`, `run_tests.py`, `timechain_web/static/index.html` — none of these needed changes. The cryptographic core, the LLM clients, the file ingestion pipeline, the inspector, and the frontend all read or write the same fields they always did. New `_meta` content sits beside them harmlessly. New reflection records carry an additional `covers_indices` field which `view_chain.py` displays correctly via its generic content renderer; old reflections (with the legacy `window_size` field) keep working too.

### Migration

Drop in the new files, leave the existing chain alone. v1 records on disk read cleanly through `read_meta`'s fallback path: missing fields get type-appropriate defaults synthesized in memory at retrieval time. The records themselves are never rewritten. New appends carry the v1.1 `_meta` block. The two coexist on the same chain indefinitely; `/verify` continues to pass.

If you want to confirm the migration: run `pytest test_timechain.py -v` (the v1 test suite passes unchanged on v1.1) and then `python view_chain.py --record N` on a freshly-written record — you'll see `_meta` inside `content`. On older records, you'll see no `_meta`, but everything still works.

