# Timechain Architecture Overview

A guide to the files that make up the timechain agent, what each is responsible for, and how they fit together.

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
│                    records using semantic search +      │
│                    recency + per-type salience weights  │
├─────────────────────────────────────────────────────────┤
│  chain.py          STORAGE — append-only signed log,    │
│                    the source of truth                  │
└─────────────────────────────────────────────────────────┘

  llm_clients.py    PLUGGABLE — LLM provider clients
                    (Claude, OpenAI, Gemini, Ollama)
                    with optional system prompt and attachment support

  file_ingest.py    INGEST — read documents/images/spreadsheets/code
                    files into chain records + content-addressed blobs

  test_timechain.py    TESTS — pytest suite covering all layers
  run_tests.py         standalone runner for environments without pytest
  view_chain.py        CLI inspector for chain contents
```

Each layer only knows about the layer below it. The chain doesn't know there's an LLM. The LLM client doesn't know there's a chain. This separation is what lets you swap providers without touching memory, and swap retrieval strategies without touching either.

---

## chain.py — The source of truth

**Responsibility:** persist records as a hash-linked, signed, append-only log. Verify integrity. Batch records into Merkle trees for external anchoring.

**Knows about:** SQLite, cryptography, data structures.
**Does not know about:** LLMs, retrieval, prompts, agents.

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

**When you'd touch this file:** rarely. The schema is intentionally minimal so it doesn't need to change as your application evolves. You'd touch it if you wanted to add a new query pattern at the storage level, change the cryptography (e.g. add post-quantum signatures), or alter how Merkle batches are anchored.

**Operational note:** the SQLite connection uses WAL (write-ahead logging) mode with `synchronous=NORMAL`. This means concurrent reads don't block on writes, the database is faster, and a future background process (e.g. a Merkle-batching daemon) can run without `database is locked` errors. Durability remains intact for an append-only log; in the worst-case crash, only an unfinished tail write could be lost, and chain integrity holds.

---

## retrieval.py — Finding the relevant past

**Responsibility:** given a query, return the prior records most worth feeding to the LLM.

**Knows about:** the chain (read-only), embeddings, similarity scoring.
**Does not know about:** LLMs, prompts, agents.

**Key components:**

`EmbeddingIndex` is the vector store. It maintains a parallel SQLite database mapping each chain record to a vector embedding. Methods:
- `index_record(rec)` — embed and store one record.
- `index_chain(chain)` — embed every record not yet indexed (catch-up after restart).
- `search(query_text, k)` — return top-k records by cosine similarity.

`Retriever` is the query interface. It combines vector search with structural access patterns and per-type salience weighting. Methods:
- `hybrid(query, k, type_filter, recency_weight, salience_weights)` — semantic search with optional filters and a recency boost, plus salience boosts that surface reflections, revisions, and identity records preferentially.
- `ancestry(record_hash, depth)` — walk reference graph backward from a record.
- `recent(n, type_filter)` — pure temporal query, no embedding.
- `build_context(query, k_semantic, n_recent)` — the workhorse. Blends semantic + recent, dedupes, returns chronologically ordered. This is what the agent loop calls every turn.
- `drift_against(anchor_query, recent_query)` — compare semantic neighborhood of two queries to detect drift over time.

**Salience weights** (defined as `Retriever.DEFAULT_SALIENCE`):

| Record type | Boost | Reason |
|-------------|-------|--------|
| `reflection` | +0.20 | Represents the agent's own judgment about what mattered |
| `revision` | +0.15 | Corrects something — important to surface |
| `file` | +0.12 | Ingested files — meaningful but not dominant over reflections |
| `genesis` | +0.10 | Foundational identity record |
| `system_prompt` | +0.05 | Behavioral configuration context |
| `observation`, `response` | 0.0 | Baseline conversational records |

These boosts are added to cosine similarity at retrieval time, so reflections and revisions surface even when they're not the closest semantic match.

`HashingEmbedder` is a fallback dependency-free embedder for the demo (bag-of-trigrams). In real use, `run.py` replaces it with a sentence-transformer.

**When you'd touch this file:** when you want different retrieval behavior. Want bigger context windows? Change how `build_context` weights its inputs. Want time-filtered queries (only records from this week)? Add a method. Want to integrate a real ANN library like FAISS or Qdrant? Replace `EmbeddingIndex` while keeping the `Retriever` interface stable.

---

## agent.py — The conversation loop

**Responsibility:** orchestrate a turn. Take user input, retrieve relevant context, format a prompt with temporal grounding, call the LLM, write everything back to the chain. Periodically reflect on history. Optionally revise prior records.

**Knows about:** the chain, the retriever, the LLM (as a callable), the system prompt, time formatting.
**Does not know about:** which LLM provider, embedding internals, storage details.

**Key components:**

`Agent` class with these methods:
- `commit_genesis(commitments)` — write record 0 with the agent's founding commitments. Called once, on first run.
- `check_genesis_drift(configured_commitments)` — compare currently-configured commitments against what's sealed at record 0. Returns `None` if they match, or a structured drift report if they differ. Used by `run.py` at startup to warn when configuration edits would silently be ignored.
- `log_system_prompt()` — write the current system prompt to the chain as a `system_prompt` record, but only if it differs from the last logged prompt. Provides an audit trail of behavioral configuration over time.
- `turn(user_input, retrieve_k)` — the heart of the loop:
  1. Append the user input as an `observation` record.
  2. Build context via the retriever.
  3. Format a prompt combining current time, retrieved context (with relative-time labels), any revisions targeting retrieved records, and the new input. Truncate to fit the context budget if needed.
  4. Call the LLM with the system prompt.
  5. Append the response as a `response` record, with refs pointing to the records that informed it.
  6. Return an `AgentTurn` containing all three.
- `reflect(window)` — read the last N records, ask the LLM to reflect on them ("what stands out, what patterns, what's worth revisiting"), and write the reflection as a `reflection` record. Reflections become retrievable memory and get a salience boost.
- `revise(target_index, correction_text)` — append a `revision` record correcting a prior record. The original is never modified. The revision references the original by hash, and the prompt formatter automatically surfaces both together when either is retrieved.
- `_truncate_to_budget(records, fixed_overhead_chars)` — when retrieved context would exceed `context_char_budget`, drop lowest-priority records first. Priority order matches retrieval salience: reflection > revision > genesis > system_prompt > response > observation. Returns kept records (chronologically ordered) and a count of dropped records, which the prompt formatter surfaces as a note to the model.

`_format_prompt` is the method that builds the actual prompt string. **This is where memory context, time, and revisions all come together.** Behavioral instructions live in the system prompt (separate channel); this method handles memory and grounding.

The prompt every turn includes:
- The current absolute time (so the model knows when "now" is).
- Total chain length and explanation that retrieval is selective.
- Any revisions that target records currently in context (so the model sees both original and correction).
- Each retrieved record with its relative time ("3 hours ago", "2 days ago").
- A gap note if it's been more than an hour since the last exchange ("the user may be returning after a pause").

Helper functions `_humanize_delta` and `_format_absolute_time` handle time formatting.

`MockLLM` is a deterministic fake LLM for the demo. Real use plugs in a function from `llm_clients.py`.

**When you'd touch this file:** when you want to change how turns work. Change `_format_prompt` to alter how memory is presented to the model. Modify `reflect()` to change reflection prompting. Add new agent capabilities (tool use, multi-step planning, scheduled actions).

---

## llm_clients.py — Provider abstraction

**Responsibility:** provide a consistent callable interface to any LLM provider, with retries, error handling, system prompt support, and sensible defaults.

**Knows about:** Anthropic, OpenAI, Google, and Ollama SDKs. Retry logic. Authentication.
**Does not know about:** the chain, the agent, retrieval, prompts.

**Key components:**

Four builder functions, all returning the same callable shape `(prompt, system=None) -> str`:
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

**When you'd touch this file:** when you want to add a new provider, change default model versions as new ones release, adjust retry behavior, or expose more provider-specific options (streaming, tool use, structured output). Each builder is independent — changing one doesn't affect the others.

---

## file_ingest.py — Reading user files into the chain

**Responsibility:** convert a user file on disk into (a) a content-addressed blob and (b) a normalized record payload ready to go on the chain.

**Knows about:** file extensions, format-specific extractors (pypdf, python-docx, openpyxl, python-pptx, Pillow, chardet).
**Does not know about:** the chain, retrieval, LLMs, the agent.

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
- `AUTO_REFLECT_EVERY` — how often the agent automatically reflects (in turns). Set to 0 to disable.
- `REFLECT_WINDOW` — how many recent records each reflection considers.

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

**When you'd touch this file:** all the time. This is your knob panel. Change `LLM_PROVIDER` to swap models. Change `SYSTEM_PROMPT` to adjust personality (auto-logged to chain). Change `SEMANTIC_K` and `RECENT_N` to tune retrieval. Change `FOUNDING_COMMITMENTS` *only on first run* — they're sealed at genesis and can't be changed without starting a new chain.

---

## How a turn flows through the system

```
You type: "What did I tell you about apples?"
           │
           ▼
       run.py REPL
           │ calls agent.turn(user_input)
           ▼
       agent.py
           │ 1. chain.append("observation", {"text": user_input})
           │       │
           │       └─► chain.py: SQLite write, signed
           │
           │ 2. retriever.build_context(user_input)
           │       │
           │       └─► retrieval.py: embed query, find similar records
           │           with salience boost (reflections + revisions surface
           │           preferentially), blend with recent records
           │
           │ 3. _format_prompt(user_input, context)
           │       │
           │       └─► adds current time, finds revisions targeting
           │           retrieved records, adds relative-time labels,
           │           checks for long-gap-since-last-turn
           │
           │ 4. llm(prompt, system=system_prompt)
           │       │
           │       └─► llm_clients.py: API call to Claude/OpenAI/etc.
           │           with system prompt routed to provider's correct field
           │
           │ 5. chain.append("response", {"text": response}, refs=[...])
           │       │
           │       └─► chain.py: SQLite write, signed, with refs
           │           pointing to what informed the response
           ▼
       run.py prints response
           │
           │ Every AUTO_REFLECT_EVERY turns:
           ▼
       agent.reflect(window=REFLECT_WINDOW)
           │
           └─► reads recent history, asks LLM to summarize what mattered,
               writes a 'reflection' record with refs to everything considered
```

Every turn writes two records: one for what you said, one for what the agent said. The agent's record references the records that informed it, so you have an auditable provenance trail. Periodic reflections add a third kind of record — the agent's evolving sense of what mattered — which then influences future retrieval through salience weighting.

---

## What's where: a quick reference

| If you want to change... | Edit this file |
|--------------------------|----------------|
| The LLM provider or model | `run.py` — `LLM_PROVIDER`, or `llm_clients.py` defaults |
| The agent's personality / tone | `run.py` — `SYSTEM_PROMPT` (auto-logged on change) |
| How often the agent reflects | `run.py` — `AUTO_REFLECT_EVERY` |
| How much context the agent sees | `run.py` — `SEMANTIC_K` and `RECENT_N` |
| Maximum prompt size before truncation | `agent.py` — `Agent` constructor `context_char_budget` (default 80,000) |
| How retrieval weights record types | `retrieval.py` — `Retriever.DEFAULT_SALIENCE` |
| How records are dropped under budget pressure | `agent.py` — `Agent._RETENTION_PRIORITY` |
| How retrieval works generally | `retrieval.py` — `Retriever` methods |
| Where data is stored | `run.py` — `DATA_DIR` |
| What the agent commits to at genesis | `run.py` — `FOUNDING_COMMITMENTS` (first run only; later edits trigger a startup warning) |
| The record schema | `chain.py` — `Record` dataclass + SCHEMA |
| Cryptographic primitives | `chain.py` — top of file |
| How memory is presented to the model | `agent.py` — `_format_prompt` method |
| How the agent reflects | `agent.py` — `reflect` method and reflection prompt |
| Adding a new LLM provider | `llm_clients.py` — add a new `make_X_client()` |
| Adding a new file type | `file_ingest.py` — add to `*_EXTS` set and add an `_extract_*` function |
| Adding tests | `test_timechain.py` — pytest classes by concern |

---

## Three things worth understanding deeply

**The chain is the source of truth, the LLM is the reasoner.** When the LLM says something about your past ("you mentioned apples earlier"), it's reasoning over what was retrieved. When you want certainty, ask the chain directly via `/length`, `/verify`, or `view_chain.py`. The model's view is partial by design; the chain's view is total. This separation is what makes the architecture trustworthy — the model can hallucinate, but it can't fake a signed record.

**Genesis commitments are permanent; the system prompt is mutable but auditable.** The founding commitments sealed at record 0 are part of the chain's identity and cannot be changed. The system prompt can be edited freely, but every change is logged to the chain as a `system_prompt` record. This gives you the right combination: stable identity values plus iterable behavioral configuration, with full history of how behavior has been shaped over time. You can always check whether the current configuration honors the sealed commitments.

**Memory is active, not passive.** Standard RAG retrieves records and stops there. This system also reflects (writes its own summaries of what mattered), revises (corrects prior records without erasing them), and weights retrieval by salience (the agent's own reflections surface preferentially). The result is closer to how human memory actually works — consolidating, revising, letting unimportant things fade — implemented as additional record types on the same append-only chain rather than as new mechanisms outside it.
