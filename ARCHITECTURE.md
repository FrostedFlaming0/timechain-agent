# Timechain Architecture Overview

A guide to the files that make up the timechain agent, what each is responsible for, and how they fit together.

**Version: 1.4.0.** This document describes the system as it currently
stands. For the release-by-release history of how it got here, see
[CHANGELOG.md](CHANGELOG.md).

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
                    (Claude, OpenAI, OpenRouter, DeepSeek, Gemini, Ollama)
                    with optional system prompt and attachment support

  tools.py          TOOLS — text-tool driver (extract/validate/execute/
                    escape) + executors; task_registry.py TASKS — per-task
                    chain registry; pending_ops.py WRITE GATE — durable
                    user-approved writes; continuum.py INGEST — code/content
                    enters as data-height blocks on task chains

  signals.py        ANALYSIS — modalities & senses: pure text detectors
                    (intent, coherence, integrity/injection, ...) → SignalReport
  sprouted_modalities.py  RUNTIME MODALITIES — data-driven (regex-spec)
                    modalities the agent can sprout without a code change;
                    validated + ReDoS-screened, fed into signals.py
  poq.py            QUALITY GATE — Proof-of-Quality: scores a candidate
                    response before commit; reads signals.py
  protected_zones.py  MEMBRANE — protected-zone policy: what may be
                    revised, what is quarantined; reads metadata.py
  cambium.py        GROWTH — scans history for recurring gaps, emits
                    skill / modality / sense / principle proposals;
                    tracks recurrence and escalates persistent ones;
                    also detects recurring OUTPUT modes and emits
                    auto-sprout specs (diversity-gated)
  apply_proposal.py REVIEW TOOL — operator-run: lists/shows proposals,
                    scaffolds detector stubs into signals.py, records
                    accept/decline decisions on the chain

  capsule.py        EXCHANGE — Experience Capsules: signed, exposure-gated
                    export of selected Rings + verify + attributed import
                    (build spec .cphyx). Uses only the chain's existing
                    crypto; no network, no tokens, no consensus. Redacted
                    (summary-only) records carry a signed summary commitment.
                    Format spec: CAPSULE.md.

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
| `file` | (legacy, pre-v1.4) | Ingested file: metadata + extracted text on chain. Still read and verified; new code enters via `continuum` blocks on task chains |
| `tool_use` | (legacy, pre-v1.4) | Per-call sanitized audit — no longer written; the identity chain carries one observation + one response per turn, with the response narrating the tool work. Old records still read fine |
| `resolution` | user approves/rejects a pending op | The USER's decision joins the stream: outcome, op id, kind, file/tool, bounded result — so the model's memory never holds a stale "pending" forever |
| `continuum` | `/task ingest`, write approvals | One data-height source chunk with source coordinates, hashes, and rolling task state |
| `attachment` | upload / paste (`ingest_blob`) | Tiny pointer ring: filename, mime, sha, refs into the artifacts chain — never the content itself, so big documents can't crowd identity retrieval |

Every record's `content` dict carries a `_meta` block (see `metadata.py`). The chain itself is unaware of this — `_meta` is just JSON inside `content` from the chain's perspective — but reader code uses it to distinguish source, salience, and supersession.

**When you'd touch this file:** rarely. The schema is intentionally minimal so it doesn't need to change as your application evolves. You'd touch it if you wanted to add a new query pattern at the storage level, change the cryptography (e.g. add post-quantum signatures), or alter how Merkle batches are anchored.

**Operational note:** the SQLite connection uses WAL (write-ahead logging) mode with `synchronous=NORMAL`. This means concurrent reads don't block on writes, the database is faster, and a future background process (e.g. a Merkle-batching daemon) can run without `database is locked` errors. Durability remains intact for an append-only log; in the worst-case crash, only an unfinished tail write could be lost, and chain integrity holds.

---

## metadata.py — The record metadata convention

**Responsibility:** define the schema and defaults for the per-record `_meta` block, plus the v1-record-fallback reader.

**Knows about:** record types and their semantic meaning. Salience and decay defaults.
**Does not know about:** SQLite, embeddings, LLMs, prompts, the chain, the retriever, the agent. Pure schema.

This module exists to make a single architectural distinction explicit: **records are evidence, beliefs are derived.** A record's `_meta` block tags it with the metadata needed for retrieval and the agent to treat different kinds of evidence appropriately.

**The `_meta` block:**

Every current-schema record carries a `_meta` dict inside its `content`:

```python
{
    "_meta": {
        "schema_version":       3,
        "source":               "user" | "assistant" | "system" | "tool",
        "salience":             0.0..1.0,   # write-time importance estimate
        "confidence":           0.0..1.0,   # how sure the writer was
        "supersedes":           int,        # record index this corrects (or absent)
        "epistemic_class":      str,        # how the content is known
        "exposure":             str,        # who may see it (protected-zone primitive)
        "poq":                  dict,       # Proof-of-Quality block (or absent)
        "truncated":            True,       # response cut off at max_tokens (or absent)
        "modalities_activated": [str, ...], # modalities that produced this (or absent)
        "senses_activated":     [str, ...]   # how this turn felt (or absent)
    },
    # ...rest of the content (text, filename, etc.)
}
```

`schema_version` is `1` for legacy records (no `_meta` present), `2` for
records written before the v3 fields existed, and `3` for current records.
`read_meta` upgrades older records in memory with sensible defaults; it never
rewrites them on disk. Several fields are emitted only when they carry
information — `supersedes`, `poq`, `truncated`, `modalities_activated`, and
`senses_activated` are absent rather than written as a null/empty value, so
a record that doesn't need them produces the same canonical JSON earlier
versions would have, and its content hash is unchanged on rebuild.

**`modalities_activated`** records which modality detectors (`signals.py`)
fired with non-trivial activation when the record's content was scored by
PoQ — in effect, *which capabilities produced this record*. It is written on
response records (the agent scores every response), defaults to `[]` on read
for records that lack it, and is stored sorted so the same set hashes
identically. The names come straight from `MODALITY_REGISTRY`, making them a
de facto stable identifier — renaming a modality later doesn't rewrite old
records, which keep the old name. Consumed by retrieval (modality anchoring,
content-aware salience).

**`senses_activated`** parallels `modalities_activated` but records which
*sense* detectors fired — the felt-quality channel rather than the
kind-of-work channel. Where modalities answer "what produced this," senses
answer "how did this feel" (`uncertainty`, `insight_markers`,
`cognitive_weather`, `emotional_contour`). Same storage discipline (sorted,
emitted only when non-empty, defaults to `[]` on read). Deliberately **not**
a retrieval input: matching feeling-to-feeling would surface memories by
mood, closer to rumination than recall. `injection_scan` lives in
`SENSE_REGISTRY` but is filtered from this field — it's a security detector,
not a felt quality.

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

Reflections and revisions are written with high salience because they represent the agent's own consolidated judgment about what mattered. Observations and responses sit at conversational baseline. These are *defaults* — a specific record can override at write time, and one does: a **response** record's salience is set by `protected_zones.salience_for_commit` from its PoQ result rather than taking the flat 0.40. A response PoQ judged low-quality (`light_log`) is demoted below baseline; a response that is substantive artifact (high `artifact_content` activation — code, structured data) is boosted up to `ARTIFACT_SALIENCE_MAX` (0.70). Demotion wins over boost. This is why the agent can retrieve code it produced several turns ago instead of letting it decay at conversational baseline. The boost still sits below reflections and revisions, so the agent's consolidated judgments outrank even a substantive response.

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
- `read_meta(record)` — extract metadata from any record. Legacy records (no `_meta`) get type-based defaults synthesized in memory; current records get their stored values, with safe fallbacks for any individually-missing fields. The result is a `RecordMeta` dataclass with an `is_default` flag indicating whether values were synthesized.
- `build_meta(rec_type, source=, salience=, confidence=, supersedes=, epistemic_class=, exposure=, poq=, truncated=, modalities_activated=)` — build a `_meta` dict for a new record. Fills any unspecified field with type-appropriate defaults, and omits the optional fields (`supersedes`, `poq`, `truncated`, `modalities_activated`) when they carry no information. Used by `agent.py` and the streaming path in `webapp.py` on every chain append.
- `half_life_days(rec_type)` — per-type half-life lookup, used by retrieval.

**The non-destructive migration rule:** `read_meta` synthesizes defaults *in memory* for legacy records. It never rewrites a record on disk. This is the point of append-only — even the schema's history is preserved. A chain written before the `_meta` block existed reads cleanly, and any new appends carry the current `_meta` block.

**When you'd touch this file:** to tune salience defaults, adjust half-lives, add a new source enum value, or extend the `_meta` schema with new fields. Adding fields is safe — `read_meta`'s fallback handles missing fields by giving them defaults, so current records remain readable under future schemas.

---

## retrieval.py — Finding the relevant past

**Responsibility:** given a query, return the prior records most worth feeding to the LLM.

**Knows about:** the chain (read-only), embeddings, similarity scoring, `metadata.py` for salience/half-lives.
**Does not know about:** LLMs, prompts, agents.

**Key components:**

`EmbeddingIndex` is the vector store. It maintains a parallel SQLite database mapping each chain record to **one or more chunk vectors** (a long record is split into chunks before embedding — see "Chunked embedding store" below). Methods:
- `index_record(rec)` — chunk a record's text and embed/store one vector per chunk. Idempotent: re-indexing a record replaces its chunks rather than duplicating them.
- `index_chain(chain)` — embed every record not yet indexed (catch-up after restart).
- `search(query_text, k)` — return top-k **records** by cosine similarity, collapsing each record's chunk hits to a single per-record score (max over its chunks). The return contract is one `(record_idx, similarity)` pair per record, exactly as before chunking — the chunking is invisible to callers.

`Retriever` is the query interface. It combines vector search with structural access patterns, per-record salience, per-kind recency decay, and revision-aware demotion. Methods:
- `hybrid(query, k, type_filter, recency_weight, salience_weights, query_modalities)` — semantic search with the score formula below. The optional kwargs are kept for backward compatibility but reinterpreted; salience now comes from the record's `_meta` block, not a global override. `query_modalities` opts the call into modality anchoring (below).
- `ancestry(record_hash, depth)` — walk reference graph backward from a record.
- `recent(n, type_filter)` — pure temporal query, no embedding.
- `build_context(query, k_semantic, n_recent, anchor_modalities)` — the workhorse. Blends semantic + recent, dedupes, auto-pulls in revisions whose targets appear in the result set, and returns chronologically ordered records. This is what the agent loop calls every turn. By default it detects the query's domain modalities and anchors on them; pass `anchor_modalities=False` to disable.
- `query_modalities(query)` — analyze a query for the domain modalities it implies (the `DOMAIN_MODALITIES` whitelist that fire above the analyzer's floor). Empty set when the query implies no domain mode.
- `drift_against(anchor_query, recent_query)` — compare semantic neighborhood of two queries to detect drift over time.

**The score formula:**

```
base    = W_SEMANTIC * cosine_similarity         # default 0.55
        + W_SALIENCE * record.salience           # default 0.25, from _meta
        + W_RECENCY  * recency_for_type          # default 0.20

score   = base − SUPERSEDED_PENALTY              # default 0.30
                 if a later revision supersedes this record
                 else 0
                − RISK_PENALTY                    # if PoQ flagged risk
```

Where `recency_for_type = 0.5 ** (age_days / half_life_days_for_type)` with half-lives from `metadata.py`.

**Modality anchoring (the fourth term, opt-in).** When `hybrid` is given a non-empty `query_modalities` set — the domain modes the current query implies, e.g. `artifact_content` when the user pasted code — a fourth term is added and the weights shift to make room without disturbing the others' balance:

```
base    = W_SEMANTIC_MODAL * cosine_similarity   # 0.45
        + W_SALIENCE_MODAL * record.salience     # 0.25
        + W_RECENCY_MODAL  * recency_for_type    # 0.15
        + W_MODALITY       * modality_overlap    # 0.15
```

`modality_overlap(query_mods, record_mods)` compares the query's domain modes to the record's stored `modalities_activated` (filtered to `DOMAIN_MODALITIES`): a record produced in the query's mode scores 1.0 (boost), a genuine mismatch scores 0.0 (mild cut), and a record with no domain modality scores the neutral `MODALITY_NEUTRAL` (0.5) — so older records and observations are treated as "unknown," not "mismatched." Only **domain** modalities (what kind of work a record is — currently just `artifact_content`) participate; **quality** modalities (`integrity_field`, `coherence`) are excluded, so a query that looks injection-y can't preferentially surface past injection-flagged records. Anchoring is strictly opt-in: a call with no `query_modalities` uses the default three-term weights, byte-identical to before, so existing callers are unaffected. Note the anchor fires on code *present in the query text* (pasted code), not coding intent expressed in prose — a prose follow-up like "make that function handle empty input" carries no artifact and so doesn't anchor; a future `coding_intent` modality could close that gap if needed.

The weights are exposed as class constants on `Retriever` (`W_SEMANTIC`, `W_SALIENCE`, `W_RECENCY`, the `*_MODAL` variants, `W_MODALITY`, `SUPERSEDED_PENALTY`) so they're tunable in one place. Each `RetrievalHit` carries a `components` dict showing the contribution of each term (including `modality_overlap`, `modality_weight_factor`, `modality_saturation`, `modality_damp`, and `modality_contribution` when anchoring is active) — useful for debugging or for the agent surfacing "why this record" downstream.

**Sprouted (runtime) modalities.** The domain set is not fixed. `sprouted_modalities.py` holds a registry of *data-driven* modalities — a name plus a few case-insensitive regex patterns plus an activation rule — loaded from `sprouted_modalities.json` in the data directory. A `Retriever` constructed with a `SproutRegistry` merges its domain-flagged entries into the live domain set (`Retriever.domain_modalities()` = baked-in + sprouted), runs their detectors at query-time analysis, and weights them in retrieval exactly like baked-in modalities. This lets the agent add to its own retrieval vocabulary at runtime with no source-code change and no restart — the data-driven counterpart to the `apply_proposal` path, which remains for modalities that need real detector logic. **Note on the safety boundary:** the codebase's stated rule was that the agent never modifies its own running behavior without a human in the loop, enforced by the manual `apply_proposal` step. Auto-sprouting deliberately relaxes that for the narrow case of pattern-based modalities; this was an explicit, owner-made decision, not an oversight. The regex surface is bounded *at validation time* (no `regex` module / thread-safe timeout is available): patterns must compile, nested-quantifier (catastrophic-backtracking) shapes are rejected, and pattern length/count and match-input length are capped — so a sprouted pattern can do bounded work at worst, never hang. (The auto-*generation* of sprout specs from recurring output — Cambium's side — is described next; this layer is the registry + retrieval integration that a sprout lands in.)

**Auto-generation and graduation (Cambium → agent).** Cambium's `_check_recurring_output_mode` trigger reads `response` records, clusters them by shared vocabulary, and when a cluster clears the **diversity gate** — at least `OUTPUT_MODE_MIN_TRIGGERS` (5) distinct responses, spanning at least `OUTPUT_MODE_MIN_SPREAD_MS` (2h), interleaved with non-matching responses — emits a `modality` proposal carrying a ready-to-stage `sprout_spec`. Patterns are derived deterministically (regex-escaped `\bword\b` over the shared vocabulary; no LLM), and a document-frequency ceiling (`OUTPUT_MODE_DOC_FREQ_CEILING`) drops ubiquitous filler words so generic chatter can't mint a mode. The agent's commit path (`_stage_and_graduate_sprouts`) stages each new sprout into the registry as **tentative** (half weight, never directly active) and writes a `sprout_status` audit record with provenance. Each later scan that re-detects the mode is a confirmation; once the live recurrence count reaches `OUTPUT_MODE_GRADUATION_CONFIRMATIONS` the modality graduates to active (full weight) with a second audit record. Two independent safety layers: the gate stops a bad sprout from being created, and tentative-by-default stops a created-but-wrong sprout from mattering much until confirmed. The chain's `proposal` / `proposal_recurrence` / `sprout_status` records are the source of truth; `sprouted_modalities.json` is the derived activation cache.

**Two damping mechanisms guard the feedback loop** that runtime sprouting otherwise invites (the agent reshaping which of its own memories surface, which shapes its next output, which keeps the modality firing):

- **Per-turn cap (`PER_TURN_MODALITY_CAP`, default 7).** `query_modalities` ranks the domain modalities a query fires by detection activation and keeps only the strongest N (all above the 0.2 floor), so a query that matches many modes doesn't blur the anchor across all of them. Exposed in `run.py`.
- **Anti-echo saturation damper.** `hybrid` is a two-pass computation: pass one scores every candidate *without* the modality term and measures what fraction of the top `MODALITY_SATURATION_TOP_N` candidates already carry the query's mode; if that saturation exceeds `MODALITY_SATURATION_THRESHOLD` (0.6), the modality term is damped by `1 − (saturation − threshold)` in pass two. So when the relevant context is already saturated with the query's mode, boosting "more of the same" is scaled down or off. Measuring saturation on the pre-boost scores is deliberate — measuring it after the boost would be circular.

A **tentative** sprouted modality (cooling-off, before it has graduated) contributes at a reduced `weight_factor` (0.5), so a not-yet-confirmed sprout nudges retrieval gently rather than at full strength. The `/modalities` REPL command lists baked-in and sprouted modalities, their status, domain flag, effective weight, and any patterns skipped at load.

**Revision-aware retrieval:**

When a record has been corrected by a later revision, two things happen at retrieval time:

1. The original gets a `−0.30` penalty applied after the weighted sum. This usually flips the ordering: the revision now ranks above its predecessor.
2. `build_context` automatically pulls any revision whose target appears in the result set, even if the revision itself wasn't a top-k semantic match. The model always sees the original *and* its correction together.

The original is never dropped — that would erase the conflict, and the conflict itself is informative ("you used to think X; you now think Y"). It's just demoted relative to the corrective record.

**Embedders.** Two ship in `retrieval.py`, and `run.py` picks between them at
startup via `make_tiered_embedder()` (see the run.py section below):
- **`HashingEmbedder`** — dependency-free bag-of-trigrams. Deterministic, no
  model, no network. Used by the test suite and as the offline fallback. It
  is lexical, not semantic: differently-worded but meaning-equivalent text
  does not score as similar.
- **`OllamaEmbedder`** — calls a local Ollama server's embeddings endpoint
  (default model `nomic-embed-text`). Real semantic embeddings; the model
  runs in Ollama's process, so no PyTorch enters the agent. Its only Python
  dependency, `requests`, is imported lazily so it stays optional. The
  constructor does one probe embed against the server, so it fails fast and
  clearly if Ollama is down or the model isn't pulled — which is what lets
  the tiered resolver fall back cleanly.

`EmbeddingIndex` records vectors at a fixed dimension. If the resolved
embedder's dimension doesn't match an existing store (because the embedder
changed between runs), the index refuses to open and tells you to delete the
store so it can be rebuilt from the chain. `OLLAMA_EMBED_DIMS` is a small
table of known model dimensions so the resolver can size the index without an
extra probe. The same identity check also catches same-dimension changes —
a different embedder coordinate space, or a different chunking scheme — by
comparing a stored embedder-identity tag against the active one.

**Chunked embedding store.** A record is not necessarily held as a single
vector. The embedder caps input (the Ollama path 500s on input that
overflows the model's token window rather than truncating), so a record
longer than the cap would be unreachable past its opening if embedded whole
— the back half of a long document, the body of a big code paste, or a long
reflection would never enter any vector. Instead, a record is split into
chunks at index time and each chunk is embedded into its own row, all sharing
the record's index as a group anchor.

- **Chunking — `chunk_text(text, target)`.** A module-level function that
  splits text on natural boundaries in priority order: paragraph breaks
  first; sentence breaks for a paragraph that alone exceeds the target; a
  hard slice for a single unbroken run (minified code, base64, a giant CSV
  line) so no chunk can exceed the embedder's request cap. Short text
  returns a single chunk, so an ordinary turn embeds to exactly one vector.
  `CHUNK_TARGET_CHARS` (default 3500)
  sits comfortably under the Ollama token window; `file` records prepend the
  `[file name kind]` header to every chunk so middle fragments stay
  self-describing.

- **Group-collapse — in `EmbeddingIndex.search`.** Per-chunk vectors create
  an obvious hazard: a long record would have *more vectors in the store* and
  so more chances to be retrieved — the "more raffle tickets" problem. The
  fix is to search over chunks (you want the most relevant *fragment*) but
  collapse the chunk hits down to one similarity per record before returning
  — the **maximum** over that record's chunks. Max, not mean: a long record
  with one strongly relevant section should rank on that section's strength,
  not be diluted by unrelated text elsewhere in it. `search` over-fetches a
  pool of chunk neighbors (so the k-th best record isn't missed when
  higher-ranked records each contribute several chunks), collapses to
  records, and returns the top-k. The return type is unchanged — one
  `(record_idx, similarity)` pair per record — so `hybrid`, `build_context`,
  `drift_against`, and every scoring term (salience, recency, supersession,
  PoQ-risk) are untouched. **The chunking is invisible above the `search`
  boundary.**

- **Completeness is free.** A retrieved record is rendered *whole* by the
  agent from the chain, not reassembled from chunks — the chain was never
  split, only the derived embedding store was. So when a record is selected,
  the model sees all of it; there is no fragment-stitching step and no risk
  of presenting half an answer.

- **Index-only; substrate untouched.** The append-only signed chain still
  stores exactly one record per turn and per file. Chunking lives entirely
  in the derived embedding store, which is rebuilt from the chain whenever
  deleted. `chain.py`, signing, Merkle batching, `/length`, and `/verify` are
  all unaffected. The chunking scheme is folded into the embedder-identity
  tag (`CHUNK_SCHEME_VERSION` + `CHUNK_TARGET_CHARS`), so changing chunk
  boundaries forces a clean store rebuild rather than silently mixing
  differently-chunked vectors.

The schema reflects this: the `embeddings` table is keyed on an autoincrement
`embed_id` with `record_idx`, `chunk_index`, and `chunk_count` columns and an
index on `record_idx` for fast per-record lookups. Re-indexing a record
deletes its existing chunk rows first, so `index_record` is idempotent under
the multi-row layout.

> **Known follow-up.** Chunking makes long records *findable* again
> but does not touch `Agent._truncate_to_budget`, which still drops whole
> lowest-salience records under context-budget pressure. A long record is now
> both more retrievable and a bigger truncation target, so making truncation
> chunk-aware (keep a record's most relevant chunks rather than drop it) is a
> natural next step — deliberately separate because that step *does* render
> fragments, reintroducing the "answer split across the cut" risk that
> whole-record rendering avoids.

**When you'd touch this file:** when you want different retrieval behavior.
Change `W_*` constants to rebalance the score components. Change
`CHUNK_TARGET_CHARS` to resize chunks (and bump `CHUNK_SCHEME_VERSION` if you
alter `chunk_text`'s boundary logic). Add a method for time-window queries.
Add an embedder (e.g. a sentence-transformers wrapper). Replace
`EmbeddingIndex` with FAISS or pgvector while keeping the `Retriever`
interface stable.

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
- `_truncate_to_budget(records, fixed_overhead_chars)` — when retrieved context would exceed `context_char_budget`, drop lowest-salience records first. Ranking uses **per-record** salience read from each record's `_meta` block (with type-based defaults for legacy records via `read_meta`). Returns kept records (chronologically ordered) and a count of dropped records, which the prompt formatter surfaces as a note to the model.

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

**Knows about:** Anthropic, OpenAI, Google, and Ollama SDKs (OpenRouter and DeepSeek reuse the OpenAI SDK). Retry logic. Authentication.
**Does not know about:** the chain, the agent, retrieval, prompts, metadata.

**Key components:**

Six builder functions, all returning a callable with shape
`(prompt, system=None, attachments=None) -> str`. Each callable also
exposes `.stream(prompt, system=None, attachments=None)` — a generator
yielding text chunks, used by the web UI for streaming responses.
- `make_claude_client(model, max_tokens, timeout_s)` — Anthropic Claude. Default: `claude-fable-5`. No sampling parameters (Fable 5 rejects `temperature`/`top_p`/`top_k`); thinking is always on (the `thinking` param is omitted); a classifier `refusal` is surfaced as an honest inline note.
- `make_openai_client(model, ...)` — OpenAI. Default: `gpt-5.5`.
- `make_openrouter_client(model, ...)` — OpenRouter (aggregator). Default: `anthropic/claude-opus-4.7`. Reuses the OpenAI SDK against OpenRouter's base URL.
- `make_deepseek_client(model, ...)` — DeepSeek. Default: `deepseek-v4-pro`. Reuses the OpenAI SDK against DeepSeek's base URL. The V4 models toggle thinking mode via a request parameter rather than a separate model name; this client runs the default (non-thinking) mode and, if a `reasoning_content` trace is returned, uses only the final answer.
- `make_gemini_client(model, ...)` — Google Gemini. Default: `gemini-3.1-pro`.
- `make_ollama_client(model, base_url, ...)` — local Ollama. Default: `llama3.1:8b`.

Each builder:
- Imports its SDK lazily so you only need the libraries for the provider you use. (OpenRouter and DeepSeek import the same `openai` package as the OpenAI client — they expose OpenAI-compatible APIs, so no extra dependency.)
- Checks the relevant API key at startup, fails fast if missing.
- Returns a callable that takes a prompt string (and optionally a system prompt) and returns response text.
- Routes the system prompt to the provider's correct API field (Anthropic's `system` parameter, OpenAI's system message role, Gemini's `system_instruction`, Ollama's `system` field; OpenRouter and DeepSeek use the OpenAI system-message role).
- Retries on transient errors (rate limits, timeouts, server errors) with exponential backoff. Does NOT retry on client errors (bad request, invalid model).

`_retry_with_backoff` is the shared retry helper.

**Truncation detection.** After every call (and stream), a client records why generation stopped on `llm.last_finish_reason`. The module-level helper `was_truncated(llm)` normalizes this across providers — OpenAI / OpenRouter / DeepSeek report `finish_reason == "length"`, Anthropic reports `stop_reason == "max_tokens"` — and returns `True` only on a confirmed max_tokens cut-off. A provider that doesn't report a reason (Gemini, Ollama, or a custom callable) reads as "complete", so the signal is never a false positive. `agent.turn()` calls `was_truncated()` and surfaces the result on `AgentTurn.truncated`; the REPL and web UI use it to show a "response was cut off — type continue" marker.

**When you'd touch this file:** when you want to add a new provider, change default model versions as new ones release, adjust retry behavior, or expose more provider-specific options (tool use, structured output, multimodal beyond images and PDFs). Each builder is independent — changing one doesn't affect the others.

---

## The code-working agent (v1.4): tools.py, task_registry.py, pending_ops.py

**Responsibility:** let the agent read, write, and audit code through tools,
with per-task continuum chains as durable memory and a three-tier safety
model. (This replaced `file_ingest.py` — code and content enter the system
through continuum ingestion now; format-aware text extraction lives in
`extractors.py`, and the content-addressed blob store remains the canonical
home of uploaded bytes.)

**The tool loop.** `llm_clients.py` has no native function calling, so tools
are TEXT: the model emits `<tool_call>{"name": …, "arguments": {…}}</tool_call>`
blocks. `tools.py` is the single shared driver — a TOLERANT extractor (strips
markdown fences, trailing commas, recovers every block) feeding a STRICT
JSON-Schema validator (unknown tools/params/types rejected), an executor with
a 64KB result cap, and an escaper that neutralizes `<tool_call>` markers in
anything re-entering the prompt, so file content can never forge a call.
`agent.turn_with_tools()` runs the bounded parse→execute→re-call loop with
one reflective retry on malformed calls, and shares its post-LLM tail
(`_finish_turn`: truncation, PoQ, verdict enforcement, commit) with the plain
`turn()` so the quality gates cannot drift. The identity chain stays a
low-noise stream of experience: ONE observation + ONE response per turn,
no per-call audit records — the response is the self-written, PoQ-gated
summary of the turn's work, and tool EFFECTS (ingest blocks with hashes
and git coordinates) live on the per-task continuum chains. The
committed response is everything the user saw: the prose of every tool-call
round (`tools.strip_tool_markup` — the `<tool_call>` blocks removed, the
surrounding text kept) joined with the final answer, in both loops and in
the web UI's live view — never just the last round's fragment, which would
read out of context on its own.

**The three safety tiers:**

| Tier | Mechanism |
|------|-----------|
| 1. Task selection | `task_registry.resolve_task()` returns exact/ambiguous/not-found and never auto-selects a fuzzy match; the system prompt forbids the model from guessing |
| 2. File scoping | `pin_file` scopes a turn's writes to one path; the pin resets at every turn start |
| 3. Write gate | `write_file` only creates a durable `PendingOperation` (0600, 1MB cap, TTL); ONLY the user can `/approve` — approval checks the pre-write hash (TOCTOU), writes atomically, ingests idempotently (`operation_id`), audits the block against live source, then deletes the op file |

Tier 3 also covers **boundary expansion**: `tools.requires_confirmation()` —
the one policy function both the REPL and web loops call — gates
`CONFIRM_TOOLS` (e.g. `task_ingest_file`, `task_reembed`) and any `task_open`
whose `source_root` resolves outside the workspace, since a task's source
root becomes an allowed read/ingest root. How the user confirms depends on
the loop: the REPL prompts inline (`proceed? yes/no`); the web loop —
which cannot prompt over one-way SSE — defers the call as a pending op of
kind `tool_call` (`tools.defer_tool_call`: exact name + arguments pinned
by hash, same TTL/0600 discipline as writes) and surfaces the existing
approve/reject card; headless loops refuse. Chat text can never satisfy
the gate — approval is `pending_ops._approve_tool_call`, reachable only
through the user-only action path, single-shot (no resumable middle state,
so a crash mid-execution can never double-run a non-idempotent tool).

**Workspace selection.** The workspace (`ctx.workspace_root`) is the
user-chosen working directory and the read/write boundary. Selection is
user-only — `POST /api/workspace` (session-gated; the UI's folder chip) or
the REPL's `/workspace` — never a model tool, so a chosen workspace is
inherently confirmed. `tools.set_workspace` validates the directory
server-side, resets the active task and pin, and persists the choice in
`<DATA_DIR>/workspace.json` (`restore_workspace` reloads it at boot).
Switching is a pure boundary move: nothing is created or sealed. A
workspace task chain appears lazily at the first action that needs durable
state (`tools.ensure_workspace_task`, called by `write_file` when no task
is active) — named after the directory, reused thereafter; reads and
unrelated turns leave no registry state. The per-turn system prompt
carries the current workspace path in both loops.

**Task chains.** `task_registry.py` tracks per-task continuum chains under
`<DATA_DIR>/tasks/<name>/` (own `chain.sqlite`, `operator.key`,
`embeddings.sqlite`). `tools.AgentContext` lazily opens chains, recalls,
continuums, and embedding indexes per task and closes them all on exit.
`Recall.retrieve_path_aware()` is the task-chain arbiter (blended semantic +
path + chronological scoring with hard filters); the identity-chain
`retrieve()` stays a pre-filter.

**Artifacts routing.** Uploads and pastes (`ingest_blob`, `/api/upload`)
default to the reserved `artifacts` task chain, lazily created on first
upload. Bytes land in the content-addressed blob store (canonical — the
vision path and `/blobs/<sha>` resolve by sha) plus a named copy in
`ARTIFACTS_DIR` (default `~/.artifacts`, which doubles as the artifacts
task's source_root so the copies are readable through the normal path
gates); extracted content is chunked and embedded in the artifacts chain's
OWN store; the identity chain gets one tiny `attachment` pointer ring (no
content). Routing into a normal task is explicit only — `task_name` from
the model, or the web UI's per-upload toggle — because an ACTIVE task
capturing uploads silently pollutes unrelated, append-only task chains,
and full text on the identity chain crowds retrieval. An upload never
moves the session's task cursor; `task_open` refuses the reserved name.

**Task-store embedder policy.** Task embedding stores default to the instant
`HashingEmbedder` regardless of the session embedder — bulk ingest must never
block on a slow embedding model (a CPU-bound Ollama at ~3–5s/chunk once
turned a 10-second walk into a 2-hour tool call). A task opts into the
session embedder only through the user-confirmed `task_reembed` tool, which
rebuilds the derived store with batched embedding
(`EmbeddingIndex.index_records_batched` → `OllamaEmbedder.embed_batch`, one
`/api/embed` request per 64 chunks) and persists the choice as
`task["embedder"] = "session"` in `tasks.json`, so later sessions reopen the
store with the same embedder instead of mismatch-deleting it back to the
hashing default. Ingest paths open the task index BEFORE sealing new blocks,
so the first-open backfill and the post-seal indexing pass can never embed
the same record twice.

**When you'd touch these files:** add a tool (schema in `TOOLS`, executor in
`EXECUTORS`), tighten write-path rules (`resolve_write_path`), add an
extraction format (`extractors.py`), or extend the pending-op state machine
(`pending_ops.execute_approve_write`).

---

## timechain_web/ — Optional browser UI

**Responsibility:** provide a browser-based chat interface as an alternative to the REPL. Same agent stack, same chain, same configuration — just a different I/O layer.

**Knows about:** FastAPI, Server-Sent Events, the same agent/chain/retriever that `run.py` uses (imported directly from `run.py` to share configuration). It also imports `metadata.build_meta` so the streaming-path inline appends carry the same `_meta` block as `agent.turn()` does.
**Does not know about:** anything `run.py` doesn't already know about. It's strictly an interface layer.

**Key components:**

`webapp.py` is a FastAPI server that:
- Boots the same Agent / Chain / Retriever stack as `run.py`, reusing `DATA_DIR`, `SYSTEM_PROMPT`, `FOUNDING_COMMITMENTS`, etc. directly.
- Exposes endpoints for chain inspection (`/api/chain/status`, `/api/chain/recent`, `/api/chain/verify`), slash commands (`/api/command`), chat turns (`/api/turn`, `/api/turn/stream`), the durable write gate (`/api/pending-ops`, `/api/pending-ops/{id}/approve|reject` — user-triggered only), and content ingestion (`/api/upload` → `ingest_blob` with an optional explicit `task` field, `GET /api/tasks/active` for the UI's routing toggle, `/blobs/{sha}` — session-gated like every other endpoint).
- The streaming turn runs the async twin of `Agent.turn_with_tools`: tool calls parsed/validated/executed/escaped by tools.py (the single shared driver), surfaced to the browser as `tool_result` / `pending_op` SSE events between token rounds. Blocking work — tool execution, approved pending ops, upload ingestion — runs in worker threads (`asyncio.to_thread`; the chain/index SQLite connections are opened `check_same_thread=False` and all access stays serialized under the web lock), so a long re-embed or tree walk never freezes the event loop.
- Streams responses via Server-Sent Events when the LLM client supports it (all six built-in providers do, via `llm.stream()`).
- **Turns are server-owned, not connection-owned.** Each turn runs in a background task (`TurnRun` / `_drive_turn`) that drains `_turn_events` into a per-turn event buffer; `/api/turn/stream` only subscribes (`_follow_turn`: replay the buffer, then follow live). Closing the page mid-turn — the audit tab, a reload, a network blip — detaches the viewer while the turn runs to completion and commits, so the chain can never strand a sealed observation without its response. The chat page probes `GET /api/turn/active` on load and reattaches with `?attach=1`; SSE event ids + `Last-Event-ID` make EventSource auto-reconnects resume the same run instead of starting a duplicate (a genuinely new input while one runs gets a 409). `POST /api/turn` rides the same runner.
- Holds a single-session lock — only one browser tab is "active" at a time, protecting the chain's single-writer guarantee. All requests that touch the chain serialize through an asyncio lock regardless of session.

`static/index.html` is a single-file frontend (vanilla JS, no build step). A typing indicator shows during the latency before tokens arrive; a sidebar surfaces recent reflections and revisions. The frontend reads `content.text`, `content.revises_index`, and other top-level content fields exactly as in v1 — the `_meta` block sits next to them and is ignored by the UI.

**The streaming endpoint** at `/api/turn/stream` deliberately inlines its chain writes (rather than calling `agent.turn()`) so it can split the LLM call into a thread. These inline writes use `metadata.build_meta()` to attach the same `_meta` block that `agent.turn()` produces — there's no path through the app that produces a record without one. If you ever extend this endpoint with new chain writes, do the same.

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
- `LLM_PROVIDER` — which client to use ("claude", "openai", "openrouter", "deepseek", "gemini", "ollama"). Read from the environment variable of the same name; defaults to "claude".
- `FOUNDING_COMMITMENTS` — committed at genesis, immutable thereafter. Define the agent's values.
- `SYSTEM_PROMPT` — sent to the LLM on every turn. Defines the agent's active behavior. Mutable; each version is logged to chain.
- `SEMANTIC_K` / `RECENT_N` — retrieval knobs.
- `OLLAMA_EMBED_MODEL` / `OLLAMA_BASE_URL` — which Ollama embedding model to use, and where to reach the server. Used by the tiered embedder resolver.
- `HASHING_EMBED_DIM` — dimension of the fallback `HashingEmbedder`. There is no longer an `EMBED_DIM` constant: the active embedding dimension is whatever the resolved embedder reports.
- `AUTO_REFLECT_EVERY` — how often the agent automatically reflects (in turns). Set to 0 to disable. Each reflection automatically covers every record since the previous reflection — there's no separate window setting; the scope sizes itself to actual activity.

Functions:
- `make_tiered_embedder()` — resolves the embedder with a fallback chain: `OllamaEmbedder` if a local Ollama server is reachable, otherwise `HashingEmbedder`. Returns an `(embedder, dim, name)` triple. Never raises — the worst case is the fallback.
- `build_llm()` — returns the configured LLM client based on `LLM_PROVIDER`.
- `run()` — the REPL. Loads everything, commits genesis if needed, checks for drift between configured and sealed founding commitments and warns if they differ, logs system prompt if changed, loops on input, dispatches commands or turns, auto-reflects on cadence.

Slash commands:
- `/verify` — cryptographically validate the entire chain.
- `/length` — current record count (ground truth, no LLM in the loop).
- `/seal` — create a Merkle batch over recent records.
- `/sysprompt` — show all system prompt versions logged on chain.
- `/reflect` — manually trigger a reflection.
- `/revise N <text>` — append a correction to record N.
- `/task list|open|ingest|resume|validate|audit` — per-task continuum chains.
- `/approve <id>` / `/reject <id>` / `/pending` — the durable write gate.

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

## Cognitive faculties (v1.3)

v1.3 adds a layer of cognitive faculties. They are deliberately
**storage-independent**: each speaks a generic "ring" shape through one small
adapter, `ring_compat.py`, which presents a repo `Record` as a ring
(`index`/`ring_type`/`payload`/`ring_hash`) on the read side and seals
ring-style payloads back through `chain.append` + `build_meta` on the write
side. So the cognitive *logic* never touches SQLite directly — it composes with
the same signed chain everything else uses, and the cryptographic core is
unchanged.

The faculties, and how each maps onto the existing substrate:

- **`poq.py` (extended) — a quality gate with teeth.** On top of the existing
  `brightness`/`action`, PoQ now returns a `verdict`
  (`SEAL`/`REVISE`/`FORCE_UNCERTAINTY`/`REJECT`) from two new measures
  (grounding, assertiveness) and the existing covenant/consistency dimensions.
  `evaluate(..., external_scores=...)` is the **seam**: a real model can
  override any score, so the lexical proxies are a runnable fallback, not the
  judge. The `verdict` is persisted inside `_meta.poq` only when it is not
  `seal`, so no schema version changed.

- **`immune.py` — active self-defense.** `screen` refuses covenant/scar inputs
  at the membrane; `scan` reuses the Ed25519 `chain.verify()` for tamper
  detection; `rollback` seals a `recovery` record and molts the wound into a
  learned scar — and, given a `faculty_dir`, offers the scar vector to
  `FacultyGarden.grow(kind_override="sense")` so the attack becomes an
  antibody faculty (the REPL `/rollback` does this automatically).
  Enforcement is a **one-line lockdown gate at the top of
  `chain.append`** (only `recovery` may be sealed while a `LOCKED` sidecar flag
  exists) — so no seal path can bypass it. State is a sidecar, never on-chain.

- **`continuum.py` — long-horizon tasking.** Streams a job too large for any
  context window into bounded data-height blocks, each carrying a full
  task-state refresh, so `resume()` re-hydrates from the head block alone.
  Built for a **per-task chain** (its own DB) so a big audit doesn't dilute the
  identity chain.

- **`recall.py` — relevance realization.** `label` self-tags a record using the
  existing `signals.py` faculties; `index`→`fetch` is the model-as-judge loop
  over a chain larger than context; `retrieve` is a cheap pre-filter that
  delegates to the existing `Retriever`, never the arbiter.

- **`chronosynaptic.py` — single-pass parallel-self MCTS.** Forks faculty-lens
  perspectives (drawn from the `signals.py` registries), scores each with PoQ,
  and collapses to one `synthesis` record — no subagents.

- **`consensus.py` — quorum attestation.** k-of-n HMAC witnesses attest each
  head over the *recomputed* `record_hash`, layering tamper-*resistance* on the
  chain's tamper-*evidence*. Sidecar config/attestations, never on-chain.
  The `Agent` holds a `Quorum` handle and auto-attests every committed
  response once `/consensus-init` has run (`Quorum.is_initialized()` checks
  only the config file — requiring an attestation file too would deadlock);
  an attestation failure never blocks a turn. The read-only `defense_status`
  tool reports chain integrity, immune posture, quorum health, and antibody
  count in one snapshot.

- **`faculties.py` + `faculties/*.json` — faculties as data + growth.**
  Descriptive data faculties (84 modalities, 107 senses) complement the
  executable `signals.py` detectors; `FacultyGarden.grow` sprouts/promotes new
  faculties when an input reveals a gap, sealing `faculty`/`promotion` records.

- **`chain.py` (extended) — optional proof-of-work.** `append(..., difficulty=N)`
  mines a nonce inside `content["_pow"]`; `difficulty=0` (the default) is
  byte-identical to before. **`migrate.py`** backfills the derived embedding
  index off-chain for historic records — the signed chain is never rewritten.

The two turn-loop changes — immune screening (on by default, narrow) and PoQ
verdict enforcement (opt-in) — live in `Agent.turn()` and the webapp
`turn_stream`, so the REPL and web UI route through the same gates. `run.py`
exposes the faculties as slash commands via `cypher_commands.py` (`/cypher-help`).

## What's where: a quick reference

| If you want to change... | Edit this file |
|--------------------------|----------------|
| The LLM provider or model | `run.py` — `LLM_PROVIDER`, or `llm_clients.py` defaults |
| Maximum response length | `run.py` — `LLM_MAX_TOKENS` (default 4096). Fed to every provider client by `build_llm()`. A response that hits this ceiling is reported as truncated. |
| The agent's personality / tone | `run.py` — `SYSTEM_PROMPT` (auto-logged on change) |
| How often the agent reflects | `run.py` — `AUTO_REFLECT_EVERY` |
| How much context the agent sees | `run.py` — `SEMANTIC_K` and `RECENT_N` |
| Maximum prompt size before truncation | `run.py` — `CONTEXT_BUDGET_CHARS` (default 150,000). Fed to the `Agent` constructor's `context_char_budget` by both `run.py` and `webapp.py`. |
| Default salience by record type | `metadata.py` — `DEFAULT_SALIENCE_BY_TYPE` |
| Per-kind decay half-lives | `metadata.py` — `DEFAULT_HALF_LIFE_DAYS_BY_TYPE` |
| Source tag enum | `metadata.py` — `SOURCE_*` constants |
| Score-formula weights (semantic / salience / recency / modality) | `retrieval.py` — `Retriever.W_SEMANTIC`, `W_SALIENCE`, `W_RECENCY`, `W_MODALITY` (and `*_MODAL` variants) |
| Which modalities anchor retrieval | `retrieval.py` — `DOMAIN_MODALITIES` (baked-in) + domain-flagged entries in `sprouted_modalities.json` |
| Per-turn modality cap / anti-echo threshold | `run.py` — `PER_TURN_MODALITY_CAP`; `retrieval.py` — `MODALITY_SATURATION_THRESHOLD`, `MODALITY_SATURATION_TOP_N` |
| Sprouted-modality registry file | `run.py` — `SPROUTED_MODALITIES_FILE` (default `<data_dir>/sprouted_modalities.json`) |
| How aggressively superseded records are demoted | `retrieval.py` — `Retriever.SUPERSEDED_PENALTY` |
| Embedding chunk size (and chunk-scheme version) | `retrieval.py` — `CHUNK_TARGET_CHARS`, `CHUNK_SCHEME_VERSION` |
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
| Adding a tool | `tools.py` — schema in `TOOLS`, executor in `EXECUTORS`, audit fields in `TOOL_AUDIT_FIELDS` |
| Write-gate rules | `tools.resolve_write_path` and `pending_ops.execute_approve_write` |
| Web UI behavior or appearance | `timechain_web/webapp.py` (server) and `timechain_web/static/index.html` (frontend) |
| Streaming responses to the browser | `llm_clients.py` — each client's `.stream()` method |
| Adding tests | `test_timechain.py` — pytest classes by concern; `test_cypher_port.py` / `test_cypher_integration.py` for the v1.3 faculties |
| PoQ verdict thresholds | `poq.py` — `PoQ_THRESHOLDS` |
| Enabling verdict enforcement / the model-judgment seam | `agent.py` — `Agent(enforce_verdict=..., score_hook=...)` |
| Disabling immune screening | `agent.py` — `Agent(enable_immune=False)` |
| The immune covenant/scar lexicon | `immune.py` — `_COVENANT_VIOLATIONS`, `SKIP_TYPES` |
| Data-faculty registries / growth thresholds | `faculties/*.json`; `faculties.py` — `DISSONANCE_FLOOR`, `PROMOTE_AT` |
| Adding a v1.3 faculty REPL command | `cypher_commands.py` — `dispatch()` |

---

## Three things worth understanding deeply

**The chain is the source of truth, the LLM is the reasoner.** When the LLM says something about your past ("you mentioned apples earlier"), it's reasoning over what was retrieved. When you want certainty, ask the chain directly via `/length`, `/verify`, or `view_chain.py`. The model's view is partial by design; the chain's view is total. This separation is what makes the architecture trustworthy — the model can hallucinate, but it can't fake a signed record.

**Genesis commitments are permanent; the system prompt is mutable but auditable.** The founding commitments sealed at record 0 are part of the chain's identity and cannot be changed. The system prompt can be edited freely, but every change is logged to the chain as a `system_prompt` record. This gives you the right combination: stable identity values plus iterable behavioral configuration, with full history of how behavior has been shaped over time. You can always check whether the current configuration honors the sealed commitments.

**Memory is active, not passive.** Standard RAG retrieves records and stops there. This system also reflects (writes its own summaries of what mattered), revises (corrects prior records without erasing them), and weights retrieval by salience (the agent's own reflections surface preferentially). The result is closer to how human memory actually works — consolidating, revising, letting unimportant things fade — implemented as additional record types on the same append-only chain rather than as new mechanisms outside it.

**Records are evidence; beliefs are derived.** Every record carries an explicit `source` tag — user, assistant, system, or tool — so the LLM can see at a glance where a claim originated. "The user said this" and "I inferred this in a reflection two months ago" are different epistemic objects, and treating them as equivalent is the failure mode that makes long-running agents drift. Source tagging plus per-kind decay plus revision-aware retrieval together make this honest: the agent's past inferences fade faster than user statements, get demoted when corrected, and never masquerade as ground truth.
