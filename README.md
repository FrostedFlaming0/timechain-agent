# Timechain Agent

A persistent-memory AI agent built on a hash-chained, cryptographically signed,
append-only memory substrate. Memory survives across sessions, is tamper-evident,
and can be cryptographically verified at any time. Provider-agnostic: works with
Claude, GPT, Gemini, DeepSeek, any model on OpenRouter, or local models via Ollama.

**Version: 1.3.0.** See [CHANGELOG.md](CHANGELOG.md) for the full release
history. Project home:
[github.com/frostedflaming0/timechain-agent](https://github.com/frostedflaming0/timechain-agent).

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
- **Per-record metadata** — every record carries a `_meta` block
  with `source` (user / assistant / system / tool), `salience`,
  `confidence`, and `schema_version`. Source is the load-bearing
  distinction: the LLM can see "this is what the user said" vs "this is
  what I inferred" rather than treating all records as equivalent.
  Response records also record `modalities_activated` — which analysis
  capabilities fired in producing the response — as a data layer for
  later retrieval, and `senses_activated` — the *felt qualities* of the
  turn (`uncertainty`, `insight_markers`, `cognitive_weather`, etc.) —
  not used for retrieval but readable when the agent revisits its own
  history.
- **Hybrid retrieval** with explicit, named score components: semantic
  similarity, per-record salience, and per-kind half-life recency decay.
  Observations decay over weeks; reflections over months; genesis and
  system prompts effectively never decay. When the query carries a domain
  mode (e.g. pasted code), a fourth **modality-anchoring** term
  preferentially surfaces records produced in the same mode — so a coding
  query pulls up the agent's past code more readily than conversational
  chatter. Opt-in and neutral when no mode is present. The domain set is
  extensible at runtime: Cambium detects a recurring *kind* of the agent's
  own output and **auto-sprouts** a new pattern-based modality for it (a name
  + deterministically-derived regex specs in `sprouted_modalities.json`)
  without a code change or restart. A new sprout must clear a diversity gate
  (≥5 distinct triggers, ≥2h spread, interleaved) to be created, lands as
  *tentative* at half weight, and only graduates to full weight after
  repeated confirmation — and two further dampers (a per-turn cap and an
  anti-echo saturation check) guard against the feedback loop of an agent
  reshaping its own retrieval. See `/modalities`.
- **Tiered embedder** — the embedder is resolved at startup with a
  fallback chain: a local Ollama server if one is reachable, otherwise a
  dependency-free hashing embedder. No required embedding model, no heavy
  ML stack, and the agent never fails to start for lack of an embedder.
- **Chunked retrieval** — long records are split into chunks at
  index time so content past the embedder's input cap is still searchable.
  A group-collapse step scores each record by its single most relevant
  chunk, so long records don't crowd out short ones, and retrieved records
  are rendered whole. Index-only; the signed chain is untouched. See
  [Chunked retrieval](#chunked-retrieval-v121).
- **Revision-aware retrieval** — when a record has been corrected
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
  records dropped first. Truncation is driven by per-record
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

`requirements.txt` uses `>=` lower bounds so the project picks up
security patches in its dependencies. For long-running deployments
that need reproducibility, freeze your working set with
`pip freeze > requirements-lock.txt` and install from the lock file
afterwards.

### Using an existing chain

The agent reads its chain from `timechain_data/` (set by `DATA_DIR` in
`run.py`). To run against a chain from elsewhere, drop that folder in — or
point `DATA_DIR` at it — and start as usual; `run.py` reuses it instead of
committing a new genesis.

- **Bring `operator.key`.** It's the Ed25519 signing key. With it, new records
  continue the same signing identity. Without it, `run.py` generates a fresh
  key: old records still verify (each carries its own pubkey), but new records
  are signed differently and `/verify` will flag the mix.
- **`embeddings.sqlite` is a derived index, not part of the chain.** The
  minimum you need is `chain.sqlite` (plus `operator.key`); the embedding store
  is rebuilt from the chain on first launch (`indexed N pre-existing records`).
  If you bring a store built with a different embedder and hit an
  `EMBEDDING STORE MISMATCH` at startup, just delete it and restart
  (`rm timechain_data/embeddings.sqlite*`).
- **Older (pre-v1.3) chains read fine** — new record types and `_meta` fields
  default non-destructively, and there are no immune/consensus sidecars to
  carry over (they're created on demand). To re-embed already-indexed records
  with your current embedder rather than just the missing ones, run `/migrate`.
- The **sealed genesis is authoritative** for the founding commitments. If
  `FOUNDING_COMMITMENTS` in `run.py` differs from what's sealed, you get a
  non-blocking drift **warning** at startup; the sealed values win.

### Embeddings (optional but recommended)

There is no required embedding dependency. At startup the agent
picks an embedder with a tiered fallback:

1. **Ollama**, if a local server is reachable — real semantic embeddings,
   runs entirely on your machine. To enable it, install
   [Ollama](https://ollama.com/download), pull an embedding model, and make
   sure `requests` is installed:

   ```bash
   pip install requests
   ollama pull nomic-embed-text
   ```

2. **Hashing fallback**, otherwise — a dependency-free bag-of-trigrams
   embedder. No install, no model. Retrieval still works but is lexical
   rather than semantic, so recall on differently-worded queries is weaker.

`run.py` prints which embedder it selected at startup. You don't configure
this — it's automatic — but you can change the Ollama model or URL via
`OLLAMA_EMBED_MODEL` / `OLLAMA_BASE_URL` in `run.py`.

The embedding store (`embeddings.sqlite`) is a **derived index, not part of
the chain.** It is rebuilt automatically from the chain whenever it is
missing, so deleting it is always safe — the chain, the signing key, and
all records are untouched. If retrieval ever looks wrong after changing the
embedder, or the agent reports an embedder-identity or dimension mismatch at
startup, delete the store and let it rebuild:

```bash
rm timechain_data/embeddings.sqlite*
python run.py   # re-embeds the whole chain
```

Slash commands inside the REPL:
- `/verify` — cryptographically validate the entire chain
- `/length` — current record count
- `/seal` — create a Merkle batch over recent records
- `/sysprompt` — show the system prompt history on chain
- `/reflect` — trigger a reflection over recent history
- `/cambium` — run a Cambium scan for recurring gaps now
- `/proposals` — list proposal records (escalated first, with recurrence counts)
- `/modalities` — list baked-in and sprouted modalities (status, domain flag, weight)
- `/revise N <text>` — append a correction record targeting record N
- `/file <path>` — ingest a file (document, image, spreadsheet, code, etc.)

To inspect the chain offline:
```bash
python view_chain.py --all
python view_chain.py --record 5
python view_chain.py --type reflection
python view_chain.py --verify
```

### Diagnosing embedding issues

If the app hangs or errors during startup indexing, run `python
diagnose_index.py`. It embeds every chain record one at a time —
chunking each record exactly as the real index path does, so a failure
here corresponds to a real boot-time indexing failure — and prints
OK/FAILED with timing and chunk count for each, so a problem record (or
a slow embedder) is pinned down immediately rather than appearing as a
silent hang.

### Reviewing Cambium proposals

`apply_proposal.py` is the operator-run tool for acting on the proposals
Cambium produces. Cambium proposes; this tool is how a human reviews and
decides. It is never called by the agent.

```bash
python apply_proposal.py --list        # all proposals, escalated first
python apply_proposal.py --show 12     # full detail for proposal #12
python apply_proposal.py --accept 12   # scaffold a stub + record the decision
python apply_proposal.py --decline 12  # record a decline
python apply_proposal.py --decline 12 --reason "out of scope"
```

`--accept` on a modality or sense proposal scaffolds a detector *stub* into
`signals.py` — correct signature, registered in the right registry, but
with a `# TODO` body and a harmless return. A human writes the real
detector logic and adds a test; the tool never writes working code. Every
decision is recorded on the chain as a `proposal_status` record, so the
audit trail shows who decided what.

This manual path is for modalities that need real detector logic. For simple
*pattern-based* modalities there is now a second path — runtime sprouting
(see `sprouted_modalities.py` and `/modalities`) — where the agent adds a
regex-spec modality to a data file with no code change and no human review.
That deliberately relaxes the "human in the loop" boundary that
`apply_proposal` enforces, for the narrow pattern-based case only; it was an
explicit owner decision, and the regex surface is bounded at validation time
(patterns must compile, catastrophic-backtracking shapes are rejected,
lengths/counts capped) so a sprouted pattern can never hang the process. The
`apply_proposal` path above is unchanged and remains the route for anything
that needs logic beyond pattern matching. The autonomous path is not
unconditional: Cambium only auto-sprouts a mode that clears a diversity gate
(repeated, time-spread, interleaved evidence), the sprout enters at half
weight as *tentative*, and it reaches full weight only after repeated
confirmation — so review is replaced by evidence thresholds, not removed
outright.

Two things to expect. First, `--list` shows `no proposal records on chain
yet` until Cambium has actually found a recurring pattern — the same
correction, failure, or confusion showing up 3+ times. A fresh or
smoothly-running chain genuinely has nothing to propose; that is normal.
You can prompt a scan with `/cambium` in the REPL rather than waiting for
the auto-scan. Second, the tool reads the chain from the `timechain_data/`
directory next to the script — the same default `run.py` uses. If you have
customized `DATA_DIR` in `run.py`, update the matching constant near the
top of `apply_proposal.py` so both point at the same chain.

## Switching LLM providers

The default is Claude. The agent also supports OpenAI, OpenRouter, DeepSeek,
Gemini, and local Ollama models.

**1. Install the SDK:**

```bash
pip install openai           # covers OpenAI, OpenRouter, AND DeepSeek
pip install google-genai     # for Gemini
pip install requests         # for Ollama
```

OpenRouter and DeepSeek both expose OpenAI-compatible APIs, so the single
`openai` package serves all three — no separate dependency for either.

**2. Set the API key** (skip for Ollama — it runs locally):

```bash
export OPENAI_API_KEY=sk-...
export OPENROUTER_API_KEY=sk-or-...
export DEEPSEEK_API_KEY=sk-...
export GEMINI_API_KEY=...
```

**3. Edit `LLM_PROVIDER` in `run.py`:**

```python
LLM_PROVIDER = "openai"   # or "openrouter", "deepseek", "gemini", "ollama", "claude"
```

That's it. The chain, retrieval, and web UI all carry over — your existing
memory works with any provider.

For Ollama, install the [Ollama app](https://ollama.com/download), pull a
model (`ollama pull llama3.1:8b`), and make sure the local server is
running before you start `run.py`.

**OpenRouter** routes to hundreds of models behind one endpoint and one key.
Its model strings are provider-namespaced — `anthropic/claude-opus-4.7`,
`deepseek/deepseek-chat`, `meta-llama/llama-3.3-70b-instruct`, and so on (see
[openrouter.ai/models](https://openrouter.ai/models)). The default is
`anthropic/claude-opus-4.7`; override it via `build_llm()` as shown below.

**DeepSeek** offers `deepseek-chat` (default) and `deepseek-reasoner`. The
reasoner model produces a separate chain-of-thought trace; this client keeps
things simple and uses only the final answer — the reasoning trace is
discarded, so the agent treats `deepseek-reasoner` like any other model.
Note that DeepSeek's API is operated from China; as with any hosted
provider, prompt content (including retrieved memory) is sent to the
provider. Use Ollama if nothing should leave your machine.

To override the default model for a provider, edit `build_llm()` in
`run.py`:

```python
def build_llm():
    if LLM_PROVIDER == "ollama":
        return make_ollama_client(model="qwen3:8b")
    if LLM_PROVIDER == "openrouter":
        return make_openrouter_client(model="deepseek/deepseek-chat")
    if LLM_PROVIDER == "deepseek":
        return make_deepseek_client(model="deepseek-reasoner")
    # ...
```

Or hoist it to a config constant if you'll be switching models often.

A few practical notes:

- You can switch providers mid-chain — your existing memory carries over.
  The chain stores observations and responses, not which model produced
  them.
- Different providers will give different answers to the same prompt.
  Same memory, different reasoner.
- Streaming works on all six providers in the web UI.
- Default model names in `llm_clients.py` may go stale as providers
  release new versions. If you get a "model not found" error, look up
  the current name and pass it explicitly via `make_X_client(model=...)`.
- **Response length.** `LLM_MAX_TOKENS` in `run.py` caps a single
  response (default 4096 ≈ 3000 words); `build_llm()` feeds it to whatever
  provider is selected. If a response does hit the ceiling, it is cut off
  mid-thought and the REPL / web UI show a "response was cut off" marker so
  you know to type `continue`. The marker appears for Claude, OpenAI,
  OpenRouter, and DeepSeek (the providers that report a finish reason).
  `LLM_MAX_TOKENS` only sets a ceiling — you are billed for tokens actually
  generated, so raising it costs nothing on short replies.
- **Context budget.** `CONTEXT_BUDGET_CHARS` in `run.py` (default 150,000)
  caps how much retrieved memory is packed into one prompt. It is a
  ceiling, not a target: turns with little relevant history use less. The
  two knobs are related — a long response becomes a large record that then
  competes for this budget on later turns — so the comments in `run.py`
  explain how to keep them balanced if you raise either.

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

The header links to two reference pages: a **Commands** page (`/commands`)
documenting every slash command — what it does and why you'd use it — and an
**Audit** dashboard (`/audit`) showing chain metrics, domain context, the
faculty surface (modality/sense categories), a searchable ring inspector that
shows each ring's full contents and `_meta`, and the blockspace.

All slash commands from the REPL work the same way:

- `/verify` `/verify-semantic` `/length` `/seal` `/sysprompt`
- `/reflect` `/revise N <text>` `/cambium` `/cambium-full` `/proposals` `/modalities`
- `/file <path>` `/export-capsule <path>` `/import-capsule <path>`
- `/verify-source <idx>` `/poq <text>` `/immune-status` `/immune-scan` `/lockdown` `/rollback <N>`
- `/recall-index` `/recall-fetch <ids>` `/recall <query>` `/think <query>`
- `/consensus-init` `/consensus-verify` `/cambium-grow <text>` `/migrate` `/continuum-resume` `/continuum-validate`

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
- Responses stream token-by-token via Server-Sent Events. All six
  providers in `llm_clients.py` (Claude, OpenAI, OpenRouter, DeepSeek,
  Gemini, Ollama) expose a `.stream()` method; the UI uses it
  automatically. Custom clients without a `.stream()` method fall back to
  non-streaming gracefully — the response just appears all at once instead
  of progressively.
- The web server doesn't append records that the REPL wouldn't append.
  Same record types, same retrieval, same reflection cadence
  (`AUTO_REFLECT_EVERY` from `run.py`). It's an I/O layer, not a
  different agent. The streaming endpoint also writes the
  same `_meta` block as `agent.turn()` — there's no path through the
  app that produces a v1 record.

You can run `run.py` and `webapp.py` against the same chain at different
times, but not simultaneously — both want exclusive access to the SQLite
database and the signing key. Pick one interface per session.

## Cognitive faculties (v1.3)

v1.3 adds a layer of cognitive self-model faculties on top of the signed
chain, keeping the project's neutral, non-experiential vocabulary. Eight new
stdlib-only modules, each a thin adaptation over `ring_compat.py` (which
presents repo `Record`s in the "ring" shape the faculty logic expects). The
cryptographic core is untouched; old chains read and verify unchanged.

| Capability | Module | What it adds |
|---|---|---|
| verify-source | `source_verify.py` | re-check an ingested file vs. disk (git-aware) |
| PoQ verdicts | `poq.py` | `SEAL`/`REVISE`/`FORCE_UNCERTAINTY`/`REJECT` + a model-judgment seam |
| immune system | `immune.py` | screen / scan / lockdown / rollback / scar-learning |
| Continuum | `continuum.py` | long-horizon tasking in data-height blocks with full state refresh |
| Recall | `recall.py` | self-label → index → fetch (the model is the relevance judge) |
| Chronosynaptic | `chronosynaptic.py` | single-pass parallel-self MCTS, collapse to a `synthesis` |
| Consensus | `consensus.py` | k-of-n witness quorum attestation over the recomputed hash |
| faculties + growth | `faculties.py` | data-faculty registries + Cambium sprout/promote |

Two behavioral changes are wired into the turn loop:

- **Immune screening is on by default** (`Agent(enable_immune=True)`). It
  refuses covenant/character violations and known attack scars at the
  membrane; prompt-injection still flows to the existing PoQ-quarantine path,
  so the two membranes are complementary.
- **PoQ verdict enforcement is opt-in** (`Agent(enforce_verdict=True)`) —
  because the verdicts run on lexical proxies by default. Pass a
  `score_hook` to drive verdicts with a real model judgment (the
  `external_scores` seam) before enabling hard output-suppression.

New REPL commands are listed by `/cypher-help` (`/verify-source`, `/poq`,
`/immune-status`, `/recall-index`, `/think`, `/consensus-verify`,
`/cambium-grow`, `/migrate`, …). See `CHANGELOG.md` (v1.3.0) for the full
release notes.

## Limitations

A short list of things the codebase doesn't try to do, so deployment
choices are informed.

- **The v1.3 immune screen and PoQ verdicts run on lexical proxies.** Like
  the rest of the signal stack, the covenant/scar/grounding/assertiveness
  scorers are deterministic English word-list heuristics, so they can
  misjudge. That is why verdict enforcement (output-suppression) is opt-in and
  immune screening is kept narrow. The real judgment is meant to enter through
  the `external_scores` / `score_hook` seam, where a model overrides the proxy.

- **Signal detectors are English-only.** The lexicons in `signals.py`
  (intent verbs, vulnerability markers, confusion markers, resolution
  cues, ...) are deliberately small English word lists. PoQ
  brightness and Cambium pattern-detection both ride on these
  signals, so on a chain conducted in another language the
  detectors fire weakly and erratically — the architecture handles
  multilingual content fine (the cryptography and storage don't
  care), but the *behavioral* signals depend on word-list hits.
  Extending to additional languages means swapping or supplementing
  the per-language lexicon constants near the top of `signals.py`;
  the detector functions themselves don't need changes.

- **Single-operator security model.** The webapp assumes you bind it
  to `127.0.0.1` and that exactly one human at a time is the
  legitimate operator. There is no multi-user auth, no per-user
  permissions, no transport encryption built in. `/api/session/claim`
  is rate-limited but a co-located attacker could still bump your
  tab off the chain. For exposure beyond localhost, put it behind a
  reverse proxy with real auth.

- **Cambium scan windows are bounded by default.** The periodic
  Cambium scan uses an incremental watermark plus a `MAX_CAMBIUM_RECORDS`
  lookback (see `run.py`). This makes scans cheap and bounded, but
  patterns whose recurrence gap exceeds the lookback can only be
  detected by an explicit `/cambium-full` deep scan, which is linear
  in chain length.

- **Provider-specific truncation reporting.** Claude, GPT, and
  DeepSeek surface a finish reason that `was_truncated()` reads;
  Gemini and Ollama now report it as well. Other providers
  back-plumbed through OpenAI-compatible endpoints may or may not
  set the field — the truncation marker silently doesn't fire if
  they don't.

- **English-style topic signatures.** Cambium's recurrence detection
  groups records by a coarse keyword fingerprint extracted from the
  text. The keyword extractor is the same English lexicon machinery
  as above; recurring patterns in another language will not cluster
  reliably under the same signature.

- **Response records under-retrieve relative to their importance.**
  `response` records commit at a flat default salience (`0.40`), so in
  the salience-weighted retrieval blend they rank below reflections,
  principles, and genesis even when the response was load-bearing — a
  substantive explanation, or a short forward commitment the agent must
  stay consistent with later. Two mechanisms mitigate this: **semantic
  relevance dominates selection** (weight `0.55` vs `0.25` for salience),
  so a response that's relevant to the query still surfaces; and
  **observation/response turn-pair stitching** pulls a response in with its
  observation (and vice versa) so a retrieved half brings its partner. An
  earlier `artifact_content` *salience boost* (lifting code/structured
  responses toward `ARTIFACT_SALIENCE_MAX`) was **removed** — it was a
  query-independent size/type proxy that biased the salience-pure budget
  truncation toward long code records regardless of relevance. Making
  response salience *earned by substance* (not by code-ness) remains the
  most promising future direction.

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
retrieval semantics, the tiered embedder fallback and dimension-mismatch
guard, chunked-embedding behavior (boundary splitting, group-collapse, and
buried-content retrieval), per-record `modalities_activated` (write/read
round-trip and the legacy-defaults-to-empty path), content-aware response
salience (artifact detection, the boost/demotion composition, and the
end-to-end boost on a code response), modality-anchored retrieval (query
modality detection, overlap scoring, and the opt-in guarantee that anchoring
off matches historical scoring), runtime sprouted modalities (schema
validation, ReDoS-pattern rejection, load/save, and live anchoring on a
sprouted mode), the anti-echo saturation damper and per-turn modality cap,
auto-sprouting (the recurring-output-mode detector, the diversity gate's
pass/fail paths, deterministic pattern derivation, tentative staging, and
cooling-off graduation), per-record `senses_activated` (the round-trip
through `_meta`, `injection_scan` exclusion, end-to-end recording on a
turn, and per-detector discrimination across all six new senses),
chunk-aware rendering of long file records (intent detection across
inflected verb forms, the short/holistic/no-match fall-through paths to
full text, the excerpt's matched/context labeling, and verified ~70%
budget savings on a 66k-char document with a targeted query), full
agent workflow integrity, genesis drift detection, context-budget
truncation, and time formatting. **310 tests pass** (3 skipped — they
need the optional webapp dependency set).

The v1.3 cognitive faculties (see below) ship two additional standalone test
suites that run without pytest or numpy where possible:

```bash
python test_cypher_port.py          # 109 tests — the ported modules (numpy-free)
python test_cypher_integration.py   # 30 tests — agent loop wiring (needs numpy)
```

They cover the PoQ verdict ladder, immune screen/scan/lockdown/rollback/scar,
Continuum chunking + state refresh + validate, Recall index/fetch/verify-source,
Chronosynaptic collapse, Consensus quorum (including a forged-record-fails check),
faculty growth/promotion, the proof-of-work nonce, the historic-chain `migrate`
backfill, and the agent screen/verdict/seam integration.

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
useful in practice. It deliberately omits components of the original in
favor of a minimal, testable foundation. The goal here is engineering
tractability, not theoretical completeness.

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

Prototype, version 1.3.0. Works end-to-end. Not production-hardened. Issues
and PRs welcome.
