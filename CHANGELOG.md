# Changelog

All notable changes to the Timechain Agent, newest first. README.md and
ARCHITECTURE.md describe the system as it currently stands; this file is the
record of how it got there.

The cryptographic core — the append-only signed chain, Ed25519 signatures,
SHA-256 linkage, and Merkle batching — has been unchanged since v1. Every
release since reads older chains without modification: old records get
sensible defaults synthesized in memory at read time, never rewritten on
disk. Append-only means append-only, including for schema migrations.

---

## v1.5.0 — 2026-06-13

### Changed — default Anthropic model is now Opus 4.8 (Fable 5 withdrawn)

On 2026-06-12 a US government export-control directive forced Anthropic to
suspend all public access to Claude Fable 5 and Mythos 5, so the default
model in `make_claude_client` moves `claude-fable-5` → `claude-opus-4-8`.
This is not a one-line swap: the client OMITTED the `thinking` parameter
because Fable 5's thinking is always on. On Opus 4.8, omitting `thinking`
turns it OFF — and with thinking off, Opus 4.8 tends to write its reasoning
into the visible answer, which this agent would then seal into the chain as
the response text. So `thinking={"type": "adaptive"}` is now set explicitly
in both request paths (the streaming path yields only text deltas, so
thinking blocks never reach the answer). Sampling parameters stay omitted
(removed on Opus 4.7/4.8 as on Fable 5). When Fable 5 access returns the
swap is trivially reversible — the docstring still names it as the
most-capable option.

### Changed — reflections are a retrieved minority, and the cadence carries across sessions

Reflections consolidate what mattered, but at the old every-10-turns cadence
with high salience (0.85) and a 180-day half-life they came to dominate
retrieval — burying the actual turns they summarize. Retuned so a reflection
is an orienting minority retrieved alongside real turns, not the majority:

- `AUTO_REFLECT_EVERY` 10 → 100. At this cadence a per-session counter would
  almost never fire (few sessions run 100 turns), so the counter is now
  CHAIN-DERIVED: `Agent.turns_since_reflection()` counts response turns since
  the last reflection ring, seeded at startup in both the REPL and the web
  app. The cadence is measured across sessions, not per-process.
- Reflection half-life 180 → 75 days. Salience stays at 0.85 on purpose: at
  the low cadence the single orienting reflection SHOULD reliably surface;
  the shorter half-life is what keeps old reflections from accumulating as
  standing retrieval magnets rather than recent-context ones.
- `reflect()` lookback cap 200 → 300 records (`MAX_REFLECT_RECORDS`, a named
  constant in `run.py`), restoring comfortable headroom over the 100-turn gap
  (a normal gap is ~100-120 records). The span logic is unchanged — a
  reflection still covers exactly the slice since the previous one.

### Changed — Cambium's cadence also carries across sessions

`Agent.turns_since_cambium()` derives the auto-Cambium trigger from the
persistent scan watermark (`cambium.last_scanned_idx`) rather than a
per-session counter, seeded at startup like the reflection counter. Cambium
never MISSED records either way (its scan is watermark-based and always
sweeps everything since the last scan), but at every-30-turns a per-session
counter rarely fired in short sessions; the cadence is now reliable. The
scan itself is unchanged.

### Changed — system prompt slimmed; tool mechanics moved to tool descriptions

The tool-safety prompt was ~60% of the system prompt, crowding the
personality section. Compressed ~44% (3732 → 2072 chars) by relocating
interface mechanics (the approval card vs inline prompt, `/approve <id>`)
into the `write_file` tool description — where they sit next to the tool —
and keeping only cross-cutting policy in the prompt. Every safety one-liner
is preserved: don't retry on `confirmation_required`, chat text can't
approve, the no-delete honesty guard, read-before-write. The
epistemic-taxonomy and `imported_capsule` notes in the personality section
were lightly compressed; the voice / honesty / push-back core is untouched.
Combined prompt 6213 → 4139 chars; personality share 40% → 50%. Both the
REPL and web UI assemble the prompt from `run.py`, so the change reaches
both.

### Changed — `task_open` auto-fills name and objective

Ingesting a repo no longer requires a name and objective up front — only
`source_root`. When `name`/`objective` are omitted, `execute_task_open`
derives the name from the source directory (`derive_task_slug`, with the
same collision-suffix loop the workspace auto-mint path uses) and defaults
the objective to `"Audit of <root>"`. An explicit name still errors on a
registry collision (old behavior preserved). Re-opening the same
`source_root` without an explicit name now REUSES the existing active task
instead of minting a `<slug>-2` duplicate (and does not re-ingest), mirroring
the workspace auto-mint path — so "ingest this repo" followed by "review this
repo" stays one chain; pass an explicit name to force a separate one.
`source_root` stays required — it's the one field that can't be guessed and
the security boundary that becomes a readable/ingestable root.

### Added — the audit dashboard shows the request behind each response

Response rings carry the user's input as `content.context` (the single-record
turn shape), but the audit detail pane rendered only the answer — and
`block_text` actually concatenated the response and the request into one
run-on blob. The detail pane now renders the pure response from
`content.text` and surfaces the request in its own labeled "request
(context)" section (left-accent quote styling). Frontend-only; the API
already returned the context.

### Added — deep-think routing (chronosynaptic in the loop, Phase A)

The skill's chronosynaptic division of labor, made the agent's reflex:
the MODEL forks perspectives of itself within its own response, judges
them, and `think_collapse` (already wired) seals the winner. No new
machinery, no extra LLM calls — the missing piece was routing:

- `tools_prompt` gains a DEEP THINK paragraph (sibling to SECOND-LOOK
  MEMORY): for hard, high-stakes, or genuinely ambiguous questions,
  name 3-5 distinct lenses, reason each to its own conclusion, score
  each on the six PoQ dimensions, call `think_collapse` with all of
  them, and build the answer on the sealed winner. Routine questions
  explicitly skip this.
- `think_collapse` now drains the sealed synthesis record's hash into
  the turn's response refs (the `recalled_refs` channel recall_fetch
  already uses) and returns `sealed_record` — the chain records that
  the answer rests on the collapse, with rejected forks preserved in
  the synthesis payload.

### Changed — retrieval no longer pays an O(chain) rebuild every turn

- `EmbeddingIndex.index_record` now appends the new record's chunk
  vectors to the live in-memory search matrix instead of invalidating
  it. Before, every indexed record threw the matrix away and the next
  search rebuilt it from SQLite — decode every stored vector, stack,
  normalize — O(whole chain) work to add ONE record, paid every turn
  because every turn writes records. Search results are identical (same
  matrix content, same cosine math); re-indexing an EXISTING record
  still takes the full-rebuild path, since its old rows are baked into
  the matrix. Measured at 50k chunks: per-turn write+search 424 ms →
  118 ms.

### Added — `bench_retrieval.py`, the retrieval latency tripwire

- Permanent harness: synthetic chains at 1k/10k/50k (or sizes you
  pass), reporting per-turn and warm-search latency. The decision rule
  it encodes: when WARM search crosses ~200 ms at the real chain size,
  that is the measured signal to build a shortlist pre-filter (FTS5 or
  a real ANN) in front of brute-force cosine — and not before, because
  every pre-filter trades recall quality for speed. Current numbers:
  65 ms warm at 50k chunks; the identity chain grows tens of records a
  day, so the tripwire is years out unless a huge ingest lands.

### Added — second-look memory (identity recall tools)

Automatic retrieval stays the baseline — deterministic, every turn, any
model. On top of it, the model can now PULL what that pass missed,
mid-turn, with its own understanding as the relevance judge (the skill's
`recall index`/`fetch` shape riding this repo's guarantees):

- `recall_index(query?)` — a bounded map of the identity chain, one line
  per record (index, type, age, salience, who-said-what snippet). With a
  query, the existing hybrid scorer shortlists ~50 candidates — the
  pre-filter, never the arbiter; the most recent records are always
  included. Quarantined records are invisible, period.
- `recall_fetch(indices)` — full records (<= 12 per call, `_meta`
  stripped, fetch budget under the tool-result cap). Superseded records
  arrive WITH their corrections (the revision pull-in covenant), and
  every fetched record is REF'D by the turn's sealed response — the
  chain records what informed the answer, mid-turn pulls included
  (`AgentContext.recalled_refs`, drained at commit, reset at turn start).
- The tools prompt teaches the reflex: if the user refers to something
  not in context, recall before guessing and before claiming to not
  remember; honest uncertainty only after an empty recall.

### The turn-model work 

The turn-model release: the user's decision moves INTO the turn (mid-turn
approval), the turn moves into ONE record (the skill shape: input, answer,
resolutions, and attachments in a single signed unit), recursive ingests
are volume-gated with real numbers (ask, never block — including streaming
and extractor-aware walking), and `write_file` can generate real .docx
documents from the model's markdown. Plus the fixes from an external
post-release review of v1.4.0 (six findings; five accepted, one declined —
at the end of this section). The cryptographic core is untouched; old
chains — observation/response pairs, standalone attachment records,
resolution records — read, render, stitch, and verify forever. **519
pytest tests pass; 137 standalone tests (106 port + 31 integration);
`python3 selftest.py` exits 0.**

### Added — real .docx output (generated formats)

`write_file` on a `.docx` path now produces a REAL Word document — a
markdown LOI is useless to an attorney; this isn't that.

- The model authors MARKDOWN (headings, bold/italic, bullet and numbered
  lists); `docx_writer.py` converts at PROPOSAL time — python-docx when
  installed (proper Heading/List styles), else a minimal stdlib
  WordprocessingML writer (zipfile + XML), so the feature needs zero new
  required dependencies and the extractors can read back what either
  backend writes.
- The pending op carries the generated BYTES (base64, ≤ 1 MB) AND the
  markdown source: every approval surface shows readable prose, never
  base64, while `proposed_content_hash` pins the binary — the whole
  crash-recovery machine (TOCTOU check, tmp verification, disk-vs-hash
  audit) works on the real bytes unchanged.
- On approval the task chain seals the SOURCE markdown (searchable —
  retrieval finds "extension notice period" later) with the binary file's
  hash recorded; `task_ingest` grew `text_override` for exactly this.
- `GENERATED_FORMATS` is a map keyed by extension — `.xlsx` etc. can join
  later without touching the write gate again.

### Added — runaway-ingest protection (ask, never block)

Opening a task on `~/` by accident no longer silently walks the world into
a chain — and deliberately ingesting something huge still works, one
confirmation away. Nothing is ever skipped, trimmed, or refused on size.

- **Volume gate.** A bounded pre-walk survey (early-exit at the threshold,
  skip-dirs pruned from descent) runs before `task_open` auto-ingest and
  `task_ingest_path`. Crossing `WALK_CONFIRM_MAX_FILES` (1,000) or
  `WALK_CONFIRM_MAX_BYTES` (1 GB) routes the call through the existing
  confirmation machinery — the approval card / REPL prompt shows the
  surveyed numbers ("1,000+ files / ~1.2+ GB under /home/james") and an
  approve runs the FULL walk untouched. Fires even inside the workspace,
  closing the workspace-is-`$HOME` hole the boundary gate cannot see.
  Every gated call now carries a human-readable reason
  (`ctx.last_gate_reason`) shown on all confirmation surfaces.
- **Streaming reads.** The walk no longer preloads every file's text (the
  whole tree used to sit in RAM before sealing began), and a file over
  `STREAM_FILE_BYTES` (8 MB) is streamed through the chunker in two passes
  — identical sealed blocks (same boundaries, line numbers, hashes; the
  list chunker is now a `list()` over the streaming one, so they cannot
  drift), peak memory of one chunk instead of one file. Trade documented:
  streamed redaction is per-chunk, so a secret spanning a chunk boundary
  can be missed.
- **Extractor-aware ingestion.** Walked trees and `task_ingest_file` now
  route document formats (`.pdf`, `.docx`, `.xlsx`, `.pptx`, `.dotx`)
  through the same `extractors.py` the upload path uses — a directory of
  agreements seals searchable prose instead of binary noise. A file with
  no extractable text is skipped VISIBLY (0 blocks in the walk summary,
  a loud error on single-file ingest), never silently.

### Changed — one record per turn 

The identity chain now seals ONE response record per turn, carrying the
whole exchange: the user's input as `content.context` (exactly the skill's
`payload.context`), the answer as `content.text`, mid-turn approval
decisions as `content.resolutions`, and upload pointers as
`content.attachments`. Chain growth and embedding count are halved, and 
a retrieval hit always carries the full Q&A. Append-only means append-only: 
old observation/response pairs and standalone attachment records read, 
render, and stitch forever — only NEW turns stop minting them.

- **No observation records.** `prepare_turn` no longer touches the chain
  before the LLM call, so the self-retrieval bug the old
  commit-then-defer-indexing ordering guarded against is now structurally
  impossible, and a stranded observation (user message with no reply) can
  no longer exist. The accepted trade: a hard crash mid-turn loses that
  turn's input from the chain — the same durability property the skill has.
- **Turn-pair stitching is legacy-only.** A context-bearing response IS the
  Q&A unit; retrieval skips partner pull-in for it. Old chains keep the
  full stitching machinery (index±1 partner, refs corroboration, pinning).
- **Attachments fold into the turn.** Upload content still goes to the
  artifacts chain + blob store immediately (durable); the identity-chain
  pointer is now STAGED (`AgentContext.staged_attachments`, persisted to
  `staged_attachments.json` across restarts) and seals into the next turn's
  response record. The staged upload is handed to that turn's prompt
  directly — a note naming each file plus native image/PDF payloads — so
  upload visibility is deterministic instead of retrieval luck (the
  invisible-upload bug class, fixed by construction). A model-initiated
  `ingest_blob` mid-turn folds into the same turn via a late drain.
- **Refused turns seal one quarantined record** (hostile input as
  `content.context` + the refusal), keeping the wound off every prompt.
- The blob index, `build_attachment` (prefix resolution included),
  `serve_blob`, multimodal attachment collection, reflection history
  formatting, and the continue-after-truncation directive all cover both
  shapes; the web UI renders a context-bearing response as YOU + AGENT
  bubbles (plus attachment chips) from the single record.

### Added — mid-turn approval (the bounded agentic loop)

The turn model changes: a pending operation now pauses the turn it was
proposed in, and the user's decision happens DURING the turn, not optionally
after it — the Claude Code / Codex approval model. All pending requests are
resolved by the time a turn ends; nothing is left lingering.

- **The turn pauses on every pending op.** In the web/SSE loop, a
  `write_file` proposal or a deferred confirmation-gated tool call parks the
  turn on an `asyncio` future (`state.approval_waiters`); the approve/reject
  endpoints — and the `/approve`/`/reject` chat commands — deliver the
  decision to the parked turn LOCK-FREE (the turn holds `state.lock`, so the
  legacy locked path would deadlock) and the turn executes the resolution
  under the lock it already owns, emits `op_resolved`, and continues. In the
  REPL, `agent.turn_with_tools` takes an `approval_hook` and prompts inline.
- **The model sees the real outcome.** The tool result fed back is
  "Written … Audit: clean." / "Write to … rejected." / "EXPIRED: …" instead
  of a dangling `confirmation_required` proposal, so the model can adjust
  within its remaining rounds (deny feeds back; it never dead-ends the turn).
- **Resolutions live in the response record.** The user's decisions embed in
  `content.resolutions` on the turn's own response record — the record
  captures the full arc (proposed → decided → outcome) as one signed unit.
  In-turn decisions no longer seal separate `resolution` records; those
  remain only for out-of-band resolutions (crash recovery, ops from before
  this change). Old chains read unchanged — append-only means append-only.
- **No decision auto-expires.** If the user walks away, the gate times out
  after the op's TTL (300s), the op is discarded, and the expiry is recorded
  in the response block — "never left lingering" holds even for absence.
- `pending_ops.resolve_inline()` is the one mid-turn resolution path (both
  loops call it); the user-action entrypoints grew a `seal_resolution` flag
  (default True) so the out-of-band contract is unchanged.
- `GET /api/pending-ops` no longer takes the chain lock: a paused turn holds
  it, and this endpoint is how the UI renders the approve button — reads are
  safe lock-free (the store's writes are atomic `tmp + os.replace`).

---

### From the external review of v1.4.0

Six findings; five accepted, one declined. Best-effort failures that were
deliberately non-fatal are no longer invisible, one latent constant-drift
bug is gone, and the pending-ops store no longer accumulates abandoned
state. No behavior change on any success path.

### Changed — silent degradation now warns

- `agent.py`, `tools.py`, and `pending_ops.py` each carry a module logger;
  the best-effort catch sites that used to swallow failures silently now
  emit `log.warning(...)` with the consequence spelled out: consensus
  attestation (sealed but not co-signed), sprout registry save (in-memory
  state will not survive restart), resolution sealing (decision executed but
  not on the identity chain), workspace-task minting (write proceeds but
  provenance ingest is skipped), and every embedding site (block sealed but
  not searchable — each warning names `task_reembed` as the repair). The
  failure-handling behavior itself is unchanged: nothing new fails the turn;
  it just stops failing invisibly.

### Fixed

- `agent.py` imported `CHUNK_TARGET_CHARS` lazily inside `_truncate_to_budget`
  with a hard-coded 3500 fallback, despite `retrieval` already being a
  module-level import — the fallback could silently diverge if the constant
  changed in `retrieval.py`. Now imported at module level; the try/except and
  magic number are gone.
- `PendingOpStore` never removed expired `pending` ops whose approval simply
  never arrived — expiry only fired if someone later called approve/reject on
  that exact id, so abandoned proposals (and their 0600 content files)
  accumulated forever. The store now runs `sweep_expired()` on construction:
  it deletes only expired ops still in `pending` (plus any tmp file of
  theirs, defensively). Partially-executed states (`writing`, `written`,
  `ingest_failed`) are never touched — they recover regardless of TTL, and a
  `writing` op's tmp file is exactly what crash recovery completes the
  `os.replace` from. The expire and reject resolution paths also discard the
  op's tmp file as a backstop against future state-machine changes.

### Documented

- The two-tier confirmation policy is now stated next to `CONFIRM_TOOLS`:
  `task_ingest_path` is deliberately ungated per-call because its volumetric
  risk is confirmed once at root-grant time (a `task_open` that expands the
  allowed roots), while `task_ingest_file` stays gated per-call because it
  can surgically target any single file within those roots — including ones
  a walk's extension filter would never touch.
- `MATCH_INPUT_CAP` (sprouted modalities) now documents its sharp edge: the
  cap is a hard head-truncation, not a windowed sample, so a modality whose
  distinctive vocabulary appears only after the first 20k chars of an input
  will never fire on it.

### Declined

- Duplicating `task_retrieve`'s `max_blocks` default into the tools prompt
  prose: the JSON schema the model reads already states the default, and a
  second copy is exactly the constant-drift bug class fixed above.

---

## v1.4.0

The code-working agent release: the agent can now read, write, and audit
code through a text-parsed tool-calling loop — in the REPL and the web UI —
with per-task continuum chains as durable memory, a three-tier safety model,
self-defense wired into the automatic turn loop, and continuum-based content
ingestion with dedicated artifacts routing. The whole batch was hardened by
a multi-angle pre-release review (19 confirmed findings, all fixed before
release — summarized at the end of this section). The cryptographic core is
untouched; old chains read and verify unchanged. **490 pytest tests pass;
136 standalone tests (106 port + 30 integration); `python3 selftest.py`
exits 0.**

### Added — tool-calling agent

- `tools.py` — tool schemas + executors + the SINGLE shared text-tool driver:
  tolerant extractor (fences, trailing commas, multiple blocks), strict
  JSON-Schema validator (unknown tools/params/types rejected), result escaper
  (`<tool_call>` in file content can never forge a call), 64KB result cap.
  `AgentContext` carries the registry, lazy per-task chains/recalls/
  continuums/embedding-indexes, and session state (`active_task`,
  `pinned_path` — reset every turn).
- `agent.turn_with_tools()` — same immune/PoQ/commit discipline as `turn()`
  (shared `_finish_turn` tail, so the gates cannot drift), plus a bounded
  parse→execute→re-call loop with ONE reflective retry on malformed calls
  and sanitized `tool_use` audit records (hashes, never content).
- `task_registry.py` — slug-validated, atomically-saved task registry with
  `repair()`; `resolve_task()` returns exact/ambiguous/not-found and NEVER
  auto-selects a fuzzy match (Safety Tier 1).

### Added — durable write gate

- `pending_ops.py` — `write_file` never writes; it creates a 0600-permission
  PendingOperation (1MB cap, 300s TTL on the pending state only). The USER
  approves via `/approve` (REPL) or the web endpoints below — never the
  model. Approval verifies the pre-write hash (TOCTOU), writes atomically
  (tmp + `os.replace`, mode preserved), ingests into the task chain
  idempotently (`operation_id` + `continuum.find_by_operation_id`), audits
  the ingested block against live source, and deletes the pending-op file.
  Crash recovery resumes deterministically from `writing`/`written`/
  `ingest_failed`.

### Added — task-chain recall

- `Recall.from_chain_db()` / `from_task_root()` — recall against any task chain.
- `Recall.retrieve_path_aware()` — the task-chain ARBITER (the identity-chain
  `retrieve()` pre-filter is unchanged): blended semantic + path-proximity +
  chronological scoring, hard role/language/ext/top-dir/exclude-dir filters,
  test/docs/vendor noise penalty, neighbor chunks. `Recall.find_by_path()`
  for path-based audits.
- `continuum.walk()` returns a `WalkResult` (files, results, sealed, state) —
  still unpackable as the legacy `(files, results)` tuple — so per-task
  `EmbeddingIndex` population is walk → seal → `index_record`. The dead
  `embed=` parameter was removed.

### REPL and Web UI tool loop

- Default turn path is `turn_with_tools` (set `TOOLS_ENABLED = False` in
  run.py to revert). New commands: `/task list|open|ingest|resume|validate|
  audit`, `/approve <id>`, `/reject <id>`, `/pending`. Tools in
  `CONFIRM_TOOLS` (e.g. `task_ingest_file`) prompt the operator inline in
  the REPL and are refused over SSE (the safe headless default).
- `/api/turn/stream` runs the async twin of `Agent.turn_with_tools`:
  per-round token streaming, `tool_result` SSE events, ONE reflective retry,
  10-round cap. Parsing/validation/execution/escaping all come from tools.py
  (the single shared driver) so the safety tiers cannot drift between REPL
  and web. `/api/turn` (non-streaming) uses `turn_with_tools` directly; the
  webapp boots a `tools.AgentContext` mirroring run.py.
- Pending-write endpoints: `GET /api/pending-ops`,
  `POST /api/pending-ops/{id}/approve|reject` — user-triggered only. A
  `write_file` during a stream emits a `pending_op` SSE event; the frontend
  renders an approve/reject card showing exactly what would be written.
- `think_collapse` tool wraps the existing
  `ChronosynapticTree.collapse_explicit_notes` — the model supplies
  perspectives + PoQ scores, the winner is sealed, rejected forks preserved.

### Self-defense loop integration

- `Quorum.is_initialized()` (cfg-only check — requiring attestations.jsonl
  too would deadlock auto-attest). `Agent` holds a `consensus` handle by
  default and auto-attests every `commit_response` once a quorum is
  initialized; attestation failure never blocks a turn. Opt out with
  `enable_consensus=False`.
- `immune.rollback(grow_antibody=True, faculty_dir=…)` offers the scar
  vector to `FacultyGarden.grow(kind_override="sense")` — the REPL
  `/rollback` does this automatically and reports the antibody.
- `task_audit_source` accepts a `path` (audits every block of that file via
  `recall.find_by_path`) as well as `block_index`.
- New read-only `defense_status` tool: chain integrity, immune posture,
  quorum health, antibody count.

### Changed — content ingestion rebuilt (file_ingest.py → ingest_blob + extractors.py)

- `file_ingest.py` and the `/file` command are gone; `chain.py`'s
  blob_index stays — old chains with `file` records read fine, and legacy
  flat-layout blobs (`blobs/<sha>`) still serve via the shared
  `tools.resolve_blob_path` (which also knows the new sharded layout).
- The replacement is the `ingest_blob` tool (utf8 or base64, 8MB cap,
  traversal-safe names). Format-aware text extraction lives in
  `extractors.py` (pdf/docx/xlsx/pptx/csv/tsv + image metadata + encoding
  detection); every format library is optional (`pip install
  "timechain-agent[ingest]"`) and degrades to a placeholder when missing.
  Ingesting never creates a NORMAL task chain — the reserved artifacts
  chain (below) is the one lazy exception.
- Multimodal attachments still reach the LLM: `Agent._collect_attachments`
  (sha-addressed, LRU-cached) ships image/PDF bytes from the blob store to
  vision-capable providers for any `file`/`attachment` record in retrieval
  context.
- `continuum.ingest()` passes custom metadata keys through to the sealed
  block (mime_type, workspace_path, source, approx_bytes).
- The webapp upload rides the new path: `POST /api/upload` routes through
  `execute_ingest_blob`; `GET /blobs/{sha}` serves identity blobs with the
  recorded MIME type — session-gated like the rest of the API (`?session=`
  query parameter, since `<img src>` can't send headers), with the MIME
  lookup indexed via `blob_index`, which is now maintained for `attachment`
  records as well as `file` records. The attach button and drag-drop still
  work.

### Added — user-selected workspace + lazy task chains

The working directory is now the USER's pick, not wherever the process
happened to start (inspired by the Claude Code / Codex working-directory
selectors). Selection is user-only — the model has no tool for it — so a
chosen workspace is inherently confirmed: reads, writes, and task_opens
inside it run without confirmation ceremony.

- `tools.set_workspace` / `restore_workspace`: validate the directory
  server-side, reset the active task and pin (a task bound to the old
  boundary must not absorb work from the new one), persist the choice in
  `<DATA_DIR>/workspace.json` (with a recents list) so restarts keep it.
- Web: `GET /api/workspace` (current + suggestions — task source roots
  and recents, never a directory listing, so the session cannot enumerate
  the filesystem) and user-only `POST /api/workspace`; a folder chip above
  the composer shows the current workspace and opens a path field with
  suggestions. REPL: `/workspace [path]`.
- The per-turn system prompt now carries `Current workspace: <path>` in
  both loops, so the model knows its repo instead of guessing at
  ~-expansions.
- **Switching creates nothing.** A task chain appears lazily at the first
  action needing durable task state: `tools.ensure_workspace_task` (called
  by `write_file` when no task is active) mints a chain named after the
  workspace directory (slug-derived, collision-suffixed), reusing any
  active task already bound to that root. Read-only poking around and
  unrelated questions leave zero registry state — code-enforced, not
  model-judged.

### Fixed — strip_tool_markup no longer eats prose that mentions the tags

When the agent audits THIS codebase it writes prose about the tool-call
machinery — "a forged `` `<tool_call>` `` in any text" — and the
destructive tail-strip (which suppresses an unclosed tag to end-of-segment
to stop half-file echoes) treated that inline mention as a real opener and
deleted the rest of the sentence. `strip_tool_markup` and the live JS
mirror now mask inline-code spans (`` `...` ``) before stripping: a tag
inside backticks is always prose (real calls and echoed results are raw,
never backtick-fenced), so it survives, while bare structural markup is
still removed. (Note: this addressed the visible mangling in the report;
the underlying repetition loop in that turn was DeepSeek output
degeneration on a large self-referential prompt — a model failure mode,
not fixed here.)

### Fixed — a mid-turn stream failure no longer strands the observation

A stream error on any tool round made the web loop `return` immediately —
but the user observation was already sealed at turn start, so the chain
was left with a user message and NO response. The next turn couldn't
retrieve a reply that was never committed ("I don't see the prior turn"),
and the turn froze mid-stream with nothing written. The larger context
budget + batched-read guidance made later-round prompts big enough to
trip provider timeouts, so this surfaced on long review turns. Now a
stream failure (first round OR mid-loop) falls through to the commit
path: the prose from completed rounds is sealed, paired with the
observation, with an honest `[stream error] this turn was cut short`
note (also streamed to the browser). No more orphaned observations.

### Changed — skill-style identity chain: two rings per turn + resolutions

The identity chain is now a low-noise stream of experience, matching the
cypher-tempre skill's design:

- **No per-call `tool_use` records.** A 33-call turn used to seal 35
  rings and evict the entire conversation from the 15-record recent
  window (pure recency, no type weighting). Now: ONE observation + ONE
  response per turn — the response is the self-written, PoQ-gated
  narrative of the turn's work (prose accumulation already carries it),
  and tool EFFECTS live on the per-task continuum chains as ingest
  blocks. `_log_tool_use` / `sanitize_audit` / `TOOL_AUDIT_FIELDS`
  removed; old chains containing `tool_use` records still read fine
  (metadata defaults kept). The tools prompt now tells the model its
  final answer is the only memory of the turn's work — make it
  self-contained.
- **Approval outcomes join the stream.** The user approving or rejecting
  a pending operation seals a `resolution` record on the identity chain
  (source=user, epistemic=known, salience 0.60): outcome, op id, kind,
  file/tool, bounded result. Approval was an out-of-band event the model
  never witnessed — its last sealed knowledge said "pending" forever,
  which is exactly the gap behind the field confabulation ("never
  approved, so never ingested"). Best-effort: a sealing failure never
  undoes an executed decision.

### Added — task chains in the audit dashboard + honesty guidance

- `/api/audit` and `/api/audit/ring/{idx}` accept `?task=<name>` to read
  a per-task continuum chain instead of the identity chain; the snapshot
  always lists available task chains, and the audit page gained a chain
  selector (identity / each task with progress + status) — so "is that
  file actually in the task chain?" is answerable by looking, block by
  block, signatures and all.
- New TASK-CHAIN STATE prompt guidance (after a field confabulation: the
  model claimed it deleted a file — it has no delete tool — and that an
  approved write "was never ingested"): never assert chain state from
  memory; check task_audit_source / task_validate first; approved writes
  are ingested by the approval itself; never claim to have deleted
  anything.

### Changed — one approval surface, bigger workspace chip, stray closers

- The in-transcript pending-op card is gone: two approve prompts for one
  operation (card + banner) read as two decisions. The banner above the
  composer is the single approval surface; the collapsed 🔧 tool_result
  card still carries the operation's details in the transcript.
- The workspace chip is ~1.7× bigger (16px text, larger padding/target),
  with the path editor scaled to match.
- `strip_tool_markup` (and the live JS mirror) also remove stray
  `</tool_call>` / `</tool_result>` closing tags — orphan echo fragments
  that rendered as literal markup lines in the agent bubble.

### Changed — budgets sized for long tasks on million-token models

Fewer turns for long tasks, by raising the right levers and not the
wrong one:

- `LLM_MAX_TOKENS` 4,096 → **16,384**: a ceiling, not a target — short
  replies cost the same, but audits/plans stop tripping the truncation
  flow, and `write_file` can propose document-sized files in one shot.
- `CONTEXT_BUDGET_CHARS` 150,000 → **400,000** (~100k tokens):
  deliberately NOT the full million-token window — retrieval stays
  selective (relevance realization, not context stuffing), ~100k tokens
  is where long-context quality holds, and inside a tool turn the prompt
  is re-sent every round, so context is paid once per round.
- The round cap stays at 24 — rounds are the quadratic resource (the
  loop re-sends the accumulated prompt each round). Instead rounds got
  FATTER: `tools_prompt` now instructs the model to BATCH independent
  reads/retrievals as multiple `<tool_call>` blocks per response, and
  `task_retrieve`'s default `max_blocks` went 8 → 16.

### Fixed — tool budget: bigger, final-round warning, visible cap notice

A plan-writing turn burned all 10 tool rounds exploring the repo, then
the model's 11th response — full of tool calls — was silently dropped:
no answer, no file, and in the web UI no explanation (the cap notice was
appended after the token stream ended). Three changes:

- `tools.DEFAULT_MAX_TOOL_ROUNDS = 24`, shared by both loops (the web
  loop's local `MAX_TOOL_ROUNDS = 10` is now a module-level constant
  reading the shared default, overridable for tests/operators).
- **Final-round nudge** (`tools.TOOL_BUDGET_NUDGE`): when the budget is
  spent, the prompt tells the model its next response is the last and
  must be the answer — so a capped turn ends with a real wrap-up from
  what it already read, not a batch of dropped calls.
- The web loop now **streams** the "[tool loop] Stopped after N tool
  rounds." notice as a token event, so the browser shows why the turn
  ended instead of just stopping.
- **"continue" resumes a capped task** (mirroring the max_tokens
  truncation flow): the nudge now asks for the answer only if completable
  — otherwise a progress checkpoint (done / remaining / next step) — and
  the sealed response carries `tool_budget_exhausted: true` in `_meta`
  (same absent-unless-True rule as `truncated`). Both UIs show a "type
  'continue' to give it a fresh budget" notice, and `_format_prompt`
  recognizes a "continue" against a budget-exhausted response with a
  mid-TASK resume directive (re-orient from the checkpoint, go straight
  back to tools) — distinct from the mid-sentence truncation directive.
  Never auto-continues: each budget grant is the user's.

### Fixed — the approval flow is discoverable (no more dead /approve)

Field-tested write flow had three papercuts: the model asked "Proceed?"
(a question chat text can't answer), suggested `/approve <id>` which the
web chat rejected as an unknown command, and the approval card could sit
off-screen above the answer. Now:

- **`/approve <id>` and `/reject <id>` work in the web chat box** —
  routed through the same user-only path as the card buttons, so the
  model's instructions are true in both interfaces.
- **A pending-operations banner sits directly above the composer**
  whenever anything awaits approval — listed with approve/reject buttons,
  refreshed on pending_op events, at turn end, after any action, and at
  boot (an op left pending last session resurfaces). An approval can no
  longer hide off-screen.
- **The write guidance stops the model asking "Proceed?"**: write_file's
  card IS the question; the model now describes the pending change in one
  short message and waits. (The 20–30s "pause" was the extra LLM round
  the old ask-then-explain guidance produced.)

### Fixed — echoed tool results no longer stream at the user

Transcript-continuation models (DeepSeek) sometimes echo the
`<tool_result>` blocks they were fed — re-streaming entire files into the
visible answer. Three layers now stop that:

- `tools.strip_tool_markup` also removes `<tool_result>` blocks and any
  unclosed/truncated tool marker (suppressed to end-of-segment), and both
  loops now strip the FINAL round before commit — echoes reach neither
  the user nor the chain. The real results are unaffected: they render as
  collapsed 🔧 cards and seal as `tool_use` records.
- The web UI renders a live-filtered view of the streaming buffer
  (mirroring the server strip), holding back partially-streamed markers —
  file contents never flash on screen mid-stream.
- `tools_prompt` now tells the model never to write `<tool_result>`
  blocks or repeat their contents verbatim.

### Changed — live tool cards render above the answer

During a streaming turn, tool-result and pending-op cards used to append
below the agent's bubble, leaving the 🔧 cards stranded under the final
answer. They now anchor ABOVE the streaming bubble — the work shows
before the answer it produced, matching the order the chain seals
(tool_use records before the response record).

### Added — turn-progress indicator

A spinner-plus-elapsed-time line (`✻ working for 1m 23s`) renders below
the live agent bubble for the whole turn. The three-dot typing indicator
only covers the wait for the FIRST token; this one covers tool-call
pauses mid-turn, when streaming stops but the turn is in progress —
previously indistinguishable from a stall. It names the current activity
(`✻ running read_file for 12s`): there is no "tool started" event, but
the raw token buffer carries the display-stripped `<tool_call>` JSON, so
the client parses this round's tool names as they finish streaming and
each `tool_result` advances the label to the next call. Removed on
`done` or stream close.

### Fixed — uploads are fetchable again (full sha in the prompt)

The prompt rendered attachment/file hashes truncated to 12 chars
(`sha256 c884391d726e...`), but that rendered line is the model's ONLY
handle for `build_attachment(blob_sha256)` — so on a text-only provider
(DeepSeek ignores image attachments by design) an upload was visible but
unfetchable: "I can't retrieve that screenshot. The SHA-256 hash in the
record is truncated." Three-part fix: the prompt now renders the FULL
hash; pointer rings with no inline text say where the content lives
(artifact path + how to fetch); and `build_attachment` resolves a unique
hash prefix (8+ hex chars, via the new `Chain.find_file_by_sha_prefix`)
so hashes quoted from older truncated displays still work.

### Fixed — one task_open ring per task

`continuum.walk()` called `open_task()` unconditionally, so every
open-with-auto-ingest (and every later walk into the same task) sealed a
redundant `task_open` ring a second after the real one and RESET the
progress metrics. walk() now reuses an already-open task's state —
extending `items_total` cumulatively; every sealed block carries the
refresh — and only opens the task itself on a never-opened chain
(standalone continuum use).

### Changed — task_open ingests in the same call

Setting up a task chain used to cost two turns (open, then ingest).
`task_open` now walks and ingests the source tree in the same tool call —
default extensions `.py`/`.md`, overridable via `extensions`, skippable
via `ingest=false`. The walk shares `tools._walk_and_index` with
`task_ingest_path` (one implementation, so the embed-once discipline
cannot drift), an ingest failure leaves a usable open task with a clear
"run task_ingest_path manually" note, and `task_ingest_path` remains for
re-ingestion (`changed_only`) and ingesting additional trees.

### Fixed — tool-round prose is no longer discarded (the disappearing reply)

When a response mixed prose with a tool call ("Hello! Let me check…" +
`<tool_call>`), the web UI deliberately wiped the streamed bubble on
`tool_result` (to hide raw tool-call JSON), and BOTH tool loops committed
only the final round's text — so the visible reply vanished mid-turn and
the chain sealed an out-of-context fragment that reloaded the same way.

- New `tools.strip_tool_markup()`: a model segment's prose with its
  `<tool_call>` blocks removed (one shared implementation, loops can't
  drift).
- `agent.turn_with_tools` and the webapp loop accumulate each tool round's
  non-empty prose and commit it joined with the final answer — PoQ scores
  and the chain seals what the user actually saw. The prompt-side
  accumulation (what the model sees of its own rounds) is unchanged.
- `index.html` strips the markup from the live buffer on `tool_result`
  but KEEPS the prose, with later rounds streaming on after a paragraph
  break — text the user already read never disappears, and the raw-JSON
  problem the old clearing solved stays solved.

### Fixed — confirmation-gated tools are approvable from the web UI

In the web UI, any confirmation-gated call (a `task_open` outside the
workspace, `task_ingest_file`, `task_reembed`) was a dead end: the SSE loop
has no inline confirm hook, so the gate refused unconditionally — and no
chat phrasing could ever satisfy it, leaving the model to confabulate
remedies (`/approve` only applied to writes). Confirmation-gated calls now
defer instead of dying:

- `PendingOperation` grows `kind="tool_call"` (back-compatible defaults;
  old on-disk write ops load unchanged): the exact tool name + canonical
  arguments, pinned by hash, same TTL and 0600 discipline as writes.
  `tools.defer_tool_call()` creates the op and returns
  `status=confirmation_required` to the model so it explains the real flow
  instead of guessing.
- The web loop surfaces the existing `pending_op` SSE event for ANY
  confirmation-required result (writes and tool calls); the UI card shows
  the tool name + arguments with the same approve/reject buttons, and the
  `/api/pending-ops` endpoints handle both kinds — approval executes the
  call through the user-only path (`pending_ops._approve_tool_call`,
  single-shot: a crash mid-execution reads as already-processed rather
  than risking a double run of a non-idempotent tool). REPL `/approve
  <id>` works for tool-call ops too; the inline REPL prompt and headless
  refusal are unchanged.
- The system prompt's tool guidance now states that confirmation is an
  interface mechanism chat text cannot satisfy, and describes the
  per-interface flow — so the model stops inventing magic confirmation
  phrases.
- **Eager validation** (`tools.precheck_gated_call`): a gated call that
  cannot possibly succeed — `task_open` on a directory that doesn't exist
  on this machine, `task_ingest_file` on an unknown task / missing file /
  out-of-bounds path, `task_reembed` on an unknown task — errors back to
  the model immediately instead of minting a pending op the user can only
  approve into a failure. The executors still validate at approval time
  for everything a precheck can't rule out (e.g. the file vanishing
  between deferral and approve).

### Fixed — task ingest no longer blocks on slow embedding (the hour-long ingest)

A full-repo `task_ingest_path` that walked and sealed 399 blocks in 10
seconds then spent **2h13m** embedding them: every chunk went through a
CPU-bound Ollama `nomic-embed-text` at ~3–5s each, and every record was
embedded TWICE (the index was opened after the walk, so the first-open
backfill embedded the just-sealed records and the post-walk pass embedded
them again). Three changes, measured back to **~7s** end-to-end:

- **Task stores default to the instant `HashingEmbedder`** regardless of
  the session embedder (`AgentContext._task_embedder`). Character-trigram
  vectors are adequate for code retrieval — queries share identifiers with
  their targets — and `retrieve_path_aware` blends path/role/recency signals
  on top. The session embedder (e.g. Ollama) is opt-in per task, below.
- **The double-embed is structurally gone**: all ingest paths
  (`task_ingest_path`, `task_ingest`, task-scoped `ingest_blob`) open the
  task index BEFORE sealing new blocks; `execute_task_ingest_path` indexes
  via `index_chain` (which only touches records missing from the store), so
  the walk's out-of-band `task_open` state record is covered too — each
  record embedded exactly once.
- **`task_reembed` (new tool, Tier-3 confirmed)** rebuilds a task's derived
  store with the session embedder for true semantic recall, batched through
  the newer Ollama `/api/embed` endpoint (`OllamaEmbedder.embed_batch`, one
  request per 64 chunks — measured ~12% over sequential; the forward pass
  dominates on CPU, so this stays a deliberate, user-confirmed operation)
  with per-batch progress and no partial chunk sets on failure
  (`EmbeddingIndex.index_records_batched`). The choice persists as
  `task["embedder"] = "session"` in `tasks.json`
  (`TaskRegistry.set_embedder`), so later sessions reopen the store with
  the same embedder instead of mismatch-deleting an expensive re-embed.

### Added — unified selftest

- `selftest.py` — every mechanism end-to-end on a throwaway chain in ~2s:
  timechain, PoQ, faculties growth, continuum + cartography (redaction,
  changed-only), path-aware + embedding recall, chronosynaptic collapse,
  consensus (incl. is_initialized), immune screen/scan/lockdown/rollback
  (incl. antibody), resolve_task, the write gate (approve/reject/TOCTOU
  discipline), defense_status, and ingest_blob routing. Exit 0 = green.

### Hardening (security review)

- Reads are bounded like writes: `read_file`, `task_ingest_path`, and
  `task_ingest_file` resolve through the same allowed roots (workspace,
  task source roots, task workspaces) with symlink-escape protection, and
  refuse chain databases and key files — the model can no longer read
  `operator.key` or walk `/etc` into a chain.
- `POST /api/upload` enforces the ingest cap while reading the request,
  not after buffering the whole file.
- Pending-op failures are machine-detectable: every non-success return
  from approve/reject starts with `ERROR:`, the web endpoints report
  `ok: false` for them (expired / not-found / TOCTOU included), and the
  UI re-enables the approve/reject buttons so a recoverable failure can
  be retried.
- The SSE consumer never waits unboundedly on its producer thread: a
  bounded poll exits cleanly if the producer dies without an end marker,
  and the webapp tool-loop tests run under hard timeouts.

### Hardening (second security review)

- **task_open can no longer expand the filesystem boundary on its own.**
  A model-chosen `source_root` becomes an allowed read/ingest root, so
  `tools.requires_confirmation()` (the one Tier-3 policy both the REPL
  and web loops call) now gates any task_open whose source_root resolves
  outside the workspace: confirmed inline in the REPL, refused headless.
  Workspace-rooted task_opens run as before. `execute_task_open` also
  requires an existing directory and stores the symlink-resolved path,
  so the boundary the user confirmed is the boundary enforced.
- **Every chain-reading endpoint is session-gated.** `/api/chain/status`,
  `/api/chain/recent`, `/api/chain/records`, `/api/chain/sidebar`,
  `/api/audit`, and `/api/audit/ring/{idx}` now demand the active session
  token like the rest of the API (record contents are the agent's
  memory). `/api/chain/recent` clamps `n` to [1, 200] — SQLite treats a
  negative LIMIT as unlimited, so `?n=-1` used to return the whole chain.
  The audit dashboard reuses the main page's token via localStorage
  instead of claiming its own session (which would bump the chat tab).
- **A corrupt tasks.json fails loudly instead of vanishing.** Treating
  unparseable registry JSON as "no tasks" meant the next save overwrote
  it; TaskRegistry now refuses to load over a corrupt file, leaves the
  bytes in place, and says how to repair.
- **Git verdicts are never optimistic.** A failed `git status` no longer
  reads as a clean worktree, and when a record pinned a commit that the
  live side can't check (git missing/errored/not a work tree any more)
  `source_verify.verify_file_record` and `recall.verify_source` return
  `git-unverifiable` (with `content_match: true` when the hash matched)
  instead of `verified`.
- **The non-streaming web turn no longer parks the event loop.** The SSE
  generator was extracted to `_turn_events()`; `/api/turn` drains it and
  returns the legacy JSON shape, so both transports share one turn
  implementation and every LLM call runs in a worker thread while chain
  and index writes stay on the loop thread.

### Added — embedding-store rebuild policy

- `retrieval.open_or_rebuild_index()` — single shared open-or-rebuild path
  for embedding stores, used by run.py, the webapp, and per-task indexes
  (which now also backfill old blocks). A store built by the cheap lexical
  HashingEmbedder is rebuilt automatically when the embedder changes (the
  "Ollama got installed" upgrade path); a mismatched SEMANTIC store refuses
  to boot with instructions instead of silently destroying hours of
  embedding work (the active embedder may be a transient fallback — e.g.
  the Ollama daemon down at boot). `force_rebuild=True` is the explicit
  wipe `task_reembed` uses.

### Added — artifacts routing (uploads & pastes)

Uploaded and pasted content gets a home of its own: the reserved
`artifacts` task chain plus a user-browsable folder on disk. The
alternatives both fail in practice — sealing uploads into whatever task
happens to be active pollutes unrelated, append-only task chains, and
sealing full extracted text on the identity chain lets big documents crowd
retrieval and drown more relevant rings.

- `ingest_blob` (and `/api/upload`) defaults to the reserved
  `artifacts` task chain, lazily created on first upload:
  - **Bytes** → the content-addressed blob store (canonical: the vision
    path and `/blobs/<sha>` resolve by sha) **plus** a named, browsable
    copy in `ARTIFACTS_DIR` (env-configurable, default `~/.artifacts`;
    name collisions with different content get a short-sha suffix).
  - **Content** → chunked, self-labeled continuum blocks in the artifacts
    chain, embedded in its OWN store — artifact text never enters
    identity retrieval.
  - **Identity chain** → ONE tiny pointer ring per upload (filename,
    mime, sha, artifact ring refs — no extracted text), so the
    conversation shows the upload, the UI renders its card, and the
    pointer can surface in retrieval without crowding anything.
- `ARTIFACTS_DIR` is the artifacts task's source_root, so the named
  copies are readable through the normal path gates (`read_file`).
- `build_attachment` follows pointer rings into the artifacts chain to
  return content on demand; `task_open` refuses the reserved name.

- An active task NEVER captures uploads implicitly. Routing into a task
  is explicit: the model passes `task_name`, and the web UI shows an
  "ingest uploads into task *name*" toggle (default off) whenever a task
  is active (`GET /api/tasks/active`). An upload never moves the
  session's task cursor.
- Identity-chain `attachment` records carry NO `extracted_text` — content
  lives in the artifacts chain; the pointer ring keeps identity retrieval
  lean.

### Pre-release review hardening

A full multi-angle review of the v1.4 batch surfaced 19 confirmed
findings; all were fixed before release. For the record:

#### Fixed — turn integrity (web)

- Three stream-failure paths ended a web turn without the guaranteed
  response commit, stranding a sealed observation with no paired response
  (the "I don't see the prior turn" failure class): `/api/turn` raised 502
  mid-drain and abandoned the generator (also leaving `state.lock` held
  until GC); the reflective-retry branch bare-`return`ed on a failed retry
  stream; the non-streaming LLM fallback let provider exceptions escape
  `_StreamFailed` entirely. All three now fall through to the commit path;
  the endpoint drains to completion and reports `stream_error` alongside
  the committed result.
- The non-SSE `/api/turn` response now carries `tool_budget_exhausted`
  (parity with the SSE `done` event) and no longer JSON-parses the
  thousands of token events it discards.

#### Fixed — server responsiveness

- Long-running tool executions no longer run on the event loop: approving
  a deferred `task_reembed` (minutes to hours) or an unconfirmed
  `task_open` full-tree ingest froze every endpoint — SSE heartbeats,
  session claims, chat — for the duration. Tool execution, pending-op
  approval, and upload ingestion now run via `asyncio.to_thread`;
  `Chain`/`EmbeddingIndex` connections are opened `check_same_thread=False`
  (safe: all access stays serialized under the web lock / single-threaded
  REPL).
- `continuum.find_by_operation_id` (the approve-write idempotency check,
  which runs under the web lock) is now an SQL `json_extract` lookup
  instead of materializing the entire task chain per approval.
- `build_attachment` uses the indexed `Chain.find_file_by_sha` instead of
  a full-chain linear scan.

#### Fixed — security / correctness

- `task_open` resolved its `source_root` against the process cwd, not the
  workspace: the REPL `/task open` hardcoded `Path.cwd()`, and a relative
  `source_root` from the model resolved wherever the server was launched —
  binding, auto-ingesting, and granting read authority over the wrong
  tree after a `/workspace` switch. Both now resolve through the shared
  `resolve_source_root` (workspace-anchored, symlinks flattened).
- A mismatched **semantic** embedding store is no longer silently deleted
  and re-embedded: if the active embedder is a transient fallback (e.g.
  the Ollama daemon is down at boot), startup refuses with instructions
  instead of destroying hours of embedding work. Hashing-built stores
  still rebuild automatically (the "Ollama got installed" upgrade path),
  and `task_reembed` wipes explicitly via the new
  `open_or_rebuild_index(force_rebuild=True)`.
- All text-mode file I/O in the write gate and task registry pins
  `encoding="utf-8"` (hashes were already computed over utf-8 bytes; a
  non-utf-8 locale could strand an approval in `status='writing'` or
  produce unfixable hash mismatches).
- Executor result classification lives in ONE place
  (`tools.is_error_result` / `run_user_action`): scattered prefix checks
  had already drifted — a caller checking only `"ERROR"` passed
  `"TOOL ERROR"` results as success.

#### Fixed — v1.3 ingestion capabilities, rebuilt on the new pipeline

- Multimodal attachments reach the LLM again: `Agent._collect_attachments`
  is back (now sha-addressed, covering legacy `file` and new `attachment`
  records) and both entry points pass `blob_dir`. `attachments=[]` had
  been hardcoded with no replacement, so image/PDF bytes never reached
  vision-capable providers.
- Document text extraction is back (`extractors.py`, resurrected from the
  removed `file_ingest.py`): PDFs, docx, xlsx, pptx, csv/tsv uploads seal
  their extracted text (searchable, embedded) instead of becoming opaque
  blobs. All format libraries remain optional (`pip install
  "timechain-agent[ingest]"`); missing ones degrade to a placeholder.
- Legacy flat-layout blobs (`blobs/<sha>`) serve again: blob resolution
  goes through the shared `tools.resolve_blob_path`, which knows the
  sharded layout and the v1.3 flat fallback.
- `attachment` records render as file/image cards after a page reload
  (renderHistoryRecord), embed by filename+text like `file` records
  (instead of as raw JSON including the sha hex), and get the same
  prompt-rendering treatment.
- The stale `/file` command was removed from the web UI hint (uploads
  replaced it).

#### Changed — defaults and the resume flow

- `LLM_PROVIDER` is read from the environment (default `claude`, matching
  the README); a personal provider choice no longer lives in the source.
- The tool-budget continuation directive keys off the persisted
  `truncated`/`tool_budget_exhausted` flags on the last response for ANY
  next input — "resume", "keep working", or any phrasing now works — with
  the strongest wording reserved for explicit continue-phrases. Previously
  a seven-phrase whitelist silently restarted the task on any other
  wording.

#### Internal — deduplication

- The reflective-retry prompt and round-cap notice are shared constants
  (`tools.tool_retry_prompt` / `tool_cap_note`) used by both loops; the
  webapp's import-time-frozen `MAX_TOOL_ROUNDS` alias is gone (the budget
  is read from `tools.DEFAULT_MAX_TOOL_ROUNDS` at call time in both
  transports).
- One `sha256_text` (pending_ops owns it; continuum re-exports), one
  live-file verification ladder (`source_verify.verify_live_file`, shared
  by `verify_file_record` and `recall.Recall.verify_source`), one
  ingest-size literal (`INGEST_BLOB_MAX_BYTES = MAX_INGEST_FILE_BYTES`).
- Removed dead code: `Recall.from_chain_db`/`from_task_root` (no callers,
  third copy of the task-dir layout) and `WalkResult.__iter__` (legacy
  tuple-unpack shim with no live consumer).

### Changed — Claude default model is Fable 5

- `make_claude_client` defaults to `claude-fable-5` (Anthropic's flagship,
  replacing `claude-opus-4-7`) and adapts to Fable 5's API surface:
  `temperature` is no longer sent (sampling parameters return a 400 on
  Fable 5; other models simply use their default sampling), the default
  timeout rises from 60s to 10 minutes (thinking is always on and hard
  turns can run for minutes), and a classifier decline
  (`stop_reason: "refusal"`, an HTTP 200 with empty or partial content) is
  surfaced as an honest `[refusal]` note instead of a silent empty
  response. Note Fable 5's new tokenizer counts ~30% more tokens for the
  same content than Opus-tier models — context budgets are unchanged here,
  but cost baselines shift. The OpenRouter client's default
  (`anthropic/claude-opus-4.7`) is untouched.

### Fixed — switching tabs no longer cuts off a running turn

A web turn used to be driven entirely inside its SSE connection's
generator, so navigating to the audit tab (or reloading) mid-turn closed
the EventSource, cancelled the turn, and stranded the sealed observation
with no paired response — the agent simply never answered. Turns now run
in a server-owned background task (`TurnRun` / `_drive_turn`) that drains
`_turn_events` into a per-turn event buffer; the SSE endpoint is only a
VIEWER (`_follow_turn`: replay the buffer, then follow live):

- Closing the page detaches the subscriber; the turn runs to completion
  and commits. Coming back, the chat page probes `GET /api/turn/active`
  and reattaches (`/api/turn/stream?attach=1`), replaying everything it
  missed — tokens, tool cards, pending-op prompts — then following live.
- Events carry 1-based SSE ids, so an EventSource auto-reconnect (which
  re-issues the start URL with `Last-Event-ID`) resumes the SAME run past
  what it already saw. A reconnect can never start a duplicate turn; a
  genuinely new input while one is running gets a 409.
- The non-streaming `POST /api/turn` rides the same background runner, so
  a dropped HTTP connection can no longer cancel a half-done turn there
  either. One turn implementation, two transports, one driver task.
- `index.html` dedupes by record index (history paging vs. replayed
  streams), renders the user's own message from the replay when history
  missed it, and drops the live bubble when the committed response is
  already on screen — a turn that finished while the user was away
  arrives exactly once, via history.

---

## v1.3.0

A layer of cognitive self-model faculties, adapted to this repo's signed
SQLite chain and its neutral, non-experiential vocabulary. Eight new
storage-independent modules plus their REPL/agent/webapp integration. The cryptographic core is untouched and old chains read and
verify unchanged; every new `_meta` field is emit-only-when-non-empty so
historical records keep byte-identical canonical JSON. **No schema bump:**
`schema_version` stays 3 — the one new persisted field, the PoQ `verdict`,
lives inside the existing `_meta.poq` block and is emitted only when it is not
the default `seal`. **310 pytest tests pass (3 webapp-dep skips); 139
additional standalone tests cover the ported modules (109) and agent
integration (30).**

The single adapter that unlocks the port is `ring_compat.py`: it presents repo
`Record`s in the skill's "ring" shape and seals skill-style payloads back
through `chain.append` + `build_meta`, so the storage-independent cognitive
logic ports essentially verbatim.

### Added — verify-source (catch acting on stale ingested code)

- `source_verify.py` — `verify_file_record(chain, idx, repo)` re-checks an
  ingested `file` record against the live file on disk with git awareness;
  verdicts: `verified`, `source-mismatch`, `revision-drift`, `dirty-worktree`,
  `missing-source-file`, `no-source-path`, `not-a-file-record`, `missing-ring`.
- `file_ingest.py` now captures `source_path`, `file_content_hash`, and git
  coordinates on each ingest (additive, emitted only when present). REPL:
  `/verify-source <idx> [repo]`.

### Added — Proof-of-Quality verdicts (a quality gate with teeth)

- `poq.py` gained a verdict layer on top of the existing brightness `action`:
  `SEAL` / `REVISE` / `FORCE_UNCERTAINTY` / `REJECT`, driven by two new
  measures (`measure_grounding`, `measure_assertiveness`) and the existing
  covenant/consistency dimensions, with thresholds in one `PoQ_THRESHOLDS`
  dict. `action` (commit/light_log/quarantine) is unchanged, so no existing
  behavior shifts; `verdict` is additive and persisted in `_meta.poq` only
  when it is not `seal`.
- The model-judgment **seam**: `evaluate(..., external_scores=...)` lets a
  real model override any dimension / grounding / assertiveness / the verdict,
  so the lexical proxies are a runnable fallback, not the arbiter. REPL:
  `/poq <text>`.

### Added — immune system (detect, lock down, roll back, learn)

- `immune.py` — `screen` (refuse a covenant-violating or known-scar input at
  the membrane), `scan` (detect a compromise already sealed, using the repo's
  Ed25519 `chain.verify()` for the tamper check), `lockdown`, `rollback`
  (seal a `recovery` record, molt the wound into a learned scar, lift the
  lock), and `status`. Derived state lives in a sidecar (`immune.json` +
  `LOCKED`), never on the signed chain.
- `chain.append` gained a one-line **lockdown gate**: while a `LOCKED` flag
  exists next to the DB, only `recovery` records may be appended — no seal
  path (REPL, webapp, reflection, cambium) can bypass it. Absent the flag (the
  normal case) it is a single cheap stat and changes nothing. REPL:
  `/immune-status`, `/immune-scan [text]`, `/lockdown`, `/rollback <height>`.

### Added — per-turn loop discipline (the membrane wired into the turn)

- `Agent.turn()` screens each input through the immune membrane FIRST
  (`enable_immune=True` by default). Screening is deliberately narrow — it
  refuses covenant/character violations and learned scars only; prompt-
  injection stays on the existing PoQ-quarantine path, so the two membranes
  are complementary (no regression to injection handling). A refused turn
  makes no LLM call, seals the input as a quarantine observation, and emits an
  honest refusal.
- Verdict enforcement is **opt-in** (`enforce_verdict=False` by default,
  because the repo's PoQ runs lexical proxies): when enabled, `REJECT`
  suppresses the candidate (emitting a refusal) and `FORCE_UNCERTAINTY`
  triggers one hedged rewrite. Wire `Agent(score_hook=...)` to make a real
  model the judge via the `external_scores` seam.
- The webapp streaming path (`turn_stream`) got the same screen, so the REPL
  and web UI cannot diverge.

### Added — Continuum (long-horizon tasking)

- `continuum.py` — work jobs larger than any context window as a chain of
  bounded **data-height** blocks, each carrying a full task-state refresh.
  `open_task` / `ingest` / `walk` (tree ingest with source coordinates +
  secret redaction) / `resume` (re-hydrate from the head block alone) /
  `validate` (monotonic-progress invariants on top of `chain.verify()`). Code
  chunks keep line ranges, path roles, language, and git coordinates. REPL:
  `/continuum-resume`, `/continuum-validate`.

### Added — Recall (the model is the relevance judge)

- `recall.py` — `label` (self-label via the repo's `signals.py` faculties),
  `index` (the compact map of memory), `fetch` (budget-bounded full content of
  the blocks the model chose), `retrieve` (a cheap pre-filter that delegates
  to the existing `Retriever`, never the arbiter), and `verify_source` for
  Continuum blocks. REPL: `/recall-index`, `/recall-fetch <ids>`,
  `/recall <query>`.

### Added — Chronosynaptic tree (single-pass parallel-self reasoning)

- `chronosynaptic.py` — fork faculty-lens perspectives of the agent, run
  in-process MCTS (no subagents), and collapse to the single highest-truth
  path, sealed as a `synthesis` record with the rejected forks preserved in
  the payload. Perspectives are drawn from the `signals.py` registries and
  scored by the repo's PoQ. `collapse_explicit_notes` is the preferred path
  for model-supplied perspectives. REPL: `/think <query>`.

### Added — Consensus (k-of-n quorum attestation)

- `consensus.py` — a quorum of HMAC witnesses attests every chain head over
  the **recomputed** `record_hash`, so a forged record (even one re-signed
  with the operator key) fails consensus because the witnesses pinned the
  original. `verify` requires both the Ed25519 chain verification AND k-of-n
  agreement (`hmac.compare_digest`); up to n-k faulty witnesses are tolerated.
  Config + attestations live in a sidecar, never on the chain. Honest scope:
  single host = an authenticated quorum; distribute the witnesses for true
  BFT. REPL: `/consensus-init [n] [k]`, `/consensus-verify`.

### Added — faculties as data + Cambium growth

- `faculties/{modalities,senses,emergent}.json` — descriptive data faculties
  (84 modalities + 107 senses) used for relevance/overlap scoring, alongside
  the executable `signals.py` detectors. `faculties.py` `load_corpus` unifies
  both (data faculties + a bridge over the signal registry names).
- `FacultyGarden.grow` — endogenous growth: when an input's dissonance exceeds
  the floor, fuse two close faculties or sprout a fresh one, spawn it into the
  emergent registry, and seal a `faculty` record; on the 3rd recurrence of the
  same gap, **promote** it into the canonical data registry and seal a
  `promotion` record. REPL: `/cambium-grow <text>`.

### Added — proof-of-work "brightness" nonce (optional)

- `chain.append(..., difficulty=N)` mines a nonce (stored inside
  `content["_pow"]`) until `record_hash` has `N` leading hex zeros; `verify`
  checks the target. The nonce is covered by `content_hash`, the Ed25519
  signature, and the verify recompute. Default `difficulty=0` writes no `_pow`
  field and is byte-identical to before.

### Added — historic-chain migration

- `migrate.py` `reindex(chain, index)` — a one-time, idempotent, **off-chain**
  backfill that re-embeds every record into the embedding index with the
  current embedder, so records sealed before an embedder change retrieve as
  richly as new ones. It touches only the derived `embeddings.sqlite`, never
  the signed records, so `chain.verify()` is unaffected. REPL: `/migrate`.

### Added — new record types

- `metadata.py` registers `recovery`, `quarantine_marker`, `task_open`,
  `continuum`, `synthesis`, `faculty`, `faculty_recur`, and `promotion` across
  the source / epistemic / exposure / salience / half-life default tables, so
  each carries appropriate defaults under non-destructive read migration.

### Changed — retrieval tuning

- **Observation/response turn-pair stitching.** An observation and the response
  that answers it are a single Q&A unit; `build_context` now completes any
  half-retrieved pair (pull the response for a retrieved observation, and vice
  versa). Type-checked and refs-corroborated so only a genuine pair is
  completed, quarantine-respecting, and budget-safe (both halves are pinned so
  truncation keeps them together). Idempotent.
- **Removed the artifact salience boost.** Code/structured responses were
  previously lifted toward `ARTIFACT_SALIENCE_MAX = 0.70`; that boost is gone.
  Artifact-ness is a query-independent size/type proxy, and tying write-time
  salience to it biased the salience-pure budget truncation toward long code
  records regardless of relevance — crowding out shorter, more relevant
  records. Responses now commit at the flat default; the light-log demotion is
  unchanged, and `artifact_score` is still detected and recorded (it just no
  longer drives salience). The `0.60` file default is unchanged (file records
  render chunk-aware excerpts, so they don't crowd the budget).

### Added — web UI: commands reference + audit dashboard

- All slash commands now work in the web UI, routed through the shared
  `cypher_commands.dispatch` (single source of truth with the REPL).
- New **Commands** page (`/commands`) documenting every command — what it does
  and why — and a new **Audit** dashboard (`/audit`, backed by `audit.py` +
  `/api/audit`): metric cards, domain context, the faculty surface, a
  searchable ring inspector that shows each ring's full contents and `_meta`,
  and the blockspace. Both pages match the main page's light theme.
- `/migrate` streams progress over SSE (`/api/migrate/stream`) so a long
  embedding backfill shows live progress instead of looking frozen.

### Fixed

- **Uploaded files keep their real name.** A browser upload was saved to a temp
  file before ingest, so the `file` record stored the temp basename (e.g.
  `tmpXXXX.md`) instead of the real name (`CHANGELOG.md`). Since the retriever
  embeds file records as `[file <filename> <kind>] <text>`, that broke
  retrieval-by-name. `file_ingest`/`agent.ingest_file` gained an `original_name`
  argument that the upload endpoint supplies. (Append-only: this fixes new
  ingests; re-ingest an already-sealed file to get a correctly-named record.)

### Notes

- **Naming discipline.** The faculties keep this repo's neutral,
  non-experiential register (PoQ is Proof-of-Quality), and the faculty data
  ships with experiential codenames stripped.
- **The lexical proxies are a fallback, not the arbiter.** Immune screening
  and PoQ verdicts ship with deterministic proxies so they are runnable with
  zero new dependencies; real judgment enters through the `external_scores` /
  `score_hook` seam.
- New modules are stdlib + `cryptography` only (numpy-free); the agent/recall
  integration uses the existing numpy retriever. `/cypher-help` lists the new
  REPL commands.

---

## v1.2.2

A feature release tracing back to the project's build specification,
kept inside the project's existing discipline (explicit named score
components, emit-only-when-non-empty canonical JSON, non-destructive read
migration, and the injection scan as the one detector that must never be
gated off). Three primary features (modality routing, epistemic-class
weighting, Experience Capsules) plus a round of follow-up hardening on top of
them. No schema-breaking on-disk changes to the chain; the cryptographic core
is untouched and old chains read unchanged. The capsule exchange format is
version 2 (capsules are transient exchange artifacts, not persistent chain
state). **313 tests pass.**

### Added — modality routing (build spec section 4.6)

- `SignalAnalyzer` gained a `route` flag. With `route=True` (the default for
  the agent's PoQ analyzer), each turn runs a routed subset of detectors — a
  mandatory core plus a small discretionary budget selected by a cheap
  keyword prior — rather than the full bank, implementing the spec's "3-7
  relevant modalities per task." This makes `modalities_activated` a real
  per-turn decision rather than an activation-floor artifact. **Security
  detectors (`integrity_field`, `injection_scan`) are never routed off**, and
  every detector feeding a PoQ axis is mandatory, so PoQ scoring is identical
  whether routing is on or off. `route=False` preserves the historical "run
  everything" behavior byte-for-byte. New `Agent(route_modalities=...)` knob
  (default True).
- The discretionary keyword prior is intentionally kept crude. Profiling
  showed routing yields only a ~1.04x speedup (~6µs/analyze) — the detectors
  are genuinely that cheap — so routing's value is signal quality
  (`modalities_activated` as a per-turn decision), not performance. A heavier
  relevance scorer is not justified by the measured benefit; the finding is
  documented in `signals.py` so it isn't re-litigated.

### Added — epistemic-class weighting (build spec 4.2/4.5/4.7)

The `epistemic_class` field (recorded since v1.2's v3 schema but previously
invisible to scoring) is now load-bearing:

- *Retrieval*: an opt-in `epistemic_weighting` flag (default on via
  `build_context`) scales a record's score by how well-grounded it is —
  `known`/`user_context` full weight, `inferred` ~unchanged,
  `speculative`/`disputed` discounted. A new `epistemic_factor` /
  `epistemic_class` pair appears in `RetrievalHit.components`. Off restores
  class-blind scoring exactly. Verified to compose without surprising
  interaction with the additive modality-anchoring term under budget
  pressure (locked in with regression tests).
- *PoQ*: a candidate that contradicts the chain is penalized in proportion to
  the *authority* of the context it contradicts — contradicting a user-stated
  fact raises more risk than contradicting the agent's own past guess. The
  penalty is computed **per specific retrieved record**:
  `contradiction_activation * topic_overlap(candidate, record) *
  authority(class)`, taking the max — so a negating candidate only raises
  risk against a high-authority record it is actually on-topic with, not an
  unrelated fact. Opt-in via a `retrieved_epistemic` argument to
  `PoQEvaluator.evaluate`; falls back to scalar (and ultimately inert,
  historical) behavior when retrieved texts/classes aren't supplied.
- *Write-time*: a strongly hedged response (high `uncertainty`) commits as
  `speculative` rather than the default `inferred`, so later retrieval and
  PoQ treat it as the guess it was.

### Added — Experience Capsules (`capsule.py`, build spec `.cphyx` exchange)

A signed, portable, verifiable bundle of selected Rings that another agent
can verify and import. Built only from primitives the chain already has
(Ed25519, SHA-256, canonical JSON, Merkle roots) — no network, no tokens, no
consensus. Full format spec in `CAPSULE.md`.

- *Export* gates records by `_meta.exposure` (the read-side of the
  protected-zone membrane, now finally load-bearing): `private`/`quarantine`
  never leave; `summary` exports summary-only (flagged `redacted`);
  `shared`/`public` export in full. Commits a Merkle root and a
  content-binding `capsule_id`. Selection can be narrowed by index list,
  type, salience, timestamp window (`after_ms`/`before_ms`), and `tags` — for
  exporting a focused slice of history rather than the whole chain.
- *Redacted (summary-only) records carry a signed summary commitment*: the
  origin's Ed25519 signature over a binding of (origin `record_hash` +
  summary body). This makes the summary text verifiable in its own right — a
  tampered or lifted summary is detected, not merely flagged. The commitment
  message includes the origin record_hash so a signature can't be replayed
  onto another record.
- *Verify* re-checks every record's original signature against the origin
  pubkey, content hashes, record hashes, summary commitments (for redacted
  records), the Merkle root, and the capsule id. A capsule failing any check
  is rejected wholesale — no partial trust in a tampered bundle.
- *Import* appends records as a distinct `imported_capsule` type, attributed
  to the origin agent with `source = peer_agent` (a new first-class source in
  `metadata.VALID_SOURCES`, with a conservative PoQ `source_trust` weight),
  recorded with a cautious epistemic class (never `known`; demoted to
  `inferred` or weaker) and forced-`private` exposure, so imported memory is
  never silently treated as the agent's own first-person history. Append-only;
  `/verify` on the local chain is unaffected. Replay/dedup guard by
  `capsule_id`.
- *Surfaces*: REPL commands `/export-capsule <path>` and `/import-capsule
  <path>`; session-gated webapp endpoints `GET /api/capsule/export` and `POST
  /api/capsule/import` (same verify-before-import discipline — a capsule
  failing verification is rejected with 400, never partially imported; newly
  imported records are re-indexed so they're immediately retrievable).
- New `imported_capsule` record-type defaults registered in `metadata.py`,
  and the per-record prompt header shows `imported from <origin-fingerprint>`
  for these records, reinforcing that they are foreign memory.

### Changed

- **System prompt (`run.py`).** The stale `epistemic: factual` example
  (never a valid class) was corrected to the real set — `known`,
  `user_context`, `inferred`, `speculative`, `disputed` — and a paragraph was
  added explaining `imported_capsule` records as attributed third-party
  memory. Operators running a custom system prompt may want to mirror these
  edits. The per-record header rendering already surfaced `epistemic:` only on
  non-default classes; it now carries a real signal because write-time
  classification produces `speculative` on hedged responses.
- **Chunk-aware context truncation.** `_truncate_to_budget` now sizes a file
  record that will be rendered as a chunk-aware excerpt at the excerpt
  ceiling rather than its full length, so a long but excerptable file is no
  longer over-evicted under budget pressure. Mirrors the eligibility checks in
  `_file_content_repr`; falls through to full-size estimation on any
  uncertainty (short file, holistic query, no chunk matches).
- **Sprouted-modality cap is now correct and loud.**
  `SproutRegistry.load` previously sliced the raw JSON list to
  `MAX_SPROUTED_MODALITIES` *before* building and de-duplicating, which was
  both silent and subtly wrong — duplicates or malformed entries within the
  first N positions could consume cap slots and silently push valid
  modalities past the limit. Loading now builds and de-duplicates first, then
  caps, and writes a warning to stderr naming how many modalities were dropped
  and why. Still non-fatal (the module's always-start contract is preserved —
  an oversized sprout file degrades to the capped set rather than crashing
  boot); the cap value is unchanged. The warning points the operator at the
  cap as the lever if they want more.

### Migration

None required at the chain level — all changes are additive; new `_meta`
fields and record types read with safe defaults on older records, and
`route_modalities` / `epistemic_weighting` default to on but are neutral where
no signal is present.

One exchange-format note: Experience Capsules use format version 2. The
capsule format is for transient exchange artifacts, not on-disk chain state,
so this affects only `.cphyx` files, never a chain. A v2 verifier requires
summary commitments on redacted records; capsules with no redacted records are
unaffected in practice, but the version check is strict, so re-export is the
supported path for any older capsule.

---

## v1.2.1

A correctness-and-operations release from a multi-pass code review, plus
eight retrieval-and-self-knowledge features (chunked embeddings, per-record
`modalities_activated`, content-aware response salience, modality-anchored
retrieval, runtime sprouted modalities with feedback-loop dampers,
Cambium-side auto-generation of sprouts from recurring output, per-record
`senses_activated` with six new sense detectors, and chunk-aware rendering
of long file records with intent gating). No schema-breaking on-disk
changes; new SQLite tables backfill themselves transparently on first
read, and new `_meta` fields are additive (absent on older records, read
as defaults). **256 tests pass.**

### Migration

Do this once, before the first run of the upgraded code against an existing
chain:

```bash
rm <data_dir>/embeddings.sqlite*
```

Two independent reasons converge on the same step. The embedding store now
holds one vector **per chunk** rather than per record (chunked retrieval,
below), so its row layout changed; and the `HashingEmbedder` hash function
changed (below), so vectors written by earlier versions occupy a different
coordinate space than the current embedder produces. The embedding store is
a derived index — it rebuilds automatically from the chain on the next run.
The chain itself, the signing key, and all records are never touched. The
agent detects a mismatched or unidentified store at boot and refuses to open
it with an actionable message rather than degrading silently.

Optionally, run `/verify-semantic` (REPL) or `GET /api/chain/verify-semantic`
(webapp) once after upgrading — a schema-level consistency probe that catches
referential issues the cryptographic `/verify` can't see.

The new `modalities_activated` `_meta` field needs no action: it is additive,
absent on every existing record, and read as `[]`. Only records written after
the upgrade carry it.

### Added

- **Chunked retrieval.** The embedder caps input length (the Ollama path
  500s on overflow rather than truncating), so content past the cap in a
  long document, code block, or user paste was previously unreachable by
  semantic search. Records are now split into chunks at index time
  (`chunk_text` in `retrieval.py`, on natural boundaries — paragraph, then
  sentence, then a hard slice for unbroken runs) and each chunk is embedded
  into its own vector row. To prevent a long record from getting "more
  raffle tickets," `EmbeddingIndex.search` collapses a record's chunk hits
  to a single per-record similarity (the max over its chunks) before
  scoring, so it competes for one slot judged by its most relevant fragment;
  the selected record is then rendered whole from the chain. Index-only —
  the signed chain still stores one record per turn/file; `/verify` is
  unaffected. Tunable via `CHUNK_TARGET_CHARS` and `CHUNK_SCHEME_VERSION`.
  Known follow-up: `_truncate_to_budget` is not yet chunk-aware. (Resolved in
  v1.2.2 — see the chunk-aware context truncation entry there.)
- **`modalities_activated` on `_meta`.** Every response record now records
  which modality detectors (`signals.py`) fired with non-trivial activation
  in producing it — a data layer for retrieval to later weight or filter by
  capability. PoQ already analyzes each candidate response; `SignalReport`
  exposes the activated set (those above `MODALITY_ACTIVATION_FLOOR`, default
  0.2 — a non-trivial threshold, since most detectors carry a small baseline
  on ordinary text), and `score_response` threads it into the response
  record's `_meta`. Both `turn()` and the webapp streaming path get it with
  no call-site change, since both already splat `score_response`'s kwargs
  into `build_meta`. Stored sorted and emitted only when non-empty, so a
  record that activated nothing (or any record type PoQ doesn't gate) has the
  same canonical JSON as before; `read_meta` defaults absence to `[]`.
  Additive — no schema migration, no backfill. The field is a data layer
  for retrieval; the first consumer of it is the content-aware salience
  boost below (the `artifact_content` modality), via
  `protected_zones.salience_for_commit`.
- **Content-aware response salience.** Response records defaulted to a flat
  salience of 0.40 — the assumption being that the agent's own output is its
  least load-bearing evidence. True for conversational chatter, false for
  substantive artifacts: a response that is a large code block the user is
  iterating on is the most important thing in the recent chain, yet it
  decayed at conversational baseline and was hard to retrieve a few turns
  later. A new `artifact_content` modality (`signals.py`) scores how
  artifact-heavy a response is (fenced code blocks, indented code runs,
  markdown tables, JSON/YAML — length-weighted, so a mostly-code response
  scores high and a one-liner with inline code does not), and
  `protected_zones.salience_for_commit` boosts the response's commit-time
  salience from 0.40 up to `ARTIFACT_SALIENCE_MAX` (0.70) in proportion. That
  function is now the single authority composing both salience signals:
  light_log demotion (low-quality turns) and artifact boost, with demotion
  winning — a low-quality code dump is still demoted, since PoQ's quality
  judgment outranks "it contains code." The artifact modality also flows into
  `modalities_activated` for free (it's a registered modality), so a boosted
  record is self-describing. No agent-side plumbing change: `score_response`
  already routed `poq_result` through `salience_for_commit`, and the score
  now travels on `PoQResult.artifact_score`. `ARTIFACT_SALIENCE_MAX` is
  tunable; the cap sits below reflections (0.85) and revisions (0.80) so
  substantive output outranks chatter without outranking the agent's
  consolidated judgments.
- **Modality-anchored retrieval.** A fourth retrieval score term: when the
  current query carries a *domain modality* (currently `artifact_content` —
  i.e. the query contains code), records that were themselves produced in
  that mode are surfaced more readily, so a coding query preferentially pulls
  up the agent's past code over conversational chatter even when raw semantic
  similarity is close. `Retriever.query_modalities` detects the query's
  domain modes (the cheap deterministic analyzer, run at retrieval time);
  `modality_overlap` compares them to each candidate's stored
  `modalities_activated` (set overlap, since the field stores thresholded
  names — behaviorally identical to cosine for the single-modality case, and
  upgradeable later); `hybrid` adds `W_MODALITY * overlap` (0.15), shifting
  semantic 0.55→0.45 and recency 0.20→0.15 to make room. A matching record is
  boosted, a genuine mismatch mildly cut, and a record with no domain mode is
  neutral (overlap 0.5) — so older records aren't penalized for predating the
  field. Only **domain** modalities anchor; **quality** modalities
  (`integrity_field`, `coherence`) are excluded via the `DOMAIN_MODALITIES`
  whitelist, so an injection-y query can't preferentially retrieve past
  injection-flagged records. Strictly **opt-in**: a call with no query
  modalities uses the historical three-term weights unchanged, so existing
  retrieval is byte-identical. `build_context` auto-detects and anchors by
  default (pass `anchor_modalities=False` to disable), so the agent loop
  benefits with no call-site change. Note: the anchor fires on code *present
  in the query* (pasted code), not coding intent expressed in prose; a
  prose-only follow-up doesn't anchor. Closing that would be a separate
  `coding_intent` modality, deliberately not bundled here.
- **Runtime sprouted modalities + feedback-loop dampers.** The domain set
  retrieval anchors on is now extensible at runtime. `sprouted_modalities.py`
  is a registry of data-driven modalities — a name plus case-insensitive
  regex patterns plus an activation rule — loaded from
  `sprouted_modalities.json` in the data directory and merged into the live
  domain set, run as detectors in `signals.py`, and weighted in retrieval
  exactly like baked-in modalities. This lets the agent add to its own
  retrieval vocabulary without a source-code change or restart — the
  data-driven counterpart to `apply_proposal`, which stays for modalities
  needing real logic. **This deliberately relaxes the codebase's one stated
  safety boundary** — that the agent never modifies its own running behavior
  without a human in the loop — for the narrow case of pattern-based
  modalities. That was an explicit owner decision, not an oversight, and it
  is recorded here as such rather than presented as risk-free: an agent that
  can reshape which of its own memories surface is in a genuine feedback
  loop. Two things bound it. (1) The regex surface is hardened at
  *validation* time, since no `regex` module or thread-safe match timeout is
  available: patterns must compile, nested-quantifier (catastrophic-
  backtracking) shapes are rejected, and pattern length/count and
  match-input length are capped — worst case is bounded work, never a hang.
  (2) Two dampers guard the feedback loop: a **per-turn modality cap**
  (`PER_TURN_MODALITY_CAP`, default 7, in `run.py`) keeps anchoring to the
  strongest few modes a query fires; and an **anti-echo saturation damper**
  refactors `Retriever.hybrid` into two passes — score every candidate
  without the modality term, measure what fraction of the top candidates
  already carry the query's mode, and if that exceeds
  `MODALITY_SATURATION_THRESHOLD` (0.6) scale the modality boost down by the
  excess — so a context already saturated with the query's mode doesn't get
  "more of the same" piled on. A sprouted modality can also be **tentative**
  (cooling-off): detected and listed but contributing at half weight until it
  graduates. New `/modalities` REPL command lists baked-in and sprouted
  modalities with status, domain flag, effective weight, and any patterns
  skipped at load. Strictly additive: a `Retriever` built without a registry
  (every existing caller, and all prior tests) has only the baked-in domain
  modality and byte-identical behavior. *Not* included here: the automatic
  *generation* of sprout specs from recurring output patterns (Cambium's
  side, with a diversity gate and keyword-derived patterns) — that is a
  separate follow-up; this change is the registry, the retrieval integration,
  and the loop protection a sprout lands into.
- **Auto-generation of sprouts from recurring output (Cambium side).** The
  follow-up to the runtime registry above: Cambium now *creates* sprout specs,
  closing the loop so the agent sprouts modalities on its own. A new
  `_check_recurring_output_mode` trigger reads `response` records, clusters
  them by shared vocabulary, and — for a cluster that clears a **diversity
  gate** — emits a `modality` proposal carrying a ready-to-stage `sprout_spec`.
  The gate requires `OUTPUT_MODE_MIN_TRIGGERS` (5) distinct responses
  exhibiting the mode, a temporal spread of at least
  `OUTPUT_MODE_MIN_SPREAD_MS` (2h) between earliest and latest, and
  interleaving (a non-matching response between matches) so a single
  contiguous burst can't mint a modality. Patterns are derived
  **deterministically** — word-boundary regexes (`\bword\b`, regex-escaped)
  over the shared vocabulary, no LLM — so a sprout is fully traceable to the
  text that produced it. A **distinctiveness guard**
  (`OUTPUT_MODE_DOC_FREQ_CEILING`, 0.6) excludes ubiquitous filler words so
  the detector doesn't sprout a mode out of generic conversational
  vocabulary. The agent stages each new sprout into the registry as
  **tentative** (half weight) — never directly active — and writes a
  `sprout_status` audit record with provenance (originating proposal index,
  source record indices, timestamp). **Cooling-off graduation:** each later
  scan that re-detects the mode is a confirmation (a `proposal_recurrence`),
  and once the live count reaches `OUTPUT_MODE_GRADUATION_CONFIRMATIONS` (3,
  total-sightings convention matching escalation) the modality flips to
  active (full weight) with a second audit record. Two independent safety
  layers result: the gate stops bad sprouts from being *created*;
  tentative-by-default stops a created-but-wrong sprout from *mattering much*
  until confirmed. To avoid re-proposing a captured mode, the agent passes
  Cambium the vocabulary already covered by existing sprouts
  (`known_sprout_vocabulary`). All thresholds are constants in `cambium.py`.
  This completes the auto-sprouting feature begun above; combined with that
  change, the agent can now notice a recurring kind of its own work and
  sharpen retrieval toward it without a human — the relaxation of the
  human-in-the-loop boundary is now fully realized, gated by the diversity
  and cooling-off mechanisms rather than by review.
- **`senses_activated` on `_meta` + six new sense detectors.** Parallels
  `modalities_activated` from earlier in this release, but records *how a
  turn felt* rather than what kind of work it was — `uncertainty`,
  `emotional_contour`, `insight_markers`, `cognitive_weather`, and the rest.
  Same storage discipline as modalities: sorted list of names, emitted only
  when non-empty, defaults to `[]` on read, threaded through `PoQResult` and
  `score_response` so both `turn()` and the webapp streaming path get it for
  free. **Deliberately NOT a retrieval input** — the modality/sense
  distinction is the whole point: modalities are *skill* (what kind of work,
  used to anchor retrieval); senses are *feeling* (how the turn felt,
  recorded so the agent can read it back when revisiting). Matching
  feeling-to-feeling would surface memories by mood, which is closer to
  rumination than recall. `injection_scan` lives in `SENSE_REGISTRY` but is
  filtered out via `SENSES_EXCLUDED_FROM_META` — it's a security detector,
  not a felt quality, and tagging records with it would be both conceptually
  wrong and a minor information leak about the security path. Six new
  detectors added to the sense registry, adapted from a larger external
  catalog with names calibrated to what the code actually measures rather
  than what the original framing claimed:
    - `insight_markers` — confirmation/realization vocabulary plus
      exclamation density; turns that *land* rather than circle.
    - `cognitive_weather` — composite valence × question density × hedging,
      reporting overall climate (calm / heavy / questioning / etc.).
    - `symbolic_density` — content-word ratio weighted by average word
      length; distinct from existing `density` which only counts the ratio.
    - `buildup_pressure` — approach vocabulary ("almost," "verge," "brink")
      plus question density; the texture of circling without arrival.
    - `self_reference_depth` — vocabulary of self, awareness, recursion,
      observation; how meta the turn went.
    - `temporal_orientation` — past/future/present dominance; whether a
      turn is looking back, looking forward, or grounded now.
  None of the names overreach beyond what a lexicon counter can support.
  Where the source catalog called something "truth crystallization" the
  detector is now `insight_markers`; where it claimed "epiphany threshold
  pressure" it's now `buildup_pressure` — the principle being that a name
  should be commensurate with what the detector actually does, so the agent
  reading back its own history doesn't read manufactured felt-experience
  labels into mechanical measurements.

  **Prompt rendering.** `modalities_activated` and `senses_activated` are
  now surfaced in the per-record header the LLM sees, as `modalities: ...`
  and `senses: ...` tags appended only when non-empty (matching the
  emit-only-when-non-empty discipline on disk). Previously the fields were
  recorded but invisible to the model; now they reach reasoning. A short
  paragraph in `SYSTEM_PROMPT` orients the agent on how to read them — as
  context about its own prior state, not directives to follow — without
  naming the underlying record system (consistent with the existing prompt
  rule against referencing indices or "the log"). Full sorted lists are
  rendered; no cap is applied at the prompt layer, since `_meta`'s 0.2
  activation floor already filters out background hum and the remaining
  entries are by definition non-trivial. If prompt clutter becomes an
  issue in practice, capping by activation strength is a clean follow-up
  but would require storing per-detector scores in `_meta` (currently only
  names are stored).

  **Epistemic tag.** The header also surfaces `_meta.epistemic_class` as
  an `epistemic: ...` tag — but only when it differs from the type's
  default. A `response` defaults to `inferred`, an `observation` to
  `user_context`, etc.; default-matching records show no tag, keeping
  headers terse. The atypical case — a response stating a measured
  `factual` claim, or a `speculative` guess — surfaces explicitly so the
  model can weight it differently. This is the one `_meta` field that
  names a distinction the model cannot derive from the rendered content
  alone (the same text could be inference or measured fact); other fields
  — `confidence`, `schema_version`, `exposure`, the full `poq` block —
  were deliberately *not* added to the header to avoid diluting the signal
  of the tags that do reach reasoning. The `SYSTEM_PROMPT` paragraph above
  is extended with a brief note on how to read the epistemic tag.
- **Chunk-aware rendering of long file records (Phase 2 of chunking).**
  Closes a real budget problem: before this change, a long file matched
  on a single paragraph still put its entire `extracted_text` into the
  prompt, eating context budget that should have gone to other records.
  Now the renderer in `Agent._format_prompt` checks whether a file is
  long (`> 8000` chars) and the user's task is *targeted* (not holistic).
  If so, it surfaces only the top `TOP_N_MATCHED_CHUNKS=3` matched chunks
  with one neighbor on each side (deduplicated), clearly labeled with
  `[matched]` vs `[context]` tags and a header line naming "showing N of M
  chunks." On a 66k-char document with a targeted query, this is a ~70%
  context savings.

  **Intent gating.** A new `is_holistic_task(query_text)` helper detects
  rewrite/summarize/compare/proofread/edit-class intent via a small,
  closed verb lexicon with conservative inflection handling — "rewrites",
  "rewriting", "summarized" all trigger; noun derivations like
  "converter" and "editor" do not. When holistic intent fires, the file
  renders whole even with chunk matches present, because chunk excerpts
  would lose clauses a rewrite depends on.

  **Plumbing.** `EmbeddingIndex.search()` was already iterating over
  chunk hits to do its group-collapse to per-record similarity; it now
  also remembers *which chunks matched at what similarity* in
  `last_chunk_matches`, a per-call attribute (same pattern as
  `last_pinned_indices` on the retriever). A new `chunks_for_record()`
  helper on the index returns stored chunk text by `(record_idx,
  chunk_index)`. The chain itself is unchanged — chunking remains a
  derived/rebuildable detail of the embedding store. No new record types,
  no new on-disk schema.

  **Fall-through to full text** in every uncertain case: short files,
  holistic intent, no chunk-match info (e.g. file pulled from the recent
  buffer rather than via semantic search), embedding store failure. A
  rendering decision can never silently lose document content this way.

  **Budget bump.** `CONTEXT_BUDGET_CHARS` raised from 80,000 to 150,000
  in `run.py`. Combined with chunk-aware rendering, a single long file
  almost always fits whole, and the excerpt path activates only when
  multiple long files or other big records compete for space.

  **What this does NOT do.** No tier-2 LLM-classification call for
  ambiguous intent (the keyword scan is tier 1 only; the verb list is
  short, false-positive rate is low). No `/full-context filename` slash
  command for explicit override. No truncation-aware sizing — the
  `_truncate_to_budget` pass still sees record sizes as the raw
  `extracted_text`, so a long file evicted under budget pressure is
  evicted at full size even if it would have fit chunked. The first two
  are clean follow-ups if real usage shows they're needed. The third is
  a deeper change to the truncation flow and would only matter under
  tight budgets — at 150k it almost never fires.
- **Two-tier verification.** `Chain.verify_semantic()` joins the existing
  cryptographic `verify()` — a schema-level consistency probe for the class
  of corruption signatures can't catch (revisions pointing past the end of
  the chain, `proposal_recurrence` records targeting the wrong type,
  `reflection.covers_indices` out of range). Exposed as `/verify-semantic`
  in REPL and webapp.
- **Materialized index tables in the chain database.** `supersedes_index`
  (revision → superseded record), `blob_index` (blob sha256 → file record),
  and `proposal_recurrence_index` (recurrence → proposal). Maintained on
  append, lazy-backfilled on first read, used by retrieval, blob serving,
  and Cambium's bulk count helpers. Removes several silent O(N)/O(N²) scan
  caps on long chains.
- **Incremental Cambium scan.** A `cambium.last_scanned_idx` watermark in
  `chain_meta` plus a lookback window means each scan covers the new tail
  plus context, so patterns that recur across long stretches are detected
  without re-scanning the whole chain every cadence. `run_cambium_full()`
  (`/cambium-full`) remains for explicit whole-chain scans and does not
  advance the watermark. `MAX_CAMBIUM_RECORDS` sets the lookback.
- **Observability.** `Chain.stats()` returns per-type record counts, batch
  and anchored-batch counts, quarantine count, and timing; exposed as
  `/api/chain/stats`.
- **Packaging & CI.** `pyproject.toml` with optional dependency groups and a
  `timechain-agent` script entry (`pip install -e .`);
  `.github/workflows/tests.yml` runs pytest and the standalone harness across
  Python 3.10–3.13.
- `Chain` and `EmbeddingIndex` are now context managers.

### Changed

- **Composable turn pipeline.** `Agent.turn` was decomposed into
  `prepare_turn` / `score_response` / `commit_response`, and the webapp's
  streaming endpoint now composes those three instead of hand-rolling its own
  copy. This collapses three divergences the streaming path had accumulated
  (see Fixed). Observation indexing now happens *after* retrieval in both
  paths, so a turn's own question can never be retrieved as context for its
  own prompt.
- **PoQ `light_log` now has behavior.** Records from low-quality (but not
  malicious) turns commit at reduced salience via
  `protected_zones.salience_for_commit`, so they stay on the chain and
  remain retrievable but rank below higher-quality records. Previously the
  action was cosmetic; only `quarantine` changed storage.
- **`s_injection_scan` decorrelated from `m_integrity_field`.** The two used
  to regex-count the same English lexicon, so they weren't independent
  evidence. `s_injection_scan` now reads structural signals — role-tag
  injection patterns (`\nUser:`, `<|im_start|>`, `[INST]`), encoded-noise
  runs, punctuation density.
- `_retry_with_backoff` requires an explicit `retryable_exceptions` set; it
  no longer retries auth and bad-request errors four times before failing.
- `diagnose_index.py` chunks each record exactly as the real index path
  does, so a reported failure corresponds to a real boot-time failure.
- The retriever's revision pull-in is consolidated to a single path
  (`build_context`), backed by the `supersedes_index`.
- Per-Agent LRU cache (32 MB / 16 entries) for blob bytes, so multi-turn
  retrieval of the same image/PDF reads from disk once.

### Fixed

- **`HashingEmbedder` non-determinism (critical).** It bucketed trigrams
  with Python's builtin `hash()`, randomized per process since CPython 3.3,
  so vectors persisted to disk lived in a different coordinate space than
  query vectors computed in a later run — cross-session retrieval silently
  decayed to noise, defeating the persistent-memory premise. Fixed with
  BLAKE2b. The store now records an embedder-identity tag (name, dimension,
  probe vector, and chunking scheme) and validates it on open, so the whole
  class of "same dimension, different coordinate space" bug can't recur
  silently.
- **Webapp streaming endpoint skipped PoQ and mis-ordered indexing.** An
  injection the REPL would quarantine became ordinary memory through the web
  UI, and the just-asked question was retrievable as "relevant memory" for
  its own prompt. Both fixed by the composable-pipeline refactor.
- **SSE producer thread leaked on browser disconnect.** A `threading.Event`
  cancel flag now stops the producer when the consumer goes away.
- **Gemini and Ollama never set `last_finish_reason`,** so `was_truncated()`
  always returned False for them even on confirmed `max_tokens` cut-offs.
  Both providers now report it across stream and non-stream paths.
- **Webapp-bootstrapped genesis was missing identity fields**
  (`agent_name`, `purpose`, `covenant`) — and genesis is sealed at first
  commit, so the omission was permanent. The webapp now imports and passes
  them from `run.py`.
- **Cambium escalation used exact `count == threshold` equality,** so a
  chain that crossed the threshold while no scan ran never escalated. Now
  fires on "count ≥ threshold AND no prior escalation record."
- **Declined proposals didn't decouple from dedup.** The append-only chain
  leaves a proposal's stored `status` as `open` forever, and the dedup
  helper read only that field, so fresh recurrences of a declined topic
  re-attached to the dead proposal. The helper now resolves effective status
  from `proposal_status` records.
- **`apply_proposal` scaffolding hardened** with sentinel-comment insertion
  points and `ast.parse` validation with rollback, instead of structure-
  dependent pattern matching.
- **`/api/chain/verify` no longer blocks the event loop** — dispatched to a
  worker thread with its own read-only SQLite connection
  (`verify_threadsafe`); requires a session token.
- **`/blobs/<sha>` is authenticated and indexed** — requires the session
  token, and resolves via `blob_index` instead of a capped linear scan.
- **`/api/session/claim` is per-IP rate-limited** against claim-storm DoS.
- **`EmbeddingIndex.search` no longer passes `n_neighbors=0` to sklearn**
  (callers like `drift_against` can pass `k=0`); returns `[]`.
- **Path traversal defense (defense-in-depth).** `verify_threadsafe`
  percent-quotes the DB path so a data directory containing URI-reserved
  characters can't silently open a different database; and both
  `Agent._collect_attachments` and `/blobs/<sha>` reject `blob_path` values
  containing path separators or leading dots and assert the resolved path
  sits under the blob directory. Neither was reachable on the documented
  happy path, but the agent's contract is "trust the chain," so the consumer
  side validates too.
- **Continue-after-truncation.** When a response was cut off at `max_tokens`
  and the user typed "continue," the truncated flag wasn't persisted to the
  response record's `_meta`, so the next turn had no signal the prior
  response was incomplete and the model reasoned aloud about what "continue"
  meant. `build_meta`/`read_meta` now carry `truncated`, and
  `_format_prompt` injects an explicit resume directive when the user asks
  to continue a flagged response.
- **`RECENT_N` was a dead config knob** — `run.py` defined it but
  `prepare_turn` hardcoded `n_recent=3`, so every deployment ran a 3-record
  recent window regardless. Now wired through `turn`/`prepare_turn` and the
  webapp endpoints. Also added an explicit-reference parser so "record N",
  "#N", and list forms pull the named records directly (semantic search
  can't match a numeric address against record *content*); the parser is
  conservative on bare numbers, years, and hex colors to avoid false pulls.

---

## v1.2

Five capabilities from the project's build spec, all layered on top of
the existing architecture rather than altering it. The framing is
deliberately neutral engineering language: the detectors measure observable
properties of text, "Proof-of-Quality" is a quality score, and no module
claims consciousness or phenomenology. The cryptographic core is untouched
and every prior test still passes; v1 and v1.1 chains read cleanly.

### Added

- **Modalities & senses (`signals.py`).** A dependency-free text-analysis
  layer of pure detectors scoring input on intent, coherence, contradiction,
  vulnerability, and — the load-bearing one — prompt-injection risk. Returns
  a `SignalReport`.
- **Proof-of-Quality (`poq.py`).** A pre-commit quality gate. Every response
  is scored before it is committed to memory; a response judged an injection
  attempt is committed *quarantined* so it never feeds future retrieval. PoQ
  gates memory, not replies — the user always sees the response.
- **Protected zones (`protected_zones.py`).** A memory-integrity boundary:
  genesis, system-prompt, and principle records cannot be revised by an
  ordinary turn, and quarantined records are filtered out of retrieval.
- **Cambium (`cambium.py`).** A growth mechanism that scans history for
  recurring gaps (repeated corrections, repeated failures, contradiction
  clusters, repeated confusion) and emits *proposals* for new skills,
  modalities, senses, or principles. Cambium proposes; it never applies.
  Recurrence tracking escalates a proposal's visibility (never its
  authority) once the same pattern is seen three times.
- **Proposal review tool (`apply_proposal.py`).** Operator-run. Lists and
  shows proposals; `--accept` scaffolds a detector *stub* into `signals.py`
  (correct signature, registered, `# TODO` body) and records the decision on
  the chain; `--decline` records a decline. The tool never writes working
  detector logic — a human implements the body and adds a test. That
  irreducible human step is the deliberate safety boundary: the agent never
  modifies its own running code.

### Changed

- `_meta` schema bumped to v3: added `epistemic_class`, `exposure` (the
  protected-zone primitive), and an optional `poq` block. New record types
  `principle`, `proposal`, `proposal_recurrence`, `proposal_status`.
- `commit_genesis` extended with `agent_name`, `purpose`, `covenant`, and a
  derived `covenant_hash`; the v1 `commitments` field is still written.
- `turn()` scores every response with PoQ and quarantines confirmed attacks.
- `retrieval.hybrid()` gained a risk-penalty term; `build_context()` filters
  quarantined records.
- New REPL commands `/cambium` and `/proposals`; new `AUTO_CAMBIUM_EVERY`
  cadence, separate from and longer than auto-reflection.

### Migration

None required. The chain format is unchanged; the new metadata fields live
inside the record `content` block, so the embedding store does not need
rebuilding.

---

## v1.11

A small, additive release. Cryptographic core, chain, retrieval scoring,
metadata, and reflection all unchanged; fully backward-compatible.

### Added

- **Tiered embedder fallback.** The embedder is resolved at startup:
  `OllamaEmbedder` (real semantic embeddings, runs in Ollama's process so no
  PyTorch enters the agent) if a local server is reachable, otherwise the
  dependency-free `HashingEmbedder`. The agent never fails to start for lack
  of an embedder, and `run.py` reports which tier it selected.
- **OpenRouter and DeepSeek providers** (`make_openrouter_client`,
  `make_deepseek_client`), both OpenAI-compatible so they reuse the `openai`
  SDK against a different base URL — no new dependency.

### Changed

- Removed the hard `sentence-transformers` (and therefore PyTorch)
  dependency; the 1–3 GB install is no longer required and the agent no
  longer exits at startup if it's absent. `EMBED_DIM` is gone — the active
  dimension is whatever the resolved embedder reports, guarded against a
  mismatch with an existing store.

### Migration

The chain is untouched. If the resolved embedder's dimension differs from
the one that built the embedding store, `EmbeddingIndex` stops at startup
with an instruction to delete the store; deleting it is safe — it rebuilds
from the chain.

---

## v1.1

A "Tier 1 sharpening" of v1's architecture, not a redesign. The principle
**records are evidence, beliefs are derived** — which v1 implemented in
spirit — is made explicit and per-record. Cryptographic core unchanged.

### Added

- `metadata.py` (new): the `_meta` block schema, source enum, salience
  defaults, half-life table, and the `read_meta` fallback reader for legacy
  records.
- Per-record `_meta` block on every new record: `schema_version`, `source`
  (user / assistant / system / tool — the load-bearing distinction),
  `salience`, `confidence`, `supersedes`.

### Changed

- Salience moved from a hardcoded per-type table to per-record values
  written at append time (type defaults still apply to legacy records).
- Recency moved from uniform linear decay to per-kind half-life decay
  (`0.5 ** (age_days / half_life_days)`): conversational records decay over
  weeks, reflections over months, genesis and system prompts effectively
  never.
- Retrieval demotes superseded records (`−0.30`) and auto-pulls their
  corrections so the model always sees original and correction together.
- Context-budget truncation switched to per-record salience.
- Reflection windows size themselves dynamically (every record since the
  previous reflection, with a safety cap) instead of a fixed lookback.

### Migration

Drop in the new files; leave the existing chain alone. Legacy records (no
`_meta`) read cleanly via `read_meta`'s in-memory fallback and are never
rewritten. New appends carry the `_meta` block; the two coexist on the same
chain indefinitely and `/verify` still passes.

---

## v1

Initial working implementation: a persistent-memory AI agent whose memory
lives in a hash-chained, Ed25519-signed, append-only log rather than chat
history or a vector-store-only RAG. SQLite storage, SHA-256 content hashing
and prior-record linking, Merkle batching with optional OpenTimestamps
anchoring to Bitcoin, sealed founding commitments at genesis plus a mutable
but chain-logged system prompt, a reflection loop, revision records,
temporal awareness, salience-weighted hybrid retrieval, file ingestion, a
pluggable multi-provider LLM layer, an optional FastAPI web UI, and a pytest
suite covering chain integrity, tamper detection, Merkle proofs, retrieval,
and agent workflows.
