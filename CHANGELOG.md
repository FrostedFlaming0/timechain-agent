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
  Known follow-up: `_truncate_to_budget` is not yet chunk-aware.
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

Five capabilities from the Cypher Tempre build spec, all layered on top of
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
