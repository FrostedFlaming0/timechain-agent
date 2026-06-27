# Contributing

Thanks for considering a contribution. This is a prototype, so the bar for
"is this worth doing" is relatively low — but the bar for "does it preserve
the integrity guarantees of the chain" is high. Most of the rules below are
aimed at that distinction.

This document tracks the current codebase (v1.2.1). For the release-by-release
history — what each version added and why — see `CHANGELOG.md`. If you're
working from an older mental model of the project, start there.

## Before you start

Read `ARCHITECTURE.md` first. The project is small but layered, and most
review friction comes from PRs that work against the layering rather than
with it. The one-line summary: **chain knows nothing about LLMs, retrieval
knows nothing about prompts, agents glue them together, `metadata.py` is a
pure schema module read by both retrieval and agent without coupling them,
`run.py` configures everything.** If your change blurs those lines, expect
questions.

For a quick orientation:

| Layer | File | Touches |
|-------|------|---------|
| Storage | `chain.py` | SQLite, Ed25519, hashes, Merkle batches |
| Metadata schema | `metadata.py` | `_meta` block layout, source enum, salience defaults, half-lives, v1-fallback reader |
| Retrieval | `retrieval.py` | embeddings, similarity, salience, recency, revision-aware demotion |
| Agent | `agent.py` | turn loop, reflection, revision, prompt formatting, metadata writes |
| Ingestion | `file_ingest.py` | file readers, blob store |
| LLM clients | `llm_clients.py` | provider adapters |
| Entry point | `run.py` | configuration, REPL |
| Web UI | `timechain_web/webapp.py` + `static/index.html` | FastAPI server, SSE streaming, frontend |

## Setup

```bash
git clone <your fork>
cd timechain
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

Run the tests to confirm everything works before you change anything:

```bash
pytest test_timechain.py -v
```

If you can't install pytest in your environment, `python run_tests.py` is a
standalone fallback. Both should pass on a clean checkout.

The v1 and v1.1 test suites pass unchanged on v1.11; if you're upgrading an
existing fork to v1.11 and a previously-passing test fails, that's a
regression, not expected
behavior.

## What kinds of contributions are welcome

In rough order of how easy they are to land:

- **Bug fixes** with a failing test that turns green. Almost always merged.
- **New file extractors** in `file_ingest.py`. Add the extension to the
  appropriate `*_EXTS` set, write an `_extract_*` function that returns
  `(text, method)`, and add a test. Self-contained, low risk.
- **New LLM providers** in `llm_clients.py`. Follow the existing
  `make_*_client` pattern. Keep the dependency optional (import inside the
  factory, not at module top).
- **Retrieval improvements** — better salience defaults, time-window
  filters, smarter context budgeting, tuned score-formula weights. These
  need benchmarks or at least a before/after example showing why the
  change is an improvement.
- **Metadata schema extensions** — adding fields to `_meta` (e.g. topic
  tags, sensitivity flags). Lower risk than a new record type, but
  requires keeping `read_meta`'s v1-fallback path intact and ensuring
  existing v1.1 records remain readable. See "Things that need extra
  care" below.
- **Documentation, examples, type hints, error message improvements.** Always
  welcome.
- **New record types or schema changes.** Possible but expensive — see the
  "Things that need extra care" section below.

If you're not sure whether something is in scope, open an issue describing
what you want to do before writing the code. The README's "Status" line
("prototype, works end-to-end, not production-hardened") is the honest
description; we're not chasing feature completeness.

## Things that need extra care

**Anything in `chain.py`.** The chain is the source of truth. A bug here
isn't "the agent answered wrong," it's "memory is silently corrupted."
Changes to record format, signature scheme, or hash linkage need:

- A clear migration story for existing chains (or a justification for why
  none is needed).
- Tests that cover both the new behavior and tamper detection against the
  new format.
- A `/verify` walk that still passes on a chain built with the change.

**Anything in `metadata.py`.** This module is pure schema, but it's loaded
by every read of the chain. Two rules:

- **Never rewrite records on disk.** `read_meta` synthesizes defaults in
  memory for records that predate a schema change. That's the v1 → v1.1
  pattern, and any future schema change should follow it. A v1 record
  read by v3 code should still produce a sensible `RecordMeta` without
  modifying the record itself.
- **Never invent values for old records that look like they were always
  there.** A v1 record didn't have a salience score; if your code needs
  one, give it the type-default and flag `is_default=True` on the
  resulting `RecordMeta`. Don't pretend the field existed.

**The analysis modules (`signals.py`, `poq.py`, `cambium.py`).** These are
pure-logic modules added in v1.2 — no I/O, no chain dependency. Two rules:

- **Detectors must stay deterministic and dependency-free.** `signals.py`
  is run on every turn and its results feed `poq.py`. A detector that
  calls the network, depends on an ML model, or returns nondeterministic
  output breaks both the offline test suite and reproducibility. Lexicon
  and regex only.
- **Cambium proposes; it never applies.** `cambium.py` may emit `proposal`
  records suggesting new structure. It must never edit code, change
  behavior, or auto-apply a suggestion. The split — model proposes, a
  human or a privileged process decides — is deliberate; keep it.

**Adding a record type.** Not hard, but has ripple effects. You'll need to:

1. Decide on a type name (lowercase, underscored).
2. Add a default salience and source mapping in `metadata.py`
   (`DEFAULT_SALIENCE_BY_TYPE` and `DEFAULT_SOURCE_BY_TYPE`).
3. Add a half-life entry in `metadata.py`'s `DEFAULT_HALF_LIFE_DAYS_BY_TYPE`
   if the new type should decay differently from the default 30 days.
4. Update `view_chain.py`'s `fmt_content` if the type has non-obvious
   content shape.
5. Document the type in `ARCHITECTURE.md`'s record-types table.
6. Add a test that round-trips the type through the chain and retrieval.

In v1, this list also included entries on `Retriever.DEFAULT_SALIENCE` and
`Agent._RETENTION_PRIORITY`. Those constants are gone in v1.1 — salience
defaults moved to `metadata.py`, and truncation now reads per-record
salience directly. Any old contributing notes referring to those names
should be updated.

**Adding a field to `_meta`.** Lower-risk than adding a record type but
not free:

1. Add the field to `RecordMeta` and `build_meta` in `metadata.py`.
2. Give it a sensible default in `read_meta` so v1 and v1.1 records that
   predate the field continue to read cleanly.
3. Bump `CURRENT_SCHEMA_VERSION` if the field is required for new records.
4. Add a test that a v1 (no `_meta`) record and a v1.1 (`_meta` without
   the new field) record both read with the default value.

**Cryptographic changes.** Don't roll your own. The current primitives
(Ed25519, SHA-256, Merkle trees over canonical JSON) are deliberately
boring. If you want to add post-quantum signatures, change the canonical
serialization, or alter the Merkle construction, that's a design discussion
in an issue first, not a PR.

**Anything that could break tamper-evidence.** If a change makes it possible
to modify a record without `chain.verify()` catching it, that's a bug
regardless of how convenient the change is. The tamper-detection tests in
`test_timechain.py` exist for this reason — they should keep passing, and if
your change requires loosening them, justify it.

**Score-formula weight changes.** The hybrid retrieval score is
`W_SEMANTIC*similarity + W_SALIENCE*salience + W_RECENCY*recency − supersession_penalty`,
with the constants on `Retriever`. Changing these affects what records
the LLM sees on every turn — so a PR that retunes them should include:

- A description of the symptom you're trying to fix (e.g. "reflections
  never surface alongside fresh observations").
- Before/after retrieval output on at least one realistic chain.
- A note in the PR description acknowledging that downstream chains will
  see different ranking after the change.

## Coding style

The codebase has a consistent feel; please match it.

- **Module docstrings explain what the file is for and how it fits in.**
  Look at the top of `metadata.py`, `file_ingest.py`, or `retrieval.py`
  for examples. New modules should do the same.
- **Comments explain *why*, not *what*.** If a line of code is non-obvious,
  the comment should answer "why is this here" rather than restate the
  syntax.
- **Type hints on public functions.** `from __future__ import annotations`
  is used throughout — keep that convention.
- **Dataclasses for structured returns.** See `IngestResult`, `RetrievalHit`,
  `Record`, `RecordMeta`. Tuples are fine for two-element returns; past
  that, use a dataclass.
- **No bare `except:`.** Catch specific exceptions. The one place we catch
  broad `Exception` is in `_extract_text`, and that's documented as a
  fallback path that records the failure on the chain rather than crashing.
- **Optional dependencies are imported inside the function that needs them**,
  not at module top. See the `_extract_pdf`, `_extract_docx` etc. pattern in
  `file_ingest.py`. This keeps `pip install -r requirements.txt` lean for
  users who don't need every extractor.
- **No new top-level dependencies without a strong reason.** If you need a
  library for one optional feature, add it as an optional dep (commented in
  `requirements.txt` with a note) and import it lazily.
- **Plain language in user-facing strings.** Match the tone of the existing
  REPL output and error messages: direct, lowercase, no exclamation marks,
  no emoji. The system prompt in `run.py` describes the tone we're aiming
  for in the agent itself; the codebase tries to match.

## Tests

Every PR that changes behavior needs a test. The suite in
`test_timechain.py` is grouped by concern (chain integrity, Merkle batching,
retrieval, agent behavior, time formatting). Add to the appropriate group
or create a new `TestX` class if your change opens a new concern.

Conventions:

- Use the existing fixtures (`workdir`, `chain`, `index`, `agent`) when
  possible. They handle setup and teardown, and `run_tests.py` knows about
  them.
- If you add a new fixture, also teach `run_tests.py`'s `build_fixtures`
  about it, otherwise the standalone runner will skip your tests.
- Keep tests deterministic. The `HashingEmbedder` (trigram-bag) is used in
  tests precisely because it has no model or network dependency. Note that
  it hashes trigrams with the builtin `hash()`, which is salted by
  `PYTHONHASHSEED` — so the exact ranking of near-tied results is not stable
  across interpreter runs. (v1.1 had a known flake here in
  `test_search_returns_results`; v1.11 fixed it by querying with a
  substantive phrase the embedder can represent stably rather than a single
  word.) If you write a new retrieval test, query with enough text to embed
  meaningfully, and prefer asserting "expected record is in top-k" over
  "expected record is at index 0" whenever scores could be close.
- The v1.11 `TestEmbedderFallback` group assumes no Ollama server is running
  (true in CI and sandboxes). If you add embedder tests, don't make them
  depend on a reachable Ollama server — the suite must pass offline.
- For cryptographic tests, the bar is "this catches the specific tampering
  vector I claim it catches." See `TestTamperDetection` for the pattern:
  tamper, then assert `verify()` returns False with a message that names
  the failure mode.
- For metadata tests, exercise both the v2 path (record with `_meta`) and
  the v1 fallback (record without). `read_meta` should produce sensible
  values in both cases.

Run the full suite before opening a PR:

```bash
pytest test_timechain.py -v
```

And for a sanity check of the standalone runner:

```bash
python run_tests.py
```

## Pull requests

Keep PRs focused. One conceptual change per PR. A PR titled "fix bug + add
feature + refactor retrieval" will get split before review.

A good PR description answers four questions:

1. What does this change?
2. Why is the change worth making?
3. What did you test, and what's the test output?
4. What did you *not* change that someone might expect you to have changed?
   (This catches scope creep early.)

If your change is user-visible — a new slash command, a new config knob, a
new file extension supported, a new `_meta` field — update `README.md` and
`ARCHITECTURE.md` in the same PR. Documentation that lags behind the code
is a bug.

If your change touches `chain.py` or any tamper-evidence guarantee, also
include the output of `pytest test_timechain.py::TestTamperDetection -v` in
the PR description. Reviewers will run it anyway, but having it in the
description shows you ran it too.

If your change touches `metadata.py`, run a quick smoke test confirming
that records appended *before* your change still read cleanly through
`read_meta` *after* your change. The non-destructive migration rule is the
one rule of metadata changes.

## Reporting issues

For bugs, the most useful issue includes:

- What you ran (the command line, the config in `run.py` if you changed it).
- What you expected.
- What happened instead, including any traceback verbatim.
- The output of `python view_chain.py --verify`, if the issue might involve
  chain state.
- The output of `python view_chain.py --record N` for a record exhibiting
  the issue, if it's a retrieval or metadata issue.
- Your Python version and OS.

For feature requests, describe the use case before the proposed solution.
"I want to do X but the current architecture makes it hard because Y" is a
much more useful starting point than "please add feature Z."

Security issues — anything that could let an attacker forge records, defeat
`/verify`, or leak the operator key — should not go in the public issue
tracker. Email the maintainer privately first.

## Licensing and attribution

This project is MIT-licensed (see `LICENSE`). By submitting a PR you agree
your contribution is licensed under the same terms.

The conceptual lineage is documented in `README.md` under "Credit and
lineage." If your contribution draws on ideas from other published work,
add the attribution there in the same PR. Standard cryptographic primitives
and well-known ML techniques don't need attribution; novel architectural
ideas do.

## A note on scope

The README is honest that this is a prototype. The temptation with a
project like this is to grow it into a framework — to add plugins, config
files, a service mode. We're trying to resist that. The value of
this codebase is that you can read it end-to-end in an afternoon and
understand exactly what it does. Contributions that preserve that property
get reviewed faster than ones that don't.

The v1 → v1.1 upgrade tried to model what "good" looks like here:
sharpening an existing concept (per-type → per-record salience) without
adding new top-level dependencies, without breaking existing chains, and
without expanding the scope of any single module. That's the bar.
