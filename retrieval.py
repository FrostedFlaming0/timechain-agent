"""
retrieval — access patterns over a chain, designed for feeding records to an LLM.

Implements the patterns we discussed:
  - Hybrid retrieval: semantic similarity + structural filters
  - Ancestry-aware retrieval: follow refs backward
  - Temporal windowing: recent / time-range queries
  - Type-filtered retrieval
  - Drift detection: compare current statements against sealed prior commitments

Vector search uses sklearn's NearestNeighbors over numpy arrays for the
prototype — replace with FAISS/Qdrant/pgvector at scale. Embeddings are
pluggable: provide any callable that maps str -> np.ndarray.

Two embedders ship in this module:
  - `HashingEmbedder` — dependency-free bag-of-trigrams. Deterministic,
    no model, no network. Used by the test suite and as the offline
    fallback. Not a real semantic embedder.
  - `OllamaEmbedder` — calls a local Ollama server's embeddings endpoint
    (default model `nomic-embed-text`). Real semantic embeddings, runs
    fully on the user's machine, no heavy ML stack in-process.

`run.py` resolves which one to use at startup via `make_tiered_embedder()`:
it probes for a reachable Ollama server and uses `OllamaEmbedder` if one is
found, otherwise falls back to `HashingEmbedder`. The `requests` dependency
for the Ollama path is imported lazily so it stays optional.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

from chain import Chain, Record
from metadata import read_meta, half_life_days, EXPOSURE_QUARANTINE


class EmbeddingStoreMismatchError(ValueError):
    """The persisted vectors are incompatible with the active embedder."""


# ---------------------------------------------------------------------------
# Modality anchoring
# ---------------------------------------------------------------------------

# Which modality detectors (signals.py) participate in retrieval anchoring.
# Only DOMAIN modalities belong here — ones that describe what KIND of work a
# record represents (code, structured output, and future kinds like
# narrative or analytical). QUALITY modalities (integrity_field, coherence)
# describe how good/safe a record is, not its domain, and must be excluded:
# a query that happens to look injection-y must NOT preferentially retrieve
# past injection-flagged records. Names are the bare SignalHit.name values
# (e.g. "artifact_content", not "m_artifact_content" — the m_ prefix is the
# function name, not the modality name stored in _meta).
DOMAIN_MODALITIES: set[str] = {"artifact_content"}

# Maximum number of distinct domain modalities that may participate in
# anchoring for a single query. Once sprouting is live a query could match
# many domain modalities at once; without a cap the modality term would
# blur across all of them and dilute the signal. The cap keeps the boost set
# tight: query_modalities ranks candidate modalities by detection activation
# and keeps the top N. Tunable via run.py's PER_TURN_MODALITY_CAP.
PER_TURN_MODALITY_CAP = 7

# Anti-echo saturation control. At retrieval time, if a large fraction of the
# top candidates ALREADY carry the query's modality, boosting them further
# just piles up "more of the same" — the self-reinforcing loop that auto-
# sprouted modalities make easy to fall into. When saturation exceeds this
# threshold, the modality term is damped in proportion to the excess (see
# Retriever.hybrid), so a context that is already saturated gets little or no
# additional modality boost. 0.6 = "start damping once >60% of the top
# candidates are already in this mode."
MODALITY_SATURATION_THRESHOLD = 0.6

# How many top candidates the saturation measurement looks at. Measured on
# the base-scored set (before the modality term is applied) so the damping
# reflects the genuinely-relevant context, not context the boost itself
# created.
MODALITY_SATURATION_TOP_N = 10

# Returned by modality_overlap when there is no information to compare on
# (query or record carries no domain modalities). Neutral prior: the term
# neither boosts nor penalizes. 0.5 because the term contributes
# W_MODALITY * overlap to the score; a neutral 0.5 lands a record exactly
# between a perfect-match boost (overlap 1.0) and a total-mismatch cut
# (overlap 0.0), so a record with no modality data is treated as "unknown,"
# not "mismatched."
MODALITY_NEUTRAL = 0.5


def modality_overlap(
    query_mods: Optional[set],
    record_mods: Optional[set],
) -> float:
    """
    How much the *kind* of work in a record matches the kind the current
    query implies — in [0, 1], where 0.5 is the neutral "no information"
    prior (see MODALITY_NEUTRAL).

    Both arguments are sets of domain-modality names (already filtered to
    DOMAIN_MODALITIES by the caller). This is set overlap, not cosine over
    score vectors: `modalities_activated` is stored as a thresholded list of
    NAMES (those above the analyzer's activation floor), not as a scored
    dict, so the activation magnitudes aren't on disk to do cosine with. For
    the current single-element domain set, set overlap and cosine are
    behaviorally identical anyway — gradation only matters once there are
    several competing domain modalities. The containment measure below
    upgrades cleanly to weighted overlap if `modalities_activated` ever
    stores scores and DOMAIN_MODALITIES grows.

    Measure: size of intersection over size of the query's domain set
    (containment), i.e. "what fraction of the modes this query is in did the
    record also exhibit." Using the query set as the denominator (rather
    than the union, plain Jaccard) means a record that exhibits the query's
    mode scores 1.0 even if it also exhibits other modes — we don't penalize
    a code response for also being, say, explanatory.

    Returns MODALITY_NEUTRAL when either side has no domain modalities: a
    neutral query (no detected mode) shouldn't reorder anything, and a
    record with no recorded mode (older records, observations, reflections)
    is "unknown," not "mismatched."
    """
    if not query_mods or not record_mods:
        return MODALITY_NEUTRAL
    inter = query_mods & record_mods
    if not inter:
        # Both sides have domain modalities, but none in common: a genuine
        # mismatch (e.g. a code query against a pure-prose response that
        # nonetheless carried some other domain mode). Pull toward 0.0 so
        # the term acts as a mild cut, symmetric with the boost.
        return 0.0
    return len(inter) / len(query_mods)


# Historical: a silent cap on revision scans, used before the materialized
# `supersedes_index` table existed in chain.py. Both the supersession and
# pull-in lookups now go through `Chain.superseded_indices()` and
# `Chain.revisions_targeting()`, which are indexed and unbounded. Kept
# exported only for backward compatibility with any external code that
# imports it; nothing in this codebase reads it any more.
REVISION_SCAN_LIMIT = 10_000


Embedder = Callable[[str], np.ndarray]


# ---------------------------------------------------------------------------
# Embedding store
# ---------------------------------------------------------------------------

EMBED_SCHEMA = """
-- One row PER CHUNK, not per record. A long record (a big file, a long
-- response, a long user paste) is split into multiple ~chunk-sized text
-- windows at index time, each embedded into its own vector row. All
-- chunks of one record share the same `record_idx` — that column is the
-- group anchor. `chunk_index` orders the chunks within a record (0-based);
-- `chunk_count` records how many chunks that record produced.
--
-- Why chunk: a single 5000-char-truncated vector loses everything past
-- the cap, so long documents and code blocks were effectively invisible
-- to retrieval beyond their opening. Chunking embeds the whole record.
--
-- Why this does NOT reintroduce the "long records get more raffle
-- tickets" problem: search() collapses a record's chunk hits down to a
-- single per-record similarity (the MAX over its chunks) BEFORE returning
-- to hybrid(). So a 30-chunk record competes for exactly one slot, scored
-- by its single most relevant fragment — see EmbeddingIndex.search.
--
-- This is an index-only change. The CHAIN still stores one signed record
-- per response/file; chunks exist only in this derived store and never
-- touch the append-only log. Rebuilt from the chain whenever deleted.
CREATE TABLE IF NOT EXISTS embeddings (
    embed_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    record_idx  INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 1,
    record_hash TEXT NOT NULL,
    vector      BLOB NOT NULL,
    text        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_record
    ON embeddings(record_idx);

-- Small key/value table for embedder identity tracking. Lets the
-- EmbeddingIndex detect not just dimension mismatches (caught by the
-- existing vector-shape guard) but the more subtle case where the
-- embedder's COORDINATE SPACE changed at the same dimension — e.g. the
-- HashingEmbedder migration from the (process-randomized) builtin
-- `hash()` to BLAKE2b. Vectors stay 256-dim either way, so dimension
-- equality is uninformative; only an explicit identity tag catches the
-- silent-noise scenario.
CREATE TABLE IF NOT EXISTS embed_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# Bump when chunk_text's boundary logic changes in a way that would change
# chunk boundaries at the same target size. Combined with CHUNK_TARGET_CHARS
# in the embedder-identity tag, so any chunking change forces a store
# rebuild rather than silently mixing old and new chunk boundaries.
CHUNK_SCHEME_VERSION = 1

# Target chunk size in characters for splitting a record before embedding.
#
# Must sit safely under OLLAMA_EMBED_MAX_CHARS (5000): every chunk becomes
# one Ollama request, and Ollama 500s on input that overflows the model's
# token window rather than truncating. 3500 chars is ~1150 tokens for
# dense 3-chars/token text — comfortable margin under the ~2048-token
# window, while still capturing a substantial span per chunk so chunk
# count stays low for typical records. The HashingEmbedder fallback has no
# request-size limit, so this is purely the Ollama-path ceiling; using the
# same value for both keeps chunk boundaries identical across embedders,
# which means a store rebuilt after an embedder swap chunks the same way.
CHUNK_TARGET_CHARS = 3500

# Hard ceiling for a single chunk. After boundary-aware splitting, any
# piece still longer than this (e.g. a 9000-char line with no whitespace —
# minified JS, a base64 blob in extracted text) is hard-sliced so no chunk
# can ever exceed the Ollama request cap. Set equal to the target: the
# boundary splitter aims for the target, and the hard-slice guarantees it.
CHUNK_HARD_MAX_CHARS = CHUNK_TARGET_CHARS


def _extract_explicit_indices(query: str) -> list[int]:
    """
    Pull explicit record-index references out of a user query.

    Catches "record 328", "record #328", "index 12", "indices 12 and 87",
    "#42", "what you said at 87". Returns a list of unique indices in
    first-mention order (so the most natural reading is preserved when
    the caller does a stable dedup).

    Deliberately conservative on the bare-number case: we ONLY treat a
    standalone "#42" as a reference; a bare "42" with no leading hash
    or word like "record"/"index" is left alone, because numbers in
    conversational text usually aren't record references ("I have 3
    pets", "two months ago", "in 2024"). False positives here would be
    annoying — the user's reference to a year would silently pull in
    some unrelated record N — so the patterns require a clue.

    Indices outside the chain are not validated here; the caller
    (`Retriever.build_context`) checks bounds and drops misses
    silently. This keeps the parser pure and testable.
    """
    if not query:
        return []
    out: list[int] = []
    seen: set[int] = set()

    # Patterns:
    #   "record 328"      -> 328
    #   "record #328"     -> 328
    #   "records 1, 2, 3" -> 1, 2, 3
    #   "index 12"        -> 12
    #   "indices 12 and 87" -> 12, 87
    #   "#42"             -> 42
    #   "at record 87"    -> 87
    #
    # Word boundary on the keyword prevents matching "indexing" or
    # "recorded". Each match consumes the keyword + one or more
    # comma/and-separated numbers immediately following it.
    keyword_pattern = re.compile(
        # Keyword + first number + zero or more (separator + next number).
        # Separator is comma and/or "and" with optional spaces, so this
        # handles "a, b", "a and b", "a, b, c", and the Oxford-comma
        # "a, b, and c" form. Each subsequent number may carry its own
        # `#` prefix.
        r"(?:record|index|indices|records|idx)\s*#?\s*"
        r"(\d+(?:\s*(?:,\s*and|,|and)\s*#?\d+)*)",
        re.IGNORECASE,
    )
    for m in keyword_pattern.finditer(query):
        for num in re.findall(r"\d+", m.group(1)):
            n = int(num)
            if n not in seen:
                seen.add(n)
                out.append(n)

    # Bare "#42" — a number preceded by # with a non-digit (or start of
    # string) before the #, so we don't match the middle of e.g. an
    # HTML color "#ff8a42".
    hash_pattern = re.compile(r"(?:^|[^\w#])#(\d+)\b")
    for m in hash_pattern.finditer(query):
        n = int(m.group(1))
        if n not in seen:
            seen.add(n)
            out.append(n)

    return out


def _embedder_identity(embedder, dim: int) -> str:
    """
    Stable identity string for an embedder instance. Persisted in the
    embedding store on first write and compared on every reopen to
    detect coordinate-space changes that don't manifest as dimension
    changes.

    Format: `<class-name>:<dim>:<variant>`. The variant tag is BLAKE2b
    on a probe string for hashing-based embedders (so the
    process-randomized vs deterministic forms of HashingEmbedder
    distinguish themselves), and the model name for hosted embedders
    that expose one.

    The point is uniqueness across coordinate-space changes, not
    cryptographic strength — a 16-hex-char digest is plenty.
    """
    cls = type(embedder).__name__
    variant = ""
    # Hosted embedders typically carry a `model` attribute.
    model = getattr(embedder, "model", None)
    if isinstance(model, str) and model:
        variant = f"model={model}"
    else:
        # Hashing-style or anonymous embedder: probe its output on a
        # fixed string. If the bucketing changed (e.g. randomized vs
        # BLAKE2b), the probe vector changes and the identity flips.
        try:
            probe_vec = embedder("identity-probe")
            digest = hashlib.blake2b(
                probe_vec.tobytes() if hasattr(probe_vec, "tobytes")
                else str(probe_vec).encode("utf-8"),
                digest_size=8,
            ).hexdigest()
            variant = f"probe={digest}"
        except Exception:
            variant = "probe=unavailable"
    # The chunking scheme is part of the store's identity. If the chunk
    # size or boundary logic changes, an existing store's chunk boundaries
    # no longer match what the current code would produce, so it must be
    # rebuilt. Bumping CHUNK_TARGET_CHARS flips this tag automatically and
    # the constructor's identity check turns the silent-mismatch into a
    # clear "delete the store" error. Bump CHUNK_SCHEME_VERSION on any
    # change to chunk_text's boundary logic that doesn't change the target.
    chunk_tag = f"chunk=v{CHUNK_SCHEME_VERSION}.{CHUNK_TARGET_CHARS}"
    return f"{cls}:{dim}:{variant}:{chunk_tag}"


def chunk_text(text: str, target: int = CHUNK_TARGET_CHARS) -> list[str]:
    """
    Split text into chunks of roughly `target` characters, preferring
    natural boundaries so a chunk break doesn't slice through the middle
    of a sentence or paragraph more than necessary.

    Boundary preference, in order:
      1. Paragraph breaks (blank lines) — accumulate paragraphs into a
         chunk until adding the next would exceed `target`.
      2. Sentence breaks — for a single paragraph already over `target`,
         split on sentence-ending punctuation.
      3. Hard slice — for a single sentence/run still over the hard max
         (a long unbroken line: minified code, base64, a CSV row), cut at
         CHUNK_HARD_MAX_CHARS so no chunk can overflow the embedder cap.

    Short text (<= target) returns a single chunk, so a normal turn still
    produces exactly one vector and behaves identically to the pre-chunking
    store. Empty/whitespace-only text returns [""] so the record still gets
    one (empty) vector row rather than vanishing from the index entirely.
    """
    if not text or not text.strip():
        return [""]
    if len(text) <= target:
        return [text]

    def hard_slice(s: str) -> list[str]:
        return [s[i:i + CHUNK_HARD_MAX_CHARS]
                for i in range(0, len(s), CHUNK_HARD_MAX_CHARS)]

    def split_sentences(s: str) -> list[str]:
        # Split after sentence-ending punctuation (., !, ?) optionally
        # followed by a closing quote/bracket, when trailed by whitespace.
        # A capturing split keeps the delimiter, which we reattach to the
        # preceding sentence — avoids the variable-width lookbehind that
        # Python's re rejects.
        tokens = re.split(r'([.!?][\'")\]]?)\s+', s)
        out: list[str] = []
        i = 0
        while i < len(tokens):
            body = tokens[i]
            delim = tokens[i + 1] if i + 1 < len(tokens) else ""
            sent = (body + delim).strip()
            if sent:
                out.append(sent)
            i += 2
        return out or ([s] if s.strip() else [])

    # Pass 1: paragraphs.
    paragraphs = re.split(r'\n\s*\n', text)
    chunks: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # A single paragraph bigger than target: flush what we have, then
        # break the paragraph down by sentence (pass 2), then hard-slice
        # any sentence still too long (pass 3).
        if len(para) > target:
            flush()
            for sent in split_sentences(para):
                if len(sent) > CHUNK_HARD_MAX_CHARS:
                    flush()
                    chunks.extend(hard_slice(sent))
                elif len(buf) + len(sent) + 1 > target:
                    flush()
                    buf = sent
                else:
                    buf = f"{buf} {sent}".strip()
            flush()
            continue
        # Normal paragraph: pack into the buffer until it would overflow.
        if len(buf) + len(para) + 2 > target:
            flush()
            buf = para
        else:
            buf = f"{buf}\n\n{para}".strip()
    flush()
    return chunks or [""]


class EmbeddingIndex:
    """
    SQLite-backed embedding store with in-memory ANN over current vectors.
    Rebuilds the ANN index on demand — fine for prototypes up to ~100k records.

    Stores one vector PER CHUNK (see EMBED_SCHEMA and chunk_text). A record
    is split into chunks at index time; search() collapses chunk hits back
    to one similarity per record (max over chunks) so long records do not
    crowd out short ones. The chain remains one signed record per turn/file;
    chunking lives entirely in this derived store.
    """

    def __init__(self, db_path: str | Path, embedder: Embedder, dim: int):
        self.db_path = str(db_path)
        self.embedder = embedder
        self.dim = dim
        # check_same_thread=False: mirrors chain.Chain — the webapp runs
        # long tool executions (which touch per-task indexes) in worker
        # threads via asyncio.to_thread; access is serialized under its
        # state.lock, never concurrent.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(EMBED_SCHEMA)
        self._conn.commit()
        self._nn: Optional[NearestNeighbors] = None
        self._idx_to_record: list[int] = []
        # Per-call cache: `{record_idx: [(chunk_idx, similarity), ...]}` for
        # records returned by the most recent search(), sorted by similarity
        # desc. Lets the renderer ask "which chunks of this record matched"
        # without changing search()'s long-standing return contract. Reset
        # on each search() call; readers must consume before the next call.
        # Empty dict before the first search.
        self.last_chunk_matches: dict[int, list[tuple[int, float]]] = {}

        # Guard against a dimension mismatch with an existing store. The
        # vectors in `embeddings.sqlite` were written at whatever dimension
        # the embedder-at-the-time produced. If the embedder changed between
        # runs (e.g. Ollama was available last time but not now, so the
        # tiered resolver fell back to HashingEmbedder), the stored vectors
        # and the new embedder disagree, and cosine search would compare
        # incompatible spaces. Detect that here and fail with an actionable
        # message rather than producing silently wrong retrieval.
        stored = self.stored_dim()
        if stored is not None and stored != dim:
            self._conn.close()
            raise EmbeddingStoreMismatchError(
                f"embedding store at {self.db_path} holds {stored}-dim vectors, "
                f"but the active embedder produces {dim}-dim vectors. The "
                f"embedder changed since this store was built.\n"
                f"Fix: delete the embedding store so it can be rebuilt with "
                f"the current embedder — `rm {self.db_path}*` (the chain "
                f"itself is untouched; it re-embeds on next start)."
            )

        # Coordinate-space identity check. Dimension equality is not
        # enough: the HashingEmbedder once used the builtin `hash()`,
        # which is process-randomized, so vectors written before the
        # BLAKE2b fix occupied a DIFFERENT 256-dim coordinate space than
        # vectors written after. Same shape, incompatible meaning,
        # silent retrieval noise. We tag the store with an embedder
        # identity on first write and compare it here.
        active_id = _embedder_identity(embedder, dim)
        stored_id = self._get_embed_meta("embedder_id")
        if stored_id is None and stored is not None:
            # Legacy store: vectors exist but no identity recorded. We
            # don't know whether they came from the old randomized
            # HashingEmbedder, so warn loudly. The dim guard above is
            # silent on this — the dim is unchanged.
            sys.stderr.write(
                f"\n  [embeddings] WARNING: store at {self.db_path} has no "
                f"embedder identity recorded.\n"
                f"  [embeddings] If this store was built before the BLAKE2b "
                f"HashingEmbedder fix\n"
                f"  [embeddings] (v1.2.1 patch round 1), its vectors are "
                f"in a different coordinate\n"
                f"  [embeddings] space than the current embedder produces — "
                f"retrieval will be silently noisy.\n"
                f"  [embeddings] Fix: delete the store so it rebuilds: "
                f"rm {self.db_path}*\n"
            )
            sys.stderr.flush()
        elif stored_id is not None and stored_id != active_id:
            self._conn.close()
            raise EmbeddingStoreMismatchError(
                f"embedding store at {self.db_path} was built by embedder "
                f"{stored_id!r}, but the active embedder is {active_id!r}. "
                f"Same dimension does not imply same coordinate space — "
                f"retrieval would be silently noisy.\n"
                f"Fix: delete the embedding store so it rebuilds with the "
                f"current embedder — `rm {self.db_path}*` (the chain itself "
                f"is untouched; it re-embeds on next start)."
            )
        # Record the active embedder identity for next-boot comparison.
        # Idempotent: same value stays put.
        self._set_embed_meta("embedder_id", active_id)

    def _get_embed_meta(self, key: str) -> Optional[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT value FROM embed_meta WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def _set_embed_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO embed_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def stored_dim(self) -> Optional[int]:
        """
        Dimension of vectors already in the store, inferred from the first
        row. Returns None for an empty store (nothing to conflict with).
        """
        cur = self._conn.cursor()
        cur.execute("SELECT vector FROM embeddings LIMIT 1")
        row = cur.fetchone()
        if row is None:
            return None
        return len(np.frombuffer(row[0], dtype=np.float32))

    @staticmethod
    def record_to_text(rec: Record) -> str:
        """Flatten a record into text for embedding. Override for fancier schemes."""
        # File/attachment records: embed by filename + extracted text.
        # Avoids embedding the giant blob_sha256 hex string which would
        # dominate the vector. (Attachment records carry mime_type where
        # legacy file records carried kind.)
        if rec.type in ("file", "attachment") and isinstance(rec.content, dict):
            filename = rec.content.get("filename", "")
            kind = rec.content.get("kind") or rec.content.get("mime_type", "")
            text = rec.content.get("extracted_text", "")
            return f"[file {filename} {kind}] {text}"
        try:
            content_str = json.dumps(rec.content, ensure_ascii=False)
        except (TypeError, ValueError):
            content_str = str(rec.content)
        return f"[{rec.type}] {content_str}"

    def _record_chunks(self, rec: Record) -> list[str]:
        """The record's flattened text split into embeddable chunks — the
        single chunking path shared by index_record and
        index_records_batched. For file records, the descriptive
        `[file name kind]` header is prepended to every chunk."""
        if rec.type in ("file", "attachment") and isinstance(rec.content, dict):
            filename = rec.content.get("filename", "")
            kind = rec.content.get("kind") or rec.content.get("mime_type", "")
            header = f"[file {filename} {kind}] "
            body = rec.content.get("extracted_text", "")
            pieces = chunk_text(body, target=CHUNK_TARGET_CHARS - len(header))
            return [header + p for p in pieces]
        text = self.record_to_text(rec)
        return chunk_text(text, target=CHUNK_TARGET_CHARS)

    def index_record(self, rec: Record) -> None:
        """
        Embed a record as one or more chunk vectors.

        The record's flattened text (record_to_text) is split by chunk_text
        into ~CHUNK_TARGET_CHARS pieces; each chunk is embedded and stored
        as its own row sharing this record's `record_idx`. Short records
        produce a single chunk and behave exactly as before.

        Idempotent: any existing rows for this record_idx are deleted first,
        so re-indexing a record (or rebuilding a chunking scheme) replaces
        its chunks cleanly rather than accumulating duplicates. The table's
        primary key is now `embed_id`, so the old INSERT OR REPLACE on
        record_idx no longer applies.

        File records get the `[file name kind]` header prepended to EVERY
        chunk so each fragment stays self-describing in the vector space —
        a chunk from the middle of a document should still carry the signal
        that it belongs to that file.
        """
        chunks = self._record_chunks(rec)
        chunk_count = len(chunks)
        cur = self._conn.cursor()
        cur.execute("DELETE FROM embeddings WHERE record_idx = ?", (rec.index,))
        for ci, chunk in enumerate(chunks):
            vec = self.embedder(chunk).astype(np.float32)
            if vec.shape != (self.dim,):
                raise ValueError(
                    f"embedder returned shape {vec.shape}, expected ({self.dim},)"
                )
            cur.execute(
                "INSERT INTO embeddings "
                "(record_idx, chunk_index, chunk_count, record_hash, vector, text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (rec.index, ci, chunk_count, rec.record_hash, vec.tobytes(), chunk),
            )
        self._conn.commit()
        self._nn = None  # invalidate

    def index_chain(self, chain: Chain) -> int:
        """
        Index every record not yet embedded. Returns the count added.

        A single record that fails to embed (e.g. a transient embedder
        error) is logged and skipped, not allowed to abort the whole
        operation. This matters most at boot: index_chain runs on startup,
        and without this guard one un-embeddable record would crash the
        entire application on every launch — the record is already
        committed to the chain, so the failure would recur forever. A
        skipped record stays on the chain (it is real history) and is
        simply not searchable until a later index_chain call succeeds on
        it. Retrieval degrades for that one record; the app still starts.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT record_idx FROM embeddings")
        existing = {r[0] for r in cur.fetchall()}

        # Records actually needing work this call. Computed up front so we
        # can show progress against a known total — a full rebuild of a
        # long chain takes minutes, and without progress output the caller
        # (notably webapp boot) looks frozen when it is simply working.
        to_index = [rec for rec in chain.iter_records()
                    if rec.index not in existing]
        total = len(to_index)
        if total > 20:
            slow = hasattr(self.embedder, "embed_batch")  # network-backed
            note = (" — this can take a few minutes, progress below:"
                    if slow else ":")
            print(f"  [index] embedding {total} records{note}")

        added = 0
        skipped = 0
        for n, rec in enumerate(to_index, start=1):
            try:
                self.index_record(rec)
                added += 1
            except Exception as e:
                # Skip this record, keep going. Boot must not die
                # because one record won't embed.
                skipped += 1
                print(f"  [index] skipped record {rec.index} "
                      f"({rec.type}): {type(e).__name__}: {e}")
            # Periodic progress so a long rebuild visibly advances.
            if total > 20 and (n % 10 == 0 or n == total):
                print(f"  [index] {n}/{total} records embedded")

        if skipped:
            print(f"  [index] {skipped} record(s) could not be embedded and "
                  f"were skipped — they remain on the chain but are not "
                  f"searchable until re-indexed.")
        return added

    def index_records_batched(self, records, batch_size: int = 64,
                              progress=None) -> dict:
        """
        Embed many records at once, sending chunk texts through the
        embedder's embed_batch() in groups of `batch_size` when it has one
        (one Ollama /api/embed call per group) and falling back to
        per-chunk embedding otherwise.

        Note the honest ceiling: on CPU the per-chunk forward pass
        dominates, so batching saves only the per-request overhead
        (measured ~12% over sequential singles with nomic-embed-text) —
        this is the natural unit for a deliberate task_reembed, not a way
        to make a slow embedder fast.

        Replaces cleanly: every record's existing rows are deleted before
        its new chunks are inserted. A record whose embedding fails is
        rolled back to "not embedded" (no partial chunk sets) and counted
        in `failed_records`; the rest proceed.

        `progress(done_chunks, total_chunks)` is called after each batch.
        Returns {"records", "chunks", "failed_records"}.
        """
        per_record = [(rec, self._record_chunks(rec)) for rec in records]
        flat = [(rec, ci, len(chunks), chunk)
                for rec, chunks in per_record
                for ci, chunk in enumerate(chunks)]
        total = len(flat)
        embed_batch = getattr(self.embedder, "embed_batch", None)
        failed: set[int] = set()
        cur = self._conn.cursor()
        for rec, _chunks in per_record:
            cur.execute("DELETE FROM embeddings WHERE record_idx = ?",
                        (rec.index,))

        done = 0
        for start in range(0, total, batch_size):
            batch = flat[start:start + batch_size]
            texts = [chunk for _, _, _, chunk in batch]
            vecs: list = []
            try:
                if embed_batch is not None:
                    vecs = list(embed_batch(texts))
                    if len(vecs) != len(texts):
                        raise RuntimeError(
                            f"embed_batch returned {len(vecs)} vectors "
                            f"for {len(texts)} inputs")
            except Exception as e:
                # One bad batch must not lose the rest: retry this batch
                # per-chunk so only the genuinely un-embeddable records
                # fail.
                print(f"  [reembed] batch failed ({type(e).__name__}: {e}) "
                      f"— retrying per chunk")
                vecs = []
            if not vecs:
                for _, _, _, chunk in batch:
                    try:
                        vecs.append(self.embedder(chunk))
                    except Exception:
                        vecs.append(None)
            for (rec, ci, chunk_count, chunk), vec in zip(batch, vecs):
                if rec.index in failed:
                    continue
                if vec is None:
                    failed.add(rec.index)
                    continue
                vec = np.asarray(vec, dtype=np.float32)
                if vec.shape != (self.dim,):
                    raise ValueError(
                        f"embedder returned shape {vec.shape}, "
                        f"expected ({self.dim},)")
                cur.execute(
                    "INSERT INTO embeddings "
                    "(record_idx, chunk_index, chunk_count, record_hash, "
                    "vector, text) VALUES (?, ?, ?, ?, ?, ?)",
                    (rec.index, ci, chunk_count, rec.record_hash,
                     vec.tobytes(), chunk),
                )
            self._conn.commit()
            done += len(batch)
            if progress is not None:
                progress(done, total)

        # No partial chunk sets: a failed record loses any rows that landed
        # before its first failure, so it reads as plainly "not embedded".
        for idx in failed:
            cur.execute("DELETE FROM embeddings WHERE record_idx = ?", (idx,))
        if failed:
            self._conn.commit()
            print(f"  [reembed] {len(failed)} record(s) could not be "
                  f"embedded and remain unindexed")
        self._nn = None  # invalidate
        return {"records": len(per_record), "chunks": total,
                "failed_records": len(failed)}

    def _rebuild_ann(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT record_idx, chunk_index, vector FROM embeddings "
            "ORDER BY record_idx ASC, chunk_index ASC"
        )
        rows = cur.fetchall()
        if not rows:
            self._nn = None
            self._idx_to_record = []
            return
        # Each ANN sample is a chunk. Map matrix row -> (record_idx,
        # chunk_index) so search() can collapse chunk hits back to records.
        self._idx_to_record = [(r[0], r[1]) for r in rows]
        matrix = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
        # Normalize for cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        self._nn = NearestNeighbors(n_neighbors=min(50, len(rows)), metric="cosine")
        self._nn.fit(matrix)
        self._matrix = matrix

    def search(self, query_text: str, k: int = 10) -> list[tuple[int, float]]:
        """Return [(record_idx, similarity_score), ...] sorted by similarity desc.

        Group-collapse: the index stores one vector per CHUNK, but this
        method returns one similarity per RECORD — the MAX similarity over
        that record's chunks. This is what prevents long records from
        getting "more raffle tickets": a 30-chunk record contributes a
        single record-level score (its best fragment), competing for one
        slot against every other record's best fragment. Max (not mean) is
        deliberate — a long record with one highly relevant section and a
        lot of unrelated text should rank on the strength of that section,
        not be penalized for its length.

        The caller's contract is unchanged from the pre-chunking store:
        `(record_idx, sim)` pairs, `sim` in [0, 1], sorted desc, at most
        `k` records. hybrid(), drift_against(), and every other caller see
        exactly what they saw before — the chunking is invisible above this
        line.

        To return the true top-k *records* we over-fetch chunk neighbors:
        the k-th best record's best chunk can sit well outside the k-th
        chunk neighbor (if higher-ranked records each contributed several
        chunks). We request a generous chunk-neighbor pool, collapse to
        records, then take the top k records. The pool is capped at the
        number of stored chunks.

        A non-positive `k`, or a `k` larger than the store, is clamped:
        sklearn's `kneighbors` raises if `n_neighbors` is 0 or exceeds the
        number of fitted samples. `hybrid()` floors its own `k` well above
        zero, but `search()` is also called directly (e.g. by
        `Retriever.drift_against` with a caller-supplied `k`), so the guard
        belongs here. A `k <= 0` request returns an empty list rather than
        raising.
        """
        if self._nn is None:
            self._rebuild_ann()
        if self._nn is None:
            return []
        if k <= 0:
            return []
        n_chunks = len(self._idx_to_record)
        # Over-fetch chunk neighbors so the top-k records are not missed
        # when highly-ranked records each contribute multiple chunks.
        # 4x headroom plus a floor; clamped to the pool size.
        pool = min(max(k * 4, 40), n_chunks)
        if pool <= 0:
            return []
        qv = self.embedder(query_text).astype(np.float32)
        qn = qv / (np.linalg.norm(qv) or 1.0)
        distances, indices = self._nn.kneighbors(qn.reshape(1, -1), n_neighbors=pool)
        # Collapse chunk hits to per-record max similarity. While doing the
        # collapse, also remember WHICH chunks matched per record — the
        # caller's contract still returns one similarity per record, but
        # downstream rendering (Agent._format_prompt's file branch) wants to
        # know "which chunks of this long record actually hit" so it can
        # show only those chunks instead of dumping the whole record text.
        # The chunk-match data is stashed on `last_chunk_matches`, a per-call
        # attribute on the index (mirroring `last_pinned_indices` on the
        # retriever). It is overwritten by every search() call, so callers
        # that want it must read it before the next search.
        best: dict[int, float] = {}
        chunk_matches: dict[int, list[tuple[int, float]]] = {}
        for d, i in zip(distances[0], indices[0]):
            rec_idx, chunk_idx = self._idx_to_record[i]
            sim = 1.0 - float(d)
            if rec_idx not in best or sim > best[rec_idx]:
                best[rec_idx] = sim
            chunk_matches.setdefault(rec_idx, []).append((chunk_idx, sim))
        # Sort each record's chunks by similarity desc so callers reading
        # the top-N matched chunks get the best ones first.
        for rec_idx in chunk_matches:
            chunk_matches[rec_idx].sort(key=lambda t: t[1], reverse=True)
        self.last_chunk_matches = chunk_matches
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]

    def chunks_for_record(self, record_idx: int) -> list[tuple[int, str]]:
        """Return all stored `(chunk_index, text)` pairs for a record, ordered
        by chunk_index. Used by chunk-aware rendering when a long file record
        is retrieved and the prompt wants to show only matched chunks (plus
        neighbors) rather than the whole record text.

        Returns an empty list if the record has no chunks in the store (e.g.
        a record that has never been indexed, or a record whose text was
        short enough that no chunking happened). Callers should treat empty
        as "fall back to record_to_text" rather than as an error.
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT chunk_index, text FROM embeddings "
            "WHERE record_idx = ? ORDER BY chunk_index ASC",
            (record_idx,),
        )
        return [(int(c), str(t)) for c, t in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EmbeddingIndex":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def open_or_rebuild_index(
    db_path: str | Path,
    embedder: "Embedder",
    dim: int,
    chain: Optional[Chain] = None,
    force_rebuild: bool = False,
    log=print,
) -> EmbeddingIndex:
    """
    Open the derived embedding index at `db_path`, rebuilding it when that
    is safe or explicitly requested.

    Rebuild policy on EmbeddingStoreMismatchError:
    - The store was built by the cheap lexical HashingEmbedder → delete and
      reopen empty (the "Ollama got installed" upgrade path: the old store
      cost seconds to build and the chain re-embeds it on the spot).
    - The store was built by anything else (a semantic store that may have
      taken hours over the network) → REFUSE and re-raise with instructions.
      The active embedder may be a transient fallback — e.g. the tiered
      resolver picked HashingEmbedder because the Ollama daemon happened to
      be down at boot — and silently deleting the store would destroy that
      work twice (once now, once again when the daemon returns).

    `force_rebuild=True` skips the policy and rebuilds unconditionally —
    the explicit path task_reembed uses. Either way the store is DERIVED
    data; the chain itself is never touched.

    When `chain` is given, index_chain() runs before returning, so any
    records missing from the (possibly freshly rebuilt) store are re-embedded
    immediately. Pass the chain everywhere a stale store must self-heal into
    a searchable one; callers that run index_chain() themselves may omit it.

    This is the SINGLE shared open-or-rebuild path — run.py, the webapp, and
    per-task indexes (tools.py) must all open stores through it so the
    recovery behavior cannot drift between entry points.
    """
    if force_rebuild:
        log(f"rebuilding embedding store at {db_path} (explicitly "
            f"requested); chain data is unchanged")
        delete_index_store(db_path)
        index = EmbeddingIndex(db_path, embedder, dim=dim)
    else:
        try:
            index = EmbeddingIndex(db_path, embedder, dim=dim)
        except EmbeddingStoreMismatchError as exc:
            stored_id = _stored_embedder_id(db_path)
            if not (stored_id or "").startswith("HashingEmbedder:"):
                raise EmbeddingStoreMismatchError(
                    f"{exc}\n"
                    f"REFUSING to delete this store automatically: it was "
                    f"built by {stored_id or 'an unknown embedder'} and may "
                    f"represent hours of embedding work, while the active "
                    f"embedder may be a temporary fallback (e.g. the Ollama "
                    f"daemon is unreachable right now). Either restore the "
                    f"original embedder and restart, or rebuild explicitly "
                    f"(task_reembed for task stores, or delete "
                    f"`{db_path}*` yourself)."
                ) from exc
            log(f"embedding store at {db_path} was built by the lexical "
                f"{stored_id} — rebuilding with the active embedder; "
                f"chain data is unchanged")
            delete_index_store(db_path)
            index = EmbeddingIndex(db_path, embedder, dim=dim)
    if chain is not None:
        index.index_chain(chain)
    return index


def delete_index_store(db_path) -> None:
    """Delete a derived embedding store plus its WAL/SHM sidecars — the ONE
    place that knows the sidecar list. The chain is never touched."""
    for path in (Path(str(db_path)),
                 Path(f"{db_path}-wal"),
                 Path(f"{db_path}-shm")):
        path.unlink(missing_ok=True)


def _stored_embedder_id(db_path) -> Optional[str]:
    """Best-effort peek at the embedder identity a store was built with.
    Used by the rebuild policy after EmbeddingIndex refused to open (it
    closed its own connection before raising)."""
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT value FROM embed_meta WHERE key = 'embedder_id'"
            ).fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    except sqlite3.Error:
        return None


# ---------------------------------------------------------------------------
# Retrieval API
# ---------------------------------------------------------------------------

@dataclass
class RetrievalHit:
    record: Record
    score: float
    reason: str  # "semantic", "recent", "type", "ancestry"
    # Score breakdown — useful for debugging and for tuning weights.
    # Empty for non-semantic hits (recency / ancestry) where the score
    # isn't a weighted blend.
    components: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.components is None:
            object.__setattr__(self, "components", {})


class Retriever:
    """
    Read-side intelligence over the chain. Three things to know:

    1. Salience is per-record, read from the v2 metadata block on the record.
       v1 records (without _meta) get type-based defaults from metadata.py.
       This replaces the old per-type DEFAULT_SALIENCE constant — the
       defaults still exist (in metadata.py) but only kick in for v1 records.

    2. Recency uses per-kind half-lives. A genesis record from a year ago
       is still as relevant as it was; an observation from a week ago has
       decayed substantially. The score is 0.5 ** (age_days / half_life).

    3. Revision-aware. Records that have been superseded by a later
       revision get a demotion penalty — they still surface (the
       conflict itself is informative) but rank below their corrections.
    """

    def __init__(self, chain: Chain, index: EmbeddingIndex, sprout_registry=None):
        self.chain = chain
        self.index = index
        # Optional SproutRegistry (sprouted_modalities.SproutRegistry) of
        # data-driven modalities. None means "baked-in modalities only" —
        # the historical behavior, so every existing caller is unaffected.
        # When present, its domain modalities join DOMAIN_MODALITIES for
        # anchoring and its detectors run inside query-time analysis.
        self.sprout_registry = sprout_registry
        # Lazily-constructed SignalAnalyzer for query-time modality
        # detection. Built on first use (not at construction) so a Retriever
        # used purely for non-modality retrieval never imports signals or
        # pays for an analyzer it won't use. See `query_modalities`.
        self._analyzer = None

    def domain_modalities(self) -> set:
        """
        The full set of domain modalities anchoring considers: the baked-in
        DOMAIN_MODALITIES plus any domain-flagged sprouted modalities. This
        is what makes a sprouted modality actually influence retrieval the
        moment it lands in the registry — no restart, no code change.
        """
        if self.sprout_registry is None:
            return set(DOMAIN_MODALITIES)
        return set(DOMAIN_MODALITIES) | self.sprout_registry.domain_names()

    def _modality_weight_factor(self, name: str) -> float:
        """
        Per-modality multiplier on the modality term. Baked-in modalities are
        always full strength (1.0). A sprouted modality may be damped — a
        `tentative` (cooling-off) sprout contributes at reduced weight until
        it graduates. See sprouted_modalities.SproutedModality.
        """
        if self.sprout_registry is None:
            return 1.0
        return self.sprout_registry.weight_factors().get(name, 1.0)

    def query_modalities(self, query: str) -> set:
        """
        Domain modalities the current query implies — the domain modalities
        (baked-in + sprouted) that fire above the analyzer's floor when the
        query text is analyzed, capped at PER_TURN_MODALITY_CAP and chosen by
        detection activation (strongest first). This tells retrieval what
        "mode" the agent is about to enter, so it can preferentially surface
        records produced in the same mode.

        Returns an empty set when no domain modality fires (a neutral query),
        which makes the modality term inactive for that retrieval.

        The per-turn cap matters once sprouting is live: a query could match
        many domain modalities at once, and anchoring across all of them
        dilutes the signal. We keep the strongest PER_TURN_MODALITY_CAP, all
        above the analyzer's activation floor, and drop the rest for this
        query. (With only the single baked-in domain modality the cap is
        moot; it is forward-protection for the sprouted world.)

        The analyzer includes any sprouted detectors so a sprouted modality
        can be detected on the *query* too, not just on stored records.
        `signals` is imported lazily here to keep it off retrieval.py's
        import path for callers that never anchor.
        """
        if self._analyzer is None:
            from signals import SignalAnalyzer
            extra = (self.sprout_registry.as_detectors()
                     if self.sprout_registry is not None else [])
            self._analyzer = SignalAnalyzer(extra_modalities=extra)
        from signals import SignalInput
        report = self._analyzer.analyze(SignalInput(content=query, source="user"))
        domain = self.domain_modalities()
        # Rank the fired modalities by activation so the cap keeps the
        # strongest. activated_modalities() returns names only (activations
        # discarded), so read activations off the report's modality hits.
        fired = [
            (h.name, h.activation)
            for h in report.modalities
            if h.name in domain and h.activation > self._activation_floor()
        ]
        fired.sort(key=lambda na: na[1], reverse=True)
        return {name for name, _ in fired[:PER_TURN_MODALITY_CAP]}

    @staticmethod
    def _activation_floor() -> float:
        """The analyzer's modality activation floor (shared with signals.py)."""
        from signals import MODALITY_ACTIVATION_FLOOR
        return MODALITY_ACTIVATION_FLOOR

    # ----- score weights -----

    # Weights for the hybrid score. Tunable; named so the impact of each
    # term is visible in `RetrievalHit.components`. They sum to 1.0 to keep
    # raw scores in roughly [0, 1.5] range (semantic + boosts).
    #
    # There are two weight sets. The DEFAULT set (no query modality
    # supplied) is the historical three-term formula — every existing caller
    # of `hybrid`/`build_context` keeps its exact prior behavior. The
    # MODALITY set is used ONLY when a query-modality vector is supplied;
    # it reapportions weight to make room for the modality-overlap term
    # without changing the relative balance of the other three much. This
    # opt-in design means modality anchoring never silently perturbs
    # retrieval for callers that don't ask for it.
    W_SEMANTIC = 0.55
    W_SALIENCE = 0.25
    W_RECENCY  = 0.20

    # Active only when query modalities are supplied. semantic 0.55->0.45,
    # recency 0.20->0.15, freeing 0.15 for the modality term; salience
    # unchanged at 0.25.
    W_SEMANTIC_MODAL = 0.45
    W_SALIENCE_MODAL = 0.25
    W_RECENCY_MODAL  = 0.15
    W_MODALITY       = 0.15

    # Penalty subtracted from a hit's score when a later revision
    # supersedes it. Applied AFTER the weighted sum, so it can push a
    # superseded record below an otherwise-weaker correction. Set high
    # enough to flip ordering when both are retrieved together, but not
    # so high the original drops out of context entirely.
    SUPERSEDED_PENALTY = 0.30

    # Penalty subtracted when a record carries an integrity-risk signal in
    # its PoQ block (see poq.py / metadata.py). A record whose own PoQ
    # scored a non-trivial `risk` dimension is either an attack the chain
    # remembered or a low-trust turn — it should still be retrievable (the
    # chain is honest about its history) but should rank below clean
    # records. This is the build spec's section 4.9 `risk_penalty` term.
    # Quarantined records (exposure=quarantine) are filtered out entirely
    # by protected_zones.filter_quarantined before the prompt is built; this
    # penalty handles the softer case of a committed-but-risky record.
    RISK_PENALTY = 0.25

    # Risk dimension at or above this level triggers the risk penalty.
    RISK_THRESHOLD = 0.4

    # ----- epistemic weighting (build spec sections 4.2 / 4.7) -----
    #
    # Every record carries an `epistemic_class` in its _meta (metadata.py):
    # how the writer knows the content — `known`, `user_context`, `inferred`,
    # `speculative`, or `disputed`. Until now this field was recorded and
    # displayed but invisible to retrieval scoring: a speculative guess and a
    # user-stated fact competed on equal footing. That is an honesty gap — the
    # agent should preferentially ground its answers in what it actually knows
    # over what it once guessed.
    #
    # The fix is a multiplicative factor on the final score, applied AFTER the
    # weighted base + penalties (like the superseded/risk penalties, not like
    # the additive modality term). A multiplier rather than an additive term
    # because epistemic standing should scale a record's whole relevance, not
    # add a fixed quantity regardless of how relevant it otherwise is: a
    # barely-relevant `known` record shouldn't leapfrog a highly-relevant one
    # just for being known, but between two similarly-relevant records the
    # better-grounded one should win.
    #
    # Calibration: `known` and `user_context` are ground truth (the user said
    # it, or it's a verified file) — full weight. `inferred` is the agent's
    # own reasoning — very slightly discounted. `speculative` is a flagged
    # guess — discounted more. `disputed` is known to conflict with another
    # record — discounted hard (but NOT removed: the chain stays honest about
    # its history, and the revision-aware pull-in already surfaces disputes
    # alongside their corrections). An unknown/again-defaulted class is
    # treated as `inferred`.
    #
    # This is OPT-IN. `epistemic_weighting=False` (the historical behavior)
    # makes every factor 1.0 so scoring is byte-for-byte unchanged for callers
    # and tests that don't ask for it.
    EPISTEMIC_FACTORS = {
        "known":        1.0,
        "user_context": 1.0,
        "inferred":     0.97,
        "speculative":  0.85,
        "disputed":     0.65,
    }
    EPISTEMIC_FACTOR_DEFAULT = 0.97  # treat unknown class as `inferred`

    # ----- helpers -----

    def _superseded_indices(self) -> set[int]:
        """
        Indices of records that have been superseded by a later revision.

        Backed by Chain's materialized `supersedes_index` table — a single
        indexed SELECT, no per-record scan. (An earlier version scanned
        every revision record on every retrieval, capped by a silent
        limit; on a chain with many revisions that cap quietly cut
        correctness.) See `Chain.superseded_indices`.
        """
        return self.chain.superseded_indices()

    @staticmethod
    def _recency_score(age_seconds: float, half_life_seconds: float) -> float:
        """0.5 ** (age / half_life). Capped to [0, 1] for numerical safety."""
        if half_life_seconds <= 0:
            return 0.0
        try:
            return float(0.5 ** (age_seconds / half_life_seconds))
        except OverflowError:
            return 0.0

    # ----- hybrid: semantic + structural + salience + recency + revision-aware -----

    def hybrid(
        self,
        query: str,
        k: int = 10,
        type_filter: Optional[str] = None,
        recency_weight: Optional[float] = None,  # back-compat; overrides W_RECENCY
        salience_weights: Optional[dict] = None,  # back-compat; ignored if None
        query_modalities: Optional[set] = None,
        epistemic_weighting: bool = False,
    ) -> list[RetrievalHit]:
        """
        Semantic search plus per-record salience, per-kind recency decay,
        and revision-aware demotion.

        Score formula (default — no query modalities):
            base  = W_SEMANTIC*similarity + W_SALIENCE*salience + W_RECENCY*recency
            score = base
                    - (SUPERSEDED_PENALTY if record is superseded else 0)
                    - (RISK_PENALTY if record's PoQ flagged a risk else 0)

        When `query_modalities` is supplied (a set of domain-modality names
        the current query implies — see `query_modalities()`), a fourth term
        is added and the weights shift to the *_MODAL set to make room:
            base  = W_SEMANTIC_MODAL*similarity + W_SALIENCE_MODAL*salience
                    + W_RECENCY_MODAL*recency + W_MODALITY*modality_overlap
        where modality_overlap compares the query's domain modes to the
        record's stored `modalities_activated` (filtered to DOMAIN_MODALITIES).
        A record produced in the same mode as the query is boosted; a genuine
        mismatch is mildly cut; a record with no modality data is neutral
        (overlap 0.5). This is opt-in: callers that pass no query_modalities
        get the exact historical three-term behavior and weights.

        - similarity: cosine sim from the embedding index, in [0, 1].
        - salience:   from the record's _meta block, in [0, 1].
        - recency:    0.5 ** (age_days / half_life_days_for_type), in [0, 1].
        - risk:       read from the record's _meta.poq block, if present.
                      A record whose own Proof-of-Quality scored a non-
                      trivial `risk` dimension ranks below clean records.

        Back-compat: `recency_weight` and `salience_weights` are still
        accepted but their interpretation has shifted —
          - recency_weight: if set, overrides the active recency weight.
          - salience_weights: now ignored. Salience is per-record from _meta,
            with type defaults from metadata.py for v1 records. Pass-through
            kept so old callers don't crash.
        """
        # Pick the weight set. Modality anchoring is active only when the
        # caller supplied a non-empty query-modality set, so existing callers
        # see identical scoring to before.
        anchoring = bool(query_modalities)
        if anchoring:
            w_semantic = self.W_SEMANTIC_MODAL
            w_salience = self.W_SALIENCE_MODAL
            w_recency_default = self.W_RECENCY_MODAL
        else:
            w_semantic = self.W_SEMANTIC
            w_salience = self.W_SALIENCE
            w_recency_default = self.W_RECENCY
        w_recency = w_recency_default if recency_weight is None else float(recency_weight)
        # Intersect against the DYNAMIC domain set (baked-in + sprouted), not
        # the module-level DOMAIN_MODALITIES constant — a sprouted modality
        # lives only in the registry, so filtering against the constant would
        # silently drop it and neutralize anchoring for sprouted modes.
        q_domain = (
            set(query_modalities) & self.domain_modalities() if anchoring else set()
        )
        candidates = self.index.search(query, k=max(k * 4, 20))
        if not candidates:
            return []
        superseded = self._superseded_indices()
        now_seconds = time.time()

        # ---- Pass 1: base scores (no modality term yet) + per-candidate
        # modality overlap and the matching record's weight factor. The
        # modality term is deferred so we can measure how saturated the
        # genuinely-relevant context already is with the query's mode BEFORE
        # the boost reshapes the ranking — that measurement is what the
        # anti-echo damper needs, and measuring it post-boost would be
        # circular.
        scratch: list = []  # (rec, meta, base_minus_penalties, m_overlap, weight_factor, components)
        for rec_idx, sim in candidates:
            rec = self.chain.get(rec_idx)
            if rec is None:
                continue
            if type_filter and rec.type != type_filter:
                continue
            meta = read_meta(rec)
            age_seconds = max(0.0, now_seconds - rec.timestamp / 1000.0)
            half_life_seconds = half_life_days(rec.type) * 86400.0
            recency = self._recency_score(age_seconds, half_life_seconds)

            # Per-candidate modality overlap and the weight factor of the
            # matching modality (tentative sprouts are damped). Only computed
            # when anchoring; otherwise neutral and unused.
            if anchoring:
                rec_domain = set(meta.modalities_activated) & q_domain
                m_overlap = modality_overlap(q_domain, rec_domain)
                # If the record matches a query modality, use that modality's
                # weight factor (min across matches, so a tentative match
                # damps). No match -> neutral overlap, factor irrelevant.
                if rec_domain:
                    weight_factor = min(
                        (self._modality_weight_factor(n) for n in rec_domain),
                        default=1.0,
                    )
                else:
                    weight_factor = 1.0
            else:
                m_overlap = MODALITY_NEUTRAL
                weight_factor = 1.0

            base = (
                w_semantic * sim
                + w_salience * meta.salience
                + w_recency * recency
            )
            penalty = self.SUPERSEDED_PENALTY if rec.index in superseded else 0.0
            # Risk penalty: a record whose stored PoQ block flagged a
            # non-trivial risk dimension is demoted. Records that were
            # quarantined outright are already filtered upstream by
            # protected_zones.filter_quarantined; this catches the softer
            # committed-but-risky case so it ranks below clean memory.
            risk_value = 0.0
            if meta.poq:
                dims = meta.poq.get("dimensions", {})
                risk_value = float(dims.get("risk", 0.0))
            risk_penalty = (
                self.RISK_PENALTY if risk_value >= self.RISK_THRESHOLD else 0.0
            )
            # Epistemic weighting (build spec 4.2/4.7): scale the positive
            # base by how well-grounded the record is, BEFORE subtracting
            # penalties. Applied to `base` (not `base_minus`) so it scales the
            # record's earned relevance, while the superseded/risk penalties
            # remain absolute subtractions that act the same regardless of
            # epistemic class. Neutral (1.0) when the feature is off, so the
            # arithmetic is identical to before for callers that don't opt in.
            if epistemic_weighting:
                epi_factor = self.EPISTEMIC_FACTORS.get(
                    meta.epistemic_class, self.EPISTEMIC_FACTOR_DEFAULT
                )
            else:
                epi_factor = 1.0
            base_minus = (base * epi_factor) - penalty - risk_penalty
            components = {
                "semantic": float(sim),
                "salience": float(meta.salience),
                "recency": float(recency),
                "superseded_penalty": float(penalty),
                "risk_penalty": float(risk_penalty),
                "risk": risk_value,
                "source": meta.source,
                "confidence": float(meta.confidence),
                "epistemic_class": meta.epistemic_class,
                "epistemic_factor": float(epi_factor),
            }
            scratch.append((rec, base_minus, m_overlap, weight_factor, components))

        if not scratch:
            return []

        # ---- Anti-echo damping factor. Measure how saturated the top
        # candidates (by base score, before any modality boost) already are
        # with the query's mode. If most of the strongest candidates ALREADY
        # match the query modality, boosting matches further just piles up
        # "more of the same" — the self-reinforcing loop sprouted modalities
        # make easy. Damp the modality term in proportion to the excess over
        # MODALITY_SATURATION_THRESHOLD. When off (no anchoring) damp is 1.0
        # and unused.
        damp = 1.0
        saturation = 0.0
        if anchoring:
            top = sorted(scratch, key=lambda s: s[1], reverse=True)[
                :MODALITY_SATURATION_TOP_N
            ]
            if top:
                # A candidate "carries the mode" when its overlap is a real
                # match (> neutral), not merely the absence of information.
                matched = sum(1 for s in top if s[2] > MODALITY_NEUTRAL)
                saturation = matched / len(top)
            excess = max(0.0, saturation - MODALITY_SATURATION_THRESHOLD)
            damp = max(0.0, 1.0 - excess)

        # ---- Pass 2: apply the (damped) modality term and finalize scores.
        hits: list[RetrievalHit] = []
        for rec, base_minus, m_overlap, weight_factor, components in scratch:
            if anchoring:
                contribution = self.W_MODALITY * m_overlap * weight_factor * damp
                score = base_minus + contribution
                components["modality_overlap"] = float(m_overlap)
                components["modality_weight_factor"] = float(weight_factor)
                components["modality_saturation"] = float(saturation)
                components["modality_damp"] = float(damp)
                components["modality_contribution"] = float(contribution)
            else:
                score = base_minus
            hits.append(RetrievalHit(rec, score, "semantic", components))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    # ----- ancestry walk -----

    def ancestry(self, record_hash: str, depth: int = 3) -> list[RetrievalHit]:
        recs = self.chain.follow_refs(record_hash, depth=depth)
        return [RetrievalHit(r, 1.0, "ancestry", {}) for r in recs]

    # ----- temporal window -----

    def recent(self, n: int = 20, type_filter: Optional[str] = None) -> list[RetrievalHit]:
        if type_filter:
            recs = self.chain.query_by_type(type_filter, limit=n)
        else:
            recs = self.chain.query_recent(limit=n)
        return [RetrievalHit(r, 1.0, "recent", {}) for r in recs]

    # ----- combined context for LLM -----

    def build_context(
        self,
        query: str,
        k_semantic: int = 5,
        n_recent: int = 15,
        type_filter: Optional[str] = None,
        anchor_modalities: bool = True,
        query_modalities: Optional[set] = None,
        epistemic_weighting: bool = True,
    ) -> list[Record]:
        """
        Blend semantic hits with recent records, dedup, return in
        chronological order suitable for an LLM context window.

        Explicit-reference pull-in: when the user's query names a record
        by number ("record 328", "index 12", "what you said at #87"),
        that record is pulled in directly, bypassing semantic retrieval.
        Without this, a user asking "please show me record 328 again"
        would get back fuzzy semantic matches — the query text rarely
        overlaps with the *content* of the record being referenced, so
        semantic search misses it. This is what makes a chain
        addressable rather than merely searchable.

        Pinning: explicitly-referenced records are also recorded in
        `self.last_pinned_indices` (a per-call attribute) so the
        caller can treat them as high-priority in any downstream
        truncation. Without this signal, a user-named record competes
        on equal salience footing with every other retrieved record,
        and a large response with default salience 0.4 can lose to
        reflections (0.85) and genesis (1.0) — making the explicit
        reference effectively useless under budget pressure. See
        `Agent._truncate_to_budget` for the consumption side.

        Revision pull-in: when a record in the result set has been
        superseded by a revision, that revision is automatically pulled
        in too (if not already present). The model needs to see both
        the original claim and its correction together — that's the
        point of keeping both around.

        Turn-pair stitching: an observation and the response that answers
        it are a single Q&A unit. When only one half is retrieved, the
        other half is pulled in (type-checked and refs-corroborated, so
        only a genuine pair is completed) and both halves are pinned so
        truncation keeps them together. Quarantined partners are never
        stitched in. Idempotent when both halves are already present.

        Quarantine filtering: records whose `_meta.exposure` is
        `quarantine` (committed prompt-injection attempts and other
        untrusted input) are dropped from the result. They remain on the
        chain and verifiable, but must never re-enter the model's context
        as if they were ordinary memory. This is the read-side half of
        the protected-zones boundary; the write-side is in agent.turn().

        Modality anchoring: by default (`anchor_modalities=True`), the query
        is analyzed for the domain modalities it implies (see
        `query_modalities`), and `hybrid` weights records that were produced
        in the same mode more highly — so a code-shaped query preferentially
        surfaces code-shaped past responses. Pass `anchor_modalities=False`
        (or `query_modalities=set()`) to disable it for a call; pass an
        explicit `query_modalities` set to skip the analysis and supply the
        modes directly. When the query implies no domain modality, anchoring
        is inert and scoring matches the historical formula exactly.

        Epistemic weighting: by default (`epistemic_weighting=True`), records
        are scaled by how well-grounded they are (`known`/`user_context` full
        weight, `inferred` ~unchanged, `speculative`/`disputed` discounted —
        see `EPISTEMIC_FACTORS`). This makes the agent prefer to ground
        answers in what it knows over what it once guessed. The penalties
        (superseded, risk) are unaffected. Pass `epistemic_weighting=False`
        to restore the historical class-blind scoring.
        """
        # Resolve query modalities for anchoring. An explicit set wins; else
        # detect from the query if anchoring is on; else none (inert).
        if query_modalities is not None:
            q_mods = query_modalities
        elif anchor_modalities:
            q_mods = self.query_modalities(query)
        else:
            q_mods = set()
        semantic = self.hybrid(
            query, k=k_semantic, type_filter=type_filter, query_modalities=q_mods,
            epistemic_weighting=epistemic_weighting,
        )
        recent = self.recent(n=n_recent, type_filter=type_filter)
        seen: set[int] = set()
        merged: list[Record] = []
        for hit in semantic + recent:
            if hit.record.index in seen:
                continue
            seen.add(hit.record.index)
            merged.append(hit.record)

        # Explicit-reference pull-in. Parses things like "record 328",
        # "index 12", "#87" out of the query and fetches them directly.
        # See `_extract_explicit_indices` for the matched patterns.
        chain_length = self.chain.length()
        pinned: set[int] = set()
        for idx in _extract_explicit_indices(query):
            if idx < 0 or idx >= chain_length:
                continue
            # Record gets pinned whether or not it was already in the
            # retrieved set — the user named it, so the caller should
            # treat it as high-priority regardless of how it arrived.
            pinned.add(idx)
            if idx in seen:
                continue
            rec = self.chain.get(idx)
            if rec is None:
                pinned.discard(idx)
                continue
            seen.add(idx)
            merged.append(rec)
        # Expose pinned indices for the caller (Agent._format_prompt
        # reads this in `_truncate_to_budget`). Per-call attribute, not
        # threaded through the return signature, so existing call sites
        # don't break.
        self.last_pinned_indices = pinned

        # Pull in revisions that supersede anything we've retrieved.
        # Without this, the model can see a stale claim and miss the
        # correction sitting on the chain. Backed by the chain's
        # materialized supersedes index — a single indexed lookup per
        # query, not a full scan of every revision record.
        merged_indices = {r.index for r in merged}
        revision_pull_ins: list[Record] = []
        for rev in self.chain.revisions_targeting(merged_indices):
            if rev.index not in merged_indices:
                revision_pull_ins.append(rev)
                merged_indices.add(rev.index)
        merged.extend(revision_pull_ins)

        # Turn-pair stitching. An observation and the response that answers it
        # are a single Q&A unit — they give each other essential context, so
        # retrieving one half without the other strips that context. The turn
        # flow seals them at consecutive indices (observation N, response N+1)
        # and the response carries a `refs` link back to its observation, so we
        # complete any half-retrieved pair by pulling in its partner. Applied to
        # the whole merged set and idempotent: when both halves are already
        # present it does nothing. Refinements:
        #   - Type-checked + refs-corroborated, so a failed-response gap (an
        #     observation whose N+1 is the *next* turn's record) can't pull an
        #     unrelated record.
        #   - Quarantine-respecting: an untrusted (quarantined) partner is never
        #     stitched in.
        #   - Budget-safe: both halves are pinned, so `_truncate_to_budget`
        #     keeps the pair together (or drops it together) rather than
        #     splitting it under prompt-budget pressure.
        stitched_in: list[Record] = []
        for rec in list(merged):
            if rec.type == "observation":
                partner_idx, partner_type = rec.index + 1, "response"
            elif rec.type == "response":
                partner_idx, partner_type = rec.index - 1, "observation"
            else:
                continue
            if partner_idx < 0 or partner_idx in merged_indices:
                continue
            partner = self.chain.get(partner_idx)
            if partner is None or partner.type != partner_type:
                continue
            # Corroborate the pair with the refs link (the response refs its
            # observation), so we only ever stitch a genuine turn pair.
            resp, obs = (rec, partner) if rec.type == "response" else (partner, rec)
            if obs.record_hash not in (resp.refs or []):
                continue
            if read_meta(partner).exposure == EXPOSURE_QUARANTINE:
                continue
            stitched_in.append(partner)
            merged_indices.add(partner_idx)
            pinned.add(rec.index)
            pinned.add(partner_idx)
        merged.extend(stitched_in)
        self.last_pinned_indices = pinned

        # Drop quarantined records. Done last so a quarantined record
        # can't sneak back in via the revision pull-in either.
        #
        # Track which indices we dropped: the filter is the right
        # security action (quarantined content must not be fed back into
        # the model as ordinary memory), but a totally silent drop is
        # bad observability. If the user asks "can you see record N?"
        # and N is quarantined, the model needs to be able to answer
        # "yes, it's on the chain but quarantined" rather than the
        # honest-but-confusing "I don't see it." The retriever returns
        # only non-quarantined records, but it stores the dropped
        # indices on its own `last_quarantined_indices` attribute so
        # the prompt builder can surface them as metadata in the
        # prompt header (NOT as retrievable content).
        quarantined = [
            r for r in merged
            if read_meta(r).exposure == EXPOSURE_QUARANTINE
        ]
        merged = [
            r for r in merged
            if read_meta(r).exposure != EXPOSURE_QUARANTINE
        ]
        self.last_quarantined_indices = [r.index for r in quarantined]

        merged.sort(key=lambda r: r.index)
        return merged

    # ----- drift detection -----

    def drift_against(
        self,
        anchor_query: str,
        recent_query: str,
        k: int = 5,
        threshold: float = 0.3,
    ) -> dict:
        """
        Compare semantic neighborhood of `anchor_query` (typically founding
        commitments) against neighborhood of `recent_query` (current behavior).
        High mean cosine distance suggests drift. Returns diagnostics, not a verdict.
        """
        anchor_hits = self.index.search(anchor_query, k=k)
        recent_hits = self.index.search(recent_query, k=k)
        if not anchor_hits or not recent_hits:
            return {"status": "insufficient_data", "drift_score": None}

        anchor_score = float(np.mean([s for _, s in anchor_hits]))
        recent_score = float(np.mean([s for _, s in recent_hits]))
        gap = anchor_score - recent_score
        return {
            "status": "drift" if gap > threshold else "ok",
            "anchor_mean_similarity": anchor_score,
            "recent_mean_similarity": recent_score,
            "gap": gap,
            "threshold": threshold,
            "anchor_records": [i for i, _ in anchor_hits],
            "recent_records": [i for i, _ in recent_hits],
        }


# ---------------------------------------------------------------------------
# Embedders
#
# Two implementations ship here. `run.py` picks between them at startup with
# `make_tiered_embedder()` — Ollama if reachable, HashingEmbedder otherwise.
# ---------------------------------------------------------------------------

class HashingEmbedder:
    """
    Bag-of-character-trigrams hashed into a fixed-dim vector.
    Deterministic, dependency-free, and good enough to demonstrate retrieval
    behavior. It is NOT a real semantic embedder — surface-form similar text
    scores high, meaning-similar-but-worded-differently text does not.

    Two roles:
      - the test suite uses it precisely because it has no model or network
        dependency (see CONTRIBUTING.md);
      - it is the offline fallback when no Ollama server is reachable.

    For real semantic retrieval, prefer `OllamaEmbedder`.

    Determinism note: the trigram-to-bucket mapping uses BLAKE2b, NOT
    Python's builtin `hash()`. `hash()` on a `str` is randomized per
    process (PYTHONHASHSEED, on by default since CPython 3.3), so the same
    trigram would land in a different bucket on every run. Because vectors
    are persisted to `embeddings.sqlite` and compared across sessions, a
    process-randomized hash put stored vectors and freshly-computed query
    vectors into different coordinate spaces — cosine similarity silently
    became noise, and nothing detected it (the dimension is unchanged, so
    the EmbeddingIndex dimension guard does not fire). BLAKE2b is stable
    across processes, Python versions, and platforms, which is what an
    on-disk, cross-session embedding store requires.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    @staticmethod
    def _bucket(tri: str, dim: int) -> int:
        """
        Map a trigram to a bucket in [0, dim) with a process-stable hash.
        BLAKE2b digest -> integer -> modulo. Deterministic everywhere.
        """
        digest = hashlib.blake2b(tri.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % dim

    def __call__(self, text: str) -> np.ndarray:
        text = text.lower()
        vec = np.zeros(self.dim, dtype=np.float32)
        if len(text) < 3:
            text = text + "   "
        for i in range(len(text) - 2):
            tri = text[i : i + 3]
            h = self._bucket(tri, self.dim)
            vec[h] += 1.0
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        return vec


# Known embedding dimensions for common Ollama embedding models. Used so the
# tiered resolver can construct an EmbeddingIndex with the right `dim`
# without a probe call. If a model isn't listed, OllamaEmbedder discovers
# its dimension with a single embed call at construction time.
OLLAMA_EMBED_DIMS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
    "bge-m3": 1024,
}

# Maximum characters sent to Ollama in one embeddings request.
#
# Embedding models have a fixed context window — nomic-embed-text is
# ~2048 tokens — and Ollama returns a 500 for input that overflows it
# rather than truncating. So input must be capped here, before the
# request.
#
# The cap is in CHARACTERS but the model's limit is in TOKENS, and the
# ratio varies: English prose averages ~4 chars/token, but code, markup,
# tables, and dense technical text can run closer to ~3. The cap must sit
# safely *under* the worst case, not at the average.
#
# 5000 characters is ~1250 tokens for prose and ~1700 tokens for dense
# 3-chars/token text — comfortably under the 2048-token window with real
# margin. As of v1.2.1 this is a *per-request* safety ceiling, not the
# point at which a record's content is lost: records are split by
# `chunk_text` into pieces of CHUNK_TARGET_CHARS (3500, well under this
# cap) before embedding, so the whole record is embedded across multiple
# chunk vectors rather than truncated to its opening. This constant is the
# backstop that guarantees no single chunk can overflow the Ollama
# request even if CHUNK_TARGET_CHARS were raised; keep it >= the chunk
# target.
#
# Do NOT raise this much further without testing: diagnostics have shown
# Ollama's behavior near the token boundary is not perfectly predictable
# (it has 500'd inconsistently on borderline input). 8000 chars — an
# earlier value — could reach ~2600 tokens and was the cause of repeated
# Ollama 500s. After changing this, run diagnose_index.py: it embeds
# every record (chunked, as the real path does) and prints OK/FAILED per
# record, so a too-high cap shows up immediately rather than as a
# mysterious boot hang.
OLLAMA_EMBED_MAX_CHARS = 5000


class OllamaEmbedder:
    """
    Embeds text by calling a local Ollama server's embeddings endpoint.

    Runs fully locally: Ollama is a local server (default localhost:11434)
    and the model weights live on the user's disk after one `ollama pull`.
    The HTTP call here never leaves the machine. Unlike sentence-transformers
    this keeps PyTorch out of the agent process — the model runs in Ollama's
    process instead.

    The `requests` dependency is imported lazily (inside __init__) so it
    stays optional for users who only ever run the HashingEmbedder fallback.

    Default model is `nomic-embed-text` (~270 MB, 768-dim, competitive
    retrieval quality). Pull it once with: `ollama pull nomic-embed-text`.

    Construction does one probe embed against the server, so building an
    OllamaEmbedder fails fast and clearly if the server is down or the model
    isn't pulled — which is exactly what the tiered resolver in run.py needs
    in order to fall back cleanly.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 30.0,
    ):
        try:
            import requests
        except ImportError:
            raise RuntimeError(
                "OllamaEmbedder needs the `requests` package: pip install requests"
            )
        self._requests = requests
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._embed_url = f"{self.base_url}/api/embeddings"
        self._batch_url = f"{self.base_url}/api/embed"

        # Resolve the output dimension AND confirm the server actually
        # answers, by doing one probe embed at construction time. The probe
        # is what makes construction fail fast — without it, an unreachable
        # server (or an un-pulled model) would only surface on the first
        # real record, far from where the tiered resolver can fall back.
        #
        # For models in the known-dimensions table we still probe (to catch
        # server/model errors early) but trust the table for `dim` rather
        # than the probe's vector length, since a quantized or patched model
        # could in principle report an unexpected size.
        probe = self._embed("dimension probe")
        known = OLLAMA_EMBED_DIMS.get(model)
        self.dim = known if known is not None else int(probe.shape[0])

    def _post(self, url: str, payload: dict, timeout: float) -> dict:
        """POST to Ollama and return the parsed JSON body, mapping every
        failure mode to a RuntimeError with an actionable message."""
        try:
            resp = self._requests.post(url, json=payload, timeout=timeout)
        except self._requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"cannot reach Ollama at {self.base_url}: {e}\n"
                "is the server running? try: `ollama serve`"
            )
        if resp.status_code == 404:
            raise RuntimeError(
                f"Ollama returned 404 for embedding model '{self.model}'. "
                f"Pull it first: ollama pull {self.model}"
            )
        # On any other non-2xx, surface the server's response body in the
        # error. Ollama's 500 responses typically include a JSON body with
        # the actual reason (tokenizer overflow, model-load failure, GPU
        # OOM, etc.) — without this, the caller only sees the generic
        # HTTP status and has no way to diagnose which Ollama failure
        # mode caused the skip. The body is truncated to 500 chars to
        # keep boot logs readable when the same error fires on many
        # records.
        if not resp.ok:
            body = (resp.text or "").strip()[:500]
            raise RuntimeError(
                f"Ollama returned {resp.status_code} for model "
                f"'{self.model}': {body or '(empty body)'}"
            )
        return resp.json()

    @staticmethod
    def _normalize(embedding) -> np.ndarray:
        vec = np.asarray(embedding, dtype=np.float32)
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n
        return vec

    def _embed(self, text: str) -> np.ndarray:
        """Single embeddings call. Raises RuntimeError with an actionable message."""
        # Cap the input length. Embedding models have a fixed context
        # window and Ollama returns a 500 for input that overflows it
        # rather than truncating, so an oversized record (a large file, a
        # long reflection) would otherwise crash indexing. Truncating here
        # is safe: the leading chunk is a representative vector for
        # topic-level retrieval.
        if len(text) > OLLAMA_EMBED_MAX_CHARS:
            text = text[:OLLAMA_EMBED_MAX_CHARS]
        data = self._post(self._embed_url,
                          {"model": self.model, "prompt": text},
                          timeout=self.timeout_s)
        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError(
                f"Ollama returned no embedding for model '{self.model}'"
            )
        return self._normalize(embedding)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed many texts in ONE request via the newer /api/embed
        endpoint (its `input` field accepts a list; the legacy
        /api/embeddings used by _embed is single-prompt only).

        Each text gets the same truncation and L2-normalization as
        _embed. The timeout scales with batch size: the server embeds
        the inputs sequentially on CPU, so a fixed timeout sized for one
        chunk would abort every real batch.
        """
        if not texts:
            return []
        capped = [t[:OLLAMA_EMBED_MAX_CHARS] for t in texts]
        timeout = self.timeout_s * max(1.0, len(capped) / 4)
        data = self._post(self._batch_url,
                          {"model": self.model, "input": capped},
                          timeout=timeout)
        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(capped):
            got = len(embeddings) if embeddings else 0
            raise RuntimeError(
                f"Ollama returned {got} embeddings for {len(capped)} "
                f"inputs (model '{self.model}')"
            )
        return [self._normalize(e) for e in embeddings]

    def __call__(self, text: str) -> np.ndarray:
        return self._embed(text)


def ollama_is_reachable(
    base_url: str = "http://localhost:11434",
    timeout_s: float = 3.0,
) -> bool:
    """
    Cheap connectivity probe: does a local Ollama server answer on /api/tags?

    Used by the tiered embedder resolver in run.py. Deliberately swallows
    every failure mode (no server, refused connection, timeout, missing
    `requests`) and returns a plain bool — the caller's job is just "Ollama
    or fallback," not diagnosing why Ollama is absent.
    """
    try:
        import requests
    except ImportError:
        return False
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout_s)
        return resp.status_code == 200
    except Exception:
        return False
