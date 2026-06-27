"""
timechain — append-only, hash-chained, signed memory for AI agents.

Design:
  - SQLite as the local store (single-writer, transactional, boring)
  - Ed25519 signatures on every record
  - SHA-256 content hashing and prior-hash linking
  - Periodic Merkle batching with roots that can be anchored externally
  - Canonical JSON serialization (RFC 8785) so hashes are stable

Threat model:
  - Detect any post-hoc modification of any past record
  - Detect any insertion or deletion in the chain
  - Verify with only the operator's public key and the chain itself
  - Optional external anchoring (e.g. OpenTimestamps -> Bitcoin) for
    third-party-verifiable integrity over long timescales

What this file deliberately does NOT include:
  - Consensus, tokens, mining, or anything blockchain-flavored beyond
    the data structure. There is one writer per chain. That is the point.
  - Schema fields for "qualia," "brightness," etc. The schema is minimal
    and extensible via the typed `content` blob.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote as _url_quote

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature


class _SigningKey:
    """Thin wrapper around cryptography's Ed25519 to mirror pynacl's API."""

    def __init__(self, raw_seed: bytes):
        self._key = Ed25519PrivateKey.from_private_bytes(raw_seed)
        self._seed = raw_seed

    @classmethod
    def generate(cls) -> "_SigningKey":
        k = Ed25519PrivateKey.generate()
        seed = k.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return cls(seed)

    def encode(self) -> bytes:
        return self._seed

    def sign(self, message: bytes) -> bytes:
        return self._key.sign(message)

    @property
    def verify_key_bytes(self) -> bytes:
        return self._key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )


def _verify_signature(pubkey_bytes: bytes, signature: bytes, message: bytes) -> None:
    Ed25519PublicKey.from_public_bytes(pubkey_bytes).verify(signature, message)


# ---------------------------------------------------------------------------
# Canonical serialization
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> bytes:
    """
    Deterministic JSON encoding suitable for hashing.

    Sorted keys, no insignificant whitespace, UTF-8. Equivalent in spirit to
    RFC 8785 for the JSON subset we use here. If you need full JCS (number
    canonicalization edge cases) swap in a JCS library — for our schema
    (strings, ints, dicts, lists) this is sufficient and stable.

    Floats: signed content MAY contain floats only if they come from a
    known small set of values (salience, confidence, PoQ dimensions),
    each rounded to a fixed number of decimal places by `build_meta`
    before they reach a record. Computed floats — multiplications,
    divisions, sqrt — must NOT be put into signed content directly,
    because their str() representation can drift between Python
    versions, and a content hash that differs across runtimes would
    break verification on the recipient side. If you need to record a
    computed float, round it to a fixed precision yourself or store
    the operands and let the consumer recompute.

    The float values currently in _meta (e.g. salience=0.4,
    confidence=0.9, poq.dimensions={...}) come from build_meta with
    explicit rounding, so this rule holds in practice. The note is here
    so future record types don't accidentally violate it.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Record schema
# ---------------------------------------------------------------------------

# A record is the smallest unit of chain state. Keep this minimal —
# anything domain-specific lives inside `content`.
#
#   index        : monotonic integer, 0 = genesis
#   prior_hash   : hex sha256 of the prior record's signed bytes (or 64 zeros for genesis)
#   timestamp    : ms since epoch, advisory only — DO NOT use for ordering
#   type         : short string, application-defined ("observation", "action", etc.)
#   content      : arbitrary JSON-serializable payload
#   refs         : list of prior record hashes this record explicitly references
#   pubkey       : hex Ed25519 public key of the writer
#   content_hash : hex sha256 of canonical_json(content)
#   record_hash  : hex sha256 of canonical_json(<everything above>)
#   signature    : hex Ed25519 signature over record_hash bytes


@dataclass(frozen=True)
class Record:
    index: int
    prior_hash: str
    timestamp: int
    type: str
    content: Any
    refs: list[str]
    pubkey: str
    content_hash: str
    record_hash: str
    signature: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "prior_hash": self.prior_hash,
            "timestamp": self.timestamp,
            "type": self.type,
            "content": self.content,
            "refs": list(self.refs),
            "pubkey": self.pubkey,
            "content_hash": self.content_hash,
            "record_hash": self.record_hash,
            "signature": self.signature,
        }

    def signing_payload(self) -> dict:
        """The fields that go into record_hash. Excludes record_hash and signature."""
        return {
            "index": self.index,
            "prior_hash": self.prior_hash,
            "timestamp": self.timestamp,
            "type": self.type,
            "content": self.content,
            "refs": list(self.refs),
            "pubkey": self.pubkey,
            "content_hash": self.content_hash,
        }


GENESIS_PRIOR_HASH = "0" * 64


# ---------------------------------------------------------------------------
# Merkle tree
# ---------------------------------------------------------------------------

def merkle_root(leaves: list[bytes]) -> bytes:
    """
    Standard Merkle root. Duplicates the last leaf if odd (Bitcoin-style)
    so root is well-defined for any non-empty input. Leaves must be 32 bytes.
    """
    if not leaves:
        raise ValueError("merkle_root requires at least one leaf")
    layer = list(leaves)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        layer = [sha256(layer[i] + layer[i + 1]) for i in range(0, len(layer), 2)]
    return layer[0]


def merkle_proof(leaves: list[bytes], target_index: int) -> list[tuple[str, bytes]]:
    """
    Return inclusion proof for leaves[target_index] as a list of
    (side, sibling_hash) pairs from leaf to root. side is 'L' or 'R'
    indicating where the sibling sits relative to the running hash.
    """
    if not (0 <= target_index < len(leaves)):
        raise ValueError("target_index out of range")
    proof: list[tuple[str, bytes]] = []
    layer = list(leaves)
    idx = target_index
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        if idx % 2 == 0:
            proof.append(("R", layer[idx + 1]))
        else:
            proof.append(("L", layer[idx - 1]))
        layer = [sha256(layer[i] + layer[i + 1]) for i in range(0, len(layer), 2)]
        idx //= 2
    return proof


def verify_merkle_proof(leaf: bytes, proof: list[tuple[str, bytes]], root: bytes) -> bool:
    h = leaf
    for side, sibling in proof:
        h = sha256(sibling + h) if side == "L" else sha256(h + sibling)
    return h == root


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    idx           INTEGER PRIMARY KEY,
    prior_hash    TEXT NOT NULL,
    timestamp     INTEGER NOT NULL,
    type          TEXT NOT NULL,
    content_json  TEXT NOT NULL,
    refs_json     TEXT NOT NULL,
    pubkey        TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    record_hash   TEXT NOT NULL UNIQUE,
    signature     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_records_type ON records(type);
CREATE INDEX IF NOT EXISTS idx_records_timestamp ON records(timestamp);

CREATE TABLE IF NOT EXISTS merkle_batches (
    batch_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    first_idx       INTEGER NOT NULL,
    last_idx        INTEGER NOT NULL,
    root_hash       TEXT NOT NULL,
    created_at      INTEGER NOT NULL,
    anchor_proof    TEXT,    -- JSON, populated by external anchorer (e.g. OpenTimestamps)
    anchor_status   TEXT     -- 'pending', 'anchored', 'verified'
);

CREATE TABLE IF NOT EXISTS chain_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Materialized index of revision -> superseded record. Avoids scanning
-- every `revision` record in the chain on every retrieval just to learn
-- which originals have corrections. Maintained on append: when a record
-- of type 'revision' is committed, we extract its supersedes pointer
-- and write a row here.
--
-- A `revision_idx` -> `superseded_idx` table is the source of truth a
-- separate `Retriever._superseded_indices` and the prompt-builder used
-- to recompute from scratch on every call, capped by a fragile silent
-- limit. With this table both can read the answer in one query.
--
-- Backfilled lazily by `Chain._backfill_supersedes_index` on first read
-- so existing chains upgrade transparently.
CREATE TABLE IF NOT EXISTS supersedes_index (
    revision_idx   INTEGER PRIMARY KEY,
    superseded_idx INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_supersedes_superseded
    ON supersedes_index(superseded_idx);

-- Materialized index of blob sha256 -> file record index. Avoids the
-- linear scan that `/blobs/<sha>` and `Agent._collect_attachments` used
-- to do per request (a `query_by_type("file", limit=500)` followed by a
-- Python-side filter — yet another silent cap on a hot path). Maintained
-- on append when a `file` record is committed; lazy-backfilled on first
-- read so existing chains upgrade transparently.
--
-- The sha is the record's payload digest, not the record's signature
-- hash — multiple `file` records can refer to the same blob (re-ingest
-- of an identical file), so the index is keyed by sha and the value is
-- the LATEST such record's index (sufficient for blob resolution; the
-- sha collision implies the bytes are identical anyway).
CREATE TABLE IF NOT EXISTS blob_index (
    blob_sha256 TEXT PRIMARY KEY,
    record_idx  INTEGER NOT NULL
);

-- Materialized index of proposal -> recurrence record indices. Avoids
-- the `query_by_type("proposal_recurrence", limit=10_000)` scan that
-- `recurrence_count`, `recurrence_counts`, and `escalated_indices` all
-- did on every call — another silent cap on a hot path (the /proposals
-- UI hits these bulk helpers on every render). Maintained on append
-- when a `proposal_recurrence` record is committed; lazy-backfilled on
-- first read so existing chains upgrade transparently.
--
-- One row per recurrence record. Keyed on the recurrence's own index
-- so re-running backfill is idempotent. The `proposal_idx` column has
-- an index so per-proposal counts and lookups are O(matching) rather
-- than O(all recurrences ever).
CREATE TABLE IF NOT EXISTS proposal_recurrence_index (
    recurrence_idx INTEGER PRIMARY KEY,
    proposal_idx   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_proposal_recurrence_proposal
    ON proposal_recurrence_index(proposal_idx);
"""


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

class ChainError(Exception):
    pass


class Chain:
    def __init__(self, db_path: str | Path, signing_key: _SigningKey):
        self.db_path = str(db_path)
        self.signing_key = signing_key
        self.pubkey_hex = signing_key.verify_key_bytes.hex()
        self._conn = sqlite3.connect(self.db_path)
        # WAL mode: faster concurrent reads, no writer-blocks-reader.
        # synchronous=NORMAL: durable on power loss in WAL mode but ~10x faster
        # than FULL. Acceptable tradeoff for an append-only log; if a crash
        # loses the very last unfinished write, the chain is still consistent.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ------- writing -------

    def append(self, type_: str, content: Any, refs: Optional[Iterable[str]] = None,
               difficulty: int = 0) -> Record:
        # Immune lockdown gate (immune.py). While a `LOCKED` flag exists next
        # to the DB, the ONLY record type that may be appended is `recovery` —
        # so no seal path (REPL, webapp, reflection, cambium) can bypass a
        # lockdown. One guard. Absent the flag (the normal case) this is a
        # single cheap stat and changes nothing. The flag is derived sidecar
        # state, never on the signed chain.
        if type_ != "recovery":
            if (Path(self.db_path).parent / "LOCKED").exists():
                raise ChainError(
                    "chain is locked (immune lockdown); only 'recovery' records "
                    "may be appended until rollback/recovery clears the lock"
                )
        refs = list(refs or [])
        cur = self._conn.cursor()
        cur.execute("SELECT idx, record_hash FROM records ORDER BY idx DESC LIMIT 1")
        row = cur.fetchone()
        if row is None:
            index = 0
            prior_hash = GENESIS_PRIOR_HASH
        else:
            index = row[0] + 1
            prior_hash = row[1]

        timestamp = int(time.time() * 1000)

        def _build(content_obj):
            ch = sha256_hex(canonical_json(content_obj))
            sp = {
                "index": index,
                "prior_hash": prior_hash,
                "timestamp": timestamp,
                "type": type_,
                "content": content_obj,
                "refs": refs,
                "pubkey": self.pubkey_hex,
                "content_hash": ch,
            }
            return ch, sp

        if difficulty and difficulty > 0:
            # Proof-of-work "brightness": mine a nonce until record_hash has
            # `difficulty` leading hex zeros. The nonce lives INSIDE content
            # under `_pow`, so it is covered by content_hash, the signature, and
            # verify()'s recompute — no Record schema change, and the default
            # difficulty=0 path below stays byte-identical to earlier versions
            # (no `_pow` field, omit-when-zero). PoW requires dict content so the
            # `_pow` key can be added without silently changing the content's
            # shape (a non-dict value would otherwise be wrapped, so the same
            # value would serialize differently at difficulty 0 vs >0).
            if not isinstance(content, dict):
                raise ChainError(
                    "proof-of-work (difficulty > 0) requires dict content")
            content = dict(content)
            prefix = "0" * difficulty
            nonce = 0
            while True:
                content["_pow"] = {"nonce": nonce, "difficulty": difficulty}
                content_hash, signing_payload = _build(content)
                record_hash = sha256_hex(canonical_json(signing_payload))
                if record_hash.startswith(prefix):
                    break
                nonce += 1
            record_hash_bytes = bytes.fromhex(record_hash)
        else:
            content_hash, signing_payload = _build(content)
            record_hash_bytes = sha256(canonical_json(signing_payload))
            record_hash = record_hash_bytes.hex()

        signature = self.signing_key.sign(record_hash_bytes).hex()

        cur.execute(
            """INSERT INTO records
               (idx, prior_hash, timestamp, type, content_json, refs_json,
                pubkey, content_hash, record_hash, signature)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                index,
                prior_hash,
                timestamp,
                type_,
                json.dumps(content, ensure_ascii=False),
                json.dumps(refs),
                self.pubkey_hex,
                content_hash,
                record_hash,
                signature,
            ),
        )
        # Maintain the materialized supersedes index. When a revision is
        # appended, extract its supersedes pointer (from _meta, the v2/v3
        # canonical location, falling back to the legacy `revises_index`
        # field for v1 revisions) and record the link here so the
        # retrieval-side lookup is O(1) per query rather than O(revisions).
        if type_ == "revision":
            superseded = _extract_supersedes(content)
            if superseded is not None:
                cur.execute(
                    "INSERT OR REPLACE INTO supersedes_index "
                    "(revision_idx, superseded_idx) VALUES (?, ?)",
                    (index, superseded),
                )
        # Maintain the blob index for file records. The blob sha lives in
        # the record content (`blob_sha256`); see file_ingest.
        if type_ == "file" and isinstance(content, dict):
            sha = content.get("blob_sha256")
            if isinstance(sha, str) and sha:
                cur.execute(
                    "INSERT OR REPLACE INTO blob_index "
                    "(blob_sha256, record_idx) VALUES (?, ?)",
                    (sha, index),
                )
        # Maintain the proposal_recurrence index. When a recurrence is
        # appended, its `recurs_proposal_index` field points at the
        # original proposal; record the link here so the bulk count
        # helpers in cambium.py (`recurrence_counts`, `is_escalated`,
        # `escalated_indices`) can answer in one indexed lookup instead
        # of scanning every recurrence record with a silent `limit=10_000`
        # cap that the same review pass already fixed for revisions.
        if type_ == "proposal_recurrence" and isinstance(content, dict):
            target = content.get("recurs_proposal_index")
            if isinstance(target, int):
                cur.execute(
                    "INSERT OR REPLACE INTO proposal_recurrence_index "
                    "(recurrence_idx, proposal_idx) VALUES (?, ?)",
                    (index, target),
                )
        self._conn.commit()

        return Record(
            index=index,
            prior_hash=prior_hash,
            timestamp=timestamp,
            type=type_,
            content=content,
            refs=refs,
            pubkey=self.pubkey_hex,
            content_hash=content_hash,
            record_hash=record_hash,
            signature=signature,
        )

    # ------- reading -------

    def head(self) -> Optional[Record]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM records ORDER BY idx DESC LIMIT 1")
        row = cur.fetchone()
        return _row_to_record(row) if row else None

    def get(self, index: int) -> Optional[Record]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM records WHERE idx = ?", (index,))
        row = cur.fetchone()
        return _row_to_record(row) if row else None

    def get_by_hash(self, record_hash: str) -> Optional[Record]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM records WHERE record_hash = ?", (record_hash,))
        row = cur.fetchone()
        return _row_to_record(row) if row else None

    def length(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM records")
        return cur.fetchone()[0]

    def iter_records(self, start: int = 0, end: Optional[int] = None) -> Iterable[Record]:
        cur = self._conn.cursor()
        if end is None:
            cur.execute("SELECT * FROM records WHERE idx >= ? ORDER BY idx ASC", (start,))
        else:
            cur.execute(
                "SELECT * FROM records WHERE idx >= ? AND idx < ? ORDER BY idx ASC",
                (start, end),
            )
        for row in cur:
            yield _row_to_record(row)

    def query_by_type(self, type_: str, limit: int = 50) -> list[Record]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM records WHERE type = ? ORDER BY idx DESC LIMIT ?",
            (type_, limit),
        )
        return [_row_to_record(r) for r in cur.fetchall()]

    def query_recent(self, limit: int = 20) -> list[Record]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM records ORDER BY idx DESC LIMIT ?", (limit,))
        return [_row_to_record(r) for r in cur.fetchall()]

    def follow_refs(self, record_hash: str, depth: int = 3) -> list[Record]:
        """Walk references backward from a record up to `depth` hops. BFS, dedup."""
        seen: set[str] = set()
        out: list[Record] = []
        frontier = [record_hash]
        for _ in range(depth):
            next_frontier: list[str] = []
            for h in frontier:
                if h in seen:
                    continue
                seen.add(h)
                rec = self.get_by_hash(h)
                if rec is None:
                    continue
                out.append(rec)
                next_frontier.extend(rec.refs)
            frontier = next_frontier
            if not frontier:
                break
        return out

    # ------- supersedes index (materialized) -------

    def _backfill_supersedes_index(self) -> None:
        """
        Populate the supersedes_index from existing revision records on
        first use. Idempotent: rows that already exist are left alone.
        This is what makes the upgrade transparent for chains created
        before the materialized index existed.
        """
        cur = self._conn.cursor()
        # Cheap probe: if the index has any row for a revision that exists,
        # we treat it as already backfilled and skip the work. Strictly
        # speaking we should check every revision, but the maintenance hook
        # in `append` ensures new revisions are always indexed; the only
        # rows that could be missing are pre-existing ones, and we either
        # backfill all of them or none.
        cur.execute("SELECT 1 FROM supersedes_index LIMIT 1")
        if cur.fetchone() is not None:
            return
        cur.execute(
            "SELECT idx, content_json FROM records WHERE type = 'revision'"
        )
        rows = cur.fetchall()
        for idx, content_json in rows:
            try:
                content = json.loads(content_json)
            except (ValueError, TypeError):
                continue
            sup = _extract_supersedes(content)
            if sup is not None:
                cur.execute(
                    "INSERT OR REPLACE INTO supersedes_index "
                    "(revision_idx, superseded_idx) VALUES (?, ?)",
                    (idx, sup),
                )
        self._conn.commit()

    def superseded_indices(self) -> set[int]:
        """
        The set of record indices that have been superseded by a later
        revision. O(1) per call after backfill (a single indexed SELECT).
        Replaces a previous O(N)-per-call scan over every revision record.
        """
        self._backfill_supersedes_index()
        cur = self._conn.cursor()
        cur.execute("SELECT DISTINCT superseded_idx FROM supersedes_index")
        return {row[0] for row in cur.fetchall()}

    def revisions_targeting(self, indices: Iterable[int]) -> list[Record]:
        """
        Return revision records whose supersedes pointer is in `indices`.
        Used by the retriever to pull the latest correction in alongside
        a retrieved original. O(len(indices)) on the materialized index.
        """
        idx_list = list(indices)
        if not idx_list:
            return []
        self._backfill_supersedes_index()
        cur = self._conn.cursor()
        placeholders = ",".join("?" for _ in idx_list)
        cur.execute(
            f"SELECT records.* FROM records "
            f"JOIN supersedes_index ON records.idx = supersedes_index.revision_idx "
            f"WHERE supersedes_index.superseded_idx IN ({placeholders}) "
            f"ORDER BY records.idx ASC",
            idx_list,
        )
        return [_row_to_record(r) for r in cur.fetchall()]

    # ------- blob index (materialized) -------

    def _backfill_blob_index(self) -> None:
        """
        Populate the blob_index from existing `file` records on first
        use. Same shape as `_backfill_supersedes_index`: idempotent,
        skipped after first run, makes the upgrade transparent for
        chains created before the index existed.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT 1 FROM blob_index LIMIT 1")
        if cur.fetchone() is not None:
            return
        cur.execute(
            "SELECT idx, content_json FROM records WHERE type = 'file' "
            "ORDER BY idx"
        )
        for idx, content_json in cur.fetchall():
            try:
                content = json.loads(content_json)
            except (ValueError, TypeError):
                continue
            if not isinstance(content, dict):
                continue
            sha = content.get("blob_sha256")
            if isinstance(sha, str) and sha:
                cur.execute(
                    "INSERT OR REPLACE INTO blob_index "
                    "(blob_sha256, record_idx) VALUES (?, ?)",
                    (sha, idx),
                )
        self._conn.commit()

    def find_file_by_sha(self, blob_sha256: str) -> Optional[Record]:
        """
        Return the file record with the given blob sha256, or None if no
        such record exists. Indexed O(1) lookup — replaces the linear
        `query_by_type("file", limit=500)` scan the webapp's `/blobs/<sha>`
        endpoint used to do per request (which had its own silent 500-
        record cap on a hot path).

        If multiple file records refer to the same blob (re-ingest of
        identical bytes), the most recent one wins — which is fine for
        blob resolution since the bytes are identical by definition of
        the sha collision.
        """
        if not blob_sha256:
            return None
        self._backfill_blob_index()
        cur = self._conn.cursor()
        cur.execute(
            "SELECT record_idx FROM blob_index WHERE blob_sha256 = ?",
            (blob_sha256,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self.get(row[0])

    # ------- proposal recurrence index (materialized) -------

    def _backfill_proposal_recurrence_index(self) -> None:
        """
        Populate the proposal_recurrence_index from existing
        `proposal_recurrence` records on first use. Same shape as the
        other two backfills: idempotent (skipped if the table has any
        rows), makes the upgrade transparent for chains created before
        this index existed.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT 1 FROM proposal_recurrence_index LIMIT 1")
        if cur.fetchone() is not None:
            return
        cur.execute(
            "SELECT idx, content_json FROM records "
            "WHERE type = 'proposal_recurrence' ORDER BY idx"
        )
        for idx, content_json in cur.fetchall():
            try:
                content = json.loads(content_json)
            except (ValueError, TypeError):
                continue
            if not isinstance(content, dict):
                continue
            target = content.get("recurs_proposal_index")
            if isinstance(target, int):
                cur.execute(
                    "INSERT OR REPLACE INTO proposal_recurrence_index "
                    "(recurrence_idx, proposal_idx) VALUES (?, ?)",
                    (idx, target),
                )
        self._conn.commit()

    def recurrence_count_for(self, proposal_idx: int) -> int:
        """
        Live recurrence count for one proposal, via the materialized
        index. Same value `cambium.recurrence_count` returned by linear
        scan, but O(matching) on an indexed lookup. Returns 0 for an
        index that isn't a proposal (caller is responsible for that
        check if it matters; the value is then "no recurrences against
        this index," which is also 0 for a valid proposal that hasn't
        recurred).
        """
        self._backfill_proposal_recurrence_index()
        cur = self._conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM proposal_recurrence_index "
            "WHERE proposal_idx = ?",
            (proposal_idx,),
        )
        # +1 because cambium counts the proposal's own creation as the
        # first sighting; the index only stores subsequent recurrences.
        # Caller must add 1 for the "proposal exists" baseline. See
        # cambium.recurrence_count for the historical semantics.
        return cur.fetchone()[0]

    def all_recurrence_counts(self) -> dict[int, int]:
        """
        Bulk version: returns `{proposal_idx: count_of_recurrences}`,
        with one indexed scan and one Python-side aggregation.
        Replaces `cambium.recurrence_counts` walking every recurrence
        record per call.

        As with `recurrence_count_for`, the returned counts do NOT
        include the proposal's own creation — callers that want the
        "live count" Cambium uses must add 1 per proposal.
        """
        self._backfill_proposal_recurrence_index()
        cur = self._conn.cursor()
        cur.execute(
            "SELECT proposal_idx, COUNT(*) FROM proposal_recurrence_index "
            "GROUP BY proposal_idx"
        )
        return {row[0]: row[1] for row in cur.fetchall()}

    # ------- chain metadata + watermark -------

    def get_meta(self, key: str) -> Optional[str]:
        """
        Read a value from the `chain_meta` key/value table. Returns None
        if the key is absent. Use for small persisted state that isn't
        worth its own table (e.g. Cambium's `last_cambium_scanned_idx`
        watermark — see Cambium.scan).
        """
        cur = self._conn.cursor()
        cur.execute("SELECT value FROM chain_meta WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Write a value to `chain_meta`. Idempotent (REPLACE)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO chain_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    # ------- observability -------

    def stats(self) -> dict:
        """
        Aggregate counts and timing for the chain — what an operator
        wants to see when answering "is this thing actually working."

        Returns a dict with:
          - length: total record count
          - first_timestamp / last_timestamp: ms epoch of the first and
            last record (None if the chain is empty)
          - by_type: {record_type: count}
          - sealed_batches: number of sealed Merkle batches
          - anchored_batches: subset of those that have an external anchor
          - quarantined_records: count of records with exposure=quarantine
            in their _meta block

        Cost: most counts are O(1) indexed SELECTs. The `quarantined`
        count is the exception — it does a sequential scan with a LIKE
        on `content_json`, so it's O(records). On a typical
        single-operator chain (10k–100k records) the whole call still
        returns in single-digit milliseconds; on a million-record chain
        it climbs to ~100ms. If that becomes the bottleneck, add an
        `exposure_index` table on the same shape as `blob_index` and
        `supersedes_index` and switch this query to use it.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM records")
        length = cur.fetchone()[0]

        first_ts: Optional[int] = None
        last_ts: Optional[int] = None
        if length > 0:
            cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM records")
            first_ts, last_ts = cur.fetchone()

        cur.execute("SELECT type, COUNT(*) FROM records GROUP BY type")
        by_type = {row[0]: row[1] for row in cur.fetchall()}

        cur.execute("SELECT COUNT(*) FROM merkle_batches")
        sealed_batches = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM merkle_batches WHERE anchor_status = 'anchored'"
        )
        anchored_batches = cur.fetchone()[0]

        # Quarantine count: scan content_json for the exposure marker.
        # SQLite has no JSON1 dependency promise here, so we do a LIKE
        # filter — a false positive is harmless (just inflates the count
        # very slightly) and the cost is one indexed pass.
        cur.execute(
            "SELECT COUNT(*) FROM records "
            "WHERE content_json LIKE '%\"exposure\":\"quarantine\"%'"
        )
        quarantined = cur.fetchone()[0]

        return {
            "length": length,
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "by_type": by_type,
            "sealed_batches": sealed_batches,
            "anchored_batches": anchored_batches,
            "quarantined_records": quarantined,
        }

    # ------- verification -------

    def verify(self, expected_pubkey: Optional[str] = None) -> tuple[bool, str]:
        """
        Walk the chain start to head, verifying:
          - prior_hash linkage
          - content_hash matches content
          - record_hash matches signing payload
          - signature is valid for the embedded pubkey
          - (optional) pubkey matches the expected operator key

        Returns (ok, message). On failure, message identifies the first bad record.
        """
        prior = GENESIS_PRIOR_HASH
        expected_index = 0
        for rec in self.iter_records():
            if rec.index != expected_index:
                return False, f"index gap at {rec.index} (expected {expected_index})"
            if rec.prior_hash != prior:
                return False, f"prior_hash mismatch at index {rec.index}"
            if expected_pubkey is not None and rec.pubkey != expected_pubkey:
                return False, f"unexpected pubkey at index {rec.index}"

            recomputed_content_hash = sha256_hex(canonical_json(rec.content))
            if recomputed_content_hash != rec.content_hash:
                return False, f"content_hash mismatch at index {rec.index}"

            recomputed_record_hash = sha256_hex(canonical_json(rec.signing_payload()))
            if recomputed_record_hash != rec.record_hash:
                return False, f"record_hash mismatch at index {rec.index}"

            # Proof-of-work check (additive). Only records that opted into a
            # difficulty (an embedded `_pow` block) are checked; all others skip
            # this, so existing chains are unaffected.
            pow_block = rec.content.get("_pow") if isinstance(rec.content, dict) else None
            if isinstance(pow_block, dict):
                d = pow_block.get("difficulty", 0)
                if isinstance(d, int) and d > 0 and not rec.record_hash.startswith("0" * d):
                    return False, f"proof-of-work below target at index {rec.index}"

            try:
                _verify_signature(
                    bytes.fromhex(rec.pubkey),
                    bytes.fromhex(rec.signature),
                    bytes.fromhex(rec.record_hash),
                )
            except (InvalidSignature, ValueError) as e:
                return False, f"bad signature at index {rec.index}: {e}"

            prior = rec.record_hash
            expected_index += 1
        return True, f"chain ok ({expected_index} records)"

    def verify_threadsafe(self, expected_pubkey: Optional[str] = None) -> tuple[bool, str]:
        """
        Same semantics as `verify()`, but opens its own short-lived
        read-only SQLite connection internally so it can be called from
        a worker thread (e.g. `asyncio.to_thread(chain.verify_threadsafe)`).

        `Chain` keeps a long-lived connection that, by default, is bound
        to the thread that created it (`check_same_thread=True`). The
        webapp's `/api/chain/verify` runs the cryptographic walk —
        seconds to minutes on long chains — and previously did so on
        the asyncio event loop, blocking the server for the duration.
        Offloading to a thread requires a connection that thread can
        use; this method provides one.

        The connection is read-only and lives only for the duration of
        the call. Writes to the chain on the main thread are unaffected;
        SQLite's WAL mode (which we don't currently enable but is the
        natural next step) would let writes proceed concurrently with
        this read.
        """
        # Open a fresh read-only connection bound to whichever thread
        # is calling us. `uri=True` lets us pass `mode=ro` so writes
        # are rejected, defending against any accidental misuse. The
        # path is percent-quoted so a directory name containing `?`,
        # `#`, or other URI-reserved characters doesn't get mis-parsed
        # — without quoting, `/data?weird/chain.sqlite` would be read
        # as path=`/data` with query=`weird/chain.sqlite`, and SQLite
        # would silently open a different (empty) database. The `/`
        # is kept literal so the path structure is preserved.
        ro_uri = f"file:{_url_quote(str(self.db_path), safe='/')}?mode=ro"
        ro_conn = sqlite3.connect(ro_uri, uri=True)
        try:
            prior = GENESIS_PRIOR_HASH
            expected_index = 0
            cur = ro_conn.cursor()
            cur.execute(
                "SELECT idx, prior_hash, timestamp, type, content_json, "
                "refs_json, pubkey, content_hash, record_hash, signature "
                "FROM records ORDER BY idx ASC"
            )
            for row in cur:
                rec = _row_to_record(row)
                if rec.index != expected_index:
                    return False, f"index gap at {rec.index} (expected {expected_index})"
                if rec.prior_hash != prior:
                    return False, f"prior_hash mismatch at index {rec.index}"
                if expected_pubkey is not None and rec.pubkey != expected_pubkey:
                    return False, f"unexpected pubkey at index {rec.index}"

                recomputed_content_hash = sha256_hex(canonical_json(rec.content))
                if recomputed_content_hash != rec.content_hash:
                    return False, f"content_hash mismatch at index {rec.index}"

                recomputed_record_hash = sha256_hex(canonical_json(rec.signing_payload()))
                if recomputed_record_hash != rec.record_hash:
                    return False, f"record_hash mismatch at index {rec.index}"

                # Proof-of-work check (additive; mirrors verify()).
                pow_block = rec.content.get("_pow") if isinstance(rec.content, dict) else None
                if isinstance(pow_block, dict):
                    d = pow_block.get("difficulty", 0)
                    if isinstance(d, int) and d > 0 and not rec.record_hash.startswith("0" * d):
                        return False, f"proof-of-work below target at index {rec.index}"

                try:
                    _verify_signature(
                        bytes.fromhex(rec.pubkey),
                        bytes.fromhex(rec.signature),
                        bytes.fromhex(rec.record_hash),
                    )
                except (InvalidSignature, ValueError) as e:
                    return False, f"bad signature at index {rec.index}: {e}"

                prior = rec.record_hash
                expected_index += 1
            return True, f"chain ok ({expected_index} records)"
        finally:
            ro_conn.close()

    def verify_semantic(self) -> tuple[bool, list[str]]:
        """
        Schema-level consistency probe — a companion to `verify()`.

        `verify()` covers the cryptographic invariants: linkage, hashes,
        signatures. It cannot catch a *semantic* corruption — a
        revision that points at a non-existent index, a
        `proposal_recurrence` against a hash that isn't a proposal, a
        `proposal_status` whose target index lives past the end of the
        chain. The chain would record these correctly (signatures valid,
        linkage intact) but the *meaning* of the data is broken.

        Returns `(ok, warnings)`. `ok` is True only when no warnings
        were generated. The probe is intentionally an LIST of warnings
        rather than a fail-fast: a long-lived chain may accumulate one
        or two stale references (e.g. after an experimental tool wrote
        a malformed record), and the operator wants to see all of them
        in one pass to triage.

        Checks performed:
          - Every `revision._meta.supersedes` (or legacy
            `revises_index`) points at an existing record.
          - Every `proposal_recurrence.recurs_proposal_index` points
            at a record that is in fact a `proposal`.
          - Every `proposal_status.marks_proposal_index` points at a
            record that is in fact a `proposal`.
          - Every `reflection.covers_indices` range fits inside the
            chain length.
          - Every `proposal.evidence` index points at an existing
            record.

        The probe is read-only and cheap (a few indexed scans). It is
        safe to run inline on the event loop for normal-sized chains;
        for very long chains, dispatch through `asyncio.to_thread` the
        same way `verify_threadsafe` is invoked.
        """
        warnings: list[str] = []
        length = self.length()

        def _exists(idx: int) -> bool:
            return 0 <= idx < length

        # 1. revisions point at existing records
        for rev in self.query_by_type("revision", limit=1_000_000):
            target = _extract_supersedes(rev.content)
            if target is None:
                # Revision with no supersedes pointer at all — possible
                # but unusual. Flag it gently.
                warnings.append(
                    f"revision at {rev.index} has no supersedes pointer "
                    f"(neither _meta.supersedes nor revises_index)"
                )
                continue
            if not _exists(target):
                warnings.append(
                    f"revision at {rev.index} supersedes index {target} "
                    f"which is past the chain (length={length})"
                )

        # 2. proposal_recurrence targets must be proposal records
        for rr in self.query_by_type("proposal_recurrence", limit=1_000_000):
            if not isinstance(rr.content, dict):
                continue
            target = rr.content.get("recurs_proposal_index")
            if not isinstance(target, int):
                warnings.append(
                    f"proposal_recurrence at {rr.index} has no integer "
                    f"recurs_proposal_index"
                )
                continue
            if not _exists(target):
                warnings.append(
                    f"proposal_recurrence at {rr.index} references "
                    f"missing index {target}"
                )
                continue
            target_rec = self.get(target)
            if target_rec is None or target_rec.type != "proposal":
                warnings.append(
                    f"proposal_recurrence at {rr.index} references "
                    f"index {target}, which is "
                    f"type={target_rec.type if target_rec else 'missing'} "
                    f"(expected 'proposal')"
                )

        # 3. proposal_status targets must be proposal records
        for sr in self.query_by_type("proposal_status", limit=1_000_000):
            if not isinstance(sr.content, dict):
                continue
            target = sr.content.get("marks_proposal_index")
            if not isinstance(target, int):
                warnings.append(
                    f"proposal_status at {sr.index} has no integer "
                    f"marks_proposal_index"
                )
                continue
            if not _exists(target):
                warnings.append(
                    f"proposal_status at {sr.index} marks missing index {target}"
                )
                continue
            target_rec = self.get(target)
            if target_rec is None or target_rec.type != "proposal":
                warnings.append(
                    f"proposal_status at {sr.index} marks index {target}, "
                    f"which is type={target_rec.type if target_rec else 'missing'} "
                    f"(expected 'proposal')"
                )

        # 4. reflection.covers_indices ranges fit in the chain
        for refl in self.query_by_type("reflection", limit=1_000_000):
            if not isinstance(refl.content, dict):
                continue
            covers = refl.content.get("covers_indices")
            if not isinstance(covers, (list, tuple)) or len(covers) != 2:
                continue
            lo, hi = covers
            if not (isinstance(lo, int) and isinstance(hi, int)):
                continue
            if lo < 0 or hi >= length or lo > hi:
                warnings.append(
                    f"reflection at {refl.index} covers_indices "
                    f"[{lo}, {hi}] is outside chain (length={length})"
                )

        # 5. proposal.evidence indices must exist
        for prop in self.query_by_type("proposal", limit=1_000_000):
            if not isinstance(prop.content, dict):
                continue
            evidence = prop.content.get("evidence")
            if not isinstance(evidence, (list, tuple)):
                continue
            bad = [i for i in evidence if not (isinstance(i, int) and _exists(i))]
            if bad:
                warnings.append(
                    f"proposal at {prop.index} has evidence indices "
                    f"outside chain: {bad[:10]}"
                )

        return (len(warnings) == 0), warnings

    # ------- merkle batching -------

    def seal_batch(self, batch_size: int = 64) -> Optional[dict]:
        """
        Take the next un-batched run of records and create a Merkle root over them.
        Returns the batch metadata (without external anchor) or None if nothing to seal.

        The root_hash is what you'd hand to OpenTimestamps to anchor on Bitcoin.
        """
        cur = self._conn.cursor()
        cur.execute("SELECT COALESCE(MAX(last_idx), -1) FROM merkle_batches")
        last_batched = cur.fetchone()[0]
        first = last_batched + 1
        last = first + batch_size - 1
        leaves: list[bytes] = []
        actual_last = first - 1
        for rec in self.iter_records(start=first, end=last + 1):
            leaves.append(bytes.fromhex(rec.record_hash))
            actual_last = rec.index
        if not leaves:
            return None
        root = merkle_root(leaves)
        created_at = int(time.time() * 1000)
        cur.execute(
            """INSERT INTO merkle_batches
               (first_idx, last_idx, root_hash, created_at, anchor_status)
               VALUES (?, ?, ?, ?, 'pending')""",
            (first, actual_last, root.hex(), created_at),
        )
        batch_id = cur.lastrowid
        self._conn.commit()
        return {
            "batch_id": batch_id,
            "first_idx": first,
            "last_idx": actual_last,
            "root_hash": root.hex(),
            "created_at": created_at,
        }

    def inclusion_proof(self, index: int) -> Optional[dict]:
        """Return Merkle inclusion proof for a record, if its batch is sealed."""
        cur = self._conn.cursor()
        cur.execute(
            """SELECT batch_id, first_idx, last_idx, root_hash
               FROM merkle_batches WHERE first_idx <= ? AND last_idx >= ?""",
            (index, index),
        )
        row = cur.fetchone()
        if not row:
            return None
        batch_id, first, last, root_hex = row
        leaves = [bytes.fromhex(r.record_hash) for r in self.iter_records(first, last + 1)]
        target = index - first
        proof = merkle_proof(leaves, target)
        rec = self.get(index)
        assert rec is not None
        return {
            "batch_id": batch_id,
            "record_hash": rec.record_hash,
            "merkle_root": root_hex,
            "proof": [(side, sib.hex()) for side, sib in proof],
        }

    def attach_anchor(self, batch_id: int, anchor_proof: dict) -> None:
        """Record an external anchor (e.g. OpenTimestamps receipt) for a batch."""
        self._conn.execute(
            "UPDATE merkle_batches SET anchor_proof = ?, anchor_status = 'anchored' WHERE batch_id = ?",
            (json.dumps(anchor_proof), batch_id),
        )
        self._conn.commit()

    def list_batches(self) -> list[dict]:
        """
        Return every sealed Merkle batch in commit order.

        Each row is a dict with keys: batch_id, first_idx, last_idx,
        root_hash, created_at, anchor_status. `view_chain.py` and any
        other inspection tool should use this rather than reaching into
        `self._conn` — the SQLite schema is an implementation detail.
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT batch_id, first_idx, last_idx, root_hash, created_at, "
            "anchor_status FROM merkle_batches ORDER BY batch_id"
        )
        return [
            {
                "batch_id": row[0],
                "first_idx": row[1],
                "last_idx": row[2],
                "root_hash": row[3],
                "created_at": row[4],
                "anchor_status": row[5],
            }
            for row in cur.fetchall()
        ]

    def close(self) -> None:
        self._conn.close()

    # Context-manager support so callers can do
    #     with Chain(path, key) as chain: ...
    # and not have to write the manual try/finally close dance. Errors
    # inside the `with` block propagate normally — close() is best-effort.
    def __enter__(self) -> "Chain":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_supersedes(content: Any) -> Optional[int]:
    """
    Read the supersedes pointer from a revision record's content. The
    canonical location since v2 metadata is `content["_meta"]["supersedes"]`;
    v1 records (pre-_meta) carry the same pointer at the top level as
    `revises_index`. Returns None if no valid pointer is present.

    Kept module-level (not on Chain) and intentionally light on
    dependencies so chain.py stays metadata-agnostic — it knows nothing
    about the broader _meta schema, just where the supersedes integer
    lives.
    """
    if not isinstance(content, dict):
        return None
    meta = content.get("_meta")
    if isinstance(meta, dict):
        v = meta.get("supersedes")
        if isinstance(v, int):
            return v
    # Legacy v1 fallback.
    v = content.get("revises_index")
    if isinstance(v, int):
        return v
    return None


def _row_to_record(row: tuple) -> Record:
    return Record(
        index=row[0],
        prior_hash=row[1],
        timestamp=row[2],
        type=row[3],
        content=json.loads(row[4]),
        refs=json.loads(row[5]),
        pubkey=row[6],
        content_hash=row[7],
        record_hash=row[8],
        signature=row[9],
    )


def load_or_create_key(path: str | Path) -> _SigningKey:
    """Load an Ed25519 signing key from disk, or generate and save one."""
    path = Path(path)
    if path.exists():
        return _SigningKey(path.read_bytes())
    key = _SigningKey.generate()
    path.write_bytes(key.encode())
    path.chmod(0o600)
    return key


def verify_inclusion(record_hash: str, proof: list[tuple[str, str]], root_hash: str) -> bool:
    """Standalone verification — only needs the proof and the (anchored) root."""
    return verify_merkle_proof(
        bytes.fromhex(record_hash),
        [(side, bytes.fromhex(sib)) for side, sib in proof],
        bytes.fromhex(root_hash),
    )
