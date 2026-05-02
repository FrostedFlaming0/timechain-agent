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
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ------- writing -------

    def append(self, type_: str, content: Any, refs: Optional[Iterable[str]] = None) -> Record:
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
        content_hash = sha256_hex(canonical_json(content))

        signing_payload = {
            "index": index,
            "prior_hash": prior_hash,
            "timestamp": timestamp,
            "type": type_,
            "content": content,
            "refs": refs,
            "pubkey": self.pubkey_hex,
            "content_hash": content_hash,
        }
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

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
