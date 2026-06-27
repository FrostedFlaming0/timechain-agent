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
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

from chain import Chain, Record
from metadata import read_meta, half_life_days


Embedder = Callable[[str], np.ndarray]


# ---------------------------------------------------------------------------
# Embedding store
# ---------------------------------------------------------------------------

EMBED_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    record_idx  INTEGER PRIMARY KEY,
    record_hash TEXT NOT NULL,
    vector      BLOB NOT NULL,
    text        TEXT NOT NULL
);
"""


class EmbeddingIndex:
    """
    SQLite-backed embedding store with in-memory ANN over current vectors.
    Rebuilds the ANN index on demand — fine for prototypes up to ~100k records.
    """

    def __init__(self, db_path: str | Path, embedder: Embedder, dim: int):
        self.db_path = str(db_path)
        self.embedder = embedder
        self.dim = dim
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(EMBED_SCHEMA)
        self._conn.commit()
        self._nn: Optional[NearestNeighbors] = None
        self._idx_to_record: list[int] = []

    @staticmethod
    def record_to_text(rec: Record) -> str:
        """Flatten a record into text for embedding. Override for fancier schemes."""
        # File records: embed by filename + extracted text. Avoids embedding
        # the giant blob_sha256 hex string which would dominate the vector.
        if rec.type == "file" and isinstance(rec.content, dict):
            filename = rec.content.get("filename", "")
            kind = rec.content.get("kind", "")
            text = rec.content.get("extracted_text", "")
            return f"[file {filename} {kind}] {text}"
        try:
            content_str = json.dumps(rec.content, ensure_ascii=False)
        except (TypeError, ValueError):
            content_str = str(rec.content)
        return f"[{rec.type}] {content_str}"

    def index_record(self, rec: Record) -> None:
        text = self.record_to_text(rec)
        vec = self.embedder(text).astype(np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"embedder returned shape {vec.shape}, expected ({self.dim},)")
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (record_idx, record_hash, vector, text) VALUES (?, ?, ?, ?)",
            (rec.index, rec.record_hash, vec.tobytes(), text),
        )
        self._conn.commit()
        self._nn = None  # invalidate

    def index_chain(self, chain: Chain) -> int:
        """Index every record not yet embedded. Returns count added."""
        cur = self._conn.cursor()
        cur.execute("SELECT record_idx FROM embeddings")
        existing = {r[0] for r in cur.fetchall()}
        added = 0
        for rec in chain.iter_records():
            if rec.index not in existing:
                self.index_record(rec)
                added += 1
        return added

    def _rebuild_ann(self) -> None:
        cur = self._conn.cursor()
        cur.execute("SELECT record_idx, vector FROM embeddings ORDER BY record_idx ASC")
        rows = cur.fetchall()
        if not rows:
            self._nn = None
            self._idx_to_record = []
            return
        self._idx_to_record = [r[0] for r in rows]
        matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        # Normalize for cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        self._nn = NearestNeighbors(n_neighbors=min(50, len(rows)), metric="cosine")
        self._nn.fit(matrix)
        self._matrix = matrix

    def search(self, query_text: str, k: int = 10) -> list[tuple[int, float]]:
        """Return [(record_idx, similarity_score), ...] sorted by similarity desc."""
        if self._nn is None:
            self._rebuild_ann()
        if self._nn is None:
            return []
        qv = self.embedder(query_text).astype(np.float32)
        qn = qv / (np.linalg.norm(qv) or 1.0)
        k = min(k, len(self._idx_to_record))
        distances, indices = self._nn.kneighbors(qn.reshape(1, -1), n_neighbors=k)
        return [
            (self._idx_to_record[i], 1.0 - float(d))
            for d, i in zip(distances[0], indices[0])
        ]

    def close(self) -> None:
        self._conn.close()


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

    def __init__(self, chain: Chain, index: EmbeddingIndex):
        self.chain = chain
        self.index = index

    # ----- score weights -----

    # Weights for the hybrid score. Tunable; named so the impact of each
    # term is visible in `RetrievalHit.components`. They sum to 1.0 to keep
    # raw scores in roughly [0, 1.5] range (semantic + boosts).
    W_SEMANTIC = 0.55
    W_SALIENCE = 0.25
    W_RECENCY  = 0.20

    # Penalty subtracted from a hit's score when a later revision
    # supersedes it. Applied AFTER the weighted sum, so it can push a
    # superseded record below an otherwise-weaker correction. Set high
    # enough to flip ordering when both are retrieved together, but not
    # so high the original drops out of context entirely.
    SUPERSEDED_PENALTY = 0.30

    # ----- helpers -----

    def _superseded_indices(self) -> set[int]:
        """
        Indices of records that have been superseded by a later revision.
        Computed by scanning revision records and reading their `supersedes`
        pointer (from _meta) or the legacy `revises_index` field.
        Cached per-call; recomputed each time hybrid()/build_context() runs.
        """
        revisions = self.chain.query_by_type("revision", limit=1000)
        out: set[int] = set()
        for rev in revisions:
            meta = read_meta(rev)
            if meta.supersedes is not None:
                out.add(meta.supersedes)
        return out

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
    ) -> list[RetrievalHit]:
        """
        Semantic search plus per-record salience, per-kind recency decay,
        and revision-aware demotion.

        Score formula:
            base = W_SEMANTIC*similarity + W_SALIENCE*salience + W_RECENCY*recency
            score = base - (SUPERSEDED_PENALTY if record is superseded else 0)

        - similarity: cosine sim from the embedding index, in [0, 1].
        - salience:   from the record's _meta block, in [0, 1].
        - recency:    0.5 ** (age_days / half_life_days_for_type), in [0, 1].

        Back-compat: `recency_weight` and `salience_weights` are still
        accepted but their interpretation has shifted —
          - recency_weight: if set, overrides the default W_RECENCY.
          - salience_weights: now ignored. Salience is per-record from _meta,
            with type defaults from metadata.py for v1 records. Pass-through
            kept so old callers don't crash.
        """
        w_recency = self.W_RECENCY if recency_weight is None else float(recency_weight)
        candidates = self.index.search(query, k=max(k * 4, 20))
        if not candidates:
            return []
        superseded = self._superseded_indices()
        now_seconds = time.time()
        hits: list[RetrievalHit] = []
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
            base = (
                self.W_SEMANTIC * sim
                + self.W_SALIENCE * meta.salience
                + w_recency * recency
            )
            penalty = self.SUPERSEDED_PENALTY if rec.index in superseded else 0.0
            score = base - penalty
            components = {
                "semantic": float(sim),
                "salience": float(meta.salience),
                "recency": float(recency),
                "superseded_penalty": float(penalty),
                "source": meta.source,
                "confidence": float(meta.confidence),
            }
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
        n_recent: int = 5,
        type_filter: Optional[str] = None,
    ) -> list[Record]:
        """
        Blend semantic hits with recent records, dedup, return in
        chronological order suitable for an LLM context window.

        Revision pull-in: when a record in the result set has been
        superseded by a revision, that revision is automatically pulled
        in too (if not already present). The model needs to see both
        the original claim and its correction together — that's the
        point of keeping both around.
        """
        semantic = self.hybrid(query, k=k_semantic, type_filter=type_filter)
        recent = self.recent(n=n_recent, type_filter=type_filter)
        seen: set[int] = set()
        merged: list[Record] = []
        for hit in semantic + recent:
            if hit.record.index in seen:
                continue
            seen.add(hit.record.index)
            merged.append(hit.record)

        # Pull in revisions that supersede anything we've retrieved.
        # Without this, the model can see a stale claim and miss the
        # correction sitting on the chain.
        merged_indices = {r.index for r in merged}
        revision_pull_ins: list[Record] = []
        all_revisions = self.chain.query_by_type("revision", limit=1000)
        for rev in all_revisions:
            meta = read_meta(rev)
            if meta.supersedes in merged_indices and rev.index not in merged_indices:
                revision_pull_ins.append(rev)
                merged_indices.add(rev.index)
        merged.extend(revision_pull_ins)

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
# Trivial deterministic embedder for the demo (no model dependency)
# ---------------------------------------------------------------------------

class HashingEmbedder:
    """
    Bag-of-character-trigrams hashed into a fixed-dim vector.
    Deterministic, dependency-free, and good enough to demonstrate retrieval
    behavior. Replace with a real sentence-transformer for production.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def __call__(self, text: str) -> np.ndarray:
        text = text.lower()
        vec = np.zeros(self.dim, dtype=np.float32)
        if len(text) < 3:
            text = text + "   "
        for i in range(len(text) - 2):
            tri = text[i : i + 3]
            h = hash(tri) % self.dim
            vec[h] += 1.0
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        return vec
