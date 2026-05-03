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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

from chain import Chain, Record


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


class Retriever:
    def __init__(self, chain: Chain, index: EmbeddingIndex):
        self.chain = chain
        self.index = index

    # ----- hybrid: semantic + structural -----

    # ----- hybrid: semantic + structural + salience -----

    # Default salience weights by record type. Reflections represent the
    # agent's own judgment about what mattered, so they get a strong boost.
    # Revisions matter because they correct prior records. System prompts and
    # genesis are foundational identity records — boosted modestly so they
    # surface when relevant. Observations and responses are the baseline.
    DEFAULT_SALIENCE = {
        "reflection": 0.20,
        "revision": 0.15,
        "genesis": 0.10,
        "system_prompt": 0.05,
        "observation": 0.0,
        "response": 0.0,
    }

    def hybrid(
        self,
        query: str,
        k: int = 10,
        type_filter: Optional[str] = None,
        recency_weight: float = 0.0,
        salience_weights: Optional[dict] = None,
    ) -> list[RetrievalHit]:
        """
        Semantic search with optional type filter, recency boost, and
        per-type salience boost. Salience lets reflection and revision
        records surface preferentially even when they're not the closest
        semantic match — modeling "the agent's own sense of what mattered."
        """
        salience = salience_weights if salience_weights is not None else self.DEFAULT_SALIENCE
        candidates = self.index.search(query, k=max(k * 4, 20))
        head_idx = self.chain.length() - 1
        hits: list[RetrievalHit] = []
        for rec_idx, sim in candidates:
            rec = self.chain.get(rec_idx)
            if rec is None:
                continue
            if type_filter and rec.type != type_filter:
                continue
            recency = 1.0 - (head_idx - rec_idx) / max(head_idx + 1, 1) if head_idx >= 0 else 0.0
            sal = salience.get(rec.type, 0.0)
            score = sim + recency_weight * recency + sal
            hits.append(RetrievalHit(rec, score, "semantic"))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    # ----- ancestry walk -----

    def ancestry(self, record_hash: str, depth: int = 3) -> list[RetrievalHit]:
        recs = self.chain.follow_refs(record_hash, depth=depth)
        return [RetrievalHit(r, 1.0, "ancestry") for r in recs]

    # ----- temporal window -----

    def recent(self, n: int = 20, type_filter: Optional[str] = None) -> list[RetrievalHit]:
        if type_filter:
            recs = self.chain.query_by_type(type_filter, limit=n)
        else:
            recs = self.chain.query_recent(limit=n)
        return [RetrievalHit(r, 1.0, "recent") for r in recs]

    # ----- combined context for LLM -----

    def build_context(
        self,
        query: str,
        k_semantic: int = 5,
        n_recent: int = 5,
        type_filter: Optional[str] = None,
    ) -> list[Record]:
        """
        Practical default: blend semantic hits with recent records, dedup,
        return chronological order suitable for an LLM context window.
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
