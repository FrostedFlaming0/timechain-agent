"""
migrate — backfill derived indexes for historic chains (the historic-chain
compatibility & migration fix: re-embed old records off-chain).

The signed chain is append-only and must stay backward-compatible (old records
keep verifying). The legitimate worry is that records sealed before the current
embedder — or before this richer build — retrieve *thin*. That is a
RETRIEVAL-LAYER problem, solved here with a one-time, idempotent, OFF-CHAIN
backfill: re-embed every record into the embedding index with the CURRENT
embedder. It touches only the derived `embeddings.sqlite` (rebuild-on-delete),
never the signed records — so `chain.verify()` is unaffected and it is safe to
re-run.

This keeps the two-layer split the plan prescribes: the signed chain stays
compatible; the derived index gets backfilled, so old and new records retrieve
with comparable richness.
"""

from __future__ import annotations

from typing import Callable, Optional


def reindex_stream(chain, index, every: int = 5):
    """Re-embed every record, yielding progress dicts so callers can show that
    a long backfill is moving (not frozen). Yields:

        {phase:'start',    done:0, total, reindexed:0, failed:0}
        {phase:'progress', done, total, reindexed, failed}   every `every` records
        {phase:'done',     done, total, reindexed, failed}

    The embedding work runs synchronously between yields (the embedding store's
    SQLite connection is thread-affine), so the caller drives this on the
    connection's own thread. Idempotent: `index.index_record` upserts."""
    total = chain.length()
    yield {"phase": "start", "done": 0, "total": total, "reindexed": 0, "failed": 0}
    done = reindexed = failed = 0
    for rec in chain.iter_records():
        done += 1
        try:
            index.index_record(rec)
            reindexed += 1
        except Exception:
            failed += 1
        if every and done % every == 0 and done != total:
            yield {"phase": "progress", "done": done, "total": total,
                   "reindexed": reindexed, "failed": failed}
    yield {"phase": "done", "done": done, "total": total,
           "reindexed": reindexed, "failed": failed}


def reindex(chain, index, on_progress: Optional[Callable] = None) -> dict:
    """Re-embed every record into `index` with its current embedder. Returns
    final counts. Idempotent: a second run is a no-op beyond recomputation.
    Thin wrapper over `reindex_stream` (the single source of truth)."""
    final = {"total": chain.length(), "reindexed": 0, "failed": 0}
    for ev in reindex_stream(chain, index):
        if ev["phase"] == "done":
            final = {"total": ev["total"], "reindexed": ev["reindexed"],
                     "failed": ev["failed"]}
        elif ev["phase"] == "progress" and on_progress is not None:
            on_progress(ev["done"])
    return final
