#!/usr/bin/env python3
"""
Retrieval latency tripwire.

Measures, on a synthetic chain of N single-chunk records (HashingEmbedder,
dim 256), the two per-turn retrieval costs:

  per-turn  — index one new record + search (the every-turn write path,
              incremental ANN append since v1.4.2)
  warm      — a search against an already-built in-memory matrix
              (brute-force cosine over all chunks, O(N * dim))

THE TRIPWIRE: rerun this occasionally at your real chain size
(`SELECT COUNT(*) FROM embeddings` in timechain_data/embeddings.sqlite,
or just pass a size). When WARM search crosses ~200 ms at that size,
that is the measured signal to build a pre-filter shortlist (FTS5 or a
real ANN) in front of brute-force cosine — and not before: every
shortlist trades recall quality for speed, so we buy it only when the
measurement says so.

Usage: python3 bench_retrieval.py [size ...]      (default: 1000 10000 50000)
"""
import os
import sys
import tempfile
import time

import numpy as np

from chain import Record
from retrieval import EmbeddingIndex, HashingEmbedder

DIM = 256
TRIPWIRE_MS = 200.0


def fake_record(i: int) -> Record:
    return Record(index=i, prior_hash="", timestamp=0, type="note",
                  content=f"synthetic record {i} about topic {i % 97}",
                  refs=[], pubkey="", content_hash="", record_hash=f"h{i}",
                  signature="")


def bench(n: int) -> tuple[float, float]:
    """Return (per_turn_seconds, warm_search_seconds) at chain size n."""
    with tempfile.TemporaryDirectory() as td:
        idx = EmbeddingIndex(os.path.join(td, "bench.sqlite"),
                             HashingEmbedder(DIM), DIM)
        # Bulk-load n-1 plausible rows straight into SQLite — setup speed
        # is not what we measure, only the search/index paths above it.
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((n - 1, DIM)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        idx._conn.executemany(
            "INSERT INTO embeddings (record_idx, chunk_index, chunk_count, "
            "record_hash, vector, text) VALUES (?, 0, 1, ?, ?, ?)",
            ((i, f"h{i}", v.tobytes(), f"synthetic {i}")
             for i, v in enumerate(vecs)))
        idx._conn.commit()
        idx.search("warm-up: build the matrix once")  # cold build, untimed

        t0 = time.perf_counter()
        idx.index_record(fake_record(n - 1))
        idx.search("what did we decide about the index")
        per_turn = time.perf_counter() - t0

        t0 = time.perf_counter()
        idx.search("warm search latency probe")
        warm = time.perf_counter() - t0
        idx.close()
        return per_turn, warm


def main() -> int:
    sizes = [int(a) for a in sys.argv[1:]] or [1_000, 10_000, 50_000]
    print(f"{'chain size':>12}  {'per-turn (write+search)':>24}  {'warm search':>12}")
    tripped = False
    for n in sizes:
        per_turn, warm = bench(n)
        mark = "  <-- TRIPWIRE" if warm * 1000 >= TRIPWIRE_MS else ""
        tripped = tripped or bool(mark)
        print(f"{n:>12,}  {per_turn * 1000:>21.1f} ms  {warm * 1000:>9.1f} ms{mark}")
    if tripped:
        print(f"\nWarm search crossed {TRIPWIRE_MS:.0f} ms: time to build the "
              f"shortlist pre-filter (see module docstring).")
    else:
        print(f"\nAll warm searches under {TRIPWIRE_MS:.0f} ms — "
              f"brute-force cosine is still the right answer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
