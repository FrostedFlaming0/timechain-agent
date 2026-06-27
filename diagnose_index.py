"""
Standalone indexing diagnostic. Run this instead of webapp.py to find
where boot hangs. It does exactly what boot's index step does, but prints
every record as it processes it, so a hang is pinned to one record.

    python diagnose_index.py

Fidelity note: this builds the per-record text with
`EmbeddingIndex.record_to_text` and then chunks it with `chunk_text` —
the SAME path the real index step (`EmbeddingIndex.index_record`) uses.
An earlier version embedded an ad-hoc f-string instead, which diverged
from the real path in two ways: it skipped the `file`-record
special-casing (filename + extracted text rather than the raw blob hash)
and it used `str(rec.content)` rather than `json.dumps(...)`. That meant
the diagnostic could pass on a record that real indexing would choke on,
or vice versa. Using the production functions keeps this diagnostic
honest — in particular, since indexing now chunks before embedding, a
record whose whole text would overflow the embedder cap embeds fine in
chunks, and this diagnostic must reflect that rather than embedding the
oversized whole.
"""
import sys, time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from chain import Chain, load_or_create_key
from retrieval import EmbeddingIndex, chunk_text, CHUNK_TARGET_CHARS
from run import DATA_DIR, make_tiered_embedder

print("opening chain...")
chain = Chain(DATA_DIR / "chain.sqlite", load_or_create_key(DATA_DIR / "operator.key"))
print(f"chain length: {chain.length()}")

print("resolving embedder...")
embedder, dim, name = make_tiered_embedder()
print(f"embedder: {name} ({dim}-dim)")

print("embedding every record one by one (chunked, as the real path does):\n")
for rec in chain.iter_records():
    # Build text and chunk it exactly as index_record does, so a failure
    # here corresponds to a real boot-time indexing failure. File records
    # carry the header on every chunk; mirror that here.
    text = EmbeddingIndex.record_to_text(rec)
    if rec.type == "file" and isinstance(rec.content, dict):
        filename = rec.content.get("filename", "")
        kind = rec.content.get("kind", "")
        header = f"[file {filename} {kind}] "
        body = rec.content.get("extracted_text", "")
        chunks = [header + p
                  for p in chunk_text(body, target=CHUNK_TARGET_CHARS - len(header))]
    else:
        chunks = chunk_text(text, target=CHUNK_TARGET_CHARS)
    print(f"  record {rec.index:>4} [{rec.type:<20}] "
          f"{len(text):>7} chars -> {len(chunks):>3} chunk(s) ... ",
          end="", flush=True)
    t0 = time.time()
    try:
        for ci, chunk in enumerate(chunks):
            vec = embedder(chunk)
        dt = time.time() - t0
        print(f"OK ({dt:.1f}s)")
    except Exception as e:
        dt = time.time() - t0
        print(f"FAILED on chunk {ci} after {dt:.1f}s: {type(e).__name__}: {e}")

print("\ndone — every record processed.")
chain.close()
