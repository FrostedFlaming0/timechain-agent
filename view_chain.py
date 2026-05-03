"""
view_chain — inspect a timechain from the command line.

Usage:
    python view_chain.py                  # summary + last 10 records
    python view_chain.py --all            # every record
    python view_chain.py --tail 20        # last 20 records
    python view_chain.py --range 5 15     # records 5-14
    python view_chain.py --record 7       # full detail of one record
    python view_chain.py --type response  # only records of a given type
    python view_chain.py --batches        # show Merkle batches
    python view_chain.py --verify         # cryptographic walk + verify

Reads the chain pointed to by DATA_DIR (matches run.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from chain import Chain, load_or_create_key


# ---- Match this to your run.py setting ----
DATA_DIR = Path(__file__).parent / "timechain_data"
# If your run.py uses a different DATA_DIR, change the line above to match.


def fmt_time(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def short(s: str, n: int = 12) -> str:
    return s[:n] + "…" if len(s) > n else s


def fmt_content(content) -> str:
    """Render a record's content for one-line display."""
    try:
        if isinstance(content, dict):
            if "filename" in content and "blob_sha256" in content:
                # File record
                return (
                    f"file: {content.get('filename', '?')} "
                    f"({content.get('kind', '?')}, "
                    f"{content.get('size_bytes', 0):,} bytes)"
                )
            if "text" in content:
                t = content["text"]
                # For revisions, prefix with what's being corrected
                if "revises_index" in content:
                    return f"corrects #{content['revises_index']}: {t[:50]}{'...' if len(t) > 50 else ''}"
                return t if len(t) <= 70 else t[:67] + "..."
            if "commitments" in content:
                return f"genesis: {len(content['commitments'])} commitments"
        return json.dumps(content, ensure_ascii=False)[:70]
    except Exception:
        return str(content)[:70]


def print_summary(chain: Chain) -> None:
    n = chain.length()
    head = chain.head()
    print(f"chain length    : {n} records")
    if head:
        print(f"head index      : {head.index}")
        print(f"head hash       : {head.record_hash}")
        print(f"operator pubkey : {head.pubkey}")
        print(f"latest timestamp: {fmt_time(head.timestamp)}")


def print_table(chain: Chain, start: int, end: int) -> None:
    print(f"\n{'idx':>4}  {'time (UTC)':19}  {'type':12}  {'hash':14}  content")
    print("-" * 90)
    for rec in chain.iter_records(start=start, end=end):
        print(
            f"{rec.index:>4}  "
            f"{fmt_time(rec.timestamp):19}  "
            f"{rec.type:12.12}  "
            f"{short(rec.record_hash, 12):14}  "
            f"{fmt_content(rec.content)}"
        )


def print_record_detail(chain: Chain, idx: int) -> None:
    rec = chain.get(idx)
    if rec is None:
        print(f"no record at index {idx}")
        return
    print(f"index        : {rec.index}")
    print(f"type         : {rec.type}")
    print(f"timestamp    : {fmt_time(rec.timestamp)}  (raw: {rec.timestamp})")
    print(f"prior_hash   : {rec.prior_hash}")
    print(f"content_hash : {rec.content_hash}")
    print(f"record_hash  : {rec.record_hash}")
    print(f"signature    : {rec.signature[:32]}...")
    print(f"pubkey       : {rec.pubkey}")
    if rec.refs:
        print(f"refs         : {len(rec.refs)} reference(s)")
        for r in rec.refs:
            print(f"               {r}")
    print(f"content      :")
    print(json.dumps(rec.content, indent=2, ensure_ascii=False))


def print_batches(chain: Chain) -> None:
    cur = chain._conn.cursor()
    cur.execute(
        "SELECT batch_id, first_idx, last_idx, root_hash, created_at, anchor_status "
        "FROM merkle_batches ORDER BY batch_id"
    )
    rows = cur.fetchall()
    if not rows:
        print("no Merkle batches sealed yet (use /seal in run.py to create one)")
        return
    print(f"\n{'batch':>5}  {'records':>15}  {'created':19}  {'status':9}  root")
    print("-" * 100)
    for batch_id, first, last, root, created, status in rows:
        print(
            f"{batch_id:>5}  "
            f"{first}-{last:<10}  "
            f"{fmt_time(created):19}  "
            f"{(status or '?'):9}  "
            f"{short(root, 16)}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="View timechain contents")
    p.add_argument("--all", action="store_true", help="show every record")
    p.add_argument("--tail", type=int, metavar="N", help="show last N records")
    p.add_argument("--range", nargs=2, type=int, metavar=("START", "END"),
                   help="show records [START, END)")
    p.add_argument("--record", type=int, metavar="IDX", help="show full detail of one record")
    p.add_argument("--type", metavar="TYPE", help="filter by record type")
    p.add_argument("--batches", action="store_true", help="show Merkle batches")
    p.add_argument("--verify", action="store_true", help="cryptographically verify the entire chain")
    args = p.parse_args()

    chain_db = DATA_DIR / "chain.sqlite"
    key_path = DATA_DIR / "operator.key"

    if not chain_db.exists():
        sys.exit(f"no chain found at {chain_db}\n(run run.py first to create one)")
    if not key_path.exists():
        sys.exit(f"no operator key at {key_path}")

    key = load_or_create_key(key_path)
    chain = Chain(chain_db, key)

    try:
        print_summary(chain)

        if args.verify:
            print("\nverifying...")
            ok, msg = chain.verify(expected_pubkey=chain.pubkey_hex)
            print(f"verify: {ok}  {msg}")
            return

        if args.batches:
            print_batches(chain)
            return

        if args.record is not None:
            print()
            print_record_detail(chain, args.record)
            return

        if args.type:
            recs = chain.query_by_type(args.type, limit=1000)
            if not recs:
                print(f"\nno records of type {args.type!r}")
                return
            print(f"\n{len(recs)} record(s) of type {args.type!r}:\n")
            print(f"{'idx':>4}  {'time (UTC)':19}  {'hash':14}  content")
            print("-" * 90)
            for rec in sorted(recs, key=lambda r: r.index):
                print(
                    f"{rec.index:>4}  "
                    f"{fmt_time(rec.timestamp):19}  "
                    f"{short(rec.record_hash, 12):14}  "
                    f"{fmt_content(rec.content)}"
                )
            return

        # Default range selection
        n = chain.length()
        if args.all:
            start, end = 0, n
        elif args.range:
            start, end = args.range
        elif args.tail is not None:
            start, end = max(0, n - args.tail), n
        else:
            # default: last 10
            start, end = max(0, n - 10), n

        print(f"\nshowing records {start} to {end - 1}:")
        print_table(chain, start, end)
    finally:
        chain.close()


if __name__ == "__main__":
    main()
