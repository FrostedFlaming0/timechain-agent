"""
cypher_commands — REPL slash-command dispatcher for the cypher-tempre port.

Keeps run.py minimal: one `dispatch(...)` hook handles every ported command
(`/verify-source`, `/poq`, `/immune-*`, `/continuum-*`, `/recall-*`, `/think`,
`/consensus-*`, `/cambium-grow`). Each command is a thin wrapper over a tested
module; `dispatch` returns True if it handled the input (so run.py `continue`s),
False otherwise (so run.py falls through to a normal turn).

Errors are caught and printed rather than crashing the REPL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


HELP = (
    "  /verify-source <idx> [repo]   re-check an ingested file record vs disk\n"
    "  /poq <text>                   show the Proof-of-Quality verdict for text\n"
    "  /immune-status                immune membrane status (locked, scars)\n"
    "  /immune-scan [text]           scan sealed memory (+ optional input) for compromise\n"
    "  /lockdown                     freeze sealing until recovery\n"
    "  /rollback <height>            roll back to the clean height before a wound\n"
    "  /recall-index                 print the compact map of memory\n"
    "  /recall-fetch <ids...>        fetch full content for chosen record ids\n"
    "  /recall <query>               cheap pre-filter (delegates to the Retriever)\n"
    "  /migrate                      re-embed all records (historic-chain backfill)\n"
    "  /think <query>                fork self-perspectives (MCTS), collapse (no seal)\n"
    "  /consensus-init [n] [k]       create a k-of-n witness quorum\n"
    "  /consensus-verify             verify chain + quorum attestation\n"
    "  /cambium-grow <text>          grow a faculty if the input reveals a gap\n"
    "  /continuum-resume             re-hydrate the latest continuum task state\n"
    "  /continuum-validate           check continuum invariants + chain integrity"
)


def _faculty_dir(repo_root: Optional[Path]) -> Path:
    base = Path(repo_root) if repo_root else Path(__file__).resolve().parent
    return base / "faculties"


def dispatch(user_input: str, chain, agent=None, repo_root: Optional[Path] = None) -> bool:
    """Handle a cypher-tempre slash command. Returns True if handled."""
    parts = user_input.split()
    if not parts:
        return False
    cmd = parts[0]
    args = parts[1:]

    try:
        if cmd == "/cypher-help":
            print(HELP)

        elif cmd == "/verify-source":
            if not args:
                print("  usage: /verify-source <record_index> [repo_path]")
                return True
            import source_verify
            repo = args[1] if len(args) > 1 else None
            v = source_verify.verify_file_record(chain, int(args[0]), repo)
            print(f"  verdict: {v.get('verdict')}")
            for k in ("source_path", "stored_sha256", "live_sha256",
                      "stored_git_commit", "live_git_commit"):
                if v.get(k) is not None:
                    print(f"    {k}: {v[k]}")

        elif cmd == "/poq":
            text = user_input.split(maxsplit=1)[1] if len(args) else ""
            if not text:
                print("  usage: /poq <text>")
                return True
            import poq as _poq
            ev = agent.poq if (agent is not None and getattr(agent, "poq", None)) else _poq.PoQEvaluator()
            cov = agent.covenant() if agent is not None else None
            r = ev.evaluate("(manual /poq probe)", text, covenant=cov)
            print(f"  brightness: {r.brightness:.3f}   verdict: {r.verdict}")
            print(f"  grounding: {r.grounding:.2f}   assertiveness: {r.assertiveness:.2f}")
            print("  dimensions: " + ", ".join(f"{k}={v:.2f}" for k, v in r.dimensions.items()))
            for n in r.notes:
                print(f"    - {n}")

        elif cmd in ("/immune-status", "/immune-scan", "/lockdown", "/rollback"):
            import immune as _immune
            im = getattr(agent, "immune", None) if agent is not None else None
            if im is None:
                im = _immune.Immune(chain)
            if cmd == "/immune-status":
                st = im.status()
                print(f"  locked: {st['locked']}   safe_height: {st['safe_height']}"
                      f"   active_head: {st['active_head']}")
                print(f"  quarantined: {st['quarantined']}   scars: {len(st['scars'])}")
            elif cmd == "/immune-scan":
                inp = user_input.split(maxsplit=1)[1] if len(args) else None
                d = im.scan(input_text=inp)
                print("  COMPROMISE DETECTED" if d["compromised"] else "  clean — no compromise")
                for s in d["signals"]:
                    print(f"    ! {s}")
                if d["first_bad_height"] is not None:
                    print(f"    -> first bad height: {d['first_bad_height']}")
            elif cmd == "/lockdown":
                im.lockdown()
                print("  IMMUNE LOCKDOWN engaged — only 'recovery' may be sealed.")
            elif cmd == "/rollback":
                if not args:
                    print("  usage: /rollback <first_bad_height>")
                    return True
                r = im.rollback(int(args[0]), grow_antibody=True,
                                faculty_dir=_faculty_dir(repo_root))
                print(f"  rolled back to clean height {r['safe_height']}; "
                      f"quarantined {r['quarantined']}; scar {r['scar']['id']} learned; "
                      f"recovery sealed as Ring {r['recovery_ring']}.")
                if r.get("antibody"):
                    ab = r["antibody"]
                    print(f"  antibody grown from scar: {ab['name']} "
                          f"[{ab['action']}] (dissonance {ab['gap_dissonance']})")

        elif cmd in ("/recall-index", "/recall-fetch", "/recall"):
            import recall as _recall
            retriever = getattr(agent, "retriever", None) if agent is not None else None
            rc = _recall.Recall(chain, retriever=retriever)
            if cmd == "/recall-index":
                for e in rc.index(limit=40):
                    kws = ",".join(e["keywords"][:4])
                    print(f"  [{e['index']:>4}] {e['type']:<12} {e['summary'][:60]}  ({kws})")
            elif cmd == "/recall-fetch":
                if not args:
                    print("  usage: /recall-fetch <id> [id ...]")
                    return True
                for blk in rc.fetch([int(a) for a in args]):
                    mark = " …[truncated]" if blk.get("truncated") else ""
                    print(f"  [{blk['index']}] {blk['type']}: {blk['content'][:200]}{mark}")
            else:  # /recall <query> — cheap pre-filter (delegates to Retriever)
                q = user_input.split(maxsplit=1)[1] if len(args) else ""
                if not q:
                    print("  usage: /recall <query>")
                    return True
                for b in rc.retrieve(q):
                    print(f"  [{b['index']:>4}] {b['type']:<12} {b['excerpt']}")
                print("  (pre-filter only — YOU judge relevance, then /recall-fetch)")

        elif cmd == "/think":
            q = user_input.split(maxsplit=1)[1] if len(args) else ""
            if not q:
                print("  usage: /think <query>")
                return True
            import chronosynaptic as _chrono
            tree = _chrono.ChronosynapticTree(chain, iterations=8, forks=3, max_depth=2)
            root = tree.search(q)
            res, _ = tree.collapse_and_seal(root, q, do_seal=False)
            if not res:
                print("  (no perspectives to collapse)")
                return True
            print(f"  forked {len(root.children)} self-perspectives (no subagents):")
            for f in res["forks"]:
                print(f"    [{f['kind'][0].upper()}] {f['perspective']:<28} "
                      f"N={f['visits']:>2} v={f['value']}")
            print(f"  -> collapsed path: {' -> '.join(p['name'] for p in res['leaf'].path)}")
            print("  (not sealed — /think is read-only; collapse-notes seals)")

        elif cmd in ("/consensus-init", "/consensus-verify"):
            import consensus as _consensus
            q = _consensus.Quorum(chain)
            if cmd == "/consensus-init":
                n = int(args[0]) if len(args) > 0 else 3
                k = int(args[1]) if len(args) > 1 else 2
                cfg = q.init(n=n, quorum=k)
                print(f"  quorum initialized: {cfg['n']} witnesses, quorum {cfg['quorum']}")
                print("  (single host = authenticated quorum; distribute witnesses for BFT)")
            else:
                ok, report = q.verify()
                for line in report[-8:]:
                    print("  " + line)
                print("  CONSENSUS:", "VALID" if ok else "BROKEN")

        elif cmd == "/cambium-grow":
            text = user_input.split(maxsplit=1)[1] if len(args) else ""
            if not text:
                print("  usage: /cambium-grow <text>")
                return True
            import faculties as _faculties
            garden = _faculties.FacultyGarden(chain, _faculty_dir(repo_root))
            result, rec = garden.grow(text)
            gap = result["gap"]
            print(f"  dissonance: {gap['dissonance']} (coverage {gap['coverage_ratio']})")
            if not result["grew"]:
                print(f"  -> {result.get('reason', 'no growth')}")
            else:
                fac = result["faculty"]
                print(f"  -> {result['action'].upper()}: {fac['name']} ({fac['kind']})")
                if rec is not None:
                    print(f"     sealed {rec.type} at index {rec.index}")

        elif cmd == "/migrate":
            idx = getattr(getattr(agent, "retriever", None), "index", None)
            if idx is None:
                print("  /migrate needs the agent's embedding index.")
                return True
            import migrate as _migrate
            # Stream progress so a long backfill is visibly moving, not frozen.
            for ev in _migrate.reindex_stream(chain, idx):
                if ev["phase"] == "start":
                    print(f"  re-embedding {ev['total']} record(s) into the index…")
                elif ev["phase"] == "progress":
                    print(f"    … {ev['done']}/{ev['total']} ({ev['reindexed']} embedded)")
                else:
                    print(f"  done: reindexed {ev['reindexed']}/{ev['total']}"
                          + (f" ({ev['failed']} failed)" if ev["failed"] else ""))
            print("  (off-chain backfill — signed records untouched; safe to re-run)")

        elif cmd in ("/continuum-resume", "/continuum-validate"):
            import continuum as _continuum
            c = _continuum.Continuum(chain)
            if cmd == "/continuum-resume":
                st = c.resume()
                if not st:
                    print("  (no continuum task on this chain)")
                else:
                    m = st["metrics"]
                    print(f"  objective: {st['objective']}")
                    print(f"  progress: {m['items_done']} items, {m['chunks_sealed']} blocks")
                    print(f"  next action: {st['next_action']}")
            else:
                ok, report = c.validate()
                for line in report[-4:]:
                    print("  " + line)
                print("  CONTINUUM:", "COHERENT" if ok else "INCOHERENT")

        else:
            return False
    except Exception as e:
        print(f"  {cmd} error: {type(e).__name__}: {e}")
    return True
