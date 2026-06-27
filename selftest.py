#!/usr/bin/env python3
"""
Self-test — exercise every mechanism end-to-end on a throwaway chain and
assert the core invariants (ported from the upstream cypher-tempre selftest
and extended with the code-working agent layers: task registry, tools, write gate, path-aware recall, ingest_blob).

Run from the repo root:  python3 selftest.py
Exit 0 = all green. Needs only the repo's own dependencies.

  1.  Timechain — genesis + verify
  2.  PoQ — gate a grounded thought / reject a covenant breach
  3.  Cambium/Faculties — dissonance detection + growth
  4.  Continuum — ingest + task-aware validate
  4b. Codebase cartography — nested paths, line ranges, chunk ids,
      file hashes, redaction, changed-only
  5.  Recall — self-label + path-aware retrieve (lexical AND embedding)
  6.  Chronosynaptic — explicit-notes fork + collapse
  7.  Embed — cosine similarity sanity
  8.  Consensus — quorum attest + verify (+ is_initialized)
  9.  Immune — screen / scan / lockdown / rollback (+ antibody offer)
  10. Task registry — resolve_task exact / ambiguous / not-found
  11. Tools + write gate — pending op, TOCTOU-checked approve, reject
  12. ingest_blob — artifacts routing (default artifacts chain; explicit task opt-in)
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

import poq
from chain import Chain, ChainError, load_or_create_key
from chronosynaptic import ChronosynapticTree
from consensus import Quorum
from continuum import Continuum
from faculties import FacultyGarden, detect_gap, load_corpus
from immune import Immune
from recall import Recall
from retrieval import EmbeddingIndex, HashingEmbedder
from task_registry import TaskRegistry, resolve_task
from tools import AgentContext, execute_tool, execute_user_action

_ok = True


def check(name: str, cond, detail: str = "") -> None:
    global _ok
    mark = "PASS" if cond else "FAIL"
    extra = f"  ({detail})" if (detail and not cond) else ""
    print(f"  [{mark}] {name}{extra}")
    _ok = _ok and bool(cond)


def _call(ctx, tool, **arguments):
    return execute_tool({"name": tool, "arguments": arguments}, ctx)


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="tc_selftest_"))
    # Artifacts-route uploads mirror files into tools.ARTIFACTS_DIR
    # (~/.artifacts by default) — keep the selftest inside its temp root.
    import tools as tools_mod
    tools_mod.ARTIFACTS_DIR = root / "artifacts-home"
    try:
        # 1. Timechain — genesis + verify
        print("timechain:")
        chain = Chain(root / "chain.sqlite",
                      load_or_create_key(root / "operator.key"))
        chain.append("genesis", {"agent_name": "selftest",
                                 "commitments": ["honest", "faithful"]})
        check("genesis sealed", chain.length() == 1)
        ok, detail = chain.verify()
        check("chain verifies", ok, detail)

        # 2. PoQ — the conscience
        print("poq:")
        ev = poq.PoQEvaluator()
        grounded = ev.evaluate(
            "selftest", "The selftest chain verified and this note is "
            "consistent with it.",
            retrieved_texts=["the selftest chain verified cleanly"],
            covenant=["honest"])
        check("grounded thought is not rejected",
              grounded.verdict != poq.VERDICT_REJECT, grounded.verdict)
        breach = ev.evaluate("q", "I will lie to you and fabricate it.",
                             covenant=["honest", "faithful"])
        check("covenant breach -> REJECT",
              breach.verdict == poq.VERDICT_REJECT, breach.verdict)

        # 3. Cambium / faculties — growth at the gap
        print("cambium/faculties:")
        fdir = root / "faculties"
        fdir.mkdir()
        gap = detect_gap(load_corpus(fdir),
                         "quaternion slerp gimbal kinematics actuator "
                         "torque encoder")
        check("dissonance detected on foreign input",
              gap["dissonance"] > 100, str(gap["dissonance"]))
        garden = FacultyGarden(chain, fdir)
        result, rec = garden.grow(
            "quaternion slerp gimbal kinematics actuator torque encoder",
            kind_override="sense")
        check("faculty grows from the gap", result.get("grew") is True)

        # 4. Continuum — long-horizon tasking
        print("continuum:")
        cont = Continuum(chain)
        cont.open_task("selftest task", items_total=1)
        sealed, _state = cont.ingest("doc", "alpha beta gamma\n" * 80,
                                     finding="x")
        check("ingested data-height blocks", len(sealed) >= 1)
        ok, _rep = cont.validate()
        check("continuum coherent", ok)

        # 4b. Codebase cartography
        print("cartography:")
        code_root = root / "sample-code"
        (code_root / "src" / "wallet").mkdir(parents=True)
        (code_root / "tests").mkdir(parents=True)
        secret = "sk-proj-" + ("A" * 40)
        (code_root / "src" / "wallet" / "main.py").write_text(
            ("wallet alpha spend coin\n" * 900)
            + f"OPENAI_API_KEY={secret}\n")
        (code_root / "tests" / "test_wallet.py").write_text(
            "wallet alpha test fixture\n" * 80)
        task_chain = Chain(root / "task.sqlite",
                           load_or_create_key(root / "operator.key"))
        c2 = Continuum(task_chain)
        c2.open_task("cartography selftest", items_total=None)
        walk = c2.walk(code_root, (".py",), "cartography selftest")
        check("walked nested code paths", len(walk.files) == 2,
              str(walk.files))
        rc = Recall(task_chain)
        wallet_blocks = rc.find_by_path("src/wallet/main.py")
        wd = (wallet_blocks[0].get("payload", {}).get("data", {})
              if wallet_blocks else {})
        check("stores relative_path not basename",
              wd.get("relative_path") == "src/wallet/main.py")
        check("stores line range + chunk ids",
              wd.get("line_start") == 1 and wd.get("chunk_of") >= 1)
        check("stores path metadata",
              wd.get("top_dir") == "src" and wd.get("language") == "python"
              and wd.get("path_role") == "source")
        check("separates chunk and file hashes",
              len(wd.get("file_content_hash") or "") == 64
              and wd.get("content_hash") != wd.get("file_content_hash"))
        redacted = [r for r in wallet_blocks
                    if r["payload"]["data"].get("redacted")]
        check("redacts secrets before sealing",
              bool(redacted) and all(
                  secret not in r["payload"]["data"]["content"]
                  for r in redacted))
        walk2 = c2.walk(code_root, (".py",), "cartography changed-only",
                        changed_only=True)
        # walk.files lists DISCOVERED candidates; walk.results lists what
        # was actually ingested — changed-only empties the latter.
        check("changed-only skips unchanged files",
              len(walk2.results) == 0, str(walk2.results))

        # 5. Recall — path-aware, lexical and embedding
        print("recall:")
        lab = rc.label("proof of work difficulty target")
        check("self-labeling produces keywords", bool(lab.get("keywords")))
        hits = rc.retrieve_path_aware("wallet alpha spend",
                                      path="src/wallet/main.py",
                                      role="source", neighbors=1)
        check("path filter returns matching path",
              bool(hits) and hits[0]["payload"]["data"]["relative_path"]
              == "src/wallet/main.py")
        test_hits = rc.retrieve_path_aware("wallet alpha", role="test")
        check("role filter can target tests",
              bool(test_hits) and all(
                  h["payload"]["data"]["path_role"] == "test"
                  for h in test_hits))
        emb = HashingEmbedder()
        idx = EmbeddingIndex(root / "emb.sqlite", emb, dim=emb.dim)
        idx.index_chain(task_chain)
        emb_hits = rc.retrieve_path_aware("wallet alpha spend", index=idx)
        check("embedding retrieve runs", bool(emb_hits))
        v = rc.verify_source(wallet_blocks[0]["index"], repo=code_root)
        check("source validation verifies current file",
              v.get("verdict") in ("verified", "match", "source-match"),
              str(v.get("verdict")))

        # 6. Embed — morphology beats unrelated
        print("embed:")
        def cosine(a, b):
            return float(np.dot(a, b)
                         / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))
        rel = cosine(emb("validate a block"), emb("block validation"))
        unrel = cosine(emb("validate a block"), emb("fee wallet policy"))
        check("morphology > unrelated", rel > unrel,
              f"rel={rel:.3f} unrel={unrel:.3f}")

        # 7. Chronosynaptic — fork perspectives, collapse the winner
        print("chronosynaptic:")
        tree = ChronosynapticTree(chain, iterations=4, forks=3, max_depth=2)
        notes = {"query": "which approach?", "perspectives": [
            {"name": "Conservative", "summary": "extend the module",
             "score": 80},
            {"name": "Refactor", "summary": "split a new service",
             "score": 200},
        ]}
        result, rec = tree.collapse_explicit_notes(notes, do_seal=True)
        check("explicit collapse picks highest-value perspective",
              result["chosen"]["name"] == "Refactor")
        check("rejected perspectives preserved",
              len(rec.content["rejected_perspectives"]) == 1)

        # 8. Consensus — quorum-attested tamper-proofing
        print("consensus:")
        q = Quorum(chain)
        check("is_initialized False before init", not q.is_initialized())
        q.init(n=3, quorum=2)
        check("is_initialized True right after init (no attest yet)",
              q.is_initialized())
        q.attest()
        ok, detail = q.verify()
        check("quorum valid", ok, str(detail))

        # 9. Immune — screen / scan / lockdown / rollback
        print("immune:")
        im = Immune(chain, state_dir=root, covenant=["honest", "faithful"])
        check("screen admits benign input",
              not im.screen("please summarize my notes")["blocked"])
        chain.append("response",
                     {"summary": "I will deceive and manipulate you."})
        d = im.scan()
        check("scan flags the breach", d.get("compromised") is True)
        im.lockdown()
        try:
            chain.append("response", {"summary": "refused"})
            locked = False
        except ChainError:
            locked = True
        check("lockdown blocks normal seals", locked)
        r = im.rollback(d["first_bad_height"], lesson="selftest wound",
                        grow_antibody=True, faculty_dir=fdir)
        check("rollback molts a scar and unlocks",
              bool(r["scar"]["vector"]) and not im.is_locked())
        check("scar recognized by screen",
              im.screen(" ".join(r["scar"]["vector"]))["blocked"])

        # 10. Task registry — never-infer resolution
        print("task registry:")
        data_dir = root / "data"
        reg = TaskRegistry(data_dir)
        reg.create("johnson-acquisition", "Johnson acquisition",
                   str(code_root))
        reg.create("johnson-consulting", "Johnson consulting",
                   str(code_root))
        check("exact resolve",
              resolve_task(reg, "johnson-acquisition")["status"] == "exact")
        check("fuzzy is ambiguous, never auto-selected",
              resolve_task(reg, "johnson")["status"] == "ambiguous")
        check("unknown is not_found",
              resolve_task(reg, "zzz")["status"] == "not_found")

        # 11. Tools + the durable write gate
        print("tools/write gate:")
        ctx = AgentContext(data_dir=data_dir, registry=reg,
                           workspace_root=code_root,
                           identity_chain=chain,
                           embedder=emb, embed_dim=emb.dim)
        out = _call(ctx, "task_open", name="selftest-task",
                    objective="selftest", source_root=str(code_root))
        check("task_open", "opened" in out, out)
        out = _call(ctx, "task_ingest_path", task_name="selftest-task",
                    path=str(code_root), extensions=[".py"])
        check("task_ingest_path", "ingested" in out, out)
        out = _call(ctx, "task_ingest_path", task_name="selftest-task",
                    path=str(code_root), extensions=[".py"],
                    changed_only=True)
        check("changed-only ingest skips clean files",
              "ingested 0" in out or "0 file" in out, out)
        out = _call(ctx, "write_file",
                    path=str(code_root / "src" / "wallet" / "new.py"),
                    content="x = 1\n", change_summary="selftest write")
        op = json.loads(out)
        check("write_file creates a pending op (never writes)",
              op["status"] == "confirmation_required"
              and not (code_root / "src" / "wallet" / "new.py").exists())
        res = execute_user_action("approve_write",
                                  {"pending_op_id": op["pending_op_id"]}, ctx)
        check("user approve writes atomically",
              (code_root / "src" / "wallet" / "new.py").read_text()
              == "x = 1\n", res)
        out = _call(ctx, "write_file",
                    path=str(code_root / "src" / "wallet" / "new2.py"),
                    content="y = 2\n", change_summary="to be rejected")
        op2 = json.loads(out)
        execute_user_action("reject_write",
                            {"pending_op_id": op2["pending_op_id"]}, ctx)
        check("user reject leaves no file and no pending op",
              not (code_root / "src" / "wallet" / "new2.py").exists()
              and ctx.pending_ops.list_ids() == [])
        out = _call(ctx, "defense_status")
        check("defense_status reports posture",
              json.loads(out)["immune"]["scars"] >= 1, out[:120])

        # 12. ingest_blob — artifacts routing (v1.4.2)
        print("ingest_blob:")
        ctx.active_task = "selftest-task"   # an ACTIVE task must not capture
        out = _call(ctx, "ingest_blob", content="pasted selftest note",
                    name="note.txt", mime_type="text/plain")
        check("default -> artifacts chain + pointer + disk copy",
              out.startswith("Attached") and "artifacts" in out
              and "into task 'selftest-task'" not in out
              and (tools_mod.ARTIFACTS_DIR / "note.txt").exists(), out)
        # Single-record turn shape: the pointer is STAGED (no standalone
        # attachment record) and seals into the next turn's response.
        staged = ctx.staged_attachments
        check("identity pointer staged, carries no content",
              [r for r in chain.iter_records()
               if r.type == "attachment"] == []
              and bool(staged)
              and "extracted_text" not in staged[-1]
              and staged[-1].get("artifact_rings"),
              str(staged[-1] if staged else None)[:120])
        out = _call(ctx, "ingest_blob", content="task-scoped note",
                    name="task-note.txt", mime_type="text/plain",
                    task_name="selftest-task")
        check("explicit task_name -> task chain + workspace",
              "into task 'selftest-task'" in out
              and (Path(reg.get("selftest-task")["root"]) / "workspace"
                   / "task-note.txt").exists(), out)

        ok, detail = chain.verify()
        check("final verify", ok, detail)
        idx.close()
        ctx.close()
        task_chain.close()
        chain.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\nSELFTEST:", "PASS" if _ok else "FAIL")
    sys.exit(0 if _ok else 1)


if __name__ == "__main__":
    main()
