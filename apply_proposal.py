"""
apply_proposal — review and scaffold a Cambium proposal.

Cambium (cambium.py) detects recurring gaps and commits `proposal`
records suggesting new skills, modalities, senses, or principles. It
deliberately never applies them: the build spec's rule is "the model
proposes, policy decides," and a proposal is generated from lexical
pattern-matching, not tested code.

This tool is the *policy decides* half. It is a deliberate, operator-run
command — not something the agent calls. It does three things:

  1. Shows a proposal in full so a human can judge it.
  2. For an `accept`, scaffolds a *stub* into the codebase:
       - modality / sense proposals  -> a detector stub in signals.py,
         with the correct signature, registered, and a `# TODO` body.
       - principle / skill proposals -> printed guidance (these are not
         code, so there is nothing to scaffold).
  3. Records the decision on the chain as a `proposal_status` record, so
     the audit trail shows *who decided what, and when* — not just that
     a file changed.

What this tool deliberately does NOT do:
  - It never writes a working detector. The stub has a `# TODO` body and
    a deliberately low, neutral return. A human writes the real logic and
    adds a test. That irreducible human step is the safety boundary —
    recurrence and escalation raise a proposal's *visibility*, never its
    authority to run unreviewed code.
  - It never edits a record on disk. The chain is append-only; a decision
    is a new `proposal_status` record referencing the proposal.

Usage:
    python apply_proposal.py --list
        Show all proposals, escalated ones first, with recurrence counts.

    python apply_proposal.py --show N
        Show proposal at record index N in full (rationale, evidence).

    python apply_proposal.py --accept N
        Scaffold a stub for proposal N (if it is a modality/sense) and
        commit a 'proposal_status' record marking it accepted.

    python apply_proposal.py --decline N --reason "..."
        Commit a 'proposal_status' record marking proposal N declined.

The data directory is the same one run.py uses (DATA_DIR in run.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chain import Chain, load_or_create_key
import cambium


# The signals.py file scaffolding is written into. Resolved relative to
# this file so the tool works regardless of the current directory.
SIGNALS_PATH = Path(__file__).parent / "signals.py"

# Data directory — mirrors run.py. If run.py's DATA_DIR was customized,
# change this to match.
DATA_DIR = Path(__file__).parent / "timechain_data"


# ---------------------------------------------------------------------------
# Chain access
# ---------------------------------------------------------------------------

def _open_chain() -> Chain:
    """Open the chain at DATA_DIR. Exits with a clear message if absent."""
    chain_db = DATA_DIR / "chain.sqlite"
    key_path = DATA_DIR / "operator.key"
    if not chain_db.exists():
        sys.exit(
            f"no chain found at {chain_db}\n"
            f"run.py has not been run yet, or DATA_DIR differs from run.py's."
        )
    key = load_or_create_key(key_path)
    return Chain(chain_db, key)


def _get_proposal(chain: Chain, index: int):
    """Fetch and validate a proposal record. Exits on any problem."""
    rec = chain.get(index)
    if rec is None:
        sys.exit(f"no record at index {index}")
    if rec.type != "proposal":
        sys.exit(f"record {index} is type {rec.type!r}, not 'proposal'")
    return rec


def _current_status(chain: Chain, proposal_index: int) -> str:
    """
    The proposal's effective status. The proposal record carries an
    initial status ("open"); later `proposal_status` records can change
    it. The most recent proposal_status wins.
    """
    rec = chain.get(proposal_index)
    status = "open"
    if rec is not None and isinstance(rec.content, dict):
        status = rec.content.get("status", "open")
    latest_idx = -1
    for sr in chain.query_by_type("proposal_status", limit=10_000):
        if not isinstance(sr.content, dict):
            continue
        if sr.content.get("marks_proposal_index") != proposal_index:
            continue
        if sr.index > latest_idx:
            latest_idx = sr.index
            status = sr.content.get("new_status", status)
    return status


# ---------------------------------------------------------------------------
# Listing and showing
# ---------------------------------------------------------------------------

def cmd_list(chain: Chain) -> None:
    proposals = chain.query_by_type("proposal", limit=200)
    if not proposals:
        print("no proposal records on chain yet")
        return
    # Bulk recurrence/escalation lookups — one scan of the chain instead
    # of one per proposal. Matters once proposals or recurrences accumulate.
    counts = cambium.recurrence_counts(chain)
    escalated_set = cambium.escalated_indices(chain)
    rows = []
    for rec in proposals:
        n = counts.get(rec.index, 1)
        esc = rec.index in escalated_set
        status = _current_status(chain, rec.index)
        rows.append((rec, n, esc, status))
    # Escalated first, then by index.
    rows.sort(key=lambda r: (not r[2], r[0].index))
    print(f"{len(rows)} proposal(s) — escalated shown first:\n")
    for rec, n, esc, status in rows:
        c = rec.content
        flag = "  ** ESCALATED **" if esc else ""
        rc = f"  (recurred {n}x)" if n > 1 else ""
        print(f"  #{rec.index}  [{c.get('proposal_kind','?')}]  "
              f"status={status}{rc}{flag}")
        print(f"      {c.get('title','')}")
    print()
    print("inspect one with:  python apply_proposal.py --show <index>")


def cmd_show(chain: Chain, index: int) -> None:
    rec = _get_proposal(chain, index)
    c = rec.content
    n = cambium.recurrence_count(chain, index)
    esc = cambium.is_escalated(chain, index)
    status = _current_status(chain, index)

    print("=" * 70)
    print(f"PROPOSAL #{index}   [{c.get('proposal_kind','?')}]")
    print("=" * 70)
    print(f"title           : {c.get('title','')}")
    print(f"status          : {status}")
    print(f"recurrence count: {n}" + ("   ** ESCALATED **" if esc else ""))
    print(f"topic signature : {c.get('topic_signature','')}")
    print(f"suggested target: {c.get('suggested_target','')}")
    print()
    print("rationale:")
    for line in str(c.get("rationale", "")).split("\n"):
        print(f"  {line}")
    print()
    evidence = c.get("evidence_indices", [])
    print(f"evidence records: {evidence}")
    print()
    kind = c.get("proposal_kind", "?")
    if kind in ("modality", "sense"):
        print("to accept and scaffold a detector stub for this proposal:")
        print(f"  python apply_proposal.py --accept {index}")
    else:
        print(f"this is a '{kind}' proposal — accepting it records the")
        print("decision but scaffolds no code (principles/skills are not")
        print("detector functions). See --accept output for guidance.")


# ---------------------------------------------------------------------------
# Scaffolding into signals.py
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    """A safe lowercase identifier fragment from arbitrary text."""
    keep = [ch if (ch.isalnum() or ch == " ") else " " for ch in text.lower()]
    words = "".join(keep).split()
    return "_".join(words[:4]) if words else "detector"


def _detector_name(kind: str, proposal_index: int, title: str) -> str:
    """
    Build a unique detector function name. Prefix m_ for modality, s_ for
    sense — matching the existing naming in signals.py — and append the
    proposal index so the name is guaranteed unique even if two proposals
    have similar titles.
    """
    prefix = "m_" if kind == "modality" else "s_"
    return f"{prefix}{_slug(title)}_p{proposal_index}"


def _build_stub(kind: str, fn_name: str, proposal_index: int,
                title: str, rationale: str) -> str:
    """The text of the detector stub function."""
    detector_kind = "modality" if kind == "modality" else "sense"
    rationale_lines = "\n".join(
        f"    {line}" for line in rationale.strip().split("\n")
    )
    return f'''

def {fn_name}(inp: SignalInput) -> SignalHit:
    """
    SCAFFOLDED STUB — from Cambium proposal #{proposal_index}.

    Proposed {detector_kind}: {title}

    Rationale from Cambium:
{rationale_lines}

    TODO: implement the detection logic. This stub returns a neutral,
    low activation so it is harmless until a human writes the real
    detector and adds a test for it (see CONTRIBUTING.md — detectors
    must be deterministic and dependency-free). Until then this stub
    does nothing meaningful; that is intentional.
    """
    # TODO: replace this stub body with real detection logic.
    return SignalHit(
        "{detector_kind}", "{fn_name}", 0.1,
        "stub detector — not yet implemented",
        {{"stub": True, "from_proposal": {proposal_index}}},
    )
'''


def _scaffold_detector(kind: str, proposal_index: int, title: str,
                       rationale: str) -> str:
    """
    Insert a detector stub into signals.py and register it. Returns the
    new function's name.

    Insertion points are sentinel comments in signals.py — not pattern
    matching on declaration text — so changes elsewhere in signals.py
    (a new annotation form, an extra detector, an import shuffle) can
    never corrupt the insert. The sentinels:

      # CAMBIUM_SCAFFOLD_INSERT_DETECTOR — the stub function goes above.
      # CAMBIUM_SCAFFOLD_INSERT_MODALITY — the registry entry goes above
                                           (inside MODALITY_REGISTRY).
      # CAMBIUM_SCAFFOLD_INSERT_SENSE    — same, inside SENSE_REGISTRY.

    After writing, the new file is parsed with `ast.parse` so any
    corruption (a stub-build bug, a half-applied edit) fails loudly
    BEFORE signals.py is left in a broken state. If parsing fails, the
    original file is restored and a RuntimeError is raised.
    """
    import ast
    src = SIGNALS_PATH.read_text()
    original_src = src  # rollback target if validation fails

    detector_sentinel = "# CAMBIUM_SCAFFOLD_INSERT_DETECTOR"
    registry_sentinel = (
        "# CAMBIUM_SCAFFOLD_INSERT_MODALITY" if kind == "modality"
        else "# CAMBIUM_SCAFFOLD_INSERT_SENSE"
    )
    for needed in (detector_sentinel, registry_sentinel):
        if needed not in src:
            raise RuntimeError(
                f"signals.py is missing scaffold sentinel {needed!r}; "
                f"the file has been edited beyond what apply_proposal "
                f"can patch automatically. Scaffold this proposal "
                f"manually instead."
            )

    fn_name = _detector_name(kind, proposal_index, title)
    if f"def {fn_name}(" in src:
        raise RuntimeError(
            f"signals.py already defines {fn_name} — proposal #{proposal_index} "
            f"appears to have been scaffolded already."
        )

    stub = _build_stub(kind, fn_name, proposal_index, title, rationale)

    # 1. Insert the stub function immediately above the detector sentinel.
    src = src.replace(detector_sentinel, stub + "\n\n" + detector_sentinel, 1)

    # 2. Insert the registry entry immediately above the registry sentinel.
    #    The sentinel itself already sits indented inside the list literal
    #    (matching the surrounding entries), so we just need to emit the
    #    new name with the same 4-space indent and a newline before it —
    #    the sentinel's own leading spaces stay where they are.
    registry_entry = f"{fn_name},\n    "
    src = src.replace(registry_sentinel, registry_entry + registry_sentinel, 1)

    # 3. Validate: the patched file must still parse as Python. A failed
    #    parse means our edit produced a broken signals.py — better to
    #    refuse the change than to ship a broken module.
    try:
        ast.parse(src, filename=str(SIGNALS_PATH))
    except SyntaxError as e:
        # Don't write anything; leave signals.py untouched.
        raise RuntimeError(
            f"scaffolded signals.py fails to parse ({e}); the proposal "
            f"was NOT applied. This is a bug in apply_proposal — please "
            f"report it. signals.py is unchanged."
        )

    SIGNALS_PATH.write_text(src)
    return fn_name

    SIGNALS_PATH.write_text(src)
    return fn_name


# ---------------------------------------------------------------------------
# Accept / decline
# ---------------------------------------------------------------------------

def _commit_status(chain: Chain, proposal_index: int, new_status: str,
                   reason: str, extra: dict | None = None) -> int:
    """
    Append a 'proposal_status' record recording a decision. Returns the
    new record's index. The chain is append-only — this is a new record
    that references the proposal, never an edit of it.
    """
    # build_meta is imported lazily so this module stays importable even
    # if metadata.py is mid-refactor; it is a hard dependency in practice.
    from metadata import build_meta, SOURCE_SYSTEM

    content = {
        "marks_proposal_index": proposal_index,
        "new_status": new_status,
        "reason": reason,
        # The decision came from a human running this tool, not the agent.
        "decided_by": "operator (apply_proposal.py)",
        "schema_version": 1,
    }
    if extra:
        content.update(extra)
    content["_meta"] = build_meta(
        "proposal_status",
        source=SOURCE_SYSTEM,   # an operator decision is system-sourced
        salience=0.85,
        confidence=1.0,
    )
    rec = chain.append("proposal_status", content)
    return rec.index


def cmd_accept(chain: Chain, index: int) -> None:
    rec = _get_proposal(chain, index)
    c = rec.content
    kind = c.get("proposal_kind", "?")
    title = c.get("title", "")
    rationale = c.get("rationale", "")

    status = _current_status(chain, index)
    if status in ("accepted", "declined"):
        sys.exit(f"proposal #{index} is already {status}; nothing to do.")

    scaffolded_name = None
    extra: dict = {}

    if kind in ("modality", "sense"):
        try:
            scaffolded_name = _scaffold_detector(kind, index, title, rationale)
        except RuntimeError as e:
            sys.exit(f"scaffolding failed: {e}")
        extra["scaffolded_function"] = scaffolded_name
        extra["scaffolded_file"] = "signals.py"
        print(f"scaffolded a {kind} detector stub into signals.py:")
        print(f"  function: {scaffolded_name}")
        print(f"  registered in: "
              f"{'MODALITY_REGISTRY' if kind == 'modality' else 'SENSE_REGISTRY'}")
        print()
        print("NEXT STEPS (these are the human's job — and the safety boundary):")
        print("  1. Open signals.py and implement the detector body where the")
        print("     `# TODO` is. Keep it deterministic and dependency-free.")
        print("  2. Add a test for it in test_timechain.py (TestSignals).")
        print("  3. Run: python run_tests.py")
        print("  Until you do (1), the stub returns a harmless low activation.")
    else:
        # principle / skill — nothing to scaffold as code.
        print(f"proposal #{index} is a '{kind}' proposal — recording the")
        print("acceptance, but there is no detector code to scaffold.")
        if kind == "principle":
            print()
            print("NEXT STEP: a principle is a durable rule. Consider adding it")
            print("as a 'principle' record (a protected-zone record type) and/or")
            print("folding it into the system prompt in run.py.")
        elif kind == "skill":
            print()
            print("NEXT STEP: a skill is a documented, repeatable procedure.")
            print("Write it up wherever the project keeps operational docs.")

    status_idx = _commit_status(
        chain, index, "accepted",
        reason=f"accepted via apply_proposal.py; kind={kind}",
        extra=extra,
    )
    print()
    print(f"recorded acceptance as proposal_status record #{status_idx}.")
    print("the chain now carries an audit trail: proposal -> human decision.")


def cmd_decline(chain: Chain, index: int, reason: str) -> None:
    _get_proposal(chain, index)
    status = _current_status(chain, index)
    if status in ("accepted", "declined"):
        sys.exit(f"proposal #{index} is already {status}; nothing to do.")
    status_idx = _commit_status(
        chain, index, "declined",
        reason=reason or "declined via apply_proposal.py (no reason given)",
    )
    print(f"proposal #{index} marked declined.")
    print(f"recorded as proposal_status record #{status_idx}.")
    print("a future Cambium scan may re-propose this topic, since declined")
    print("proposals are not used to suppress fresh detection.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Review and scaffold Cambium proposals.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true",
                   help="list all proposals, escalated first")
    g.add_argument("--show", type=int, metavar="N",
                   help="show proposal at record index N in full")
    g.add_argument("--accept", type=int, metavar="N",
                   help="accept proposal N (scaffolds a stub if it is a "
                        "modality/sense) and record the decision")
    g.add_argument("--decline", type=int, metavar="N",
                   help="decline proposal N and record the decision")
    p.add_argument("--reason", default="",
                   help="reason string, used with --decline")
    args = p.parse_args()

    chain = _open_chain()
    try:
        if args.list:
            cmd_list(chain)
        elif args.show is not None:
            cmd_show(chain, args.show)
        elif args.accept is not None:
            cmd_accept(chain, args.accept)
        elif args.decline is not None:
            cmd_decline(chain, args.decline, args.reason)
    finally:
        chain.close()


if __name__ == "__main__":
    main()
