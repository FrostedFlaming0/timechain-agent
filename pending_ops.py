"""
pending_ops — durable write-confirmation state (Safety Tier 3).

`write_file` never touches the filesystem. It creates a PendingOperation,
persists it to `<data_dir>/pending_ops/<id>.json` (0600), and returns the id.
The USER approves via `approve_write` (REPL "yes" / web endpoint) — the model
never calls it autonomously. Approval verifies optimistic concurrency
(pre-write hash), writes atomically (temp file + os.replace), ingests the
change into the task chain idempotently (operation_id), audits the ingested
block against live source, and deletes the pending-op file.

State machine:

    pending -> writing -> written -> approved
       |                     |
       v                     v
    rejected/expired     ingest_failed -> (retry approve_write) -> approved

Crash recovery inspects live hash, proposed hash, pre-write hash, and the
temp file to resume deterministically. Expiry applies ONLY to `pending` —
partially-executed states recover regardless of TTL.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Literal, Optional

PENDING_TTL_SECONDS = 300
MAX_CONTENT_BYTES = 1024 * 1024          # 1MB cap on proposed content

PendingStatus = Literal[
    "pending", "writing", "written", "ingest_failed",
    "approved", "rejected", "expired",
]


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class PendingOperation:
    id: str
    task_name: str
    file_path: str                   # resolved absolute path
    proposed_content: str
    proposed_content_hash: str       # SHA-256 of proposed content
    pre_write_file_hash: Optional[str]   # SHA-256 of file BEFORE edit; None if new
    change_summary: str
    created_at: float
    expires_at: float
    status: str = "pending"
    target_existed: bool = False     # True if the file existed at read time

    # kind="tool_call" defers a confirmation-gated tool call (e.g. a
    # boundary-expanding task_open) for the loops that have no inline
    # confirm hook (web/SSE). The write fields above are unused ("");
    # proposed_content_hash holds the SHA-256 of tool_args_json so the
    # approved call is pinned exactly like approved write content is.
    # Defaults keep older on-disk write ops loading unchanged.
    kind: str = "write"
    tool_name: str = ""
    tool_args_json: str = ""

    def expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def tmp_path(self) -> str:
        return self.file_path + ".tmp." + self.id[:8]


class PendingOpStore:
    """Load/save/delete pending operations under `<data_dir>/pending_ops/`.

    NOT served through any generic file route — webapp must never expose
    this directory (proposed content may hold unreviewed secrets)."""

    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir) / "pending_ops"

    def _path(self, op_id: str) -> Path:
        # op ids are uuid4 hex — refuse anything path-like outright.
        safe = os.path.basename(op_id)
        if safe != op_id or not safe or safe in (".", ".."):
            raise ValueError(f"invalid pending op id: {op_id!r}")
        return self.dir / f"{safe}.json"

    def create(self, task_name: str, file_path: str, proposed_content: str,
               change_summary: str, ttl: float = PENDING_TTL_SECONDS) -> PendingOperation:
        if len(proposed_content.encode("utf-8")) > MAX_CONTENT_BYTES:
            raise ValueError(
                f"proposed content exceeds {MAX_CONTENT_BYTES} bytes — refuse")
        target = Path(file_path)
        existed = target.is_file()
        now = time.time()
        op = PendingOperation(
            id=uuid.uuid4().hex,
            task_name=task_name,
            file_path=str(file_path),
            proposed_content=proposed_content,
            proposed_content_hash=sha256_text(proposed_content),
            pre_write_file_hash=sha256_file(target) if existed else None,
            change_summary=change_summary,
            created_at=now,
            expires_at=now + ttl,
            status="pending",
            target_existed=existed,
        )
        self.save(op)
        return op

    def create_tool_call(self, tool_name: str, arguments: dict,
                         ttl: float = PENDING_TTL_SECONDS) -> PendingOperation:
        """Defer a confirmation-gated TOOL CALL for user approval (the
        web loop's stand-in for the REPL's inline confirm hook). The exact
        call is pinned: canonical-JSON arguments, hashed into
        proposed_content_hash, verified again at approval."""
        args_json = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        if len(args_json.encode("utf-8")) > MAX_CONTENT_BYTES:
            raise ValueError(
                f"tool arguments exceed {MAX_CONTENT_BYTES} bytes — refuse")
        now = time.time()
        op = PendingOperation(
            id=uuid.uuid4().hex,
            task_name=str(arguments.get("task_name")
                          or arguments.get("name") or ""),
            file_path="",
            proposed_content="",
            proposed_content_hash=sha256_text(args_json),
            pre_write_file_hash=None,
            change_summary=f"tool call: {tool_name}",
            created_at=now,
            expires_at=now + ttl,
            status="pending",
            target_existed=False,
            kind="tool_call",
            tool_name=tool_name,
            tool_args_json=args_json,
        )
        self.save(op)
        return op

    def save(self, op: PendingOperation) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(op.id)
        tmp = path.with_suffix(".json.tmp")
        # encoding pinned: hashes are computed over utf-8 bytes (sha256_text),
        # so the bytes on disk must be utf-8 regardless of locale.
        tmp.write_text(json.dumps(asdict(op), indent=2), encoding="utf-8")
        os.chmod(tmp, 0o600)        # only the agent process reads pending content
        os.replace(tmp, path)

    def load(self, op_id: str) -> Optional[PendingOperation]:
        try:
            path = self._path(op_id)
        except ValueError:
            return None
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return PendingOperation(**data)
        except (OSError, json.JSONDecodeError, TypeError):
            return None

    def delete(self, op_id: str) -> None:
        try:
            self._path(op_id).unlink(missing_ok=True)
        except ValueError:
            pass

    def list_ids(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("*.json"))


# --------------------------------------------------------------- execution


def _atomic_write(op: PendingOperation) -> None:
    """Write proposed content via temp file + os.replace, preserving mode."""
    tmp = op.tmp_path
    # utf-8 pinned: proposed_content_hash was computed over utf-8 bytes; a
    # locale-default write would either raise UnicodeEncodeError mid-approval
    # (stranding the op in status='writing') or write bytes whose hash can
    # never match verification.
    Path(tmp).write_text(op.proposed_content, encoding="utf-8")
    if Path(op.file_path).exists():
        shutil.copymode(op.file_path, tmp)    # preserve existing mode
    else:
        os.chmod(tmp, 0o644)                  # new file default
    os.replace(tmp, op.file_path)             # atomic on POSIX


def _finalize_approved(op: PendingOperation, ctx, ingest_result, note: str = "") -> str:
    """Shared tail for the primary path AND ingest_failed recovery:
    mark approved, audit the just-ingested block, delete the pending-op
    file. Neither path may skip the audit or the cleanup."""
    op.status = "approved"
    ctx.pending_ops.save(op)
    suffix = f" ({note})" if note else ""

    # Automatic post-write audit: verify ONLY the block we just ingested.
    try:
        latest_ring = ingest_result["ring_index"]
        task = ctx.registry.get(op.task_name) or {}
        recall = ctx.get_task_recall(op.task_name)
        audit = recall.verify_source(latest_ring, repo=task.get("source_root"))
        if audit.get("verdict") == "source-mismatch":
            msg = (f"Written {op.file_path}.{suffix} "
                   f"WARNING: source mismatch after write. "
                   f"This should not happen — check for external modification.")
        else:
            msg = f"Written {op.file_path}.{suffix} Audit: {audit.get('verdict')}."
    except Exception as e:                    # noqa: BLE001 — non-fatal
        msg = f"Written {op.file_path}.{suffix} Audit skipped: {e}"

    # Cleanup on resolution: delete the on-disk pending-op JSON now that the
    # op is terminal. The save above covers the crash window — a crash before
    # this delete leaves status=approved, so a retry reports "already
    # processed" instead of re-executing.
    ctx.pending_ops.delete(op.id)
    _seal_resolution(ctx, op, "approved", msg)
    return msg


def _seal_resolution(ctx, op: PendingOperation, outcome: str,
                     result: str) -> None:
    """Seal the user's approve/reject decision on the IDENTITY chain as a
    `resolution` record. Approval is an out-of-band event the model never
    witnesses — without this record its last sealed knowledge says
    "pending" forever, and it fills the gap with fiction (the field
    confabulation: "never approved, so never ingested"). Best-effort: a
    sealing failure must not undo a write that already happened."""
    chain = getattr(ctx, "identity_chain", None)
    if chain is None:
        return
    try:
        from metadata import build_meta
        content = {
            "event": outcome,                  # "approved" | "rejected"
            "pending_op_id": op.id,
            "op_kind": getattr(op, "kind", "write"),
            "summary": (op.change_summary if op.kind != "tool_call"
                        else f"tool call: {op.tool_name}"),
            "result": (result or "")[:300],
            "_meta": build_meta("resolution"),
        }
        if op.kind == "tool_call":
            content["tool"] = op.tool_name
        else:
            content["file"] = op.file_path
        chain.append("resolution", content)
    except Exception:        # noqa: BLE001 — the decision already executed
        pass


def _approve_tool_call(op: PendingOperation, ctx) -> str:
    """USER-triggered approval of a deferred tool call. Single-shot: there
    is no resumable middle state like the write machine — once execution
    starts the op is terminal, and a crash mid-execution reads as
    'already processed' rather than risking a double run of a
    non-idempotent tool."""
    if op.status != "pending":
        return f"ERROR: Operation already processed (status={op.status})."
    if op.expired():
        op.status = "expired"
        ctx.pending_ops.save(op)
        ctx.pending_ops.delete(op.id)
        return "ERROR: Operation expired."
    # The approved call must be exactly the deferred call.
    if sha256_text(op.tool_args_json) != op.proposed_content_hash:
        return "ERROR: Tool arguments hash mismatch — refusing to execute."
    # Lazy import: tools.py imports this module at load time.
    import tools as tools_mod
    if (op.tool_name not in tools_mod.TOOL_MAP
            or op.tool_name in tools_mod.USER_ONLY_TOOLS):
        return f"ERROR: {op.tool_name!r} is not an approvable tool."
    try:
        arguments = json.loads(op.tool_args_json)
    except json.JSONDecodeError:
        return "ERROR: Stored tool arguments are not valid JSON."

    op.status = "writing"            # running; terminal from here on
    ctx.pending_ops.save(op)
    result = tools_mod.execute_tool(
        {"name": op.tool_name, "arguments": arguments}, ctx)
    op.status = "approved"
    ctx.pending_ops.save(op)
    ctx.pending_ops.delete(op.id)    # cleanup on resolution
    _seal_resolution(ctx, op, "approved", result)
    if result.startswith("TOOL ERROR"):
        return f"ERROR: {result}"    # keep the ok/failed prefix contract
    return result


def execute_approve_write(kwargs: dict, ctx) -> str:
    """USER-triggered approval. For kind="write": the full write-gate state
    machine, including crash recovery for the writing/written/ingest_failed
    states. For kind="tool_call": single-shot deferred-call execution
    (_approve_tool_call).

    CONTRACT: every non-success return starts with "ERROR:" or "REFUSED" —
    the web pending-op endpoints (and anything else that needs ok/failed
    without parsing prose) rely on those prefixes."""
    op = ctx.pending_ops.load(kwargs.get("pending_op_id", ""))
    if op is None:
        return "ERROR: Operation not found — invalid or expired pending_op_id."

    if getattr(op, "kind", "write") == "tool_call":
        return _approve_tool_call(op, ctx)

    if op.status == "ingest_failed":
        # Recovery: file already written, retry only the ingest step.
        # Verify disk content first — external modification after the
        # original write must not be sealed as the approved content.
        live_hash = sha256_file(op.file_path) if Path(op.file_path).exists() else None
        if live_hash != op.proposed_content_hash:
            return (f"ERROR: File content changed since original write. "
                    f"Expected {op.proposed_content_hash[:8]}, "
                    f"got {live_hash[:8] if live_hash else 'file missing'}. "
                    f"Manual intervention required.")
        ingest_result = ctx.task_ingest(op.task_name, op.file_path,
                                        finding=op.change_summary,
                                        content_hash=op.proposed_content_hash,
                                        operation_id=op.id)
        return _finalize_approved(op, ctx, ingest_result,
                                  note="recovered from ingest failure")

    if op.status in ("writing", "written"):
        # Crash-recovery: inspect live hash, proposed hash, pre-write hash,
        # and the temp file to determine what actually happened.
        live_hash = sha256_file(op.file_path) if Path(op.file_path).exists() else None
        tmp_exists = Path(op.tmp_path).exists()
        if op.status == "writing":
            if live_hash == op.proposed_content_hash:
                # os.replace succeeded before the crash.
                if tmp_exists:
                    Path(op.tmp_path).unlink(missing_ok=True)
                op.status = "written"
                ctx.pending_ops.save(op)
            elif tmp_exists:
                # Temp file exists but replace didn't run. Verify tmp content
                # before replacing — stale or altered tmp must be caught.
                tmp_hash = sha256_file(op.tmp_path)
                if tmp_hash != op.proposed_content_hash:
                    Path(op.tmp_path).unlink(missing_ok=True)
                    return (f"ERROR: Temporary file content mismatch. "
                            f"Expected {op.proposed_content_hash[:8]}, "
                            f"got {tmp_hash[:8]}. Temp file discarded. "
                            f"Manual intervention required.")
                if Path(op.file_path).exists():
                    shutil.copymode(op.file_path, op.tmp_path)
                else:
                    os.chmod(op.tmp_path, 0o644)
                os.replace(op.tmp_path, op.file_path)
                live_hash = sha256_file(op.file_path)   # recompute post-replace
                op.status = "written"
                ctx.pending_ops.save(op)
            else:
                # Nothing was written yet. Safe to restart from pending ONLY
                # if the live file still matches the recorded pre-write state.
                if op.target_existed and live_hash != op.pre_write_file_hash:
                    return (f"ERROR: File was externally modified during crash window. "
                            f"Expected pre-write hash "
                            f"{(op.pre_write_file_hash or '')[:8]}, "
                            f"got {live_hash[:8] if live_hash else 'file missing'}. "
                            f"Manual intervention required.")
                if not op.target_existed and live_hash is not None:
                    return ("ERROR: File was externally created during crash window. "
                            "Manual intervention required.")
                op.status = "pending"
                ctx.pending_ops.save(op)
        if op.status == "written":
            # File written but ingest may not have run. Verify disk first.
            if live_hash != op.proposed_content_hash:
                return (f"ERROR: File content mismatch after crash recovery. "
                        f"Expected {op.proposed_content_hash[:8]}, "
                        f"got {live_hash[:8] if live_hash else 'file missing'}. "
                        f"Manual intervention required.")
            ingest_result = ctx.task_ingest(op.task_name, op.file_path,
                                            finding=op.change_summary,
                                            content_hash=op.proposed_content_hash,
                                            operation_id=op.id)
            return _finalize_approved(op, ctx, ingest_result,
                                      note="recovered from crash")

    if op.status != "pending":
        return f"ERROR: Operation already processed (status={op.status})."

    # Expiry ONLY applies to pending — partially-executed states recover
    # regardless of TTL, because real work happened.
    if op.expired():
        op.status = "expired"
        ctx.pending_ops.save(op)
        ctx.pending_ops.delete(op.id)   # cleanup on resolution (expire)
        return "ERROR: Operation expired."

    # Optimistic concurrency: verify the file hasn't changed since read.
    if op.target_existed and not Path(op.file_path).exists():
        return "ERROR: File was deleted since read — aborting."
    if not op.target_existed and Path(op.file_path).exists():
        return "ERROR: File was created since read — aborting."
    if op.target_existed:
        live_hash = sha256_file(op.file_path)
        if live_hash != op.pre_write_file_hash:
            return "ERROR: File changed since read — aborting. Re-read and try again."

    # Phase: writing -> atomic write -> written
    op.status = "writing"
    ctx.pending_ops.save(op)
    _atomic_write(op)
    op.status = "written"
    ctx.pending_ops.save(op)

    # Phase: ingest into task chain. Idempotent — continuum stores
    # operation_id and find_by_operation_id() detects duplicates, so a crash
    # after ingest but before approved cannot double-seal on retry.
    try:
        ingest_result = ctx.task_ingest(op.task_name, op.file_path,
                                        finding=op.change_summary,
                                        content_hash=op.proposed_content_hash,
                                        operation_id=op.id)
    except Exception as e:                    # noqa: BLE001 — recovery state
        op.status = "ingest_failed"
        ctx.pending_ops.save(op)
        return (f"ERROR: Written {op.file_path} but ingest failed: {e} "
                f"(op {op.id}, status=ingest_failed). Retry with "
                f"approve_write to re-attempt ingest.")
    return _finalize_approved(op, ctx, ingest_result)


def execute_reject_write(kwargs: dict, ctx) -> str:
    """USER-triggered (or expiry-driven) rejection — never the model's call.
    Covers both kinds: a rejected tool_call op is simply discarded."""
    op = ctx.pending_ops.load(kwargs.get("pending_op_id", ""))
    if op is None or op.status != "pending":
        return "ERROR: Operation already processed."
    op.status = "rejected"
    ctx.pending_ops.save(op)
    ctx.pending_ops.delete(op.id)   # cleanup on resolution (reject)
    if getattr(op, "kind", "write") == "tool_call":
        msg = f"Tool call {op.tool_name} rejected."
    else:
        msg = f"Write to {op.file_path} rejected."
    _seal_resolution(ctx, op, "rejected", msg)
    return msg
