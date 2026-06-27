"""
tools — tool schemas, execution functions, and the shared text-tool driver
for the code-working agent (v1.4).

The LLM client layer has no native function calling: tools are TEXT. The
model emits

    <tool_call>
    {"name": "read_file", "arguments": {"path": "agent.py"}}
    </tool_call>

and this module supplies the single shared driver both the sync REPL loop
and the async web loop must use (extraction, validation, execution,
escaping — never duplicated, so the safety gates cannot drift):

    extract_tool_calls()  TOLERANT syntax layer — strips code fences,
                          tolerates trailing commas, recovers every block.
    validate_tool_call()  STRICT contract layer — JSON-Schema-checked args,
                          unknown tools/params/types rejected.
    execute_tool()        dispatch with size-capped results.
    escape_tool_markup()  neutralizes <tool_call> in results so file content
                          can never inject forged calls.

Safety tiers: task selection is explicit (resolve_task never infers);
file scoping via pin_file; writes go through the durable PendingOperation
gate in pending_ops.py — `write_file` only ever returns a pending_op_id,
and approve_write/reject_write are USER-triggered, never model-callable.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pending_ops as pending_ops_mod
from pending_ops import PendingOpStore, sha256_text
from task_registry import TaskRegistry, TaskRegistryError, resolve_task

MAX_TOOL_RESULT_BYTES = 64 * 1024     # reject/truncate oversized tool results

# Tool-loop round budget, shared by the REPL loop (agent.turn_with_tools)
# and the web loop so the two can't drift. 10 proved too small for real
# work — a model that reads files one round at a time burns the budget
# exploring a repo and gets cut off before it can write anything.
DEFAULT_MAX_TOOL_ROUNDS = 24

# Injected into the prompt with the LAST round's results, so the model's
# final response is useful instead of a batch of silently-dropped tool
# calls. It deliberately does NOT demand a complete answer: forcing a
# big task to "finish now" produces a thin, premature deliverable. A
# checkpoint is the resume state the user's next "continue" runs on.
TOOL_BUDGET_NUDGE = (
    "\n[tool loop] FINAL round: your tool budget is now exhausted — do "
    "NOT emit any more tool calls; they will not be executed. If you can "
    "complete the task from what you already have, give the complete "
    "answer now. If you cannot, do NOT force a premature result: produce "
    "a clear progress checkpoint instead — what you have done, what "
    "remains, and the next concrete step — and tell the user they can "
    "type 'continue' to give you a fresh tool budget to resume.\n")


def tool_retry_prompt(parse_errors) -> str:
    """The ONE reflective-retry instruction, shared by both loops (REPL
    agent.turn_with_tools and the web _turn_events) so a prompt-engineering
    fix reaches both. Appended after the model's unparseable output."""
    return ("\n\n[tool loop] Your tool call did not parse as JSON ("
            + ("; ".join(parse_errors) or "no valid block found")
            + "). Re-emit it as ONE valid <tool_call> block, or "
            "answer in plain text.")


def tool_cap_note(max_rounds: int) -> str:
    """The ONE round-cap notice, shared by both loops."""
    return f"\n\n[tool loop] Stopped after {max_rounds} tool rounds."


MAX_READ_BYTES = 256 * 1024           # refuse reading huge files outright
MAX_INGEST_FILE_BYTES = 8 * 1024 * 1024   # cap on a single-file task ingest
# (task_ingest reads the whole file into memory before chunking, so an
# unbounded file would balloon both the process and the task chain. The
# approve-write path stays well under this — proposed content is capped at
# 1MB in pending_ops.)
# ingest_blob (pastes/uploads) shares the same budget — ONE literal, so the
# two caps cannot drift: a file that uploads fine is never refused by the
# provenance ingest after approval.
INGEST_BLOB_MAX_BYTES = MAX_INGEST_FILE_BYTES

# Tools the model may call that still need an explicit per-call user
# confirmation in the driving loop (Tier 3). write_file is NOT here because
# it self-gates: it only creates a pending op. approve/reject are user-only.
# task_open is gated CONDITIONALLY via requires_confirmation() below.
# task_reembed is here because it is long-running (a full-chain re-embed
# through a CPU embedding model can take hours), not because it expands
# any boundary.
CONFIRM_TOOLS = frozenset({"task_ingest_file", "task_reembed"})


def resolve_source_root(src: str, ctx: "AgentContext") -> Path:
    """Resolve a task source_root the way every other path tool resolves
    paths (see _resolve_allowed_path): relative values are anchored at the
    WORKSPACE root — the contract the tool schema and workspace_prompt
    promise the model — never at the process cwd, which is wherever the
    server happened to be launched from. Symlinks are flattened so the
    boundary that gets confirmed is the boundary that gets enforced."""
    p = Path(src).expanduser()
    if not p.is_absolute():
        p = ctx.workspace_root / p
    return Path(os.path.realpath(p))


def requires_confirmation(name: str, arguments: dict,
                          ctx: "AgentContext") -> bool:
    """Tier-3 gate used by BOTH driving loops (REPL and web — keep them
    calling this one function so the policy cannot drift).

    task_open is conditional: its source_root becomes an allowed read/ingest
    root in _resolve_allowed_path, so the MODEL choosing one is a boundary
    expansion. Opening a task on the current workspace (or inside it) grants
    no new authority and runs unconfirmed; any other root — `/`, $HOME, a
    sibling repo — needs the user's explicit yes (and is refused in headless
    loops that have no confirm hook)."""
    if name in CONFIRM_TOOLS:
        return True
    if name == "task_open":
        src = arguments.get("source_root") if isinstance(arguments, dict) else None
        if not isinstance(src, str) or not src:
            return True              # malformed → let the strict validator
                                     # reject it, but never skip the gate
        resolved = resolve_source_root(src, ctx)
        workspace = ctx.workspace_root.resolve()
        return not (resolved == workspace or workspace in resolved.parents)
    return False

# Executors that exist for the REPL/web approval endpoints but are NEVER
# offered to the model as callable tools.
USER_ONLY_TOOLS = frozenset({"approve_write", "reject_write"})


# ------------------------------------------------------------ AgentContext


@dataclass
class AgentContext:
    """Carries everything tool executors need. One per session."""

    data_dir: Path
    registry: TaskRegistry
    identity_chain: Any = None
    identity_recall: Any = None
    workspace_root: Path = field(default_factory=Path.cwd)
    embedder: Any = None              # for per-task EmbeddingIndex (Phase 10)
    embed_dim: int = 0

    task_chains: dict = field(default_factory=dict)       # name -> Chain
    task_recalls: dict = field(default_factory=dict)      # name -> Recall
    task_continuums: dict = field(default_factory=dict)   # name -> Continuum
    _task_indexes: dict = field(default_factory=dict)     # name -> EmbeddingIndex

    # --- Session / turn state (NOT persisted to tasks.json) ---
    # active_task is a runtime cursor; persisting it causes stale-pointer
    # bugs. pinned_path is Tier-2 scoping, reset at the START of every turn
    # so a pin never leaks across turns.
    active_task: Optional[str] = None
    pinned_path: Optional[str] = None

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.workspace_root = Path(self.workspace_root)
        self.pending_ops = PendingOpStore(self.data_dir)

    # ----- per-task lazy resources -----

    def _require_task(self, task_name: str) -> dict:
        task = self.registry.get(task_name)
        if task is None:
            raise TaskRegistryError(
                f"unknown task {task_name!r} — use list_tasks / resolve_task")
        return task

    def get_task_chain(self, task_name: str):
        if task_name not in self.task_chains:
            from chain import Chain, load_or_create_key
            task = self._require_task(task_name)
            root = Path(task["root"])
            root.mkdir(parents=True, exist_ok=True)
            self.task_chains[task_name] = Chain(
                root / "chain.sqlite", load_or_create_key(root / "operator.key"))
        return self.task_chains[task_name]

    def get_task_recall(self, task_name: str):
        if task_name not in self.task_recalls:
            from recall import Recall
            self.task_recalls[task_name] = Recall(self.get_task_chain(task_name))
        return self.task_recalls[task_name]

    def get_task_continuum(self, task_name: str):
        if task_name not in self.task_continuums:
            from continuum import Continuum
            self.task_continuums[task_name] = Continuum(self.get_task_chain(task_name))
        return self.task_continuums[task_name]

    def _task_embedder(self, task: dict):
        """The embedder a task's derived store should be built with.

        Defaults to the instant HashingEmbedder REGARDLESS of the session
        embedder: bulk ingest must never block on a slow embedding model
        (a CPU-bound Ollama at ~3-5s/chunk once turned a 10-second walk
        into a 2-hour tool call). A task uses the session embedder only
        after an explicit, user-confirmed task_reembed persisted it as
        task["embedder"] = "session"."""
        from retrieval import HashingEmbedder
        if task.get("embedder") == "session" and self.embedder is not None:
            return self.embedder, (self.embed_dim
                                   or getattr(self.embedder, "dim", 0))
        embedder = HashingEmbedder()
        return embedder, embedder.dim

    def get_task_index(self, task_name: str):
        """Per-task EmbeddingIndex at `<task_root>/embeddings.sqlite` (lazy).
        Embedder choice is per task (see _task_embedder); the database is
        separate from the identity index."""
        if task_name not in self._task_indexes:
            from retrieval import open_or_rebuild_index
            task = self._require_task(task_name)
            embedder, dim = self._task_embedder(task)
            # Pass the task chain so a store built by a different embedder
            # is rebuilt AND backfilled — task stores otherwise only fill
            # incrementally (index_record at seal time), so without the
            # backfill a rebuilt store would silently lose all old blocks.
            self._task_indexes[task_name] = open_or_rebuild_index(
                Path(task["root"]) / "embeddings.sqlite", embedder, dim,
                chain=self.get_task_chain(task_name))
        return self._task_indexes[task_name]

    def ensure_artifacts_task(self) -> dict:
        """The reserved artifacts task (lazy-created on first upload).
        Holds the CONTENT of uploaded/pasted artifacts in its own chain
        and embedding store, keeping artifact text out of identity
        retrieval. Its source_root is ARTIFACTS_DIR, which makes the
        named file copies readable through the normal path gates."""
        task = self.registry.get(ARTIFACTS_TASK_NAME)
        if task is None:
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            objective = ("Reserved: uploaded and pasted artifacts "
                         "(content, provenance, embeddings)")
            task = self.registry.create(
                ARTIFACTS_TASK_NAME, objective, str(ARTIFACTS_DIR))
            # Open the continuum head block too (what task_open does) —
            # ingest refuses to seal into a never-opened task chain.
            self.get_task_continuum(ARTIFACTS_TASK_NAME).open_task(
                objective, items_total=None)
        return task

    # ----- write-approval support -----

    def task_ingest(self, task_name: str, file_path: str, finding: str = "",
                    content_hash: Optional[str] = None,
                    operation_id: Optional[str] = None) -> dict:
        """Ingest ONE live file into a task chain, idempotently when an
        operation_id is supplied. Returns {"ring_index": int|None, ...}."""
        if not task_name:
            return {"ring_index": None, "note": "no task — ingest skipped"}
        import continuum as continuum_mod
        cont = self.get_task_continuum(task_name)
        if operation_id:
            prior = continuum_mod.find_by_operation_id(
                self.get_task_chain(task_name), operation_id)
            if prior:
                return {"ring_index": prior[-1], "duplicate": True}
        live = resolve_read_path(self, file_path)   # PermissionError -> caller
        size = live.stat().st_size
        if size > MAX_INGEST_FILE_BYTES:
            raise ValueError(
                f"{live} is {size} bytes; the single-file ingest cap is "
                f"{MAX_INGEST_FILE_BYTES} (use task_ingest_path for trees, "
                f"or split the file)")
        text = live.read_text(encoding="utf-8", errors="replace")
        task = self._require_task(task_name)
        rel = file_path
        src = task.get("source_root") or ""
        try:
            if src:
                rel = str(live.resolve().relative_to(Path(src).resolve()))
        except ValueError:
            rel = live.name
        metadata = {
            "relative_path": rel,
            "file_content_hash": content_hash or pending_ops_mod.sha256_file(live),
            "extension": live.suffix.lower(),
        }
        if operation_id:
            metadata["operation_id"] = operation_id
        if cont.resume() is None:
            cont.open_task(task.get("objective") or f"task {task_name}",
                           items_total=None)
        # Open the index BEFORE sealing: the first open backfills the chain
        # (open_or_rebuild_index), so opening it after the seal would embed
        # the just-sealed records in the backfill AND in the loop below —
        # the double-embed that once doubled a multi-hour ingest.
        index = self.get_task_index(task_name)
        sealed, state = cont.ingest(rel, text, finding=finding or "ingested",
                                    metadata=metadata)
        for rec, _tokens in sealed:
            try:
                index.index_record(rec)
            except Exception:        # noqa: BLE001 — embedding is best-effort
                pass
        m = state["metrics"]
        self.registry.update_state(task_name, m["items_done"],
                                   m["items_total"] or m["items_done"])
        return {"ring_index": sealed[-1][0].index if sealed else None,
                "blocks": len(sealed)}

    def close(self):
        """Close all cached resources. Call on REPL exit / web shutdown.
        EmbeddingIndex owns a SQLite connection; task chains own DB handles."""
        for index in self._task_indexes.values():
            try:
                index.close()
            except Exception:        # noqa: BLE001
                pass
        self._task_indexes.clear()
        closed = set()
        for chain in self.task_chains.values():
            if id(chain) not in closed:
                try:
                    chain.close()
                except Exception:    # noqa: BLE001
                    pass
                closed.add(id(chain))
        self.task_chains.clear()
        self.task_recalls.clear()
        self.task_continuums.clear()


# ------------------------------------------------------- path safety (writes)

PROTECTED_SUFFIXES = (".sqlite", ".sqlite-wal", ".sqlite-shm", ".key")


def _resolve_allowed_path(ctx: AgentContext, raw_path: str,
                          verb: str) -> Path:
    """Resolve a path the model wants to touch, with symlink-escape
    protection. Reads AND writes must land inside the workspace root, a
    task source_root, or a task workspace — and never touch chain
    databases, key files, or pending-op state (a read of operator.key
    would hand the signing key to an external LLM; a write would corrupt
    the chain)."""
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = ctx.workspace_root / candidate
    resolved = Path(os.path.realpath(str(candidate)))
    if resolved.suffix in PROTECTED_SUFFIXES:
        raise PermissionError(
            f"refusing to {verb} chain database or key file: {resolved}")
    # data-dir internals (chains, registry, pending_ops) are off-limits;
    # only a task's workspace/ subdirectory inside it is reachable.
    data_root = ctx.data_dir.resolve()
    inside_data = resolved == data_root or data_root in resolved.parents
    if inside_data and "workspace" not in resolved.parts:
        raise PermissionError(
            f"refusing to {verb} inside the chain data directory: {resolved}")
    allowed = [ctx.workspace_root.resolve()]
    for _, task in ctx.registry.list_all():
        if task.get("source_root"):
            allowed.append(Path(task["source_root"]).resolve())
        allowed.append(Path(task["root"]).resolve() / "workspace")
    for base in allowed:
        try:
            resolved.relative_to(base)
            return resolved
        except ValueError:
            continue
    raise PermissionError(
        f"path {resolved} is outside the allowed {verb} roots "
        f"(workspace, task source roots, task workspaces)")


def resolve_write_path(ctx: AgentContext, raw_path: str) -> Path:
    return _resolve_allowed_path(ctx, raw_path, "write")


def resolve_read_path(ctx: AgentContext, raw_path: str) -> Path:
    return _resolve_allowed_path(ctx, raw_path, "read")


# ------------------------------------------------------------- tool schemas


TOOLS: list[dict] = [
    {
        "name": "read_file",
        "description": "Read a file (or line range) from the filesystem.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or workspace-relative path"},
                "start_line": {"type": "integer", "description": "Optional: first line (1-indexed)"},
                "end_line": {"type": "integer", "description": "Optional: last line (inclusive)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": ("Propose writing content to a file. Does NOT write: it returns a "
                        "pending_op_id and the USER must approve. After calling this, STOP "
                        "and show the user exactly what you intend to change."),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "change_summary": {"type": "string", "description": "One-line human-readable description of the change"},
            },
            "required": ["path", "content", "change_summary"],
        },
    },
    {
        "name": "task_open",
        "description": ("Create a new per-task continuum chain for a "
                        "long-horizon code task AND ingest the source tree "
                        "in the same call (default extensions .py/.md — one "
                        "turn, no separate ingest step). Set ingest=false "
                        "to open without ingesting."),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short slug for the task directory"},
                "objective": {"type": "string"},
                "source_root": {"type": "string", "description": "Absolute path to the live source repository. Required for audits and post-write verification."},
                "extensions": {"type": "array", "items": {"type": "string"},
                               "description": "File extensions to ingest, e.g. ['.py', '.md'] (the default)"},
                "ingest": {"type": "boolean", "description": "Set false to open the task without ingesting the tree"},
            },
            "required": ["name", "objective", "source_root"],
        },
    },
    {
        "name": "task_ingest_path",
        "description": "Walk a directory tree and ingest matching source files into the task chain as data-height blocks.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_name": {"type": "string"},
                "path": {"type": "string"},
                "extensions": {"type": "array", "items": {"type": "string"},
                               "description": "e.g. ['.py', '.md']"},
                "changed_only": {"type": "boolean", "description": "Skip files whose stored hash still matches"},
            },
            "required": ["task_name", "path"],
        },
    },
    {
        "name": "task_ingest_file",
        "description": "Ingest a single file into the task chain (e.g. after editing it). Requires user confirmation.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_name": {"type": "string"},
                "path": {"type": "string"},
                "finding": {"type": "string", "description": "One-line summary of what changed or was found"},
            },
            "required": ["task_name", "path", "finding"],
        },
    },
    {
        "name": "task_reembed",
        "description": ("Re-embed an entire task chain with the session "
                        "embedding model (e.g. Ollama) for true semantic "
                        "recall. Task stores use the instant hashing "
                        "embedder by default; only run this when the user "
                        "asks for semantic embeddings — it is SLOW on CPU "
                        "(can take minutes to hours for a large task) and "
                        "requires user confirmation."),
        "parameters": {
            "type": "object",
            "properties": {
                "task_name": {"type": "string"},
            },
            "required": ["task_name"],
        },
    },
    {
        "name": "task_resume",
        "description": "Re-hydrate the full task state from the head block. Use at session start or after a gap.",
        "parameters": {
            "type": "object",
            "properties": {"task_name": {"type": "string"}},
            "required": ["task_name"],
        },
    },
    {
        "name": "task_retrieve",
        "description": "Semantic search across an ingested task chain. Returns scored blocks with source paths and snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_name": {"type": "string"},
                "query": {"type": "string"},
                "path": {"type": "string", "description": "Optional: pin a specific file"},
                "dir": {"type": "string", "description": "Optional: scope to a directory"},
                "max_blocks": {"type": "integer",
                               "description": "How many blocks to return (default 16)"},
            },
            "required": ["task_name", "query"],
        },
    },
    {
        "name": "task_audit_source",
        "description": "Verify that task-chain code blocks still match the live file on disk (source-mismatch / revision-drift / dirty-worktree). Audit ONE block by block_index, or every block of a file by its relative path.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_name": {"type": "string"},
                "block_index": {"type": "integer"},
                "path": {"type": "string", "description": "Relative path as ingested; audits all blocks of that file. Supply this OR block_index."},
            },
            "required": ["task_name"],
        },
    },
    {
        "name": "task_validate",
        "description": "Check continuum invariants and chain integrity for a task.",
        "parameters": {
            "type": "object",
            "properties": {"task_name": {"type": "string"}},
            "required": ["task_name"],
        },
    },
    {
        "name": "task_fetch_block",
        "description": "Fetch the full content of specific task-chain blocks by ring index.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_name": {"type": "string"},
                "indices": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["task_name", "indices"],
        },
    },
    {
        "name": "list_tasks",
        "description": "List all open task chains with objectives and progress.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "resolve_task",
        "description": ("Resolve a user-provided task name. Returns exact match, candidates, or "
                        "not found. NEVER call this with a name you inferred — only with the name "
                        "the user literally provided. If ambiguous, ask the user; never choose."),
        "parameters": {
            "type": "object",
            "properties": {
                "name_hint": {"type": "string", "description": "The task name or fragment the USER provided"},
            },
            "required": ["name_hint"],
        },
    },
    {
        "name": "pin_file",
        "description": ("Pin a file path as the scoped working file for this turn. Subsequent "
                        "write_file calls are validated against it. Use when the user says "
                        "@filename or names an explicit file to edit."),
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "build_attachment",
        "description": ("Retrieve an EXISTING attachment (image, document) from the identity "
                        "chain by blob hash. Read-only; returns stored metadata and extracted "
                        "text. To ingest NEW content use ingest_blob."),
        "parameters": {
            "type": "object",
            "properties": {
                "blob_sha256": {"type": "string", "description": "SHA-256 hash from an attachment/file record — full hash preferred; a unique prefix (8+ hex chars, e.g. from a truncated display) also resolves"},
            },
            "required": ["blob_sha256"],
        },
    },
    {
        "name": "think_collapse",
        "description": ("Collapse multiple self-generated perspectives into a sealed winner. "
                        "Supply perspective summaries with PoQ scores; the winning synthesis is "
                        "sealed on the identity chain, rejected forks preserved."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "perspectives": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "summary": {"type": "string"},
                            "scores": {"type": "object", "description": "PoQ dims 0-255: coherence, relevance, novelty, consistency, depth, covenant"},
                        },
                        "required": ["name", "summary", "scores"],
                    },
                },
                "winner": {"type": "string", "description": "Optional explicit winner name"},
            },
            "required": ["query", "perspectives"],
        },
    },
    {
        "name": "ingest_blob",
        "description": ("Ingest pasted or uploaded content (image, text, "
                        "file). By default it lands in the reserved "
                        "'artifacts' chain (content searchable there, a "
                        "named file copy on disk, and a small pointer "
                        "record in the conversation) — NOT in the active "
                        "task. Pass task_name ONLY when the content "
                        "genuinely belongs to that open task's work."),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Base64-encoded content (binary) or raw text"},
                "name": {"type": "string", "description": "Human-readable name, e.g. 'screenshot-20260608.png'"},
                "mime_type": {"type": "string", "description": "MIME type: image/png, text/plain, application/pdf, …"},
                "description": {"type": "string", "description": "Optional: what this content shows or contains"},
                "encoding": {"type": "string", "enum": ["utf8", "base64"]},
                "task_name": {"type": "string", "description": "Optional: seal into this open task's chain + workspace instead of the artifacts chain (explicit opt-in)"},
            },
            "required": ["content", "name", "mime_type"],
        },
    },
    {
        "name": "defense_status",
        "description": ("Report self-defense posture: identity-chain integrity, "
                        "immune status (lockdown / quarantine / scars), consensus "
                        "quorum health, and antibody faculties grown from scars. "
                        "Read-only."),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

TOOL_MAP = {t["name"]: t for t in TOOLS}

# Per-call tool_use audit records (and their sanitize_audit /
# TOOL_AUDIT_FIELDS machinery) were removed in v1.4: the identity chain is
# ONE observation + ONE response per turn, skill-style — the response
# narrates the tool work, and tool EFFECTS (ingest blocks with hashes and
# git coordinates) live on the per-task continuum chains.


# --------------------------------------------- tolerant extraction (syntax)


TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def extract_tool_calls(model_text: str) -> tuple[list[dict], list[str]]:
    """TOLERANT extractor: recover every <tool_call> block in MODEL-generated
    text (never run this on tool results or file content). Strips markdown
    fences and trailing commas, then hands off to the STRICT validator.
    Returns (calls, parse_errors)."""
    calls, errors = [], []
    for m in TOOL_CALL_RE.finditer(model_text or ""):
        raw = FENCE_RE.sub("", m.group(1)).strip()
        raw = TRAILING_COMMA_RE.sub(r"\1", raw)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            errors.append(f"unparseable tool_call ({e}): {raw[:120]}")
            continue
        if isinstance(obj, dict):
            calls.append(obj)
        else:
            errors.append(f"tool_call is not a JSON object: {raw[:120]}")
    return calls, errors


def looks_like_intended_tool_call(model_text: str) -> bool:
    """True when the model clearly tried to call a tool but no block parsed —
    triggers the single reflective retry."""
    t = model_text or ""
    return ("<tool_call" in t or '"arguments"' in t) and not extract_tool_calls(t)[0]


# ----------------------------------------------- strict validation (contract)


_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def _validate_against(schema: dict, value: Any, where: str) -> Optional[str]:
    typ = schema.get("type")
    if typ and typ in _TYPE_CHECKS and not _TYPE_CHECKS[typ](value):
        return f"{where}: expected {typ}, got {type(value).__name__}"
    if typ == "object":
        # A schema that DECLARES properties (even {}) is closed: unknown
        # keys are rejected. One that omits them is an open object (e.g.
        # think_collapse's free-form `scores` dict).
        props = schema.get("properties")
        required = schema.get("required", [])
        if isinstance(value, dict):
            for r in required:
                if r not in value:
                    return f"{where}: missing required parameter {r!r}"
            if props is not None:
                for k, v in value.items():
                    if k not in props:
                        return f"{where}: unknown parameter {k!r}"
                    err = _validate_against(props[k], v, f"{where}.{k}")
                    if err:
                        return err
    if typ == "array" and isinstance(value, list):
        items = schema.get("items")
        if items:
            for i, v in enumerate(value):
                err = _validate_against(items, v, f"{where}[{i}]")
                if err:
                    return err
    return None


def validate_tool_call(call: dict) -> Optional[str]:
    """STRICT contract check. Returns an error string, or None if valid."""
    name = call.get("name")
    if isinstance(name, str) and name in USER_ONLY_TOOLS:
        return f"tool {name!r} is user-triggered only — the model may not call it"
    if not isinstance(name, str) or name not in TOOL_MAP:
        return f"unknown tool {name!r}"
    args = call.get("arguments", {})
    if not isinstance(args, dict):
        return f"{name}: arguments must be a JSON object"
    return _validate_against(TOOL_MAP[name]["parameters"], args, name)


# ---------------------------------------------------------- result handling


def escape_tool_markup(text: str) -> str:
    """Neutralize tool-call markers in any text that re-enters the prompt
    (tool results, file content) so it can never forge a call."""
    return (text or "").replace("<tool_call>", "&lt;tool_call&gt;") \
                       .replace("</tool_call>", "&lt;/tool_call&gt;")


TOOL_RESULT_ECHO_RE = re.compile(r"<tool_result\b[^>]*>.*?</tool_result>",
                                 re.DOTALL)
# An opening marker the segment never closes — a truncated echo. Suppress
# to the end rather than letting half a file through.
TOOL_MARKUP_TAIL_RE = re.compile(r"<tool_(?:call|result)\b[^>]*>.*\Z",
                                 re.DOTALL)
# A closing tag with no opener — models emit these as stray echo
# fragments; nothing pairs them once the passes above ran.
TOOL_STRAY_CLOSER_RE = re.compile(r"</tool_(?:call|result)>")
# Inline-code spans (single-line `...`). A tool tag INSIDE backticks is
# always prose discussing the syntax — when an agent audits THIS codebase
# it writes "a forged `<tool_call>`" — never a real call or an echoed
# result (those are raw, never backtick-fenced). Masked before stripping
# so the destructive tail-strip can't eat the rest of a sentence the
# moment the prose mentions a tag.
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


def strip_tool_markup(text: str) -> str:
    """The PROSE of a model segment: everything outside its tool markup,
    whitespace-tidied. Both tool loops use this on every round INCLUDING
    the final one before commit/display.

    Removes <tool_call> blocks (the calls the loop executed) AND
    <tool_result> blocks (transcript-continuation models like DeepSeek
    echo the results they were fed — whole files re-streamed at the user;
    the REAL results are already shown as cards in the UI). An unclosed
    marker is stripped to the end of the segment.

    Inline-code mentions of the tags (`` `<tool_call>` ``) are preserved:
    they are prose about the syntax, never real markup."""
    text = text or ""
    # Mask inline-code spans so a tag mentioned in prose survives.
    spans: list[str] = []

    def _mask(m):
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"

    masked = INLINE_CODE_RE.sub(_mask, text)
    prose = TOOL_CALL_RE.sub("", masked)
    prose = TOOL_RESULT_ECHO_RE.sub("", prose)
    prose = TOOL_MARKUP_TAIL_RE.sub("", prose)
    prose = TOOL_STRAY_CLOSER_RE.sub("", prose)
    # Restore the masked inline-code spans.
    prose = re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], prose)
    # Collapse the seams the removed blocks leave behind.
    prose = re.sub(r"\n{3,}", "\n\n", prose)
    return prose.strip()


def format_tool_result(name: str, result: str) -> str:
    body = escape_tool_markup(result)
    raw = body.encode("utf-8")
    if len(raw) > MAX_TOOL_RESULT_BYTES:
        body = raw[:MAX_TOOL_RESULT_BYTES].decode("utf-8", errors="replace") \
            + f"\n…[truncated at {MAX_TOOL_RESULT_BYTES} bytes]"
    return f'\n<tool_result name="{name}">\n{body}\n</tool_result>\n'


def tools_prompt(include: Optional[list[str]] = None) -> str:
    """Render the tool schemas + calling convention as system-prompt text."""
    chosen = [t for t in TOOLS if include is None or t["name"] in include]
    lines = [
        "You can call tools. To call one, emit EXACTLY this block (one JSON "
        "object per block, multiple blocks allowed):",
        "",
        "<tool_call>",
        '{"name": "<tool_name>", "arguments": {…}}',
        "</tool_call>",
        "",
        "Tool results come back in <tool_result> blocks. NEVER write "
        "<tool_result> blocks yourself and never repeat their contents "
        "back verbatim — the system provides them and the user can already "
        "see them. When you have enough information, answer normally with "
        "no <tool_call> blocks.",
        "",
        "BATCH your tool calls: every <tool_call> block in one response "
        "runs in the same round, and rounds are the scarce resource (each "
        "one is a full model round-trip against a growing prompt). When "
        "reads/retrievals are independent — several files, several "
        "queries — emit them ALL in one response instead of one per "
        "round. Go round-by-round only when a call's arguments depend on "
        "an earlier call's result.",
        "",
        "Your final answer is the ONLY record of this turn's tool work in "
        "your long-term memory — individual tool calls are not stored. "
        "Make it self-contained: name what you examined, what you found, "
        "and what you did, so your future self can rely on it.",
        "",
        "Available tools:",
    ]
    for t in chosen:
        lines.append(json.dumps(
            {"name": t["name"], "description": t["description"],
             "parameters": t["parameters"]}, ensure_ascii=False))
    return "\n".join(lines)


# --------------------------------------------------------------- executors


def execute_read_file(kwargs: dict, ctx: AgentContext) -> str:
    try:
        path = resolve_read_path(ctx, kwargs["path"])
    except PermissionError as e:
        return f"REFUSED: {e}"
    if not path.is_file():
        return f"ERROR: not a file: {path}"
    if path.stat().st_size > MAX_READ_BYTES:
        return (f"ERROR: {path} is {path.stat().st_size} bytes "
                f"(cap {MAX_READ_BYTES}). Read a line range instead.")
    text = path.read_text(encoding="utf-8", errors="replace")
    start, end = kwargs.get("start_line"), kwargs.get("end_line")
    if start or end:
        lines = text.splitlines()
        s = max(1, start or 1)
        e = min(len(lines), end or len(lines))
        text = "\n".join(f"{i}\t{l}" for i, l in
                         zip(range(s, e + 1), lines[s - 1:e]))
        return f"{path} lines {s}-{e} of {len(lines)}:\n{text}"
    return f"{path} ({len(text.splitlines())} lines):\n{text}"


def execute_write_file(kwargs: dict, ctx: AgentContext) -> str:
    try:
        target = resolve_write_path(ctx, kwargs["path"])
    except PermissionError as e:
        return f"REFUSED: {e}"
    if ctx.pinned_path and str(target) != str(Path(ctx.pinned_path)):
        return (f"REFUSED: turn is pinned to {ctx.pinned_path}; "
                f"write to {target} is out of scope (Tier 2).")
    if not ctx.active_task:
        # First durable action in this workspace: lazily mint its task
        # chain so the approved write's provenance ingest has somewhere to
        # land (without a task it is silently skipped). Best-effort — a
        # registry hiccup must not block the write proposal itself.
        try:
            ensure_workspace_task(ctx)
        except Exception:        # noqa: BLE001
            pass
    try:
        op = ctx.pending_ops.create(
            task_name=ctx.active_task or "",
            file_path=str(target),
            proposed_content=kwargs["content"],
            change_summary=kwargs["change_summary"],
        )
    except ValueError as e:
        return f"REFUSED: {e}"
    return json.dumps({
        "status": "confirmation_required",
        "pending_op_id": op.id,
        "task": op.task_name or "(no active task)",
        "file": op.file_path,
        "change": op.change_summary,
        "new_file": not op.target_existed,
        "expires_in_seconds": int(op.expires_at - op.created_at),
    }, indent=2)


def _walk_and_index(ctx: AgentContext, task_name: str, ingest_root: Path,
                    exts: list, changed_only: bool = False) -> str:
    """Shared walk → seal → index body for task_ingest_path AND task_open's
    auto-ingest — one implementation so the embed-once discipline cannot
    drift between the two entry points."""
    task = ctx._require_task(task_name)
    cont = ctx.get_task_continuum(task_name)
    # Open the index BEFORE the walk: the first open backfills the chain
    # (open_or_rebuild_index), so opening it after the walk would embed
    # every just-sealed record twice — once in the backfill, once in the
    # post-walk pass — doubling the cost of a first full-tree ingest.
    index = ctx.get_task_index(task_name)
    result = cont.walk(ingest_root, exts,
                       task.get("objective") or task_name,
                       changed_only=changed_only)
    files, per_file = result.files, result.results
    # index_chain (not a loop over result.sealed): it indexes exactly the
    # records missing from the store, which covers both the walk's data
    # blocks and the task_open state record walk seals outside
    # result.sealed — each embedded once, failures skipped.
    try:
        indexed = index.index_chain(ctx.get_task_chain(task_name))
    except Exception:                # noqa: BLE001 — embedding is best-effort
        indexed = 0
    m = result.state["metrics"] if result.state else {"items_done": 0,
                                                      "items_total": 0}
    ctx.registry.update_state(task_name, m["items_done"],
                              m["items_total"] or m["items_done"])
    ctx.active_task = task_name
    summary = ", ".join(f"{rel}({n})" for rel, n in per_file[:20])
    more = f" …+{len(per_file) - 20} more" if len(per_file) > 20 else ""
    return (f"Walked {len(files)} file(s), ingested {len(per_file)} "
            f"({sum(n for _, n in per_file)} blocks, {indexed} embedded): "
            f"{summary}{more}")


def execute_task_open(kwargs: dict, ctx: AgentContext) -> str:
    if str(kwargs.get("name", "")).strip().lower() == ARTIFACTS_TASK_NAME:
        return (f"ERROR: {ARTIFACTS_TASK_NAME!r} is a reserved task name "
                f"(the upload/paste artifacts chain, created lazily by "
                f"ingest_blob)")
    # source_root becomes an allowed read/ingest root, so validate it here
    # regardless of which entry point called us: it must be a real,
    # existing directory, and it is stored RESOLVED (symlinks flattened,
    # relative paths anchored at the workspace — NOT the process cwd) so
    # the boundary the user confirmed is the boundary that gets enforced.
    source_root = resolve_source_root(str(kwargs["source_root"]), ctx)
    if not source_root.is_dir():
        return (f"ERROR: source_root {kwargs['source_root']!r} is not an "
                f"existing directory")
    try:
        task = ctx.registry.create(kwargs["name"], kwargs["objective"],
                                   str(source_root))
    except TaskRegistryError as e:
        return f"ERROR: {e}"
    cont = ctx.get_task_continuum(task["name"])
    state, _rec = cont.open_task(kwargs["objective"], items_total=None)
    ctx.active_task = task["name"]
    opened = (f"Task '{task['name']}' opened (root {task['root']}). "
              f"Objective: {kwargs['objective']}.")
    if kwargs.get("ingest") is False:
        return f"{opened} Ingestion skipped (ingest=false)."
    # Open-and-ingest in ONE call: setup must not cost the user a second
    # turn. The walk is seconds with the hashing-embedder default, and a
    # failure leaves a perfectly usable open task.
    exts = kwargs.get("extensions") or [".py", ".md"]
    try:
        ingested = _walk_and_index(ctx, task["name"], source_root, exts)
    except Exception as e:           # noqa: BLE001 — task stays open
        return (f"{opened} Auto-ingest failed ({type(e).__name__}: {e}) — "
                f"run task_ingest_path to ingest manually.")
    return f"{opened} {ingested}"


def execute_task_ingest_path(kwargs: dict, ctx: AgentContext) -> str:
    ctx._require_task(kwargs["task_name"])
    try:
        # Recursive ingestion sends file contents to the LLM — bound it to
        # the same roots as reads (workspace, task source roots, task
        # workspaces), so /etc or $HOME can never be walked into a chain.
        ingest_root = resolve_read_path(ctx, kwargs["path"])
    except PermissionError as e:
        return f"REFUSED: {e}"
    exts = kwargs.get("extensions") or [".py", ".md"]
    return _walk_and_index(ctx, kwargs["task_name"], ingest_root, exts,
                           changed_only=bool(kwargs.get("changed_only")))


def execute_task_ingest_file(kwargs: dict, ctx: AgentContext) -> str:
    out = ctx.task_ingest(kwargs["task_name"], kwargs["path"],
                          finding=kwargs["finding"])
    ctx.active_task = kwargs["task_name"]
    return json.dumps(out)


def execute_task_reembed(kwargs: dict, ctx: AgentContext) -> str:
    """Rebuild a task's derived embedding store with the SESSION embedder
    (batched), and persist that choice so later sessions reopen the store
    with the same embedder instead of mismatch-deleting it back to the
    hashing default. The chain itself is never touched — the store is
    derived data."""
    import time as time_mod
    task = ctx._require_task(kwargs["task_name"])
    name = task["name"]
    if ctx.embedder is None:
        return ("REFUSED: no session embedder is configured — start the "
                "app with an embedding model (e.g. Ollama) to re-embed.")
    records = list(ctx.get_task_chain(name).iter_records())

    old = ctx._task_indexes.pop(name, None)
    if old is not None:
        old.close()
    from retrieval import open_or_rebuild_index
    db = Path(task["root"]) / "embeddings.sqlite"
    dim = ctx.embed_dim or getattr(ctx.embedder, "dim", 0)
    # force_rebuild: the user explicitly asked for a re-embed, so the
    # shared open-or-rebuild path (which owns the delete-and-reopen
    # mechanics, sidecars included) wipes unconditionally.
    index = open_or_rebuild_index(db, ctx.embedder, dim=dim,
                                  force_rebuild=True)

    def progress(done: int, total: int) -> None:
        print(f"  [reembed {name}] {done}/{total} chunks")

    t0 = time_mod.monotonic()
    stats = index.index_records_batched(records, progress=progress)
    elapsed = time_mod.monotonic() - t0
    ctx._task_indexes[name] = index
    ctx.registry.set_embedder(name, "session")
    ctx.active_task = name
    failed = stats.get("failed_records", 0)
    note = (f"; {failed} record(s) failed and remain unembedded"
            if failed else "")
    return (f"Re-embedded task '{name}': {stats['records']} records / "
            f"{stats['chunks']} chunks in {elapsed:.0f}s with "
            f"{type(ctx.embedder).__name__}{note}. Choice persisted — "
            f"future sessions reopen this store with the session embedder.")


def execute_task_resume(kwargs: dict, ctx: AgentContext) -> str:
    cont = ctx.get_task_continuum(kwargs["task_name"])
    st = cont.resume()
    if not st:
        return f"No continuum state on task {kwargs['task_name']!r} yet."
    ctx.active_task = kwargs["task_name"]
    m = st["metrics"]
    findings = "\n".join(f"  - {f}" for f in st.get("findings", [])[-6:])
    return (f"objective: {st['objective']}\n"
            f"cursor: {st['cursor']}\n"
            f"progress: {m['items_done']}/{m['items_total']} items, "
            f"{m['chunks_sealed']} blocks, ~{m['approx_tokens_ingested']} tokens\n"
            f"findings (last {min(6, st.get('findings_total', 0))}):\n{findings}\n"
            f"NEXT ACTION: {st['next_action']}")


def execute_task_retrieve(kwargs: dict, ctx: AgentContext) -> str:
    recall = ctx.get_task_recall(kwargs["task_name"])
    rings = recall.retrieve_path_aware(
        kwargs["query"],
        index=ctx.get_task_index(kwargs["task_name"]),
        path=kwargs.get("path"),
        dir=kwargs.get("dir"),
        # 16 (was 8): with the larger context budget, a fatter default
        # retrieval saves whole tool rounds — rounds, not result size,
        # are the scarce resource in the loop.
        max_blocks=kwargs.get("max_blocks") or 16,
    )
    if not rings:
        return "No matching blocks (is the task ingested and embedded?)."
    out = []
    for r in rings:
        data = r.get("payload", {}).get("data", {}) or {}
        snippet = str(data.get("content") or "")[:200].replace("\n", " ")
        out.append(f"[{r['index']:>4}] {data.get('relative_path', '?')}"
                   f":{data.get('line_start', '?')}-{data.get('line_end', '?')}"
                   f"  score={r.get('_final_score', 0):.3f}  {snippet}")
    return "\n".join(out)


def execute_task_audit_source(kwargs: dict, ctx: AgentContext) -> str:
    task = ctx._require_task(kwargs["task_name"])
    recall = ctx.get_task_recall(kwargs["task_name"])
    repo = task.get("source_root") or None
    if kwargs.get("block_index") is not None:
        v = recall.verify_source(kwargs["block_index"], repo=repo)
        return json.dumps(v, indent=2, default=str)
    if kwargs.get("path"):
        matches = recall.find_by_path(kwargs["path"])
        if not matches:
            return f"No continuum blocks found for path {kwargs['path']!r}."
        results = [recall.verify_source(m["index"], repo=repo)
                   for m in matches]
        return json.dumps(results, indent=2, default=str)
    return "ERROR: task_audit_source needs block_index or path"


def execute_task_validate(kwargs: dict, ctx: AgentContext) -> str:
    cont = ctx.get_task_continuum(kwargs["task_name"])
    ok, report = cont.validate()
    return "\n".join(report) + f"\nCONTINUUM: {'COHERENT' if ok else 'INCOHERENT'}"


def execute_task_fetch_block(kwargs: dict, ctx: AgentContext) -> str:
    recall = ctx.get_task_recall(kwargs["task_name"])
    blocks = recall.fetch(kwargs["indices"], budget_tokens=4000)
    out = []
    for b in blocks:
        mark = " …[truncated]" if b.get("truncated") else ""
        out.append(f"=== block {b['index']} ({b['type']}){mark}\n{b['content']}")
    return "\n".join(out) or "No blocks found for those indices."


def execute_list_tasks(kwargs: dict, ctx: AgentContext) -> str:
    pairs = ctx.registry.list_all()
    if not pairs:
        return "No tasks. Create one with task_open."
    lines = []
    for name, t in pairs:
        active = " *active*" if name == ctx.active_task else ""
        lines.append(f"{name} [{t['status']}] {t['items_done']}/{t['items_total']}"
                     f" — {t['objective'][:80]}{active}")
    return "\n".join(lines)


def execute_resolve_task(kwargs: dict, ctx: AgentContext) -> str:
    out = resolve_task(ctx.registry, kwargs["name_hint"])
    slim = dict(out)
    for key in ("task",):
        if key in slim:
            slim[key] = {k: slim[key][k] for k in
                         ("name", "objective", "status") if k in slim[key]}
    for key in ("candidates", "all_tasks"):
        if key in slim:
            slim[key] = [{k: t[k] for k in ("name", "objective", "status")
                          if k in t} for t in slim[key]]
    if out["status"] == "exact":
        ctx.active_task = out["task"]["name"]
    return json.dumps(slim, indent=2)


def execute_pin_file(kwargs: dict, ctx: AgentContext) -> str:
    path = Path(kwargs["path"])
    if not path.is_absolute():
        path = ctx.workspace_root / path
    ctx.pinned_path = os.path.realpath(str(path))
    return f"Pinned this turn's writes to {ctx.pinned_path}."


def execute_build_attachment(kwargs: dict, ctx: AgentContext) -> str:
    if ctx.identity_chain is None:
        return "ERROR: no identity chain available."
    sha = (kwargs["blob_sha256"] or "").strip().rstrip(".…")
    # Indexed O(1) via blob_index (covers file AND attachment records) —
    # never a full-chain scan, which would grow with uptime and run inside
    # the tool loop. Identical bytes share a sha, so "most recent record
    # wins" is the right resolution for a re-ingest.
    rec = ctx.identity_chain.find_file_by_sha(sha)
    if rec is None and len(sha) < 64:
        # Truncated hash (quoted from a display or old conversation):
        # resolve a unique prefix instead of dead-ending.
        rec = ctx.identity_chain.find_file_by_sha_prefix(sha)
    if rec is not None and isinstance(rec.content, dict):
        meta = {k: rec.content.get(k) for k in
                ("filename", "kind", "mime_type", "approx_bytes",
                 "artifact_path")
                if rec.content.get(k) is not None}
        text = str(rec.content.get("extracted_text") or "")[:8000]
        if not text and rec.content.get("artifact_rings"):
            # Pointer ring (v1.4.2 artifacts routing): the content lives
            # in the artifacts chain — fetch its blocks on demand.
            try:
                art_chain = ctx.get_task_chain(
                    rec.content.get("artifact_task") or ARTIFACTS_TASK_NAME)
                parts: list[str] = []
                for idx in rec.content["artifact_rings"]:
                    block = art_chain.get(idx)
                    data = (block.content.get("data")
                            if block is not None
                            and isinstance(block.content, dict) else None)
                    if isinstance(data, dict) and data.get("content"):
                        parts.append(str(data["content"]))
                    if sum(len(p) for p in parts) >= 8000:
                        break
                text = "\n".join(parts)[:8000]
            except Exception:    # noqa: BLE001 — pointer may outlive the
                pass             # artifacts dir; metadata still answers
        return json.dumps({"record_index": rec.index, **meta,
                           "extracted_text": text})
    return f"No attachment record found for blob {sha[:12]}…"


# (INGEST_BLOB_MAX_BYTES — the raw cap for ingest_blob — is defined next to
# MAX_INGEST_FILE_BYTES at the top of this module: one literal, two names.)

# The reserved artifacts task: uploaded/pasted content is sealed HERE by
# default — chunked and embedded in the artifacts chain's OWN store —
# never into whatever task happens to be active (which silently polluted
# unrelated, append-only task chains), and never as full content on the
# identity chain (where big extracted texts crowded retrieval and drowned
# more relevant rings). The identity chain gets one tiny pointer ring per
# upload instead.
ARTIFACTS_TASK_NAME = "artifacts"

# Named, user-browsable copies of uploaded files live here (the
# content-addressed blob store stays the CANONICAL bytes — names can be
# renamed or deleted by the user, hashes cannot). Registered as the
# artifacts task's source_root, so the normal path gates make it readable.
ARTIFACTS_DIR = Path(
    os.environ.get("ARTIFACTS_DIR", "~/.artifacts")).expanduser()


def resolve_blob_path(blob_root: Path, sha: str) -> Optional[Path]:
    """Locate a content-addressed blob by sha256: the sharded layout
    `blobs/<sha[:2]>/<sha>` (what ingest_blob writes), falling back to the
    legacy flat `blobs/<sha>` written by the removed file_ingest pipeline —
    pre-existing data dirs still hold their blobs there. The ONE place that
    knows the layout; serve_blob and the attachment collector both resolve
    through it. Returns None when the blob is absent in both layouts."""
    sharded = blob_root / sha[:2] / sha
    if sharded.exists():
        return sharded
    flat = blob_root / sha
    if flat.exists():
        return flat
    return None


def execute_ingest_blob(kwargs: dict, ctx: AgentContext) -> str:
    """Continuum-based content ingestion (Phase 14, rerouted in v1.4.2).

    DEFAULT: the reserved artifacts chain. Bytes go to the content-
    addressed blob store (canonical — vision and /blobs serve by sha)
    plus a named copy in ARTIFACTS_DIR; the CONTENT is chunked and
    embedded into the artifacts chain's own store; the identity chain
    gets ONE tiny pointer ring (filename/mime/sha + artifact ring refs,
    NO extracted text) so the conversation shows the upload without
    artifact content crowding identity retrieval.

    EXPLICIT `task_name`: seal into that task's chain + workspace (the
    old behavior) — but only ever by explicit opt-in. An ACTIVE task no
    longer captures uploads silently: that polluted unrelated append-only
    task chains and kept image bytes out of the vision path.

    Ingesting never creates a NORMAL task chain (only task_open does);
    the reserved artifacts task is the one lazy exception."""
    import base64
    import hashlib
    import os

    name = kwargs["name"]
    mime_type = kwargs["mime_type"]
    encoding = kwargs.get("encoding", "utf8")
    description = kwargs.get("description", "")
    task_name = (kwargs.get("task_name") or "").strip()
    if task_name.lower() == ARTIFACTS_TASK_NAME:
        task_name = ""               # explicit artifacts == the default

    if encoding == "base64":
        try:
            raw = base64.b64decode(kwargs["content"], validate=True)
        except Exception:                    # noqa: BLE001
            return "ERROR: content is not valid base64"
    else:
        raw = kwargs["content"].encode("utf-8")
    if not raw:
        return "ERROR: empty content"
    if len(raw) > INGEST_BLOB_MAX_BYTES:
        return (f"ERROR: content is {len(raw)} bytes; the ingest cap is "
                f"{INGEST_BLOB_MAX_BYTES} (write large files to disk and "
                f"ingest by path instead)")
    content_hash = hashlib.sha256(raw).hexdigest()
    is_text = mime_type.startswith("text/")
    # Format-aware extraction (extractors.py): PDFs, Office documents,
    # spreadsheets, and slide decks keep their text searchable instead of
    # silently becoming opaque blobs. Plain text passes through; unknown
    # binary yields "" (method "none"). Never raises.
    from extractors import extract_text as _extract_text
    extracted, extraction_method, extraction_truncated = _extract_text(
        raw, name, mime_type)

    # Sanitize the name FIRST — before any filesystem op — so a traversal
    # attempt (../../etc/passwd) collapses to its basename.
    safe_name = os.path.basename(name)
    if not safe_name or safe_name in (".", ".."):
        return f"ERROR: invalid attachment name {name!r}"
    source = "clipboard" if encoding == "base64" else "upload"
    finding = description or f"{mime_type}, {len(raw)} bytes"
    # Text pastes seal their CONTENT (chunked, self-labeled, searchable);
    # binary formats with an extractor (PDF, docx, xlsx, pptx…) seal
    # their EXTRACTED text; only truly opaque binary (and images, whose
    # extraction is just a metadata placeholder) falls back to the
    # finding — the bytes live on disk either way.
    if is_text:
        block_content = raw.decode("utf-8", "replace")
    elif extracted and not mime_type.startswith("image/"):
        block_content = extracted
    else:
        block_content = finding

    if task_name:
        # EXPLICIT task routing: store in the task workspace, seal a
        # continuum block in that task's chain.
        task = ctx._require_task(task_name)
        task_root = Path(task["root"])
        workspace = task_root / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        dest = workspace / safe_name
        dest.write_bytes(raw)

        continuum = ctx.get_task_continuum(task_name)
        # Open the index BEFORE sealing (same reason as task_ingest: the
        # first open backfills the chain, so opening it after the seal
        # would embed the new records twice).
        index = ctx.get_task_index(task_name)
        metadata = {
            "source": source,
            "mime_type": mime_type,
            "workspace_path": str(dest.relative_to(task_root)),
            "file_content_hash": content_hash,
            "approx_bytes": len(raw),
        }
        if extraction_method not in ("utf8", "none"):
            metadata["extraction_method"] = extraction_method
        sealed, _state = continuum.ingest(
            name=safe_name,
            content=block_content,
            finding=finding,
            metadata=metadata,
        )
        for rec, _tok in sealed:
            try:
                index.index_record(rec)
            except Exception:    # noqa: BLE001 — embedding is best-effort
                pass             # (the block is sealed; erroring here would
                                 # invite a duplicate re-ingest)
        first_ring = sealed[0][0].index if sealed else "?"
        ctx.active_task = task_name
        return (f"Ingested '{safe_name}' into task '{task_name}' "
                f"(ring {first_ring}, workspace/{safe_name}). {finding}")

    # DEFAULT: the artifacts route.
    if ctx.identity_chain is None:
        return "ERROR: no identity chain to attach to"

    # 1. Canonical bytes: the content-addressed blob store (what the
    #    vision path and /blobs/<sha> resolve against).
    blob_dir = ctx.data_dir / "blobs" / content_hash[:2]
    blob_dir.mkdir(parents=True, exist_ok=True)
    (blob_dir / content_hash).write_bytes(raw)

    # 2. Named, user-browsable copy in ARTIFACTS_DIR. A name collision
    #    with DIFFERENT content gets a short-sha suffix; identical
    #    content is left as-is (re-upload of the same file).
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARTIFACTS_DIR / safe_name
    if dest.exists() and hashlib.sha256(
            dest.read_bytes()).hexdigest() != content_hash:
        stem, suffix = os.path.splitext(safe_name)
        dest = ARTIFACTS_DIR / f"{stem}-{content_hash[:8]}{suffix}"
    if not dest.exists():
        dest.write_bytes(raw)

    # 3. Content into the reserved artifacts chain (its own embedding
    #    store — artifact text never enters identity retrieval).
    ctx.ensure_artifacts_task()
    continuum = ctx.get_task_continuum(ARTIFACTS_TASK_NAME)
    index = ctx.get_task_index(ARTIFACTS_TASK_NAME)   # before sealing
    metadata = {
        "source": source,
        "mime_type": mime_type,
        "artifact_path": str(dest),
        "file_content_hash": content_hash,
        "blob_sha256": content_hash,
        "approx_bytes": len(raw),
    }
    if extraction_method not in ("utf8", "none"):
        metadata["extraction_method"] = extraction_method
        if extraction_truncated:
            metadata["extraction_truncated"] = True
    sealed, _state = continuum.ingest(
        name=safe_name,
        content=block_content,
        finding=finding,
        metadata=metadata,
    )
    for rec, _tok in sealed:
        try:
            index.index_record(rec)
        except Exception:        # noqa: BLE001 — embedding is best-effort
            pass
    artifact_rings = [rec.index for rec, _tok in sealed]
    # NOTE: ctx.active_task is deliberately NOT touched — an upload must
    # never hijack the session's task cursor.

    # 4. ONE tiny pointer ring on the identity chain — filename, mime,
    #    sha, and where the content lives. NO extracted text: the
    #    pointer is a couple of embedding chunks of mostly filename, so
    #    it can surface in retrieval without crowding anything; the
    #    agent pulls content from the artifacts chain (build_attachment)
    #    or disk (read_file on artifact_path) on demand.
    pointer = {
        "filename": safe_name,
        "mime_type": mime_type,
        "blob_sha256": content_hash,
        "approx_bytes": len(raw),
        "source": source,
        "description": description or "",
        "artifact_task": ARTIFACTS_TASK_NAME,
        "artifact_rings": artifact_rings,
        "artifact_path": str(dest),
    }
    rec = ctx.identity_chain.append("attachment", pointer)
    ring_span = (f"rings {artifact_rings[0]}–{artifact_rings[-1]}"
                 if len(artifact_rings) > 1
                 else f"ring {artifact_rings[0]}" if artifact_rings
                 else "no rings")
    return (f"Attached '{safe_name}' ({mime_type}, {len(raw)} bytes): "
            f"content in the '{ARTIFACTS_TASK_NAME}' chain ({ring_span}), "
            f"file at {dest}, pointer record {rec.index}, "
            f"blob {content_hash[:12]}…. "
            f"{description or 'No description provided.'}")


def execute_defense_status(kwargs: dict, ctx: AgentContext) -> str:
    """Phase 13: one read-only snapshot of the whole defense posture."""
    chain = ctx.identity_chain
    if chain is None:
        return "ERROR: defense_status needs the identity chain"
    out: dict = {}
    try:
        ok, msg = chain.verify()
        out["chain"] = {"intact": ok, "detail": msg,
                        "length": chain.length()}
    except Exception as e:                   # noqa: BLE001
        out["chain"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        from immune import Immune
        s = Immune(chain).state()
        out["immune"] = {
            "locked": s["locked"],
            "safe_height": s["safe_height"],
            "quarantined_blocks": len(s["quarantine"]),
            "scars": len(s["scars"]),
            "scar_lessons": [sc.get("lesson", "") for sc in s["scars"]],
        }
    except Exception as e:                   # noqa: BLE001
        out["immune"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        from consensus import Quorum
        q = Quorum(chain)
        if q.is_initialized():
            ok, detail = q.verify()
            out["consensus"] = {"initialized": True, "quorum_ok": ok,
                                "detail": detail}
        else:
            out["consensus"] = {"initialized": False,
                                "note": "no quorum on this chain "
                                        "(run /consensus-init to harden)"}
    except Exception as e:                   # noqa: BLE001
        out["consensus"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        from cypher_commands import _faculty_dir
        from faculties import load_emergent
        emergent = load_emergent(_faculty_dir(None)).get("faculties", [])
        antibodies = [
            f["name"] for f in emergent
            if f.get("kind") == "sense" and any(
                "scar" in (h.get("context") or "")
                for h in f.get("history", []))
        ]
        out["antibodies"] = {"count": len(antibodies), "names": antibodies}
    except Exception as e:                   # noqa: BLE001
        out["antibodies"] = {"error": f"{type(e).__name__}: {e}"}
    return json.dumps(out, indent=2, default=str)


def execute_think_collapse(kwargs: dict, ctx: AgentContext) -> str:
    from chronosynaptic import ChronosynapticTree
    notes = {"query": kwargs["query"], "perspectives": kwargs["perspectives"]}
    tree = ChronosynapticTree(ctx.identity_chain)
    result, _rec = tree.collapse_explicit_notes(
        notes, query=kwargs["query"], winner=kwargs.get("winner"), do_seal=True)
    chosen = result.get("chosen") or {}
    rejected = result.get("rejected") or []
    return json.dumps({
        "chosen": chosen.get("name") if isinstance(chosen, dict) else chosen,
        "rejected": [p.get("name", str(p)) for p in rejected],
        "synthesis": str(result.get("synthesis", ""))[:500],
    })


EXECUTORS: dict[str, Callable[[dict, AgentContext], str]] = {
    "read_file": execute_read_file,
    "write_file": execute_write_file,
    "task_open": execute_task_open,
    "task_ingest_path": execute_task_ingest_path,
    "task_ingest_file": execute_task_ingest_file,
    "task_reembed": execute_task_reembed,
    "task_resume": execute_task_resume,
    "task_retrieve": execute_task_retrieve,
    "task_audit_source": execute_task_audit_source,
    "task_validate": execute_task_validate,
    "task_fetch_block": execute_task_fetch_block,
    "list_tasks": execute_list_tasks,
    "resolve_task": execute_resolve_task,
    "pin_file": execute_pin_file,
    "build_attachment": execute_build_attachment,
    "think_collapse": execute_think_collapse,
    "defense_status": execute_defense_status,
    "ingest_blob": execute_ingest_blob,
    # USER-triggered only — never offered to the model (see USER_ONLY_TOOLS):
    "approve_write": pending_ops_mod.execute_approve_write,
    "reject_write": pending_ops_mod.execute_reject_write,
}


def execute_tool(call: dict, ctx: AgentContext) -> str:
    """Validate + dispatch one MODEL-emitted tool call. Exceptions become
    error strings so the loop survives a misbehaving tool. User-only actions
    (approve/reject) are rejected here — use execute_user_action."""
    err = validate_tool_call(call)
    if err:
        return f"TOOL ERROR: {err}"
    name = call["name"]
    try:
        return EXECUTORS[name](call.get("arguments", {}), ctx)
    except Exception as e:           # noqa: BLE001 — surface, don't crash
        return f"TOOL ERROR: {name} raised {type(e).__name__}: {e}"


# ------------------------------------------------------------- workspace

# The workspace is the USER's choice of working directory — the read/write
# boundary the model operates inside. Selection is user-only (web endpoint
# or REPL command, never a model tool), so it is inherently confirmed:
# task_opens inside it run unconfirmed, writes resolve against it, and the
# lazy task chain (ensure_workspace_task) is named after it. Switching is a
# pure boundary move: nothing is created, sealed, or ingested by a switch.

_WORKSPACE_RECENT_MAX = 8


def _workspace_file(data_dir) -> Path:
    return Path(data_dir) / "workspace.json"


def load_workspace_choice(data_dir) -> tuple[Optional[str], list]:
    """(persisted current workspace or None, recent list) — tolerant of a
    missing or corrupt file; a bad workspace.json must never block boot."""
    try:
        data = json.loads(_workspace_file(data_dir).read_text(
            encoding="utf-8"))
        current = data.get("current")
        recent = [r for r in data.get("recent", []) if isinstance(r, str)]
        return (current if isinstance(current, str) else None, recent)
    except (OSError, ValueError):
        return None, []


def set_workspace(ctx: "AgentContext", path: str) -> str:
    """USER-only workspace switch. Validates server-side (the same truth
    the executors enforce), resolves symlinks, resets the active task (a
    task bound to the OLD boundary must not silently absorb work from the
    new one), and persists the choice so restarts keep it. Returns the
    resolved path; raises ValueError on a bad one."""
    resolved = Path(os.path.realpath(str(path or "")))
    if not resolved.is_dir():
        raise ValueError(
            f"{path!r} is not an existing directory on this machine")
    ctx.workspace_root = resolved
    ctx.active_task = None
    ctx.pinned_path = None
    try:
        _current, recent = load_workspace_choice(ctx.data_dir)
        entry = str(resolved)
        recent = [entry] + [r for r in recent if r != entry]
        _workspace_file(ctx.data_dir).parent.mkdir(parents=True,
                                                   exist_ok=True)
        _workspace_file(ctx.data_dir).write_text(json.dumps(
            {"current": entry, "recent": recent[:_WORKSPACE_RECENT_MAX]},
            indent=2), encoding="utf-8")
    except OSError:
        pass                     # persistence is best-effort; the switch holds
    return str(resolved)


def restore_workspace(ctx: "AgentContext") -> Optional[str]:
    """Boot-time restore of the persisted workspace choice. Silently keeps
    the default when nothing was saved or the saved directory no longer
    exists (a stale pointer must not wedge startup)."""
    saved, _recent = load_workspace_choice(ctx.data_dir)
    if saved and Path(saved).is_dir():
        ctx.workspace_root = Path(saved)
        return saved
    return None


def workspace_suggestions(ctx: "AgentContext") -> list:
    """Workspace candidates for the selector UI: every registered task's
    source_root plus recently chosen workspaces — no directory browsing,
    so the web session can never enumerate the filesystem."""
    roots = {t.get("source_root") for _, t in ctx.registry.list_all()
             if t.get("source_root")}
    _current, recent = load_workspace_choice(ctx.data_dir)
    return sorted(roots | set(recent))


def workspace_prompt(ctx: "AgentContext") -> str:
    """One system-prompt line so the model KNOWS its working directory
    instead of guessing at ~-expansions."""
    return (f"\nCurrent workspace (the user-selected working directory): "
            f"{ctx.workspace_root}")


def derive_task_slug(name: str) -> str:
    """A registry-legal slug from a directory name."""
    slug = re.sub(r"[^a-z0-9_.-]+", "-", (name or "").lower()).strip("-.")
    slug = re.sub(r"^[^a-z0-9]+", "", slug)[:64]
    return slug or "workspace"


def ensure_workspace_task(ctx: "AgentContext") -> dict:
    """Lazy task-chain creation: the chain appears at the FIRST action that
    needs durable task state (an approved write's provenance ingest), never
    on a workspace switch or read-only poking around. Reuses the active
    task, then any active task already bound to this workspace, and only
    then creates one named after the workspace directory."""
    if ctx.active_task:
        task = ctx.registry.get(ctx.active_task)
        if task:
            return task
    root = str(ctx.workspace_root.resolve())
    for _name, task in ctx.registry.list_all():
        if (task.get("source_root") == root
                and task.get("status") == "active"):
            ctx.active_task = task["name"]
            return task
    base = derive_task_slug(ctx.workspace_root.name)
    name, n = base, 2
    while ctx.registry.get(name) is not None:
        name, n = f"{base}-{n}", n + 1
    task = ctx.registry.create(name, f"Work in {root}", root)
    cont = ctx.get_task_continuum(name)
    cont.open_task(task["objective"], items_total=None)
    ctx.active_task = name
    return task


def precheck_gated_call(name: str, arguments: dict,
                        ctx: AgentContext) -> Optional[str]:
    """Cheap validity checks for a confirmation-gated call BEFORE a pending
    op is created. The executors validate too, but only at approval time —
    by then the user has already confirmed a card that was doomed when it
    was minted (a task_open on a directory that never existed). Failing
    eagerly hands the error straight back to the model, which can ask the
    user for the right path in the same turn. Returns an error string, or
    None when the call is worth deferring."""
    if not isinstance(arguments, dict):
        return "arguments must be an object"
    if name == "task_open":
        src = arguments.get("source_root")
        if not isinstance(src, str) or not src:
            return "task_open needs a source_root string"
        if not Path(os.path.realpath(src)).is_dir():
            return (f"source_root {src!r} is not an existing directory on "
                    f"this machine — ask the user where the repo lives")
    elif name in ("task_ingest_file", "task_reembed"):
        task_name = arguments.get("task_name") or ""
        if ctx.registry.get(task_name) is None:
            return (f"unknown task {task_name!r} — use list_tasks / "
                    f"resolve_task first")
        if name == "task_ingest_file":
            try:
                live = resolve_read_path(ctx, arguments.get("path") or "")
            except PermissionError as e:
                return str(e)
            if not live.is_file():
                return f"{arguments.get('path')!r} is not an existing file"
    return None


def defer_tool_call(call: dict, ctx: AgentContext) -> str:
    """Defer a confirmation-gated tool call as a pending operation — the
    stand-in for an inline confirm hook in loops that cannot prompt (the
    web/SSE turn loop). Returns the confirmation_required JSON handed to
    the model as the tool result, so it can tell the user accurately what
    to do (approve/reject the card) instead of guessing. Execution happens
    ONLY through the user-only approve path (execute_user_action ->
    approve_write -> _approve_tool_call)."""
    name = call.get("name", "?")
    arguments = call.get("arguments", {}) or {}
    problem = precheck_gated_call(name, arguments, ctx)
    if problem:
        return (f"ERROR: {problem} (no pending operation was created — "
                f"correct the call or ask the user)")
    try:
        op = ctx.pending_ops.create_tool_call(name, arguments)
    except ValueError as e:
        return f"REFUSED: {e}"
    return json.dumps({
        "status": "confirmation_required",
        "kind": "tool_call",
        "pending_op_id": op.id,
        "tool": name,
        "arguments": arguments,
        "expires_in_seconds": int(op.expires_at - op.created_at),
        "message": (f"{name} requires explicit user confirmation. A "
                    f"pending operation was created — ask the user to "
                    f"approve or reject it (the approval card's buttons, "
                    f"or /approve {op.id} typed in either interface). "
                    f"Do NOT retry the call; wait for their decision."),
    }, indent=2)


def execute_user_action(name: str, kwargs: dict, ctx: AgentContext) -> str:
    """REPL/web entrypoint for USER-triggered actions (approve_write,
    reject_write). This is the ONLY path that may run them — the model-side
    execute_tool refuses them by design."""
    if name not in USER_ONLY_TOOLS:
        return f"ERROR: {name!r} is not a user action"
    try:
        return EXECUTORS[name](kwargs, ctx)
    except Exception as e:           # noqa: BLE001
        return f"ERROR: {name} raised {type(e).__name__}: {e}"


def is_error_result(message: str) -> bool:
    """The ONE classifier for executor result strings. Executors signal
    failure with an ERROR:/REFUSED: prefix (and execute_tool's crash
    wrapper with TOOL ERROR:); every caller that needs an ok/failed signal
    must classify through here, never prefix-match the message itself —
    scattered copies already drifted once (a caller checking only "ERROR"
    silently passed "TOOL ERROR" results as success)."""
    return message.startswith(("ERROR", "REFUSED", "TOOL ERROR"))


def run_user_action(name: str, kwargs: dict,
                    ctx: AgentContext) -> tuple[bool, str]:
    """Structured form of execute_user_action: `(ok, message)`."""
    message = execute_user_action(name, kwargs, ctx)
    return not is_error_result(message), message
