"""
task_registry — the ledger of open continuum task chains (Safety Tier 1).

A lightweight JSON registry that tracks per-task chain directories under
`<data_dir>/tasks/<name>/`. It exists so the AGENT never has to guess which
task chain the user means: `resolve_task()` returns a deterministic
exact/ambiguous/not-found result and the calling convention (see the system
prompt) forbids the model from picking among candidates on its own.

Layout:

    <data_dir>/
    ├── chain.sqlite          # identity chain
    ├── tasks.json            # this registry
    └── tasks/                # per-task chain directories
        └── my-code-audit/
            ├── chain.sqlite
            ├── embeddings.sqlite
            └── operator.key

Contract rules:
  - `list_all()` returns list[tuple[str, dict]] — consistent with
    `resolve_task()` unpacking.
  - Task names match ^[a-z0-9][a-z0-9_.-]{0,63}$ and are unique
    case-insensitively.
  - All writes go through `_save_atomic()` (temp file + os.replace).
  - `repair()` reconciles tasks.json with the tasks/ directory.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskRegistryError(ValueError):
    """Invalid registry operation (bad slug, duplicate name, missing task)."""


class TaskRegistry:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.registry_path = self.data_dir / "tasks.json"
        self.tasks_dir = self.data_dir / "tasks"
        self._tasks: dict[str, dict] = self._load()

    # ----- persistence -----

    def _load(self) -> dict[str, dict]:
        """Load tasks.json. A MISSING file is a fresh start ({}); a file
        that exists but cannot be read or parsed is NOT — treating it as
        empty would let the next _save_atomic() overwrite it, silently
        losing every objective, source root, and status. Fail loudly and
        leave the corrupt file in place for the human to inspect."""
        if not self.registry_path.is_file():
            return {}
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise TaskRegistryError(
                f"task registry at {self.registry_path} exists but cannot be "
                f"loaded ({type(e).__name__}: {e}). Refusing to continue with "
                f"an empty registry — the next save would overwrite it. "
                f"Inspect and fix the file, or move it aside and run "
                f"repair() to re-register task directories from disk."
            ) from e
        tasks = data.get("tasks") if isinstance(data, dict) else None
        if not isinstance(tasks, dict):
            raise TaskRegistryError(
                f"task registry at {self.registry_path} parsed but has no "
                f"'tasks' object — malformed. Refusing to continue; inspect "
                f"the file or move it aside and run repair().")
        return dict(tasks)

    def _save_atomic(self) -> None:
        """Atomic save — never corrupts tasks.json on crash."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"tasks": self._tasks}, indent=2)
        fd, tmp = tempfile.mkstemp(
            dir=self.data_dir, prefix=".tasks.json.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, self.registry_path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # ----- CRUD -----

    def create(self, name: str, objective: str, source_root: str) -> dict:
        name = (name or "").strip()
        if not SLUG_RE.match(name):
            raise TaskRegistryError(
                f"invalid task name {name!r} — must match {SLUG_RE.pattern}")
        lower = name.lower()
        for existing in self._tasks:
            if existing.lower() == lower:
                raise TaskRegistryError(f"task {existing!r} already exists")
        task = {
            "name": name,
            "root": str(self.tasks_dir / name),
            "source_root": str(source_root),
            "objective": objective,
            "created_at": now_iso(),
            "status": "active",
            "items_total": 0,
            "items_done": 0,
        }
        (self.tasks_dir / name).mkdir(parents=True, exist_ok=True)
        self._tasks[name] = task
        self._save_atomic()
        return dict(task)

    def get(self, name: str) -> Optional[dict]:
        task = self._tasks.get(name)
        return dict(task) if task else None

    def list_all(self) -> list[tuple[str, dict]]:
        """Returns (name, task_dict) tuples — consistent with resolve_task
        unpacking."""
        return [(name, dict(task)) for name, task in self._tasks.items()]

    def update_state(self, name: str, items_done: int, items_total: int) -> dict:
        task = self._tasks.get(name)
        if task is None:
            raise TaskRegistryError(f"unknown task {name!r}")
        task["items_done"] = int(items_done)
        task["items_total"] = int(items_total)
        self._save_atomic()
        return dict(task)

    def set_embedder(self, name: str, embedder: str) -> dict:
        """Persist which embedder family the task's derived embedding store
        was built with: "hash" (the instant default) or "session" (the
        app-level embedder, e.g. Ollama, after an explicit task_reembed).
        get_task_index reads this on open, so a deliberately re-embedded
        store survives restarts instead of being deleted by the next
        embedder-mismatch check."""
        task = self._tasks.get(name)
        if task is None:
            raise TaskRegistryError(f"unknown task {name!r}")
        if embedder not in ("hash", "session"):
            raise TaskRegistryError(
                f"invalid embedder {embedder!r} — expected 'hash' or 'session'")
        task["embedder"] = embedder
        self._save_atomic()
        return dict(task)

    def mark_complete(self, name: str) -> dict:
        task = self._tasks.get(name)
        if task is None:
            raise TaskRegistryError(f"unknown task {name!r}")
        task["status"] = "complete"
        task["completed_at"] = now_iso()
        self._save_atomic()
        return dict(task)

    def repair(self) -> dict:
        """Reconcile tasks.json with tasks/ directory contents.

        Orphan directories (a chain on disk with no registry entry) are
        re-registered with a placeholder objective so they stay reachable;
        stale entries (registry points at a missing directory) are marked
        status='missing' rather than deleted — history is surfaced, not
        silently erased."""
        orphans, stale = [], []
        on_disk = set()
        if self.tasks_dir.is_dir():
            for child in sorted(self.tasks_dir.iterdir()):
                if child.is_dir() and (child / "chain.sqlite").is_file():
                    on_disk.add(child.name)
        for name in on_disk:
            if name not in self._tasks:
                self._tasks[name] = {
                    "name": name,
                    "root": str(self.tasks_dir / name),
                    "source_root": "",
                    "objective": "(recovered by repair — objective unknown; "
                                 "run task_resume to re-hydrate)",
                    "created_at": now_iso(),
                    "status": "active",
                    "items_total": 0,
                    "items_done": 0,
                }
                orphans.append(name)
        for name, task in self._tasks.items():
            if name not in on_disk and task.get("status") != "missing":
                task["status"] = "missing"
                stale.append(name)
        if orphans or stale:
            self._save_atomic()
        return {"orphans_recovered": orphans, "stale_marked": stale}


def resolve_task(registry: TaskRegistry, name_hint: str) -> dict:
    """
    Resolve a user-provided task name fragment to zero, one, or many matches.

    Returns:
        {"status": "exact", "task": <registry entry>}
        {"status": "ambiguous", "candidates": [<registry entries>]}
        {"status": "not_found", "all_tasks": [<registry entries>]}

    Never returns a single task for a fuzzy match — 'exact' requires the
    name_hint to be a case-insensitive exact match of an existing task name.
    """
    name_lower = (name_hint or "").strip().lower()

    for name, task in registry.list_all():
        if name.lower() == name_lower:
            return {"status": "exact", "task": task}

    candidates = []
    for name, task in registry.list_all():
        obj = (task.get("objective") or "").lower()
        if name_lower and (name_lower in name.lower() or name_lower in obj):
            candidates.append(task)

    if candidates:
        # Even a SINGLE fuzzy match stays ambiguous — never infer.
        return {"status": "ambiguous", "candidates": candidates}
    return {"status": "not_found",
            "all_tasks": [t for _, t in registry.list_all()]}
