"""
source_verify — re-check an ingested `file` record against the live file on
disk, with git awareness. Ported from cypher-tempre-self-model's
`recall.verify_source`, keyed off this repo's `file` records.

The repo ingests files (`file_ingest.py`) and stores `blob_sha256` + the
extracted text, but never re-checks that a recalled file record still matches
what is on disk now. Acting on stale ingested code is a real failure mode;
this module catches it. It is read-only and additive — it adds no record
types and changes no storage.

Verdicts (in `verify_file_record(...)["verdict"]`):
    verified            live file matches the ingested bytes (and git coords, if any)
    source-mismatch     live file's sha256 differs from what was ingested
    revision-drift      same path, but the git commit moved since ingest
    dirty-worktree      git worktree has uncommitted changes at verify time
    missing-source-file the recorded source path no longer exists
    no-source-path      the record predates source-coordinate capture
    not-a-file-record   the record exists but is not a `file` record
    missing-ring        no record at that index

Stdlib only (uses `subprocess` for git, like the skill).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Optional


def current_git_info(path: str | Path) -> dict:
    """Return `{commit, branch, dirty}` for the git repo containing `path`,
    or `{}` if `path` is not inside a git worktree / git is unavailable.

    Pure stdlib `subprocess`; never raises — a non-repo or missing git just
    yields an empty dict so callers treat the file as un-versioned.
    """
    p = Path(path)
    cwd = str(p if p.is_dir() else p.parent)

    def _git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        return out.stdout.strip()

    inside = _git("rev-parse", "--is-inside-work-tree")
    if inside != "true":
        return {}
    info: dict = {}
    commit = _git("rev-parse", "HEAD")
    if commit:
        info["commit"] = commit
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch:
        info["branch"] = branch
    status = _git("status", "--porcelain")
    # `status` is "" (clean) or a non-empty listing (dirty); None on error.
    info["dirty"] = bool(status)
    return info


def _get_record(chain, index: int):
    """Fetch the single record at `index`, or None. Prefers the chain's O(1)
    `get(index)`; falls back to a bounded `iter_records` scan for duck-typed
    chains that don't expose `get`."""
    getter = getattr(chain, "get", None)
    if callable(getter):
        try:
            return getter(index)
        except Exception:
            pass
    try:
        return next(iter(chain.iter_records(start=index, end=index + 1)), None)
    except Exception:
        return None


def verify_file_record(chain, record_index: int, repo_path: str | Path | None = None) -> dict:
    """Re-check the ingested `file` record at `record_index` against disk.

    `repo_path`, if given, is the root to resolve a relative `source_path`
    against (and the repo whose git HEAD to compare). Returns a verdict dict
    (see module docstring) always carrying `verdict` and `record_index`.
    """
    result: dict = {"record_index": record_index}
    rec = _get_record(chain, record_index)
    if rec is None:
        result["verdict"] = "missing-ring"
        return result
    if getattr(rec, "type", None) != "file":
        result["verdict"] = "not-a-file-record"
        result["record_type"] = getattr(rec, "type", None)
        return result

    content = rec.content if isinstance(rec.content, dict) else {}
    result["filename"] = content.get("filename")
    stored_sha = content.get("file_content_hash") or content.get("blob_sha256")
    source_path = content.get("source_path")
    if not source_path:
        result["verdict"] = "no-source-path"
        return result

    live = Path(source_path)
    if not live.is_absolute() and repo_path is not None:
        live = Path(repo_path) / live
    if not live.is_file():
        result["verdict"] = "missing-source-file"
        result["source_path"] = str(live)
        return result

    # Content check: live bytes vs ingested bytes.
    live_sha = hashlib.sha256(live.read_bytes()).hexdigest()
    result["source_path"] = str(live)
    result["stored_sha256"] = stored_sha
    result["live_sha256"] = live_sha
    if stored_sha and live_sha != stored_sha:
        result["verdict"] = "source-mismatch"
        return result

    # Git awareness: only meaningful if the record captured git coords.
    stored_commit = content.get("git_commit")
    if stored_commit:
        git = current_git_info(live)
        result["stored_git_commit"] = stored_commit
        result["live_git_commit"] = git.get("commit")
        if git.get("dirty"):
            result["verdict"] = "dirty-worktree"
            return result
        if git.get("commit") and git["commit"] != stored_commit:
            result["verdict"] = "revision-drift"
            return result

    result["verdict"] = "verified"
    return result
