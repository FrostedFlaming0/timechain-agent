"""
source_verify — re-check an ingested `file` record against the live file on
disk, with git awareness. Ported from cypher-tempre-self-model's
`recall.verify_source`, keyed off this repo's `file` records.

The repo seals `file` records carrying source coordinates (`source_path`,
`file_content_hash`), but never re-checks that a recalled file record still matches
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
    # `status` is "" (clean) or a non-empty listing (dirty); None on ERROR —
    # which must stay distinguishable from clean (dirty=None, not False), or
    # a failed `git status` would read as a clean worktree downstream.
    info["dirty"] = bool(status) if status is not None else None
    return info


def verify_live_file(path: str | Path, stored_hashes,
                     stored_commit: Optional[str] = None,
                     repo_path: str | Path | None = None,
                     include_text_hash: bool = False) -> dict:
    """The SHARED live-side verification ladder: path resolution → missing
    file → content-hash compare → git dirty/drift/unverifiable. Both
    verify_file_record (identity `file` records) and recall.Recall
    .verify_source (continuum blocks) verify through here, so a verdict
    or dirty-semantics change can never drift between them.

    `stored_hashes`: any one matching the live content passes (falsy
    entries ignored; no stored hash → no content check).
    `include_text_hash`: also compare/expose a hash of the file's TEXT
    decoded as utf-8 (continuum stores sha256_text of possibly-redacted
    text, so byte-hash equality alone would false-alarm).

    Returns a dict always carrying `verdict` and `source_path`, plus
    `live_sha256` (and `live_text_sha256` when requested) once the file
    was read, and the git fields once a stored commit was checked.
    """
    out: dict = {}
    live = Path(path)
    if not live.is_absolute() and repo_path is not None:
        live = Path(repo_path) / live
    out["source_path"] = str(live)
    if not live.is_file():
        out["verdict"] = "missing-source-file"
        return out

    raw = live.read_bytes()
    live_sha = hashlib.sha256(raw).hexdigest()
    out["live_sha256"] = live_sha
    live_hashes = [live_sha]
    if include_text_hash:
        text_sha = hashlib.sha256(
            raw.decode("utf-8", errors="replace").encode("utf-8")
        ).hexdigest()
        out["live_text_sha256"] = text_sha
        live_hashes.append(text_sha)
    stored = [h for h in (stored_hashes or []) if h]
    if stored and not any(h in live_hashes for h in stored):
        out["verdict"] = "source-mismatch"
        return out

    # Git awareness: only meaningful if the record captured git coords.
    if stored_commit:
        git = current_git_info(live)
        out["stored_git_commit"] = stored_commit
        out["live_git_commit"] = git.get("commit")
        if git.get("dirty"):
            out["verdict"] = "dirty-worktree"
            return out
        if git.get("commit") and git["commit"] != stored_commit:
            out["verdict"] = "revision-drift"
            return out
        if not git.get("commit") or git.get("dirty") is None:
            # The record pinned a commit but the live side can't be checked
            # (git missing, errored, or the dir is no longer a work tree).
            # The CONTENT hash above did match — say that, but never claim
            # full verification on a comparison that didn't run.
            out["content_match"] = True
            out["verdict"] = "git-unverifiable"
            return out

    out["verdict"] = "verified"
    return out


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

    result["stored_sha256"] = stored_sha
    result.update(verify_live_file(
        source_path, [stored_sha], content.get("git_commit"),
        repo_path=repo_path))
    if result["verdict"] == "missing-source-file":
        # Parity with the historical shape: a missing file never carried
        # the hash keys (it was never read).
        result.pop("stored_sha256", None)
    return result
