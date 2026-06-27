"""
continuum — long-horizon tasking via data-height blocks with full state refresh.
Ported from cypher-tempre-self-model/continuum.py onto this repo's signed chain.

A task can be far larger than any context window (an enterprise codebase, a
months-long investigation). Holding it all in context causes rot; forgetting it
causes drift. The Continuum turns the task into a self-validating chain of
bounded blocks:

  1. DATA-HEIGHT BOUND — each block ingests ONE chunk sized to a sweet-spot band
     (>= MIN so blocks hold real data, <= MAX so no single block can rot the
     context). Any size of task is tackled at constant granularity.
  2. FULL STATE REFRESH — each block carries the COMPLETE task state (objective,
     cursor, metrics, rolling findings, next action), not a diff. Reading the
     single HEAD block fully re-hydrates the task, so the agent can resume at any
     block — a new session, hours or weeks later — and know exactly where it is.

`validate` checks the running invariants (monotonic progress, one chunk per
block, non-decreasing tokens) on top of the chain's Ed25519 `verify()`.

The chunking, redaction, file-metadata, and validate-invariant logic are all
storage-independent and ported verbatim; only sealing/reading is re-pointed at
`Chain` through `ring_compat`. For big jobs use a PER-TASK chain (a separate
Chain/DB) so a large code audit doesn't dilute the identity chain.

Stdlib + cryptography only (numpy-free).
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ring_compat
# ONE hashing rule for content: pending_ops owns sha256_text (the write
# gate hashes proposed content with it; continuum blocks store
# file_content_hash with it) so approval-time checks and post-write audits
# can never drift on normalization. Re-exported here for existing callers.
from pending_ops import sha256_text  # noqa: F401


@dataclass
class WalkResult:
    """Return value of Continuum.walk(). Use the named attributes; `sealed`
    and `state` carry what the per-task embedding index needs
    (walk -> seal -> index_record)."""
    files: list = field(default_factory=list)            # discovered file paths
    results: list = field(default_factory=list)          # (relative_path, chunk_count)
    sealed: list = field(default_factory=list)           # (Record, token_count)
    state: Optional[dict] = None                         # refreshed task state


def find_by_operation_id(chain, operation_id: str) -> Optional[list]:
    """Ring indices of continuum blocks sealed with this operation_id, or
    None if the id has never been sealed. The write-approval flow calls this
    BEFORE ingesting so a crash-retry cannot double-seal (idempotency).

    Runs as an SQL-level json_extract lookup — this fires on EVERY
    approve_write (under the webapp's global lock), and the old
    implementation materialized the whole task chain through
    ring_compat.load_rings per call. Falls back to that scan only when
    the SQLite build lacks JSON1."""
    if not operation_id:
        return None
    try:
        hits = chain.find_indices_by_content_field(
            "$.data.operation_id", operation_id)
    except sqlite3.OperationalError:
        hits = [
            ring["index"]
            for ring in ring_compat.load_rings(chain,
                                               exclude_quarantined=False)
            if (ring.get("payload", {}).get("data") or {})
            .get("operation_id") == operation_id
        ]
    return hits or None


# Data-height band, measured in approximate tokens (~4 chars/token).
TARGET_TOKENS = 1024   # the sweet spot per block
MIN_TOKENS = 256       # below this, merge — blocks must hold real data
MAX_TOKENS = 1536      # hard ceiling — no single block may exceed this (anti-rot)
FINDINGS_WINDOW = 6    # rolling cap so the state refresh stays bounded
DEFAULT_SKIP_DIRS = {".git", ".hg", ".svn", ".venv", "__pycache__",
                     "node_modules", "vendor"}

LANGUAGE_BY_EXT = {
    ".c": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".go": "go",
    ".h": "c", ".hpp": "cpp", ".java": "java", ".js": "javascript",
    ".jsx": "javascript", ".json": "json", ".md": "markdown", ".py": "python",
    ".rb": "ruby", ".rs": "rust", ".sh": "shell", ".ts": "typescript",
    ".tsx": "typescript", ".yaml": "yaml", ".yml": "yaml",
}

SECRET_PATTERNS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
     "[REDACTED_PRIVATE_KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(
        r"(?i)\b((?:api|access|secret|private|auth|bearer|token|password|passwd|pwd)"
        r"[A-Za-z0-9_.-]*\s*[:=]\s*)(['\"]?)[^'\"\s]{8,}(['\"]?)"
    ), None),
]


def approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def chunk_text_with_lines(text: str, target=TARGET_TOKENS, min_=MIN_TOKENS, max_=MAX_TOKENS):
    """Split text into chunks and retain 1-based inclusive source line ranges."""
    chunks, cur, cur_start, cur_end = [], "", None, None

    def flush():
        nonlocal cur, cur_start, cur_end
        if cur:
            chunks.append({"content": cur, "line_start": cur_start or 1,
                           "line_end": cur_end or cur_start or 1})
            cur, cur_start, cur_end = "", None, None

    lines = text.splitlines(keepends=True)
    for line_no, ln in enumerate(lines, start=1):
        if approx_tokens(ln) > max_:                      # a single oversized line
            flush()
            step = max_ * 4
            for j in range(0, len(ln), step):
                chunks.append({"content": ln[j:j + step], "line_start": line_no,
                               "line_end": line_no})
            continue
        if cur and approx_tokens(cur + ln) > target:
            flush()
        if not cur:
            cur_start = line_no
        cur += ln
        cur_end = line_no
    flush()

    if not chunks:
        chunks.append({"content": "", "line_start": 1, "line_end": 1})
    if (len(chunks) >= 2 and approx_tokens(chunks[-1]["content"]) < min_
            and approx_tokens(chunks[-2]["content"] + chunks[-1]["content"]) <= max_):
        chunks[-2]["content"] += chunks[-1]["content"]
        chunks[-2]["line_end"] = chunks[-1]["line_end"]
        chunks.pop()
    return chunks


def chunk_text(text: str, target=TARGET_TOKENS, min_=MIN_TOKENS, max_=MAX_TOKENS):
    """Backward-compatible content-only chunking helper."""
    return [c["content"] for c in chunk_text_with_lines(text, target, min_, max_)]


def language_for_extension(ext: str):
    return LANGUAGE_BY_EXT.get(ext.lower())


def redact_secrets(text: str):
    """Return text with common secrets masked, plus the number of replacements."""
    total = 0
    out = text
    for pattern, replacement in SECRET_PATTERNS:
        if replacement is None:
            def repl(match):
                return f"{match.group(1)}{match.group(2)}[REDACTED_SECRET]{match.group(3)}"
            out, count = pattern.subn(repl, out)
        else:
            out, count = pattern.subn(replacement, out)
        total += count
    return out, total


def git_value(path: Path, *args):
    try:
        proc = subprocess.run(
            ["git", "-C", str(path)] + list(args),
            check=False, capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return None
    value = proc.stdout.strip()
    return value if proc.returncode == 0 and value else None


def git_info_for(path: Path):
    status = git_value(path, "status", "--porcelain")
    return {
        "git_commit": git_value(path, "rev-parse", "HEAD"),
        "git_branch": git_value(path, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "git_root": git_value(path, "rev-parse", "--show-toplevel"),
        "git_remote": git_value(path, "config", "--get", "remote.origin.url"),
    }


def path_role(relative_path: str, ext: str):
    parts = [p.lower() for p in relative_path.split("/") if p]
    name = parts[-1] if parts else ""
    stem = name.rsplit(".", 1)[0] if "." in name else name
    if any(p in {"generated", "gen", "dist", "build", "out", "target"} for p in parts):
        return "generated"
    if any(p in {"vendor", "third_party", "node_modules"} for p in parts):
        return "vendor"
    if any(p in {"test", "tests", "__tests__", "spec", "specs"} for p in parts):
        return "test"
    if stem.startswith("test_") or stem.endswith("_test") or stem.endswith(".test") or stem.endswith(".spec"):
        return "test"
    if any(p in {"doc", "docs", "documentation"} for p in parts) or ext in {".md", ".rst", ".txt"}:
        return "docs"
    if ext in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"} or name in {
        "makefile", "dockerfile", ".env", ".gitignore", ".dockerignore",
    }:
        return "config"
    if language_for_extension(ext):
        return "source"
    return "other"


def should_skip_file(path: Path, skip_dirs):
    parts = set(path.parts)
    return bool(parts & set(skip_dirs or ()))


def file_metadata(base_path: Path, file_path: Path, file_index: int, content: str,
                  git_info=None, redaction_count=0):
    rel = file_path.relative_to(base_path).as_posix()
    parts = rel.split("/")
    ext = file_path.suffix.lower()
    h = sha256_text(content)
    role = path_role(rel, ext)
    git_info = dict(git_info or {})
    return {
        "relative_path": rel,
        "filename": file_path.name,
        "file_index": file_index,
        "top_dir": parts[0] if len(parts) > 1 else "",
        "extension": ext,
        "language": language_for_extension(ext),
        "path_role": role,
        "is_test": role == "test",
        "is_generated": role == "generated",
        "git_commit": git_info.get("git_commit"),
        "git_branch": git_info.get("git_branch"),
        "git_dirty": git_info.get("git_dirty"),
        "git_root": git_info.get("git_root"),
        "git_remote": git_info.get("git_remote"),
        "file_content_hash": h,
        "redacted": redaction_count > 0,
        "redaction_count": redaction_count,
    }


class Continuum:
    """Long-horizon task ledger over a `Chain`. Pass a PER-TASK chain for big
    jobs (a separate Chain/DB) so the work-ledger stays out of the identity
    chain; the identity chain can seal one pointer record to the task head.
    """

    def __init__(self, chain, target=TARGET_TOKENS, min_=MIN_TOKENS, max_=MAX_TOKENS,
                 labeler=None):
        self.chain = chain
        self.target, self.min, self.max = target, min_, max_
        self._state = None       # cached rolling state across a walk -> no reload
        self._labeler = labeler  # optional callable(content)->labels dict (recall)

    def _rings(self):
        return ring_compat.load_rings(self.chain, exclude_quarantined=False)

    def _labels(self, content):
        """Self-label a chunk at ingest so retrieval reads sealed labels instantly.
        Uses the injected labeler if present, else the repo's recall.Recall;
        degrades to no labels if recall isn't importable in this environment."""
        if self._labeler is None:
            import recall
            self._labeler = recall.Recall(self.chain).label
        return self._labeler(content)

    def _head_state(self):
        for r in reversed(self._rings()):
            st = r.get("payload", {}).get("state")
            if st:
                return st
        return None

    def open_task(self, objective, items_total=None):
        state = {
            "objective": objective,
            "cursor": {"item_index": 0, "item": None, "chunk_index": 0, "chunk_of": 0},
            "metrics": {"items_total": items_total, "items_done": 0,
                        "chunks_sealed": 0, "approx_tokens_ingested": 0},
            "findings": [], "findings_total": 0,
            "next_action": "ingest first item",
            "data_height": {"target_tokens": self.target, "min_tokens": self.min,
                            "max_tokens": self.max},
        }
        rec = ring_compat.seal_ring(
            self.chain, "task_open",
            {"event": "task_open", "objective": objective, "state": state},
            source="tool")
        self._state = state
        return state, rec

    def ingest(self, name, content, finding=None, label=True, metadata=None):
        st = self._state if self._state is not None else self._head_state()
        if st is None:
            raise RuntimeError("No open task on this chain — run open_task first.")
        metadata = dict(metadata or {})
        rel_path = metadata.get("relative_path") or name
        chunks = chunk_text_with_lines(content, self.target, self.min, self.max)
        sealed = []
        for i, chunk in enumerate(chunks):
            ch = chunk["content"]
            file_content_hash = metadata.get("file_content_hash") or metadata.get("content_hash")
            st = json.loads(json.dumps(st))   # deep copy the prior state
            last = (i == len(chunks) - 1)
            st["cursor"] = {"item_index": st["cursor"]["item_index"] + (1 if i == 0 else 0),
                            "item": rel_path, "file_index": metadata.get("file_index"),
                            "chunk_index": i + 1, "chunk_of": len(chunks)}
            st["metrics"]["chunks_sealed"] += 1
            st["metrics"]["approx_tokens_ingested"] += approx_tokens(ch)
            if last:
                st["metrics"]["items_done"] += 1
                if finding:
                    st["findings"] = (st["findings"] + [f"{rel_path}: {finding}"])[-FINDINGS_WINDOW:]
                    st["findings_total"] += 1
                it = st["metrics"]["items_total"]
                done = st["metrics"]["items_done"]
                st["next_action"] = ("task complete" if (it and done >= it)
                                     else f"ingest next item (done {done}" + (f"/{it}" if it else "") + ")")
            else:
                st["next_action"] = f"continue ingesting {rel_path}: chunk {i + 2}/{len(chunks)}"
            data = {
                "item": rel_path,
                "relative_path": rel_path,
                "filename": metadata.get("filename") or Path(name).name,
                "file_index": metadata.get("file_index"),
                "chunk_index": i + 1,
                "chunk_of": len(chunks),
                "line_start": chunk["line_start"],
                "line_end": chunk["line_end"],
                "top_dir": metadata.get("top_dir"),
                "extension": metadata.get("extension") or Path(name).suffix.lower(),
                "language": metadata.get("language"),
                "path_role": metadata.get("path_role"),
                "is_test": metadata.get("is_test"),
                "is_generated": metadata.get("is_generated"),
                "git_commit": metadata.get("git_commit"),
                "git_branch": metadata.get("git_branch"),
                "git_dirty": metadata.get("git_dirty"),
                "git_root": metadata.get("git_root"),
                "git_remote": metadata.get("git_remote"),
                "content_hash": sha256_text(ch),
                "file_content_hash": file_content_hash,
                "operation_id": metadata.get("operation_id"),
                "redacted": bool(metadata.get("redacted")),
                "redaction_count": metadata.get("redaction_count", 0),
                "approx_tokens": approx_tokens(ch),
                "content": ch,
            }
            # Pass through custom metadata keys (Phase 14: mime_type,
            # workspace_path, source, approx_bytes, …). setdefault so a
            # custom key can never shadow the canonical chunk fields above.
            for k, v in metadata.items():
                data.setdefault(k, v)
            payload = {"event": "continuum", "task": st["objective"][:48],
                       "state": st, "data": data}
            if label:
                try:
                    payload["labels"] = self._labels(ch)   # self-label -> instant recall
                except Exception:
                    pass
            rec = ring_compat.seal_ring(self.chain, "continuum", payload, source="tool")
            sealed.append((rec, approx_tokens(ch)))
        self._state = st                       # cache rolling state -> next ingest needs no reload
        return sealed, st

    def resume(self):
        return self._head_state()

    def validate(self):
        ok, report = self.chain.verify()
        report = [report] if isinstance(report, str) else list(report)
        prev, sizes, heights, issues = None, [], [], []
        for r in self._rings():
            p = r.get("payload", {})
            if p.get("event") != "continuum":
                continue
            sizes.append(len(json.dumps(r)))
            h = p["data"]["approx_tokens"]
            heights.append(h)
            m = p["state"]["metrics"]
            if m["chunks_sealed"] == 1:        # first block of a (new) task segment
                prev = None                    # invariants are per-task
            if h > self.max:
                issues.append(f"ring {r['index']}: data-height {h} > max {self.max}")
            if prev:
                if m["items_done"] < prev["items_done"]:
                    issues.append(f"ring {r['index']}: items_done regressed")
                if m["chunks_sealed"] != prev["chunks_sealed"] + 1:
                    issues.append(f"ring {r['index']}: chunks_sealed not monotonic +1")
                if m["approx_tokens_ingested"] < prev["approx_tokens_ingested"]:
                    issues.append(f"ring {r['index']}: tokens ingested regressed")
            prev = m
        out = list(report)
        if heights:
            out.append(f"continuum blocks: {len(heights)}")
            out.append(f"data-height (tokens) min/avg/max: "
                       f"{min(heights)}/{sum(heights)//len(heights)}/{max(heights)}  "
                       f"(band {self.min}-{self.max})")
            out.append(f"block size (bytes)   min/avg/max: "
                       f"{min(sizes)}/{sum(sizes)//len(sizes)}/{max(sizes)}")
        out.append("invariant issues: " + "; ".join(issues) if issues
                   else "state invariants coherent: monotonic progress, +1 chunk/block, "
                        "every block within data-height band")
        return ok and not issues, out

    def latest_file_hashes(self):
        latest = {}
        for ring in self._rings():
            data = ring.get("payload", {}).get("data") or {}
            rel = data.get("relative_path")
            h = data.get("file_content_hash")
            if rel and h:
                latest[rel] = h
        return latest

    def walk(self, path, exts, objective, label=True,
             redact=True, changed_only=False, skip_dirs=None) -> "WalkResult":
        path = Path(path)
        skip_dirs = DEFAULT_SKIP_DIRS if skip_dirs is None else set(skip_dirs)
        files = sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix in exts
            and not should_skip_file(p.relative_to(path), skip_dirs)
        )
        prior_hashes = self.latest_file_hashes() if changed_only else {}
        planned = []
        for file_index, f in enumerate(files, start=1):
            text = f.read_text(errors="replace")
            rel = f.relative_to(path).as_posix()
            if changed_only and prior_hashes.get(rel) == sha256_text(text):
                continue
            planned.append((file_index, f, rel, text))
        git_info = git_info_for(path)
        # Reuse an already-open task instead of re-opening. open_task seals
        # a task_open ring, and walk used to call it unconditionally — so
        # every open-with-auto-ingest (and every later walk into the same
        # task) wrote a redundant task_open record 1s after the real one.
        # One task, one open ring: when state already exists, just extend
        # its metrics in memory (`objective` is the task's, not this
        # walk's) — every sealed block carries the refreshed state anyway.
        state = self._state if self._state is not None else self._head_state()
        if state is None:
            self.open_task(objective, items_total=len(planned))
        else:
            st = json.loads(json.dumps(state))   # never mutate sealed state
            m = st.setdefault("metrics", {})
            m["items_total"] = (m.get("items_done") or 0) + len(planned)
            st["next_action"] = f"ingest {len(planned)} item(s)"
            self._state = st
        results = []
        sealed_all = []
        state = self._state
        for file_index, f, rel, text in planned:
            sealed_text, redaction_count = redact_secrets(text) if redact else (text, 0)
            ndef = text.count("def "); ncls = text.count("class ")
            finding = f"{text.count(chr(10)) + 1} lines, {ndef} defs, {ncls} classes"
            if redaction_count:
                finding += f", {redaction_count} secret(s) redacted"
            meta = file_metadata(path, f, file_index, text, git_info=git_info,
                                 redaction_count=redaction_count)
            sealed, state = self.ingest(rel, sealed_text, finding=finding,
                                        label=label, metadata=meta)
            sealed_all.extend(sealed)
            results.append((rel, len(sealed)))
        return WalkResult(files=files, results=results,
                          sealed=sealed_all, state=state)
