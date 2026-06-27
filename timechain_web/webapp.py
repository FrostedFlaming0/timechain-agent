"""
webapp — FastAPI frontend for the timechain agent.

Wraps the same Agent / Chain / Retriever stack that run.py uses, and serves
a single-page browser UI. The chain stays single-writer because there's
still one process (this one) holding the signing key and appending records.

What you get:
  - Streaming responses (Server-Sent Events) when the LLM client supports it,
    falling back to non-streaming otherwise.
  - File ingestion via drag-and-drop or file picker.
  - Image rendering for ingested file blobs (served from /blobs/<sha256>).
  - Sidebar showing recent reflections + revisions.
  - All slash commands from run.py: /verify, /length, /seal, /sysprompt,
    /reflect, /cambium, /proposals, /revise N <text>, /file <path>.
  - Single-session lock: only one browser tab is "active" at a time. A second
    tab can take over, but they don't run concurrently — protects the chain's
    single-writer guarantee.

What's NOT included on purpose:
  - Authentication. This binds to 127.0.0.1 by default. Don't expose it on
    a network without putting auth in front of it; the operator key lives
    in this process.
  - Multi-user. One operator, one chain, one signing key.
  - Background reflection and Cambium cadence — these run the same way as
    run.py: reflection every AUTO_REFLECT_EVERY turns, Cambium every
    AUTO_CAMBIUM_EVERY turns (a separate, longer counter).

Run:
    pip install fastapi uvicorn sse-starlette python-multipart
    python timechain_web/webapp.py

Then open the URL it prints (default http://127.0.0.1:8765).
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import secrets
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    sys.exit(
        "webapp requires fastapi, uvicorn, sse-starlette, python-multipart.\n"
        "install with: pip install fastapi uvicorn sse-starlette python-multipart"
    )

# Reuse run.py's configuration verbatim — same chain, same provider, same
# system prompt. The web UI is just an alternative I/O layer.
from chain import Chain, load_or_create_key
from retrieval import EmbeddingIndex, Retriever
from agent import Agent, ProtectedZoneError
from run import (
    DATA_DIR,
    LLM_PROVIDER,
    FOUNDING_COMMITMENTS,
    SYSTEM_PROMPT,
    SEMANTIC_K,
    RECENT_N,
    OLLAMA_EMBED_MODEL,
    AUTO_REFLECT_EVERY,
    AUTO_CAMBIUM_EVERY,
    MAX_CAMBIUM_RECORDS,
    PER_TURN_MODALITY_CAP,
    SPROUTED_MODALITIES_FILE,
    CONTEXT_BUDGET_CHARS,
    LLM_MAX_TOKENS,
    # v1.2 genesis identity fields. The webapp MUST pass these to
    # commit_genesis so a chain bootstrapped here ends up identical to
    # one bootstrapped from the REPL. Genesis is sealed at first run; if
    # these are omitted, the chain has no agent_name/purpose/covenant at
    # record 0 — forever.
    AGENT_NAME,
    AGENT_PURPOSE,
    AGENT_COVENANT,
    build_llm,
    make_tiered_embedder,
)


# ---------------------------------------------------------------------------
# Server config
# ---------------------------------------------------------------------------

HOST = "127.0.0.1"
PORT = 8765
STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# AppState — a single shared instance carries the agent, session lock, etc.
# ---------------------------------------------------------------------------

class AppState:
    """
    Holds the long-lived agent and a single-session lock.

    The lock is intentionally simple: one token is "active" at a time. Any
    request without that token, or with a stale one, gets a polite 409 and
    a hint to take over. Take-over is explicit (the user clicks a button)
    so a second tab opening doesn't silently steal the session.
    """

    def __init__(self) -> None:
        self.chain: Optional[Chain] = None
        self.index: Optional[EmbeddingIndex] = None
        self.agent: Optional[Agent] = None
        # Serialize all chain-touching work; the chain assumes single-writer.
        self.lock = asyncio.Lock()
        # Session token — only one tab is "active" at a time.
        self.active_token: Optional[str] = None
        # Counter for auto-reflection (mirrors run.py's behavior).
        self.turns_since_reflect = 0
        # Counter for auto-Cambium — a separate, longer cadence than
        # auto-reflection (mirrors run.py's AUTO_CAMBIUM_EVERY behavior).
        self.turns_since_cambium = 0

    def boot(self) -> None:
        """Set up chain, index, agent. Idempotent — safe to call once at startup."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        chain_db = DATA_DIR / "chain.sqlite"
        embed_db = DATA_DIR / "embeddings.sqlite"
        key_path = DATA_DIR / "operator.key"
        blob_dir = DATA_DIR / "blobs"
        blob_dir.mkdir(parents=True, exist_ok=True)

        print(f"[boot] data dir: {DATA_DIR}")
        print(f"[boot] llm provider: {LLM_PROVIDER}")
        print(f"[boot] max response tokens: {LLM_MAX_TOKENS}  |  "
              f"context budget: {CONTEXT_BUDGET_CHARS:,} chars")
        print("[boot] setting up chain, embedder, and agent...")

        key = load_or_create_key(key_path)
        self.chain = Chain(chain_db, key)

        # Resolve the embedder with the same tiered fallback run.py uses:
        # Ollama if reachable, HashingEmbedder otherwise. Never aborts.
        embedder, embed_dim, embed_name = make_tiered_embedder()
        if embed_name == "hashing-fallback":
            print(f"[boot] embedder: {embed_name} ({embed_dim}-dim, lexical only)")
            print(f"[boot] no Ollama server reachable — retrieval will be lexical.")
            print(f"[boot] for semantic recall: ollama pull {OLLAMA_EMBED_MODEL}")
        else:
            print(f"[boot] embedder: {embed_name} ({embed_dim}-dim, semantic)")

        try:
            self.index = EmbeddingIndex(embed_db, embedder, dim=embed_dim)
        except ValueError as e:
            # Embedding store built with a different embedder. The chain is
            # intact; only the derived index is stale. Surface the rebuild
            # instructions and abort boot — the server can't serve sensible
            # retrieval against a mismatched store.
            self.chain.close()
            raise RuntimeError(f"embedding store mismatch — {e}") from e

        added = self.index.index_chain(self.chain)
        if added:
            print(f"[boot] indexed {added} pre-existing records")

        import retrieval as _retrieval
        from sprouted_modalities import SproutRegistry
        _retrieval.PER_TURN_MODALITY_CAP = PER_TURN_MODALITY_CAP
        sprout_registry = SproutRegistry.load(SPROUTED_MODALITIES_FILE)
        if sprout_registry.names():
            print(f"[boot] sprouted modalities: {', '.join(sprout_registry.names())}")
        retriever = Retriever(self.chain, self.index, sprout_registry=sprout_registry)
        llm = build_llm()
        self.agent = Agent(
            self.chain, retriever, llm,
            system_prompt=SYSTEM_PROMPT, blob_dir=blob_dir,
            context_char_budget=CONTEXT_BUDGET_CHARS,
        )

        # First-run genesis (mirrors run.py exactly).
        if self.chain.length() == 0:
            print("[boot] first run — committing genesis")
            # Pass the same v1.2 identity fields run.py does — agent_name,
            # purpose, and (optional) covenant. Without these, a webapp-
            # bootstrapped chain has a stripped-down genesis record forever,
            # since genesis is sealed at first commit.
            genesis = self.agent.commit_genesis(
                FOUNDING_COMMITMENTS,
                agent_name=AGENT_NAME,
                purpose=AGENT_PURPOSE,
                covenant=AGENT_COVENANT,
            )
            self.index.index_record(genesis)
        else:
            drift = self.agent.check_genesis_drift(FOUNDING_COMMITMENTS)
            if drift and drift["status"] == "drift":
                print("[boot] WARNING: genesis drift detected — sealed commitments differ from config")

        # Log system prompt change if any (also mirrors run.py).
        sp_record = self.agent.log_system_prompt()
        if sp_record:
            self.index.index_record(sp_record)
            print(f"[boot] logged system prompt change at index {sp_record.index}")

        print(f"[boot] chain length: {self.chain.length()}")
        print(f"[boot] ready at http://{HOST}:{PORT}")

    def shutdown(self) -> None:
        if self.chain is not None:
            try: self.chain.close()
            except Exception: pass
        if self.index is not None:
            try: self.index.close()
            except Exception: pass


state = AppState()


# ---------------------------------------------------------------------------
# Lifespan — boot/shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    state.boot()
    try:
        yield
    finally:
        state.shutdown()


app = FastAPI(lifespan=lifespan, title="Timechain Agent")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_session(request: Request) -> str:
    """Return the session token if the request owns the active session; else 409."""
    token = request.headers.get("x-session-token") or request.query_params.get("session")
    if token and state.active_token == token:
        return token
    raise HTTPException(
        status_code=409,
        detail={
            "error": "session_inactive",
            "message": (
                "another browser tab holds the active session, or you haven't "
                "claimed one yet. POST /api/session/claim to take over."
            ),
        },
    )


def _record_to_dict(rec) -> dict:
    return {
        "index": rec.index,
        "type": rec.type,
        "timestamp": rec.timestamp,
        "timestamp_iso": datetime.fromtimestamp(
            rec.timestamp / 1000, tz=timezone.utc
        ).isoformat(),
        "content": rec.content,
        "record_hash": rec.record_hash,
    }


def _cambium_result_to_dict(result: dict) -> dict:
    """
    Flatten an Agent.run_cambium() result ({proposals, recurrences,
    escalations} of Records) into a JSON-friendly summary for the UI.
    """
    return {
        "proposals": [
            {
                "index": r.index,
                "proposal_kind": r.content.get("proposal_kind", "?"),
                "title": r.content.get("title", ""),
            }
            for r in result.get("proposals", [])
        ],
        "recurrences": [
            {
                "index": r.index,
                "recurs_proposal_index": r.content.get("recurs_proposal_index"),
            }
            for r in result.get("recurrences", [])
        ],
        "escalations": [
            {
                "index": r.index,
                "marks_proposal_index": r.content.get("marks_proposal_index"),
                "recurrence_count": r.content.get("recurrence_count"),
            }
            for r in result.get("escalations", [])
        ],
    }


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

@app.get("/api/session")
async def session_status(request: Request):
    token = request.headers.get("x-session-token") or request.query_params.get("session")
    return {
        "active_token_present": state.active_token is not None,
        "you_are_active": bool(token and state.active_token == token),
    }


# Rate-limit state for /api/session/claim. Without this, anyone with
# reach to the port could spam claim requests and bump the active tab
# off the chain repeatedly — a one-line denial-of-service against the
# documented "single operator" model. Per-IP, decaying counter; a real
# operator only claims once per tab open.
_claim_limit: dict[str, list[float]] = {}
_CLAIM_MAX_PER_MINUTE = 6


def _claim_rate_limit_ok(ip: str) -> bool:
    """
    True iff this IP has made fewer than _CLAIM_MAX_PER_MINUTE claim
    attempts in the past 60 seconds. The window is sliding: timestamps
    older than 60s are evicted from the per-IP list on every call. The
    dict grows to at most one entry per unique IP that has ever
    claimed; in single-operator use that is a handful of entries.
    """
    now = time.time()
    window = _claim_limit.setdefault(ip, [])
    # Evict expired timestamps. The list is small (capped by the limit).
    cutoff = now - 60.0
    while window and window[0] < cutoff:
        window.pop(0)
    if len(window) >= _CLAIM_MAX_PER_MINUTE:
        return False
    window.append(now)
    return True


@app.post("/api/session/claim")
async def session_claim(request: Request):
    """
    Mint a fresh token. Any previously active tab is bumped.

    Rate-limited per source IP (`_CLAIM_MAX_PER_MINUTE` per 60 seconds)
    so an attacker who can reach the port cannot bump the active tab
    repeatedly. The single-operator model assumes a low ceiling on
    legitimate claims — opening a fresh tab a few times an hour at
    most. Six per minute gives plenty of headroom for a real operator
    reloading the page while making claim-storm DoS impractical.
    """
    client_ip = request.client.host if request.client else "unknown"
    if not _claim_rate_limit_ok(client_ip):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limited",
                "message": (
                    f"too many session claims from this IP "
                    f"(max {_CLAIM_MAX_PER_MINUTE} per minute)"
                ),
            },
        )
    new = secrets.token_urlsafe(24)
    state.active_token = new
    return {"session_token": new}


# ---------------------------------------------------------------------------
# Chain inspection — read-only, no session required
# ---------------------------------------------------------------------------

@app.get("/api/chain/status")
async def chain_status():
    head = state.chain.head()
    return {
        "length": state.chain.length(),
        "head_index": head.index if head else None,
        "head_hash": head.record_hash if head else None,
        "pubkey": state.chain.pubkey_hex,
        "data_dir": str(DATA_DIR),
        "llm_provider": LLM_PROVIDER,
    }


@app.get("/api/chain/recent")
async def chain_recent(n: int = 30, type_filter: Optional[str] = None):
    if type_filter:
        recs = state.chain.query_by_type(type_filter, limit=n)
    else:
        recs = state.chain.query_recent(limit=n)
    recs = sorted(recs, key=lambda r: r.index)
    return {"records": [_record_to_dict(r) for r in recs]}


@app.get("/api/chain/records")
async def chain_records(before: Optional[int] = None, limit: int = 50):
    """
    Paginated window into the full chain, newest-first.

    Powers the webapp's "load earlier" history view. The chain is
    append-only with stable, monotonic indices, so pagination needs no
    cursors: `before` is a record index (exclusive) and the endpoint
    returns up to `limit` records with strictly smaller indices.

    - `before` defaults to head index + 1, i.e. the most recent page.
    - `limit` is clamped to [1, 200] to bound the response size and the
      cost of a single page (a malicious or fat-fingered `limit=10**9`
      would otherwise try to serialize the whole chain at once).
    - `has_more` is true when older records exist before this page, so
      the frontend knows whether to keep offering "load earlier".

    Returns records ordered newest-first (descending index) to match how
    the frontend prepends them above existing messages. Stays on the
    event-loop thread and reuses the main Chain connection (same as
    /recent); iter_records is a single indexed SELECT, cheap per page.
    """
    head = state.chain.head()
    if head is None:
        return {"records": [], "has_more": False, "oldest_index": None}

    limit = max(1, min(limit, 200))
    if before is None:
        before = head.index + 1
    # Guard against negative / absurd `before`; clamp to the valid window.
    before = max(0, min(before, head.index + 1))

    start = max(0, before - limit)
    end = before
    recs = list(state.chain.iter_records(start, end))  # ascending by idx
    recs.reverse()  # newest-first for prepend-friendly rendering

    oldest_index = recs[-1].index if recs else None
    return {
        "records": [_record_to_dict(r) for r in recs],
        "has_more": start > 0,
        "oldest_index": oldest_index,
    }


@app.get("/api/chain/sidebar")
async def chain_sidebar():
    """Reflections and revisions for the sidebar — most recent first."""
    reflections = state.chain.query_by_type("reflection", limit=10)
    revisions = state.chain.query_by_type("revision", limit=20)
    return {
        "reflections": [_record_to_dict(r) for r in reflections],
        "revisions": [_record_to_dict(r) for r in revisions],
    }


@app.get("/api/chain/verify")
async def chain_verify(request: Request):
    """
    Walk the chain end-to-end checking signatures, hashes, and linkage.

    `Chain.verify_threadsafe` is O(records) and CPU-heavy on long
    chains — seconds to minutes for a 100K-record chain. Running it
    directly on the asyncio event loop would block every other request
    for that whole time, so we dispatch to a worker thread.
    `verify_threadsafe` opens its own short-lived read-only SQLite
    connection internally (the main `Chain` object's connection was
    opened with `check_same_thread=True` and cannot legally be used
    from another thread).

    Requires a session token. An anonymous caller hammering /verify on
    a big chain would be a trivial CPU DoS.
    """
    _require_session(request)
    ok, msg = await asyncio.to_thread(
        state.chain.verify_threadsafe, state.chain.pubkey_hex
    )
    return {"ok": ok, "message": msg, "length": state.chain.length()}


@app.get("/api/chain/stats")
async def chain_stats(request: Request):
    """
    Aggregate counts and timing for the chain. Cheap (indexed SELECTs),
    safe to call from a status page. See `Chain.stats` for the field
    list.
    """
    _require_session(request)
    return state.chain.stats()


@app.get("/api/chain/verify-semantic")
async def chain_verify_semantic(request: Request):
    """
    Schema-level consistency probe — a companion to `/api/chain/verify`.

    `verify` covers the cryptographic invariants (linkage, hashes,
    signatures). This endpoint covers the schema-level ones: do
    revision pointers reference real records, do `proposal_recurrence`
    records actually point at proposals, do `reflection.covers_indices`
    ranges fit inside the chain, and so on. The cryptography can be
    perfect while the *meaning* of the data is broken — a malformed
    record from a buggy tool, a referenced index from a since-trimmed
    chain — and that's what this catches.

    Cheap (a few indexed scans of typed records, not a full walk), so
    it runs inline rather than via `asyncio.to_thread`. Returns
    `warnings: []` and `ok: true` on a clean chain; otherwise a list of
    one-line strings, sorted by record index where applicable.

    Requires a session token. Same reasoning as `/verify`: an
    unauthenticated probe is a CPU footgun on long chains.
    """
    _require_session(request)
    ok, warnings = state.chain.verify_semantic()
    return {
        "ok": ok,
        "warnings": warnings,
        "length": state.chain.length(),
    }


# ---------------------------------------------------------------------------
# Experience Capsules — signed export / verified import (issue #8)
# ---------------------------------------------------------------------------

@app.get("/api/capsule/export")
async def capsule_export(request: Request):
    """
    Export shareable history as a signed Experience Capsule (capsule.py).

    Mirrors the `/export-capsule` REPL command. Exposure gating is enforced
    in `capsule.export_capsule`: `private` and `quarantine` records never
    leave; `summary`-exposed records export summary-only with a signed
    summary commitment; `shared`/`public` export in full. The response is the
    JSON capsule document — the caller saves it as a `.cphyx` file.

    Optional query params narrow the selection (all optional):
      - type: only records of this type
      - min_salience: float floor
      - after_ms / before_ms: timestamp window (ms since epoch)
      - tags: comma-separated tag list

    Requires a session token: an export crosses the agent's trust boundary
    (it emits signed memory), so it must not be anonymous.

    Returns 400 if the selection is empty after filtering (CapsuleError),
    rather than writing a meaningless empty capsule.
    """
    _require_session(request)
    import capsule as _capsule
    qp = request.query_params
    kwargs: dict = {"title": qp.get("title", "webapp export")}
    if qp.get("type"):
        kwargs["type_filter"] = qp.get("type")
    if qp.get("min_salience"):
        try:
            kwargs["min_salience"] = float(qp.get("min_salience"))
        except ValueError:
            raise HTTPException(status_code=400, detail="min_salience must be a number")
    for bound in ("after_ms", "before_ms"):
        if qp.get(bound):
            try:
                kwargs[bound] = int(qp.get(bound))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"{bound} must be an integer")
    if qp.get("tags"):
        kwargs["tags"] = [t.strip() for t in qp.get("tags").split(",") if t.strip()]
    try:
        cap = _capsule.export_capsule(state.chain, **kwargs)
    except _capsule.CapsuleError as e:
        raise HTTPException(status_code=400, detail=str(e))
    ok, msg = _capsule.verify_capsule(cap)
    return {
        "ok": ok,
        "message": msg,
        "capsule_id": cap["capsule_id"],
        "record_count": cap["header"]["record_count"],
        "capsule": cap,
    }


@app.post("/api/capsule/import")
async def capsule_import(request: Request):
    """
    Verify and import an Experience Capsule (capsule.py).

    Mirrors the `/import-capsule` REPL command and keeps the same
    verify-before-import discipline: the capsule is fully verified (every
    record's origin signature, content hashes, record hashes, summary
    commitments, the Merkle root, and the capsule id) BEFORE anything is
    imported. A capsule that fails any check is rejected wholesale (400) —
    never partially imported. Imported records are appended as
    `imported_capsule`, attributed to the origin agent with `peer_agent`
    source, recorded with a cautious epistemic class and forced-`private`
    exposure, and deduplicated by `capsule_id`. Append-only — the local
    chain's own `/verify` is unaffected.

    Body: the capsule JSON, either as the raw capsule object or wrapped as
    `{"capsule": {...}}` (the shape `/api/capsule/export` returns).

    Requires a session token: import mutates the chain.
    """
    _require_session(request)
    import capsule as _capsule
    from metadata import build_meta as _build_meta
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be valid JSON")
    cap = body.get("capsule") if isinstance(body, dict) and "capsule" in body else body
    if not isinstance(cap, dict):
        raise HTTPException(status_code=400, detail="no capsule object in body")
    ok, msg = _capsule.verify_capsule(cap)
    if not ok:
        # Verification failure is a client/data error, not a server fault.
        raise HTTPException(status_code=400, detail=f"capsule did not verify: {msg}")
    try:
        # Run inline on the event-loop thread, NOT via asyncio.to_thread:
        # import_capsule writes to `state.chain`, whose SQLite connection was
        # opened with check_same_thread=True and cannot legally be used from a
        # worker thread (the same reason the streaming turn does its
        # commit_response on the loop thread while only the LLM call is
        # offloaded). The expensive part — full cryptographic verification —
        # already ran inline above via verify_capsule; the append loop here is
        # cheap, so running it on the loop is fine.
        res = _capsule.import_capsule(
            state.chain, cap, build_meta_fn=_build_meta
        )
    except _capsule.CapsuleError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Re-index any newly appended records so they are immediately retrievable.
    if not res.get("skipped"):
        try:
            state.index.index_chain(state.chain)
        except Exception:
            # Indexing is best-effort here; the records are committed
            # regardless and will be indexed on next boot if this fails.
            pass
    return {
        "ok": True,
        "verify_message": msg,
        "imported_count": res["imported_count"],
        "skipped": res["skipped"],
        "reason": res["reason"],
        "length": state.chain.length(),
    }


@app.get("/blobs/{sha}")
async def serve_blob(sha: str, session: Optional[str] = None,
                     request: Request = None):
    """
    Serve an ingested file's bytes by sha256 — used for image rendering
    in chat.

    Requires a session token (passed as a `?session=` query parameter so
    `<img src>` tags can carry it). Without auth, anyone with reach to
    the port could pull every ingested file by guessing or harvesting
    sha values — the rest of the API is single-session-locked; this
    must match.

    Lookup goes through the chain's indexed `find_file_by_sha`, not a
    linear scan of every file record. Both are O(1)-ish, but the old
    scan also had a silent `limit=500` cap that would silently 404 the
    501st file onward.
    """
    # Session check — same shape as `_require_session`, but the query
    # parameter is named `session` and we accept the header form too.
    token = (session
             or (request.headers.get("x-session-token") if request else None))
    if not token or state.active_token != token:
        raise HTTPException(
            status_code=409,
            detail={"error": "session_inactive",
                    "message": "blob access requires the active session token"},
        )

    rec = state.chain.find_file_by_sha(sha)
    if rec is None:
        raise HTTPException(404, "no file record references that hash")
    blob_dir = DATA_DIR / "blobs"
    blob_filename = rec.content.get("blob_path", "")

    # Defense-in-depth: validate that `blob_filename` is a plain basename
    # within blob_dir, not a relative path that escapes it. file_ingest
    # only ever stores the sha256 basename, so this is normally a no-op
    # — but the chain is the source of truth, and a corrupted chain
    # (or a buggy ingestion tool) could plant `../../etc/passwd`. The
    # session check protects against random callers; this protects
    # against a malformed record. We resolve both paths to absolute and
    # check the candidate is under blob_dir.
    if not blob_filename or "/" in blob_filename or "\\" in blob_filename \
            or blob_filename in (".", "..") or blob_filename.startswith("."):
        raise HTTPException(
            400, "file record's blob_path is not a plain basename")
    candidate = (blob_dir / blob_filename).resolve()
    try:
        candidate.relative_to(blob_dir.resolve())
    except ValueError:
        raise HTTPException(
            400, "blob_path escapes the blob directory")
    path = candidate
    if not path.exists():
        raise HTTPException(404, "blob missing on disk")
    # Map ext -> media type for the common cases.
    ext = rec.content.get("ext", "").lower()
    media = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
        ".pdf": "application/pdf",
    }.get(ext, "application/octet-stream")
    return FileResponse(path, media_type=media, filename=rec.content.get("filename"))


# ---------------------------------------------------------------------------
# Slash commands — non-turn actions that mirror run.py's REPL
# ---------------------------------------------------------------------------

@app.post("/api/command")
async def run_command(request: Request, body: dict):
    """
    Handle a slash command. Body: {"command": "/reflect"} etc.
    Returns a structured result that the UI renders inline as a system message.
    """
    _require_session(request)
    cmd = (body.get("command") or "").strip()
    if not cmd:
        raise HTTPException(400, "empty command")

    async with state.lock:
        if cmd == "/verify":
            ok, msg = state.chain.verify(expected_pubkey=state.chain.pubkey_hex)
            return {"kind": "verify", "ok": ok, "message": msg}

        if cmd == "/verify-semantic":
            ok, warnings = state.chain.verify_semantic()
            return {
                "kind": "verify_semantic",
                "ok": ok,
                "warnings": warnings,
                "length": state.chain.length(),
            }

        if cmd == "/length":
            return {"kind": "length", "length": state.chain.length()}

        if cmd == "/seal":
            batch = state.chain.seal_batch()
            return {"kind": "seal", "batch": batch}

        if cmd == "/sysprompt":
            history = state.chain.query_by_type("system_prompt", limit=10)
            entries = [
                {
                    "index": r.index,
                    "timestamp": r.timestamp,
                    "text": r.content.get("text", ""),
                }
                for r in sorted(history, key=lambda r: r.index)
            ]
            return {"kind": "sysprompt", "entries": entries}

        if cmd == "/reflect":
            rec = state.agent.reflect()
            if rec is None:
                return {"kind": "reflect", "result": "not_enough_history"}
            state.index.index_record(rec)
            state.turns_since_reflect = 0
            return {"kind": "reflect", "record": _record_to_dict(rec)}

        if cmd == "/cambium":
            # Scan chain history for recurring gaps and commit proposals,
            # recurrences, and escalations. Cambium proposes; never applies.
            # Incremental + lookback: covers everything new since the
            # last scan plus a MAX_CAMBIUM_RECORDS lookback window for
            # cross-boundary pattern detection.
            result = state.agent.run_cambium(max_records=MAX_CAMBIUM_RECORDS)
            for group in ("proposals", "recurrences", "escalations"):
                for rec in result.get(group, []):
                    state.index.index_record(rec)
            # A manual scan resets the auto counter, so auto-Cambium
            # doesn't fire again right after the operator just ran it.
            state.turns_since_cambium = 0
            summary = _cambium_result_to_dict(result)
            summary["kind"] = "cambium"
            return summary

        if cmd == "/cambium-full":
            # Explicit one-shot deep scan of the entire chain. Linear in
            # chain length and unbounded — use sparingly. Does NOT
            # advance the incremental watermark, so the periodic scan
            # keeps its rolling coverage afterwards.
            result = state.agent.run_cambium_full()
            for group in ("proposals", "recurrences", "escalations"):
                for rec in result.get(group, []):
                    state.index.index_record(rec)
            summary = _cambium_result_to_dict(result)
            summary["kind"] = "cambium"
            return summary

        if cmd == "/proposals":
            import cambium
            proposals = state.chain.query_by_type("proposal", limit=50)
            # Bulk recurrence/escalation lookups — one scan total, not
            # one per proposal.
            counts = cambium.recurrence_counts(state.chain)
            escalated_set = cambium.escalated_indices(state.chain)
            entries = []
            for r in proposals:
                entries.append({
                    "index": r.index,
                    "proposal_kind": r.content.get("proposal_kind", "?"),
                    "status": r.content.get("status", "open"),
                    "title": r.content.get("title", ""),
                    "recurrence_count": counts.get(r.index, 1),
                    "escalated": r.index in escalated_set,
                })
            # Escalated proposals first, then by index.
            entries.sort(key=lambda e: (not e["escalated"], e["index"]))
            return {"kind": "proposals", "entries": entries}

        if cmd.startswith("/revise"):
            parts = cmd.split(maxsplit=2)
            if len(parts) < 3:
                raise HTTPException(400, "usage: /revise <index> <correction text>")
            try:
                target_idx = int(parts[1])
            except ValueError:
                raise HTTPException(400, f"invalid index: {parts[1]!r}")
            try:
                rec = state.agent.revise(target_idx, parts[2])
            except ProtectedZoneError as e:
                # Genesis / system_prompt / principle records are a
                # protected zone and cannot be revised by a turn.
                raise HTTPException(403, str(e))
            if rec is None:
                return {"kind": "revise", "result": "no_such_record", "target": target_idx}
            state.index.index_record(rec)
            return {
                "kind": "revise",
                "record": _record_to_dict(rec),
                "target_index": target_idx,
            }

        if cmd.startswith("/file"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                raise HTTPException(400, "usage: /file <path>")
            file_path = parts[1].strip().strip('"').strip("'")
            try:
                rec = state.agent.ingest_file(file_path)
            except FileNotFoundError:
                raise HTTPException(404, f"no such file: {file_path}")
            except (ValueError, OSError) as e:
                raise HTTPException(400, str(e))
            state.index.index_record(rec)
            return {"kind": "file", "record": _record_to_dict(rec)}

        # cypher-tempre port commands (/verify-source, /poq, /immune-*,
        # /recall-*, /think, /consensus-*, /cambium-grow, /migrate,
        # /continuum-*, /cypher-help). One dispatcher (cypher_commands.py) is
        # the single source of truth shared with the REPL; it prints its
        # output, which we capture and hand back for the UI to render as a
        # system message. Runs under state.lock, like every other command.
        import io
        import contextlib
        import cypher_commands

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            handled = cypher_commands.dispatch(cmd, state.chain, state.agent)
        if handled:
            return {
                "kind": "cypher",
                "command": cmd.split()[0],
                "output": buf.getvalue().rstrip() or "(no output)",
            }

        raise HTTPException(400, f"unknown command: {cmd}")


# ---------------------------------------------------------------------------
# File upload (drag-and-drop) — calls agent.ingest_file under the hood
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    _require_session(request)
    # Save to a temp path (basename is random), keeping the original suffix.
    # The REAL filename is passed to ingest_file via original_name so the record
    # stores "CHANGELOG.md", not the temp basename — the retriever embeds the
    # filename, so the user must be able to search by the real name.
    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        async with state.lock:
            try:
                rec = state.agent.ingest_file(tmp_path, original_name=file.filename)
            except (ValueError, OSError) as e:
                raise HTTPException(400, str(e))
            except HTTPException:
                raise
            except Exception as e:
                # Anything else (e.g. an embedder failure, a dependency
                # missing, a Python-version incompatibility) would otherwise
                # surface as an opaque 500. Return the real error type and
                # message so the cause is visible in the browser, and also
                # print a full traceback to the server console.
                import traceback
                traceback.print_exc()
                raise HTTPException(
                    500, f"ingest failed: {type(e).__name__}: {e}")
            try:
                state.index.index_record(rec)
            except Exception as e:
                # The file WAS committed to the chain — indexing is a
                # separate, derived step. Surface the indexing error
                # rather than masking it as a generic 500.
                import traceback
                traceback.print_exc()
                raise HTTPException(
                    500,
                    f"file committed to chain (record {rec.index}) but "
                    f"indexing failed: {type(e).__name__}: {e}")
            return {"record": _record_to_dict(rec)}
    finally:
        try: tmp_path.unlink()
        except Exception: pass


# ---------------------------------------------------------------------------
# Chat turn — streaming via SSE when the LLM client supports it
# ---------------------------------------------------------------------------

def _llm_supports_streaming(llm) -> bool:
    """Look for a stream() method or a stream=True kwarg path."""
    return callable(getattr(llm, "stream", None))


@app.post("/api/turn")
async def turn(request: Request, body: dict):
    """
    Non-streaming turn endpoint. Always available regardless of LLM streaming
    support. Returns the full response after the turn commits. The UI prefers
    /api/turn/stream when streaming is supported.
    """
    _require_session(request)
    user_input = (body.get("input") or "").strip()
    if not user_input:
        raise HTTPException(400, "empty input")

    async with state.lock:
        # SQLite connections in chain/index were opened in the main thread,
        # so chain ops run inline. Only the LLM call needs a thread (it
        # blocks on the network), and Agent.turn() does that synchronously
        # inside a single function — we can't isolate just the network call
        # without rewriting Agent.turn(), so we accept that this endpoint
        # blocks the loop while the LLM responds. The streaming endpoint
        # below decomposes the turn so the LLM call CAN go to a thread.
        turn_obj = state.agent.turn(user_input, SEMANTIC_K, n_recent=RECENT_N)
        state.index.index_record(turn_obj.observation_record)
        state.index.index_record(turn_obj.response_record)
        state.turns_since_reflect += 1
        state.turns_since_cambium += 1

        # Auto-reflection mirrors run.py's behavior.
        reflection_rec = None
        if AUTO_REFLECT_EVERY > 0 and state.turns_since_reflect >= AUTO_REFLECT_EVERY:
            reflection_rec = state.agent.reflect()
            if reflection_rec is not None:
                state.index.index_record(reflection_rec)
            state.turns_since_reflect = 0

        # Auto-Cambium on its own separate, longer cadence (mirrors
        # run.py's AUTO_CAMBIUM_EVERY). The scan is LLM-free and cheap.
        cambium_result = None
        if AUTO_CAMBIUM_EVERY > 0 and state.turns_since_cambium >= AUTO_CAMBIUM_EVERY:
            result = state.agent.run_cambium(max_records=MAX_CAMBIUM_RECORDS)
            for group in ("proposals", "recurrences", "escalations"):
                for rec in result.get(group, []):
                    state.index.index_record(rec)
            cambium_result = _cambium_result_to_dict(result)
            state.turns_since_cambium = 0

    return {
        "observation": _record_to_dict(turn_obj.observation_record),
        "response": _record_to_dict(turn_obj.response_record),
        "response_text": turn_obj.response_text,
        "retrieved_indices": [r.index for r in turn_obj.retrieved],
        "reflection": _record_to_dict(reflection_rec) if reflection_rec else None,
        "cambium": cambium_result,
        "truncated": turn_obj.truncated,
    }


@app.get("/api/turn/stream")
async def turn_stream(request: Request, input: str, session: str):
    """
    Streaming turn via Server-Sent Events.

    If the LLM client exposes a .stream(prompt, **kwargs) method that yields
    text chunks, we stream those to the browser as 'token' events, then
    commit the full response to the chain and emit a 'done' event with the
    final record metadata. If the client doesn't support streaming, we fall
    back to a single-chunk emission (so the UI works either way).

    Implementation note: the LLM call sits between two chain-write halves
    that need to share state (the observation, the context, the prompt).
    We use `Agent.prepare_turn` -> LLM -> `Agent.score_response` ->
    `Agent.commit_response` so the streaming path goes through the same
    PoQ scoring, quarantine routing, and observation-indexing order as
    `Agent.turn()`. An earlier version of this endpoint inlined its own
    hand-rolled copy of those steps and diverged from `turn()` in three
    ways: (1) it indexed the observation BEFORE retrieval so the prompt
    could include the user's own just-asked question as "relevant
    memory"; (2) it skipped PoQ entirely so an injection that the REPL
    would quarantine became ordinary memory through the web UI; (3) the
    SSE producer thread had no cancellation path, so a disconnect leaked
    the running LLM call. All three are fixed here.
    """
    _require_session(request)
    user_input = input.strip()
    if not user_input:
        raise HTTPException(400, "empty input")

    async def event_source():
        # Acquire the chain lock for the whole turn — single-writer guarantee.
        async with state.lock:
            agent = state.agent
            chain = state.chain

            # 0. Immune screen FIRST — parity with Agent.turn(). The
            # non-streaming /turn path gets this for free (it calls
            # agent.turn()); the streaming path composes the steps by hand, so
            # it must screen here too or it would silently diverge. A blocked
            # input is refused at the membrane: an honest refusal is sealed (no
            # LLM call), emitted, and the turn ends.
            if agent.immune is not None:
                _screen = agent.immune.screen(user_input)
                if _screen.get("blocked"):
                    refused = agent._refused_turn(user_input, _screen)
                    state.index.index_record(refused.observation_record)
                    state.index.index_record(refused.response_record)
                    yield {
                        "event": "observation",
                        "data": json.dumps(_record_to_dict(refused.observation_record)),
                    }
                    yield {
                        "event": "done",
                        "data": json.dumps({
                            "response": _record_to_dict(refused.response_record),
                            "retrieved_indices": [],
                            "truncated": False,
                            "refused": True,
                        }),
                    }
                    return

            # 1. Pre-LLM half: commit observation, retrieve, build prompt.
            # `prepare_turn` is the SAME method `Agent.turn` uses, so the
            # streaming path can never silently diverge in metadata or
            # quarantine handling. It does NOT index the observation —
            # that waits until after retrieval, so the just-asked question
            # cannot be retrieved as context for its own prompt.
            prep = agent.prepare_turn(
                user_input, retrieve_k=SEMANTIC_K, n_recent=RECENT_N
            )
            obs = prep.observation_record

            # Now safe to index — retrieval has already run against the
            # pre-existing chain. The browser can still see the obs
            # index immediately via the SSE event below.
            state.index.index_record(obs)

            yield {
                "event": "observation",
                "data": json.dumps(_record_to_dict(obs)),
            }

            # 2. Call the model. THIS is what needs the thread — it's a
            # blocking network call that can take seconds. Chain ops
            # before and after stay on the main thread.
            full_text_parts: list[str] = []
            llm = agent.llm

            # Cancellation: the producer thread checks this event after
            # every chunk it tries to enqueue. If the consumer's `finally`
            # fires (browser disconnect, request cancelled), we set the
            # event so the producer stops iterating the LLM stream
            # instead of generating tokens nobody reads. Without this the
            # producer thread leaks for the rest of the LLM call's
            # natural lifetime — including the API cost of every
            # generated-but-unread token.
            import threading
            cancel_event = threading.Event()

            try:
                if _llm_supports_streaming(llm):
                    queue: asyncio.Queue = asyncio.Queue()
                    loop = asyncio.get_running_loop()

                    def producer():
                        try:
                            for chunk in llm.stream(prep.prompt, **prep.llm_kwargs):
                                if cancel_event.is_set():
                                    return
                                asyncio.run_coroutine_threadsafe(
                                    queue.put(("chunk", chunk)), loop
                                )
                        except Exception as e:
                            if not cancel_event.is_set():
                                asyncio.run_coroutine_threadsafe(
                                    queue.put(("error", f"{type(e).__name__}: {e}")),
                                    loop,
                                )
                        finally:
                            # Always emit `end` so the consumer loop
                            # terminates, even on cancel.
                            try:
                                asyncio.run_coroutine_threadsafe(
                                    queue.put(("end", None)), loop
                                )
                            except RuntimeError:
                                # Loop is gone — nothing more to do.
                                pass

                    loop.run_in_executor(None, producer)
                    while True:
                        kind, payload = await queue.get()
                        if kind == "chunk":
                            full_text_parts.append(payload)
                            yield {"event": "token",
                                   "data": json.dumps({"text": payload})}
                        elif kind == "error":
                            yield {"event": "error",
                                   "data": json.dumps({"message": payload})}
                            return
                        else:  # end
                            break
                else:
                    # Fallback: non-streaming client. Run the LLM call in
                    # a thread so the event loop can keep serving other
                    # requests while we wait.
                    if prep.llm_kwargs:
                        full_text = await asyncio.to_thread(
                            llm, prep.prompt, **prep.llm_kwargs)
                    else:
                        full_text = await asyncio.to_thread(llm, prep.prompt)
                    full_text_parts.append(full_text)
                    yield {"event": "token",
                           "data": json.dumps({"text": full_text})}
            finally:
                # Always signal cancel on the way out — covers both
                # normal completion (already-set is a no-op) and
                # asyncio.CancelledError from a disconnect.
                cancel_event.set()

            response_text = "".join(full_text_parts)

            # Did the model hit its max_tokens ceiling? Read the finish
            # reason before any later call overwrites it.
            from llm_clients import was_truncated
            response_was_truncated = was_truncated(llm)

            # 3. Post-LLM half: PoQ scoring + commit. score_response
            # produces the response `_meta` dict (including PoQ block and
            # quarantine exposure when warranted); commit_response writes
            # the record. Both come from Agent so the streaming path
            # cannot diverge from `Agent.turn`.
            poq_result, response_meta_kwargs = agent.score_response(
                user_input, response_text, prep.context
            )
            # Persist the truncation flag on the response record so a
            # later "continue" turn can detect it. Mirrors `Agent.turn`;
            # see `_format_prompt`'s continuation-after-truncation logic.
            if response_was_truncated:
                response_meta_kwargs["truncated"] = True
            response = agent.commit_response(
                prep, response_text, response_meta_kwargs
            )
            state.index.index_record(response)
            state.turns_since_reflect += 1
            state.turns_since_cambium += 1

            done_payload = {
                "response": _record_to_dict(response),
                "retrieved_indices": [r.index for r in prep.context],
                "truncated": response_was_truncated,
            }
            # Tell the browser if PoQ quarantined this turn — same signal
            # the REPL prints, so the operator knows the response was
            # shown but the memory was routed off the belief path.
            if poq_result is not None:
                done_payload["poq"] = poq_result.to_meta()
                if poq_result.action == "quarantine":
                    done_payload["quarantined"] = True

            # Run auto-reflection and auto-Cambium BEFORE emitting `done`: the
            # browser closes the EventSource on `done`, which cancels this
            # generator, so anything yielded (or even run) after `done` can be
            # skipped — diverging from the REPL / non-streaming path. Do it now.

            # Auto-reflection. The reflect() call makes an LLM call internally,
            # so it blocks the loop briefly; acceptable, and mirrors the REPL.
            if AUTO_REFLECT_EVERY > 0 and state.turns_since_reflect >= AUTO_REFLECT_EVERY:
                reflection_rec = agent.reflect()
                if reflection_rec is not None:
                    state.index.index_record(reflection_rec)
                    yield {
                        "event": "reflection",
                        "data": json.dumps(_record_to_dict(reflection_rec)),
                    }
                state.turns_since_reflect = 0

            # 5. Auto-Cambium on its own separate, longer cadence (mirrors
            # run.py's AUTO_CAMBIUM_EVERY). Unlike reflection, the Cambium
            # scan is LLM-free, so it does not block on the network.
            if AUTO_CAMBIUM_EVERY > 0 and state.turns_since_cambium >= AUTO_CAMBIUM_EVERY:
                result = agent.run_cambium(max_records=MAX_CAMBIUM_RECORDS)
                for group in ("proposals", "recurrences", "escalations"):
                    for rec in result.get(group, []):
                        state.index.index_record(rec)
                summary = _cambium_result_to_dict(result)
                if (summary["proposals"] or summary["recurrences"]
                        or summary["escalations"]):
                    yield {
                        "event": "cambium",
                        "data": json.dumps(summary),
                    }
                state.turns_since_cambium = 0

            # `done` is the LAST event — the client closes the stream on it, so
            # everything above (response commit, reflection, cambium) is already
            # committed and streamed by the time the browser disconnects.
            yield {"event": "done", "data": json.dumps(done_payload)}

    return EventSourceResponse(event_source())


@app.get("/api/migrate/stream")
async def migrate_stream(request: Request, session: str):
    """Stream the historic-chain backfill (re-embed every record) as SSE, so a
    long migration shows progress instead of looking frozen. Emits `start`,
    periodic `progress`, and a final `done` event. The work runs on the chain
    connection's own thread (SQLite is thread-affine); `await sleep(0)` between
    batches flushes each event and lets the loop serve other requests."""
    _require_session(request)
    import migrate as _migrate

    async def event_source():
        async with state.lock:
            for ev in _migrate.reindex_stream(state.chain, state.index):
                yield {"event": ev["phase"], "data": json.dumps(ev)}
                await asyncio.sleep(0)

    return EventSourceResponse(event_source())


# ---------------------------------------------------------------------------
# Static files + index page
# ---------------------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _serve_page(name: str) -> HTMLResponse:
    path = STATIC_DIR / name
    if not path.exists():
        return HTMLResponse(f"<h1>missing static/{name}</h1>", status_code=500)
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
async def index():
    return _serve_page("index.html")


@app.get("/commands", response_class=HTMLResponse)
async def commands_page():
    """Reference page: what each slash command does and why you'd use it."""
    return _serve_page("commands.html")


@app.get("/audit", response_class=HTMLResponse)
async def audit_page():
    """The audit dashboard page (data from /api/audit)."""
    return _serve_page("audit.html")


@app.get("/api/audit")
async def api_audit():
    """Read-only audit snapshot for the dashboard. Ungated, like
    /api/chain/status — it exposes the same chain summary data, no writes."""
    import audit as _audit
    from pathlib import Path as _Path
    ok, _msg = state.chain.verify()
    faculty_dir = _Path(__file__).resolve().parent.parent / "faculties"
    blob_dir = DATA_DIR / "blobs"
    result = _audit.compute(state.chain, faculty_dir, blob_dir=blob_dir, integrity=ok)
    result["data_dir"] = str(DATA_DIR)
    return result


@app.get("/api/audit/ring/{idx}")
async def api_audit_ring(idx: int):
    """Full contents of one ring, for the audit dashboard's detail pane. The
    /api/audit ring list carries only truncated summaries (to keep that payload
    small); this returns the complete record on demand."""
    rec = state.chain.get(idx)
    if rec is None:
        raise HTTPException(404, "no such ring")
    import ring_compat as _rc
    import recall as _recall
    # rec.to_dict() carries everything: content (incl. _meta), refs, and all the
    # cryptographic fields (prior/content/record hashes, signature, pubkey).
    d = rec.to_dict()
    d["timestamp_iso"] = datetime.fromtimestamp(
        rec.timestamp / 1000, tz=timezone.utc).isoformat()
    d["text"] = _recall.block_text(_rc.record_to_ring(rec))
    return d


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
