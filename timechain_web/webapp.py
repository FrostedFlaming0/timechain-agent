"""
webapp — FastAPI frontend for the timechain agent.

Wraps the same Agent / Chain / Retriever stack that run.py uses, and serves
a single-page browser UI. The chain stays single-writer because there's
still one process (this one) holding the signing key and appending records.

What you get:
  - Streaming responses (Server-Sent Events) when the LLM client supports it,
    falling back to non-streaming otherwise.
  - Tool turns with the same shared driver as the REPL: tool_result /
    pending_op SSE events, user-only approve/reject endpoints for the
    durable write gate.
  - Content ingestion via drag-and-drop or file picker (POST /api/upload →
    ingest_blob: task workspace when a task is active, identity-chain
    attachment + content-addressed blob otherwise).
  - Sidebar showing recent reflections + revisions.
  - All slash commands from run.py: /verify, /length, /seal, /sysprompt,
    /reflect, /cambium, /proposals, /revise N <text>.
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
from retrieval import EmbeddingIndex, Retriever, open_or_rebuild_index
from agent import Agent, ProtectedZoneError
from pending_ops import (PENDING_TTL_SECONDS,
                         resolve_inline as pending_ops_resolve_inline)

# (The tool-loop round budget is tools.DEFAULT_MAX_TOOL_ROUNDS — read
# fresh at each turn, the same knob the REPL loop reads, so an override
# changes both transports and the two can never drift.)
from run import (
    DATA_DIR,
    LLM_PROVIDER,
    FOUNDING_COMMITMENTS,
    SYSTEM_PROMPT,
    SEMANTIC_K,
    RECENT_N,
    OLLAMA_EMBED_MODEL,
    AUTO_REFLECT_EVERY,
    MAX_REFLECT_RECORDS,
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
    TOOLS_ENABLED,
    TOOL_SAFETY_PROMPT,
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
        # Tool execution context (tools.AgentContext) — task registry,
        # per-task chains/indexes, and the durable write gate. None when
        # TOOLS_ENABLED is off in run.py.
        self.tool_ctx = None
        # Serialize all chain-touching work; the chain assumes single-writer.
        self.lock = asyncio.Lock()
        # Mid-turn approval gate (v1.4.x): pending_op_id -> asyncio.Future.
        # While a streaming turn is PAUSED on a pending op, its future sits
        # here; the approve/reject endpoints deliver the decision by
        # resolving the future WITHOUT touching state.lock (the turn holds
        # that lock for its whole duration — going through the lock would
        # deadlock). Execution then happens inside the turn, which already
        # owns the lock.
        self.approval_waiters: dict = {}
        # Session token — only one tab is "active" at a time.
        self.active_token: Optional[str] = None
        # Counter for auto-reflection (mirrors run.py's behavior).
        self.turns_since_reflect = 0
        # Counter for auto-Cambium — a separate, longer cadence than
        # auto-reflection (mirrors run.py's AUTO_CAMBIUM_EVERY behavior).
        self.turns_since_cambium = 0
        # The in-flight (or most recently finished) background turn — a
        # TurnRun. A turn's lifetime belongs to the SERVER, not to any one
        # SSE connection: navigating away mid-turn only detaches a viewer,
        # never cancels the turn (see TurnRun / _drive_turn).
        self.active_turn = None

    def boot(self) -> None:
        """Set up chain, index, agent. Idempotent — safe to call once at startup."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        chain_db = DATA_DIR / "chain.sqlite"
        embed_db = DATA_DIR / "embeddings.sqlite"
        key_path = DATA_DIR / "operator.key"

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

        self.index = open_or_rebuild_index(
            embed_db, embedder, embed_dim,
            log=lambda msg: print(f"[boot] {msg}"))

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
        # Same system-prompt composition as run.py: the tool-safety prompt
        # rides along whenever tools are enabled, so web turns and REPL
        # turns see identical instructions.
        system_prompt = SYSTEM_PROMPT + (TOOL_SAFETY_PROMPT
                                         if TOOLS_ENABLED else "")
        self.agent = Agent(
            self.chain, retriever, llm,
            system_prompt=system_prompt,
            blob_dir=DATA_DIR / "blobs",
            context_char_budget=CONTEXT_BUDGET_CHARS,
        )

        # Tool execution context (mirrors run.py). Built even before first
        # genesis — it only touches the registry/pending-op dirs lazily.
        if TOOLS_ENABLED:
            from task_registry import TaskRegistry
            from tools import AgentContext, restore_workspace
            self.tool_ctx = AgentContext(
                data_dir=DATA_DIR,
                registry=TaskRegistry(DATA_DIR),
                identity_chain=self.chain,
                identity_recall=retriever,   # powers recall_index pre-filter
                workspace_root=Path.cwd(),
                embedder=embedder,
                embed_dim=embed_dim,
            )
            restored = restore_workspace(self.tool_ctx)
            if restored:
                print(f"[boot] workspace restored: {restored}")

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

        # Seed both auto-cadence counters from the chain so they carry
        # across sessions (mirrors run.py). At every-100 / every-30 turns a
        # per-session counter would rarely fire — few web sessions run that
        # long — so the chain, not the process, is the source of truth.
        self.turns_since_reflect = self.agent.turns_since_reflection()
        self.turns_since_cambium = self.agent.turns_since_cambium()
        print(f"[boot] turns since last reflection: {self.turns_since_reflect} "
              f"(auto-reflect every {AUTO_REFLECT_EVERY}); "
              f"since last cambium scan: {self.turns_since_cambium} "
              f"(auto-cambium every {AUTO_CAMBIUM_EVERY})")

        print(f"[boot] chain length: {self.chain.length()}")
        print(f"[boot] ready at http://{HOST}:{PORT}")

    def shutdown(self) -> None:
        if self.tool_ctx is not None:
            try: self.tool_ctx.close()
            except Exception: pass
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
        # Stop an in-flight background turn before tearing down the chain —
        # otherwise its task would race the connection close below. This is
        # the ONE place a running turn is cancelled (server shutdown); a
        # browser disconnect never reaches here.
        run = getattr(state, "active_turn", None)
        if run is not None and run.task is not None and not run.task.done():
            run.task.cancel()
            try:
                await run.task
            except BaseException:
                pass
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
# Chain inspection — read-only, but session-gated: record contents ARE the
# agent's memory (conversations, ingested code, reflections). Serving them
# anonymously would hand the whole chain to anyone with reach to the port,
# while every mutating endpoint demands the token — same rule as /blobs.
# ---------------------------------------------------------------------------

@app.get("/api/chain/status")
async def chain_status(request: Request):
    _require_session(request)
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
async def chain_recent(request: Request, n: int = 30,
                       type_filter: Optional[str] = None):
    _require_session(request)
    # Clamp n: SQLite treats a negative LIMIT as unlimited, so n=-1 would
    # serialize the entire chain in one response.
    n = max(1, min(n, 200))
    if type_filter:
        recs = state.chain.query_by_type(type_filter, limit=n)
    else:
        recs = state.chain.query_recent(limit=n)
    recs = sorted(recs, key=lambda r: r.index)
    return {"records": [_record_to_dict(r) for r in recs]}


@app.get("/api/chain/records")
async def chain_records(request: Request, before: Optional[int] = None,
                        limit: int = 50):
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
    _require_session(request)
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
async def chain_sidebar(request: Request):
    """Reflections and revisions for the sidebar — most recent first."""
    _require_session(request)
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

    # /approve & /reject BEFORE the lock: while a streaming turn is parked
    # on its approval gate it HOLDS state.lock, so the legacy path below
    # would deadlock the very command meant to unblock it. Deliver the
    # decision to the waiter lock-free (same as the card buttons); fall
    # through to the legacy locked path when no turn is waiting.
    _parts = cmd.split()
    if _parts[0] in ("/approve", "/reject") and len(_parts) == 2:
        _require_tools()
        delivered = _deliver_to_waiter(
            _parts[1], "approve_write" if _parts[0] == "/approve"
            else "reject_write")
        if delivered is not None:
            return {"kind": "pending_op_action", "ok": True,
                    "message": delivered["result"]}

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
            rec = state.agent.reflect(max_records=MAX_REFLECT_RECORDS)
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

        # /approve <id> and /reject <id> — REPL parity in the chat box, so
        # the muscle memory (and the model's instructions) work in both
        # interfaces. Same user-only path the card buttons hit.
        parts = cmd.split()
        if parts[0] in ("/approve", "/reject"):
            _require_tools()
            if len(parts) != 2:
                raise HTTPException(400, f"usage: {parts[0]} <pending_op_id>")
            import tools as tools_mod
            action = ("approve_write" if parts[0] == "/approve"
                      else "reject_write")
            ok, result = tools_mod.run_user_action(
                action, {"pending_op_id": parts[1]}, state.tool_ctx)
            return {"kind": "pending_op_action", "ok": ok, "message": result}

        raise HTTPException(400, f"unknown command: {cmd}")


# ---------------------------------------------------------------------------
# Chat turn — streaming via SSE when the LLM client supports it
# ---------------------------------------------------------------------------

def _llm_supports_streaming(llm) -> bool:
    """Look for a stream() method or a stream=True kwarg path."""
    return callable(getattr(llm, "stream", None))


class TurnRun:
    """One background turn: the event buffer + completion flag.

    The turn's lifetime belongs to the SERVER, not to any one connection.
    `_drive_turn` (an asyncio task) drains `_turn_events` into `events`;
    any number of subscribers replay the buffer and then follow live via
    `_follow_turn`. A browser that navigates away (the audit tab, a reload)
    only detaches its subscriber — the turn keeps running and COMMITS, so
    the chain can never be left with a sealed observation and no paired
    response just because the viewer left.

    Events are stamped with 1-based sequential SSE ids, so a reconnecting
    EventSource (which sends Last-Event-ID) resumes exactly where it left
    off instead of replaying — or worse, re-running — the turn.
    """

    def __init__(self, turn_id: str, user_input: str) -> None:
        self.id = turn_id
        self.input = user_input
        self.events: list = []
        self.done = False
        self.cond = asyncio.Condition()
        self.task: Optional[asyncio.Task] = None


async def _drive_turn(run: TurnRun) -> None:
    """Drain ONE turn's events into its buffer, then mark it done.

    This coroutine — not any HTTP response — owns the turn. It is only
    ever cancelled at server shutdown (see lifespan); a client disconnect
    cancels its `_follow_turn` subscriber instead, which this never sees.
    """
    def _stamp(ev: dict) -> dict:
        ev = dict(ev)
        ev["id"] = str(len(run.events) + 1)
        return ev

    try:
        async for ev in _turn_events(run.input):
            async with run.cond:
                run.events.append(_stamp(ev))
                run.cond.notify_all()
    except Exception as e:
        # _turn_events converts LLM/stream failures into a committed turn,
        # so reaching here is an unexpected crash — surface it as an event
        # rather than dying silently with subscribers still waiting.
        async with run.cond:
            run.events.append(_stamp({
                "event": "error",
                "data": json.dumps(
                    {"message": f"{type(e).__name__}: {e}"}),
            }))
            run.cond.notify_all()
    finally:
        async with run.cond:
            run.done = True
            run.cond.notify_all()


def _start_turn(user_input: str) -> TurnRun:
    """Create a TurnRun and start its background driver task."""
    run = TurnRun(secrets.token_hex(8), user_input)
    state.active_turn = run
    run.task = asyncio.create_task(_drive_turn(run))
    return run


async def _follow_turn(run: TurnRun, after: int = 0):
    """Replay a run's buffered events, then follow live until done.

    `after` skips events already delivered (the Last-Event-ID contract:
    ids are 1-based, so `after=N` resumes at event N+1).
    """
    i = after
    while True:
        async with run.cond:
            while i >= len(run.events) and not run.done:
                await run.cond.wait()
            if i >= len(run.events):
                return
            ev = run.events[i]
            i += 1
        yield ev


def _last_event_seq(request: Request) -> int:
    """The Last-Event-ID a reconnecting EventSource sends, as an int."""
    try:
        return int(request.headers.get("last-event-id", "0"))
    except ValueError:
        return 0


_TURN_IN_PROGRESS = {
    "error": "turn_in_progress",
    "message": "a turn is already running; wait for it to finish "
               "(GET /api/turn/active to check, or attach to its stream "
               "with /api/turn/stream?attach=1).",
}


@app.get("/api/turn/active")
async def turn_active(request: Request):
    """Probe: is a turn currently running in the background?

    The chat page calls this on load and re-attaches to the live stream
    when it reports active — the half of the design that makes navigating
    away mid-turn (audit tab, reload) lossless for the viewer too."""
    _require_session(request)
    run = getattr(state, "active_turn", None)
    if run is None or run.done:
        return {"active": False}
    return {"active": True, "input": run.input}


@app.post("/api/turn")
async def turn(request: Request, body: dict):
    """
    Non-streaming turn endpoint. Always available regardless of LLM streaming
    support. Rides the SAME background TurnRun the SSE endpoint serves
    (token events are discarded, the terminal events become the JSON
    payload): one turn implementation, two transports, one driver task. The
    background task also makes this endpoint disconnect-proof — starlette
    cancels a handler whose client went away, but the turn commits anyway.
    """
    _require_session(request)
    user_input = (body.get("input") or "").strip()
    if not user_input:
        raise HTTPException(400, "empty input")
    prior = getattr(state, "active_turn", None)
    if prior is not None and not prior.done:
        raise HTTPException(status_code=409, detail=_TURN_IN_PROGRESS)
    run = _start_turn(user_input)

    observation = None
    reflection = None
    cambium = None
    done = None
    error_message = None
    async for ev in _follow_turn(run):
        kind = ev.get("event")
        if kind == "token":
            # Token payloads are discarded here — skip the parse.
            continue
        try:
            data = json.loads(ev.get("data") or "null")
        except (ValueError, TypeError):
            data = None
        if kind == "observation":
            observation = data
        elif kind == "reflection":
            reflection = data
        elif kind == "cambium":
            cambium = data
        elif kind == "error":
            # Do NOT raise here: the turn converts stream failures into a
            # committed turn, so drain to the `done` event and report the
            # failure after. (Abandoning the subscriber early would not
            # hurt the turn — the background driver owns it — but it would
            # return before `done` carries the committed record.)
            error_message = (data or {}).get("message", "?")
        elif kind == "done":
            done = data
    if done is None:
        # The generator is exhausted (commit path ran or genuinely never
        # produced a result), so raising is safe now.
        raise HTTPException(
            502 if error_message else 500,
            f"LLM call failed: {error_message}" if error_message
            else "turn ended without a result")

    response = done.get("response") or {}
    out = {
        "observation": observation,
        "response": response,
        "response_text": (response.get("content") or {}).get("text", ""),
        "retrieved_indices": done.get("retrieved_indices", []),
        "reflection": reflection,
        "cambium": cambium,
        "truncated": done.get("truncated", False),
        "tool_budget_exhausted": done.get("tool_budget_exhausted", False),
    }
    if error_message:
        # The turn committed (with the stream-error note sealed into the
        # response) but was cut short — let non-SSE clients see why.
        out["stream_error"] = error_message
    # Extra signals the stream carries (poq, quarantined, refused) ride along.
    for key in ("poq", "quarantined", "refused"):
        if key in done:
            out[key] = done[key]
    return out


async def _turn_events(user_input: str):
    """Drive ONE full turn as an async event stream.

    Shared by /api/turn/stream (which serves the events as SSE) and
    /api/turn (which drains them and returns the final payload as JSON):
    one implementation, two transports. Both get the same immune screen,
    PoQ scoring, and commit discipline, and the same thread-dispatched LLM
    calls — the event loop is never parked on the network. Chain and index
    writes stay on the loop thread (SQLite same-thread).
    """
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
                # Single-record turn shape: the refused turn is ONE
                # quarantined record (input as content.context + refusal).
                state.index.index_record(refused.response_record)
                yield {
                    "event": "observation",
                    "data": json.dumps({"type": "observation",
                                        "content": {"text": user_input}}),
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

        # 1. Pre-LLM half: retrieve, build prompt. `prepare_turn` is the
        # SAME method `Agent.turn` uses, so the streaming path can never
        # silently diverge in metadata or quarantine handling.
        # Single-record turn shape: NO observation record is committed —
        # the user's input seals into the response record as
        # content.context at commit. The browser still gets its YOU
        # bubble immediately via a synthetic observation event.
        prep = agent.prepare_turn(
            user_input, retrieve_k=SEMANTIC_K, n_recent=RECENT_N
        )

        # Uploads staged since the last turn ride THIS turn: prompt note
        # + native payloads now, pointer entries sealed into the response
        # record at commit (content.attachments).
        staged_attachments = agent.consume_staged_attachments(
            prep, state.tool_ctx)

        yield {
            "event": "observation",
            "data": json.dumps({"type": "observation",
                                "content": {"text": user_input}}),
        }

        # 2. Call the model — via inner helpers, because the tool loop
        # below re-calls the LLM once per round and every round must
        # stream identically. Each blocking network call runs in a
        # thread; chain ops stay on the main thread.
        import threading
        llm = agent.llm

        class _StreamFailed(Exception):
            """The LLM stream raised; message is client-ready."""

        async def stream_one_call(call_prompt: str, call_kwargs: dict):
            """Yield text chunks for ONE LLM call.

            Cancellation: the producer thread checks cancel_event after
            every chunk it tries to enqueue. If the consumer's `finally`
            fires, the producer stops iterating the LLM stream instead of
            generating tokens nobody reads. Since turns moved to a
            server-owned background task, the only consumer is
            `_drive_turn` — so this now fires only at server shutdown,
            never on a browser disconnect (a disconnect detaches a
            `_follow_turn` subscriber and the turn keeps generating).
            """
            cancel_event = threading.Event()
            try:
                if _llm_supports_streaming(llm):
                    queue: asyncio.Queue = asyncio.Queue()
                    loop = asyncio.get_running_loop()

                    def producer():
                        try:
                            for chunk in llm.stream(call_prompt,
                                                    **call_kwargs):
                                if cancel_event.is_set():
                                    return
                                asyncio.run_coroutine_threadsafe(
                                    queue.put(("chunk", chunk)), loop
                                )
                        except Exception as e:
                            if not cancel_event.is_set():
                                asyncio.run_coroutine_threadsafe(
                                    queue.put(
                                        ("error",
                                         f"{type(e).__name__}: {e}")),
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

                    producer_future = loop.run_in_executor(None, producer)
                    while True:
                        # Never wait unboundedly on the queue: if the
                        # producer thread dies without enqueueing its
                        # 'end' marker (e.g. the loop was closing when
                        # its finally ran), a bare queue.get() would
                        # hang this stream forever. Poll with a timeout
                        # and exit once the producer is gone and the
                        # queue is drained.
                        try:
                            kind, payload = await asyncio.wait_for(
                                queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            if producer_future.done() and queue.empty():
                                break
                            continue
                        if kind == "chunk":
                            yield payload
                        elif kind == "error":
                            raise _StreamFailed(payload)
                        else:  # end
                            break
                else:
                    # Fallback: non-streaming client. Run the LLM call
                    # in a thread so the event loop can keep serving
                    # other requests while we wait. Failures are wrapped
                    # in _StreamFailed so run_llm_round's handler converts
                    # them into a committed turn, exactly like a streaming
                    # failure — a raw exception here would escape the
                    # whole generator and skip the guaranteed commit.
                    try:
                        if call_kwargs:
                            full_text = await asyncio.to_thread(
                                llm, call_prompt, **call_kwargs)
                        else:
                            full_text = await asyncio.to_thread(
                                llm, call_prompt)
                    except Exception as e:
                        raise _StreamFailed(
                            f"{type(e).__name__}: {e}") from e
                    yield full_text
            finally:
                # Always signal cancel on the way out — covers both
                # normal completion (already-set is a no-op) and
                # asyncio.CancelledError from a disconnect.
                cancel_event.set()

        # One LLM round: stream tokens as SSE events, leave the full
        # text in holder["text"] (None when the stream errored). On
        # failure the error message lands in holder["error"] — the turn
        # must STILL commit (a sealed observation with no paired response
        # strands the chain and makes the turn invisible to memory, the
        # "I don't see the prior turn" failure), so callers fall through
        # to commit whatever prose completed rather than returning.
        holder: dict = {"text": None, "error": None}

        async def run_llm_round(call_prompt: str, call_kwargs: dict):
            parts: list[str] = []
            holder["text"] = None
            holder["error"] = None
            try:
                async for chunk in stream_one_call(call_prompt,
                                                   call_kwargs):
                    parts.append(chunk)
                    yield {"event": "token",
                           "data": json.dumps({"text": chunk})}
            except _StreamFailed as e:
                holder["error"] = str(e)
                yield {"event": "error",
                       "data": json.dumps({"message": str(e)})}
                return
            holder["text"] = "".join(parts)

        # Tool setup — parity with Agent.turn_with_tools: the Tier-2
        # pin never leaks across turns, and the tool schemas ride the
        # system prompt as TEXT for this call only.
        import tools as tools_mod
        tools_on = TOOLS_ENABLED and state.tool_ctx is not None
        llm_kwargs = dict(prep.llm_kwargs)
        if tools_on:
            state.tool_ctx.pinned_path = None
            state.tool_ctx.recalled_refs = []   # refs never leak across turns
            base_system = llm_kwargs.get("system") or ""
            llm_kwargs["system"] = (
                base_system + "\n\n" + tools_mod.tools_prompt()
                + tools_mod.workspace_prompt(state.tool_ctx)).strip()

        prompt = prep.prompt
        stream_failed = False
        async for ev in run_llm_round(prompt, llm_kwargs):
            yield ev
        if holder["text"] is None:
            # First-round stream failure: still commit (an empty response
            # carrying the error note) so the observation is paired.
            stream_failed = True
            response_text = ""
        else:
            response_text = holder["text"]

        # 2b. The tool loop — the async twin of Agent.turn_with_tools.
        # Parsing, validation, execution, and escaping all come from
        # tools.py (the SINGLE shared driver), so the Tier-2/Tier-3
        # safety gates cannot drift between the REPL and the web. Only
        # how the LLM is awaited and how output is emitted differ.
        # The round budget is read fresh from tools.DEFAULT_MAX_TOOL_ROUNDS
        # — the same call-time read the REPL loop does.
        # Prose that accompanied tool-call rounds (parity with
        # turn_with_tools): the committed response is ALL of it plus the
        # final answer, matching what streamed to the browser.
        prose_segments: list = []
        resolutions: list = []
        rounds = 0
        reflected = False
        budget_exhausted = False
        max_tool_rounds = tools_mod.DEFAULT_MAX_TOOL_ROUNDS
        while tools_on and not stream_failed and rounds < max_tool_rounds:
            # Mitigation 1: ONLY the fresh model segment is scanned —
            # never the accumulated prompt, whose tool results and file
            # content are escaped on entry (mitigation 2).
            calls, parse_errors = tools_mod.extract_tool_calls(
                response_text)
            if not calls:
                if (not reflected
                        and (parse_errors
                             or tools_mod.looks_like_intended_tool_call(
                                 response_text))):
                    # Mitigation 5: ONE reflective retry when the model
                    # clearly intended a tool call that failed to parse.
                    reflected = True
                    prompt += ("\n" + response_text
                               + tools_mod.tool_retry_prompt(parse_errors))
                    async for ev in run_llm_round(prompt, llm_kwargs):
                        yield ev
                    if holder["text"] is None:
                        # Retry-round stream failure: same rule as the
                        # mid-loop handler below — never drop the turn.
                        # Break to the commit path so the prose from
                        # completed rounds is sealed and paired.
                        stream_failed = True
                        break
                    response_text = holder["text"]
                    continue
                break    # a final answer — leave the loop
            rounds += 1
            prompt += "\n" + response_text
            prose = tools_mod.strip_tool_markup(response_text)
            if prose:
                prose_segments.append(prose)
            for call in calls:
                name = call.get("name", "?")
                if (isinstance(name, str)
                        and tools_mod.requires_confirmation(
                            name, call.get("arguments", {}),
                            state.tool_ctx)):
                    # No inline confirm hook over SSE — defer the call as
                    # a pending op instead of dead-ending it: the UI pops
                    # the approve/reject card (pending_op event below) and
                    # ONLY the user-only approve path can execute it.
                    # (Covers CONFIRM_TOOLS and a task_open whose
                    # source_root would expand the read boundary.)
                    result = tools_mod.defer_tool_call(call, state.tool_ctx)
                else:
                    # Run in a worker thread: an unconfirmed task_open on
                    # the current workspace walks, seals, and embeds an
                    # entire source tree — inline it would park the event
                    # loop (frozen SSE, stalled endpoints) for the
                    # duration. The chain/index connections are opened
                    # with check_same_thread=False and all access stays
                    # serialized under state.lock, so the thread hop is
                    # safe.
                    result = await asyncio.to_thread(
                        tools_mod.execute_tool, call, state.tool_ctx)
                yield {
                    "event": "tool_result",
                    "data": json.dumps({
                        "tool": name,
                        "round": rounds,
                        "result": (result if len(result) <= 4000
                                   else result[:4000] + "…"),
                    }),
                }
                # Mid-turn approval gate (v1.4.x). A freshly created pending
                # op (a write_file proposal OR a deferred confirmation-gated
                # tool call) PAUSES the turn: the UI pops its approve/reject
                # card (pending_op event, inline=true), the loop parks on a
                # future the endpoints resolve lock-free, and the REAL
                # outcome — written/rejected/expired — is what the model
                # sees as the tool result. The turn cannot end with the op
                # unresolved: no decision within the op's TTL auto-expires
                # it (recommendation: never leave requests lingering).
                try:
                    parsed = json.loads(result)
                except (ValueError, TypeError):
                    parsed = None
                if (isinstance(parsed, dict)
                        and parsed.get("status") == "confirmation_required"
                        and parsed.get("pending_op_id")):
                    parsed["inline"] = True   # tells the UI the turn waits
                    yield {"event": "pending_op",
                           "data": json.dumps(parsed)}
                    op_id = parsed["pending_op_id"]
                    fut = asyncio.get_running_loop().create_future()
                    state.approval_waiters[op_id] = fut
                    try:
                        decision = await asyncio.wait_for(
                            fut, timeout=parsed.get("expires_in_seconds")
                            or PENDING_TTL_SECONDS)
                    except asyncio.TimeoutError:
                        decision = "expired"
                    finally:
                        state.approval_waiters.pop(op_id, None)
                    entry, result = await asyncio.to_thread(
                        pending_ops_resolve_inline,
                        op_id, decision, state.tool_ctx)
                    if entry is not None:
                        resolutions.append(entry)
                        yield {"event": "op_resolved",
                               "data": json.dumps(entry)}
                prompt += tools_mod.format_tool_result(name, result)
            if rounds >= max_tool_rounds:
                # The next LLM call is the last one this turn gets — tell
                # it so, or it spends the call emitting tool calls that
                # will be silently dropped instead of an answer.
                prompt += tools_mod.TOOL_BUDGET_NUDGE
            async for ev in run_llm_round(prompt, llm_kwargs):
                yield ev
            if holder["text"] is None:
                # Mid-loop stream failure: DON'T drop the turn. Break to
                # the commit path so the prose from completed rounds is
                # sealed (paired with the observation) instead of lost.
                stream_failed = True
                break
            response_text = holder["text"]
        else:
            if tools_on and not stream_failed:
                # Round cap hit: surface it rather than silently
                # truncating work (parity with turn_with_tools). The
                # notice is also STREAMED — it is appended after the
                # token stream ended, so without this yield the browser
                # never sees why the turn stopped.
                budget_exhausted = True
                cap_note = tools_mod.tool_cap_note(max_tool_rounds)
                response_text += cap_note
                yield {"event": "token",
                       "data": json.dumps({"text": cap_note})}

        # The committed response is everything the user saw: prose from
        # every tool round plus the final answer (parity with
        # turn_with_tools — only the last fragment would otherwise be
        # sealed and reload out of context). The final round is stripped
        # too: echoed <tool_result> walls must not reach the chain.
        final = tools_mod.strip_tool_markup(response_text)
        response_text = "\n\n".join(prose_segments
                                    + ([final] if final else []))

        # A stream that failed mid-turn still commits — with an honest note
        # so the record (and the next turn's memory) shows the turn was cut
        # short rather than silently losing it. Streamed to the browser too.
        if stream_failed:
            err = holder.get("error") or "the model stream failed"
            note = (f"\n\n[stream error] this turn was cut short before "
                    f"completing: {err}")
            response_text = (response_text + note).strip()
            yield {"event": "token", "data": json.dumps({"text": note})}

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
        # Same persistence rule for the tool-budget flag: a later
        # "continue" resumes the task with a fresh budget (see
        # _format_prompt's continue-after-budget handling).
        if budget_exhausted:
            response_meta_kwargs["tool_budget_exhausted"] = True
        # Late drain: a model-initiated ingest_blob DURING this turn staged
        # its pointer after the turn-start drain — fold those in too, so a
        # pointer never waits a turn it didn't have to.
        if state.tool_ctx is not None:
            staged_attachments += state.tool_ctx.drain_staged_attachments()
        recalled = (state.tool_ctx.drain_recalled_refs()
                    if state.tool_ctx is not None else [])
        response = agent.commit_response(
            prep, response_text, response_meta_kwargs,
            resolutions=resolutions,
            attachments=staged_attachments,
            extra_refs=recalled,
        )
        state.index.index_record(response)
        state.turns_since_reflect += 1
        state.turns_since_cambium += 1

        done_payload = {
            "response": _record_to_dict(response),
            "retrieved_indices": [r.index for r in prep.context],
            "truncated": response_was_truncated,
            "tool_budget_exhausted": budget_exhausted,
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
            reflection_rec = agent.reflect(max_records=MAX_REFLECT_RECORDS)
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


@app.get("/api/turn/stream")
async def turn_stream(request: Request, input: str = "", session: str = "",
                      attach: str = ""):
    """
    Streaming turn via Server-Sent Events — a VIEWER onto a background turn.

    The turn itself runs in a server-owned task (`_drive_turn`); this
    endpoint only subscribes to its event buffer. That split is what makes
    the chat survive navigation: closing the EventSource (audit tab,
    reload, network blip) detaches the subscriber, while the turn runs to
    completion and commits. An earlier version drove the whole turn inside
    this response's generator, so leaving the page cancelled the turn
    mid-flight and stranded the sealed observation with no paired response.

    Modes:
      - `input=...` (no Last-Event-ID): start a new background turn and
        follow it. 409 if one is already running — except an identical
        re-issue of the running turn's input, which re-attaches (an
        EventSource auto-reconnect replays the same URL).
      - `attach=1`: follow the active turn without starting one (the chat
        page reattaching after navigation; 404 when there is none).
      - Last-Event-ID header present: always a reconnect — resume the
        active run past the already-delivered events, and NEVER start a
        fresh turn from a stale start-URL.

    Implementation note: the LLM call sits between two chain-write halves
    that need to share state (the observation, the context, the prompt).
    `_turn_events` uses `Agent.prepare_turn` -> LLM -> `Agent.score_response`
    -> `Agent.commit_response` so the streaming path goes through the same
    PoQ scoring, quarantine routing, and observation-indexing order as
    `Agent.turn()`.
    """
    _require_session(request)
    run = getattr(state, "active_turn", None)
    last_seq = _last_event_seq(request)

    if attach:
        if run is None:
            raise HTTPException(404, "no turn to attach to")
        return EventSourceResponse(_follow_turn(run, after=last_seq))

    user_input = input.strip()
    if not user_input:
        raise HTTPException(400, "empty input")

    if run is not None and not run.done:
        if user_input == run.input:
            # Idempotent re-issue of the running turn (EventSource
            # auto-reconnect repeats the start URL) — re-attach, never
            # start a duplicate.
            return EventSourceResponse(_follow_turn(run, after=last_seq))
        raise HTTPException(status_code=409, detail=_TURN_IN_PROGRESS)

    if last_seq:
        # A reconnect that outlived its turn: replay the tail (which ends
        # with `done`) when it matches, otherwise report it gone. A
        # reconnect must never START a turn — the input in the URL was
        # already consumed by the run it belongs to.
        if run is not None and user_input == run.input:
            return EventSourceResponse(_follow_turn(run, after=last_seq))
        raise HTTPException(404, "that turn already finished")

    run = _start_turn(user_input)
    return EventSourceResponse(_follow_turn(run))



# ---------------------------------------------------------------------------
# Pending write operations — the durable write gate (Tier 3). The model only
# ever CREATES a PendingOperation (via write_file); these endpoints are how
# the USER approves or rejects it. They are never called autonomously.
# ---------------------------------------------------------------------------

def _require_tools() -> None:
    if not (TOOLS_ENABLED and state.tool_ctx is not None):
        raise HTTPException(404, "tools are disabled")


def _pending_op_to_dict(op) -> dict:
    """Session-holder's view of a PendingOperation. Includes the proposed
    content — the approval dialog must show the user exactly what will be
    written (capped at 1MB by PendingOpStore.create) — or, for a deferred
    tool call, the exact tool name and arguments that would run."""
    out = {
        "id": op.id,
        "kind": getattr(op, "kind", "write"),
        "status": op.status,
        "task": op.task_name or "(no active task)",
        "file": op.file_path,
        "change": op.change_summary,
        "new_file": not op.target_existed,
        "content_chars": len(op.proposed_content),
        "proposed_content": op.proposed_content,
        # Generated formats (docx): proposed_content above is the readable
        # SOURCE; this names the binary that will actually be written.
        "generated_format": getattr(op, "generated_format", ""),
        "expired": op.expired(),
        "expires_at": op.expires_at,
    }
    if out["kind"] == "tool_call":
        out["tool"] = op.tool_name
        try:
            out["arguments"] = json.loads(op.tool_args_json)
        except (ValueError, TypeError):
            out["arguments"] = {}
    return out


@app.get("/api/workspace")
async def workspace_get(request: Request):
    """Current workspace + selector suggestions (task source_roots and
    recent choices — never a directory listing, so the web session cannot
    enumerate the server's filesystem)."""
    _require_session(request)
    _require_tools()
    import tools as tools_mod
    async with state.lock:
        ctx = state.tool_ctx
        return {"current": str(ctx.workspace_root),
                "suggestions": tools_mod.workspace_suggestions(ctx)}


@app.post("/api/workspace")
async def workspace_set(request: Request):
    """USER-only workspace switch — the selector is the user's hand on the
    read/write boundary, which is exactly why the model has no tool for
    it. A pure boundary move: nothing is created, sealed, or ingested."""
    _require_session(request)
    _require_tools()
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="body must be valid JSON")
    path = body.get("path") if isinstance(body, dict) else None
    if not isinstance(path, str) or not path:
        raise HTTPException(status_code=400, detail="missing 'path' string")
    import tools as tools_mod
    async with state.lock:
        try:
            current = tools_mod.set_workspace(state.tool_ctx, path)
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "current": current}


@app.get("/api/pending-ops")
async def pending_ops_list(request: Request):
    _require_session(request)
    _require_tools()
    # Deliberately NOT under state.lock: a turn paused on its mid-turn
    # approval gate HOLDS the lock, and this endpoint is how the UI's
    # banner learns there is something to approve — locking here would
    # deadlock the very refresh that renders the approve button. Reading
    # is safe lock-free: the store's writes are atomic (tmp + os.replace),
    # so a concurrent load sees either the old or the new file, never a
    # torn one.
    ops = []
    for op_id in state.tool_ctx.pending_ops.list_ids():
        op = state.tool_ctx.pending_ops.load(op_id)
        if op is not None:
            ops.append(_pending_op_to_dict(op))
    return {"pending": ops}


def _deliver_to_waiter(op_id: str, action: str) -> Optional[dict]:
    """Mid-turn decision delivery. If a streaming turn is parked on this
    op, resolve its future and return the response — WITHOUT acquiring
    state.lock (the turn holds it; the legacy path below would deadlock).
    The turn itself executes the approval under the lock it already owns.
    Returns None when no turn is waiting (legacy post-turn op)."""
    fut = getattr(state, "approval_waiters", {}).get(op_id)
    if fut is None or fut.done():
        return None
    decision = "approved" if action == "approve_write" else "rejected"
    fut.set_result(decision)
    # delivered=True tells the UI to stay quiet: the turn's op_resolved
    # event reports the real outcome, and two messages for one decision
    # read as two events.
    return {"ok": True, "delivered": True,
            "result": f"decision ({decision}) delivered to the running "
                      f"turn — it resolves the operation and continues"}


async def _pending_op_action(request: Request, op_id: str,
                             action: str) -> dict:
    _require_session(request)
    _require_tools()
    delivered = _deliver_to_waiter(op_id, action)
    if delivered is not None:
        return delivered
    import tools as tools_mod
    # The lock is held for the whole approve (atomic write + idempotent
    # ingest + audit) — same single-writer guarantee the REPL gets for
    # free. The execution itself runs in a worker thread: write approvals
    # are milliseconds, but an approved DEFERRED tool call can be a
    # task_reembed or a full-tree task_open ingest (minutes to hours) —
    # run inline it would freeze the event loop, killing SSE heartbeats
    # and every other endpoint for the duration. Other turns still queue
    # behind the lock, but the server stays responsive.
    async with state.lock:
        ok, result = await asyncio.to_thread(
            tools_mod.run_user_action,
            action, {"pending_op_id": op_id}, state.tool_ctx)
    return {"ok": ok, "result": result}


@app.post("/api/pending-ops/{op_id}/approve")
async def pending_op_approve(request: Request, op_id: str):
    return await _pending_op_action(request, op_id, "approve_write")


@app.post("/api/pending-ops/{op_id}/reject")
async def pending_op_reject(request: Request, op_id: str):
    return await _pending_op_action(request, op_id, "reject_write")


# ---------------------------------------------------------------------------
# Content ingestion (Phase 14) — the upload path, rebuilt on ingest_blob.
# USER-triggered (attach button / drag-drop); routes by context exactly like
# the model-callable tool: active task -> task chain + workspace, otherwise
# identity-chain attachment + content-addressed blob.
# ---------------------------------------------------------------------------

@app.get("/api/tasks/active")
async def tasks_active(request: Request):
    """The session's task cursor — the UI uses it to offer (never assume)
    routing an upload into the active task. The reserved artifacts task is
    never reported: it is the default destination, not a user task."""
    _require_session(request)
    import tools as tools_mod
    active = None
    if state.tool_ctx is not None:
        active = state.tool_ctx.active_task
        if active == tools_mod.ARTIFACTS_TASK_NAME:
            active = None
    return {"active": active}


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...),
                 task: str = Form("")):
    """`task` (optional form field): explicitly seal the upload into that
    open task's chain + workspace. Default (empty) routes to the reserved
    artifacts chain — an active task never captures uploads silently."""
    _require_session(request)
    _require_tools()
    import base64
    import tools as tools_mod
    # Enforce the ingest cap BEFORE buffering the whole upload: read at most
    # cap+1 bytes and reject on overflow, so an oversized upload can't
    # exhaust memory just to be refused afterwards.
    cap = tools_mod.INGEST_BLOB_MAX_BYTES
    data = await file.read(cap + 1)
    if len(data) > cap:
        raise HTTPException(
            413, f"upload exceeds the {cap}-byte ingest cap")
    async with state.lock:
        # Worker thread: ingest hashes, extracts, and embeds the upload —
        # blocking work that must not park the event loop (same rule as
        # tool execution; connections are check_same_thread=False and
        # serialized under the lock).
        ingest_kwargs = {
            "content": base64.b64encode(data).decode("ascii"),
            "name": file.filename or "upload.bin",
            "mime_type": file.content_type or "application/octet-stream",
            "encoding": "base64",
        }
        if task.strip():
            ingest_kwargs["task_name"] = task.strip()
        result = await asyncio.to_thread(
            tools_mod.execute_ingest_blob, ingest_kwargs, state.tool_ctx,
        )
        if tools_mod.is_error_result(result):
            raise HTTPException(400, result)
        # Identity-route ingests STAGE a pointer (single-record turn
        # shape): it seals into the next turn's response record, with the
        # message it accompanied. The UI renders its chip from `staged`.
        # Task-route ingests land in the task chain (no staging).
        staged_entry = None
        staged = getattr(state.tool_ctx, "staged_attachments", None) or []
        if staged and not task.strip():
            staged_entry = staged[-1]
    return {"record": None, "staged": staged_entry,
            "staged_count": len(staged), "result": result}


@app.get("/blobs/{sha}")
async def serve_blob(sha: str, session: Optional[str] = None,
                     request: Request = None):
    """Serve a content-addressed blob (identity-route ingest_blob storage).

    Requires the active session token — passed as a `?session=` query
    parameter so `<img src>` tags can carry it, or as the x-session-token
    header. The rest of the API is single-session-locked; without this,
    anyone with reach to the port could pull every ingested blob by
    guessing or harvesting sha values.
    """
    token = (session
             or (request.headers.get("x-session-token") if request else None))
    if not token or state.active_token != token:
        raise HTTPException(
            status_code=409,
            detail={"error": "session_inactive",
                    "message": "blob access requires the active session token"},
        )
    if not all(c in "0123456789abcdef" for c in sha) or len(sha) != 64:
        raise HTTPException(400, "not a blob hash")
    # resolve_blob_path knows both layouts: sharded (current ingest_blob)
    # and legacy flat (the removed file_ingest pipeline) — pre-existing
    # data dirs keep serving their old images/PDFs.
    import tools as tools_mod
    path = tools_mod.resolve_blob_path(DATA_DIR / "blobs", sha)
    if path is None:
        raise HTTPException(404, "no such blob")
    # Recover the MIME type from the record that sealed this blob: indexed
    # O(1) via blob_index (which covers file/attachment records AND
    # attachment entries embedded in response records), with a linear-scan
    # fallback for attachments sealed before the index covered them.
    def _pointer_for(record) -> Optional[dict]:
        c = record.content if isinstance(record.content, dict) else None
        if c is None:
            return None
        if record.type in ("file", "attachment") and c.get("blob_sha256") == sha:
            return c
        if record.type == "response":
            for e in c.get("attachments") or []:
                if isinstance(e, dict) and e.get("blob_sha256") == sha:
                    return e
        return None

    media_type = "application/octet-stream"
    pointer = None
    rec = state.chain.find_file_by_sha(sha) if state.chain is not None else None
    if rec is not None:
        pointer = _pointer_for(rec)
    if pointer is None and state.chain is not None:
        for r in state.chain.iter_records():
            pointer = _pointer_for(r)
            if pointer is not None:
                break
    if pointer is not None:
        media_type = pointer.get("mime_type") or media_type
    return FileResponse(path, media_type=media_type)


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


def _audit_chain_for(task: Optional[str]):
    """Resolve which chain the audit endpoints read: the identity chain
    (no `task` param), or a per-task continuum chain by registry name.
    Task chains and the identity chain share the Chain class, so the
    whole audit pipeline works on either."""
    if not task:
        return state.chain
    if state.tool_ctx is None:
        raise HTTPException(400, "tools are disabled — no task chains")
    if state.tool_ctx.registry.get(task) is None:
        raise HTTPException(404, f"no such task: {task}")
    return state.tool_ctx.get_task_chain(task)


@app.get("/api/audit")
async def api_audit(request: Request, task: Optional[str] = None):
    """Read-only audit snapshot for the dashboard. Session-gated like
    /api/chain/* — the ring list carries record summaries (memory content).
    The audit page reads the token index.html stored in localStorage.
    Pass ?task=<name> to audit a task chain instead of the identity
    chain; the response always lists the available task chains so the
    dashboard can render its chain selector."""
    _require_session(request)
    import audit as _audit
    from pathlib import Path as _Path
    chain = _audit_chain_for(task)
    ok, _msg = chain.verify()
    faculty_dir = _Path(__file__).resolve().parent.parent / "faculties"
    # blob storage was removed with file_ingest (v1.4); audit's
    # blockspace section degrades gracefully without a blob_dir.
    result = _audit.compute(chain, faculty_dir, integrity=ok)
    result["data_dir"] = str(DATA_DIR)
    result["viewing"] = task or ""
    result["task_chains"] = ([
        {"name": name, "status": t.get("status", "?"),
         "items_done": t.get("items_done", 0),
         "items_total": t.get("items_total", 0),
         "source_root": t.get("source_root", "")}
        for name, t in state.tool_ctx.registry.list_all()
    ] if state.tool_ctx is not None else [])
    return result


@app.get("/api/audit/ring/{idx}")
async def api_audit_ring(idx: int, request: Request,
                         task: Optional[str] = None):
    """Full contents of one ring, for the audit dashboard's detail pane. The
    /api/audit ring list carries only truncated summaries (to keep that payload
    small); this returns the complete record on demand. Same ?task= rule
    as /api/audit."""
    _require_session(request)
    rec = _audit_chain_for(task).get(idx)
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
