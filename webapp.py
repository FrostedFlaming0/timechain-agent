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
    /reflect, /revise N <text>, /file <path>.
  - Single-session lock: only one browser tab is "active" at a time. A second
    tab can take over, but they don't run concurrently — protects the chain's
    single-writer guarantee.

What's NOT included on purpose:
  - Authentication. This binds to 127.0.0.1 by default. Don't expose it on
    a network without putting auth in front of it; the operator key lives
    in this process.
  - Multi-user. One operator, one chain, one signing key.
  - Background reflection cadence — that runs the same way as run.py:
    every AUTO_REFLECT_EVERY successful turns.

Run:
    pip install fastapi uvicorn sse-starlette python-multipart
    python webapp.py

Then open the URL it prints (default http://127.0.0.1:8765).
"""

from __future__ import annotations

import asyncio
import json
import secrets
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
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
from agent import Agent
from run import (
    DATA_DIR,
    LLM_PROVIDER,
    FOUNDING_COMMITMENTS,
    SYSTEM_PROMPT,
    SEMANTIC_K,
    EMBED_DIM,
    AUTO_REFLECT_EVERY,
    REFLECT_WINDOW,
    build_llm,
    make_sentence_embedder,
)


# ---------------------------------------------------------------------------
# Server config
# ---------------------------------------------------------------------------

HOST = "127.0.0.1"
PORT = 8765
STATIC_DIR = Path(__file__).parent


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
        print("[boot] loading model + embedder...")

        key = load_or_create_key(key_path)
        self.chain = Chain(chain_db, key)

        embedder = make_sentence_embedder()
        self.index = EmbeddingIndex(embed_db, embedder, dim=EMBED_DIM)
        added = self.index.index_chain(self.chain)
        if added:
            print(f"[boot] indexed {added} pre-existing records")

        retriever = Retriever(self.chain, self.index)
        llm = build_llm()
        self.agent = Agent(
            self.chain, retriever, llm,
            system_prompt=SYSTEM_PROMPT, blob_dir=blob_dir,
        )

        # First-run genesis (mirrors run.py exactly).
        if self.chain.length() == 0:
            print("[boot] first run — committing genesis")
            genesis = self.agent.commit_genesis(FOUNDING_COMMITMENTS)
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


@app.post("/api/session/claim")
async def session_claim():
    """Mint a fresh token. Any previously active tab is bumped."""
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
async def chain_verify():
    ok, msg = state.chain.verify(expected_pubkey=state.chain.pubkey_hex)
    return {"ok": ok, "message": msg, "length": state.chain.length()}


@app.get("/blobs/{sha}")
async def serve_blob(sha: str):
    """
    Serve an ingested file's bytes by sha256 — used for image rendering in
    chat. We look up the file record to pick a sensible content-type and
    confirm the blob actually exists on the chain (don't serve random files).
    """
    # Find a file record with this sha. Linear scan but bounded; the file
    # record count is small in practice.
    file_records = state.chain.query_by_type("file", limit=500)
    rec = next(
        (r for r in file_records if r.content.get("blob_sha256") == sha),
        None,
    )
    if rec is None:
        raise HTTPException(404, "no file record references that hash")
    blob_dir = DATA_DIR / "blobs"
    path = blob_dir / rec.content.get("blob_path", "")
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
            rec = state.agent.reflect(window=REFLECT_WINDOW)
            if rec is None:
                return {"kind": "reflect", "result": "not_enough_history"}
            state.index.index_record(rec)
            state.turns_since_reflect = 0
            return {"kind": "reflect", "record": _record_to_dict(rec)}

        if cmd.startswith("/revise"):
            parts = cmd.split(maxsplit=2)
            if len(parts) < 3:
                raise HTTPException(400, "usage: /revise <index> <correction text>")
            try:
                target_idx = int(parts[1])
            except ValueError:
                raise HTTPException(400, f"invalid index: {parts[1]!r}")
            rec = state.agent.revise(target_idx, parts[2])
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

        raise HTTPException(400, f"unknown command: {cmd}")


# ---------------------------------------------------------------------------
# File upload (drag-and-drop) — calls agent.ingest_file under the hood
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    _require_session(request)
    # Save to a temp path that preserves the original filename so the
    # ingest pipeline records the right .ext and metadata.
    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        async with state.lock:
            try:
                rec = state.agent.ingest_file(tmp_path)
            except (ValueError, OSError) as e:
                raise HTTPException(400, str(e))
            state.index.index_record(rec)
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
        turn_obj = state.agent.turn(user_input, SEMANTIC_K)
        state.index.index_record(turn_obj.observation_record)
        state.index.index_record(turn_obj.response_record)
        state.turns_since_reflect += 1

        # Auto-reflection mirrors run.py's behavior.
        reflection_rec = None
        if AUTO_REFLECT_EVERY > 0 and state.turns_since_reflect >= AUTO_REFLECT_EVERY:
            reflection_rec = state.agent.reflect(window=REFLECT_WINDOW)
            if reflection_rec is not None:
                state.index.index_record(reflection_rec)
            state.turns_since_reflect = 0

    return {
        "observation": _record_to_dict(turn_obj.observation_record),
        "response": _record_to_dict(turn_obj.response_record),
        "response_text": turn_obj.response_text,
        "retrieved_indices": [r.index for r in turn_obj.retrieved],
        "reflection": _record_to_dict(reflection_rec) if reflection_rec else None,
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

            # Chain ops are SQLite-on-local-disk: microseconds. Run them
            # inline in the event loop. The SQLite connection was opened
            # in the main thread and can't be used from asyncio.to_thread
            # workers, so we MUST keep these in the main thread.

            # 1. Commit observation up front so the user can see the index
            # immediately, and the response can ref it.
            obs = chain.append("observation", {"text": user_input})
            state.index.index_record(obs)
            yield {
                "event": "observation",
                "data": json.dumps(_record_to_dict(obs)),
            }

            # 2. Build context + prompt (reuses agent's logic).
            context = agent.retriever.build_context(
                query=user_input, k_semantic=SEMANTIC_K, n_recent=3,
            )
            prompt = agent._format_prompt(user_input, context)
            attachments = agent._collect_attachments(context)
            llm_kwargs = {}
            if agent.system_prompt:
                llm_kwargs["system"] = agent.system_prompt
            if attachments:
                llm_kwargs["attachments"] = attachments

            # 3. Call the model. THIS is what needs the thread — it's a
            # blocking network call that can take seconds. Chain ops above
            # and below stay on the main thread.
            full_text_parts: list[str] = []
            llm = agent.llm
            if _llm_supports_streaming(llm):
                # Streaming path — call llm.stream() in a thread and queue
                # chunks back to the event loop.
                queue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def producer():
                    try:
                        for chunk in llm.stream(prompt, **llm_kwargs):
                            asyncio.run_coroutine_threadsafe(
                                queue.put(("chunk", chunk)), loop
                            )
                    except Exception as e:
                        asyncio.run_coroutine_threadsafe(
                            queue.put(("error", f"{type(e).__name__}: {e}")), loop
                        )
                    finally:
                        asyncio.run_coroutine_threadsafe(
                            queue.put(("end", None)), loop
                        )

                loop.run_in_executor(None, producer)
                while True:
                    kind, payload = await queue.get()
                    if kind == "chunk":
                        full_text_parts.append(payload)
                        yield {"event": "token", "data": json.dumps({"text": payload})}
                    elif kind == "error":
                        yield {"event": "error", "data": json.dumps({"message": payload})}
                        return
                    else:  # end
                        break
            else:
                # Fallback: non-streaming client. Run the LLM call in a
                # thread so the event loop can keep serving other requests
                # while we wait.
                if llm_kwargs:
                    full_text = await asyncio.to_thread(llm, prompt, **llm_kwargs)
                else:
                    full_text = await asyncio.to_thread(llm, prompt)
                full_text_parts.append(full_text)
                yield {"event": "token", "data": json.dumps({"text": full_text})}

            response_text = "".join(full_text_parts)

            # 4. Commit response with refs. Back on the main thread.
            refs = [r.record_hash for r in context] + [obs.record_hash]
            response = chain.append("response", {"text": response_text}, refs=refs)
            state.index.index_record(response)
            state.turns_since_reflect += 1

            yield {
                "event": "done",
                "data": json.dumps({
                    "response": _record_to_dict(response),
                    "retrieved_indices": [r.index for r in context],
                }),
            }

            # 5. Auto-reflection. The reflect() call itself makes an LLM
            # call internally, so it blocks the loop briefly. Acceptable
            # for now; if it becomes an issue, refactor Agent.reflect to
            # split the LLM call out the way we did above.
            if AUTO_REFLECT_EVERY > 0 and state.turns_since_reflect >= AUTO_REFLECT_EVERY:
                reflection_rec = agent.reflect(window=REFLECT_WINDOW)
                if reflection_rec is not None:
                    state.index.index_record(reflection_rec)
                    yield {
                        "event": "reflection",
                        "data": json.dumps(_record_to_dict(reflection_rec)),
                    }
                state.turns_since_reflect = 0

    return EventSourceResponse(event_source())


# ---------------------------------------------------------------------------
# Static files + index page
# ---------------------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            "<h1>missing static/index.html</h1>"
            "<p>expected at: " + str(index_path) + "</p>",
            status_code=500,
        )
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
