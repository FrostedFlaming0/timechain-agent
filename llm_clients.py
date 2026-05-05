"""
llm_clients — production-ready LLM client builders for the timechain agent.

Each builder returns a callable str -> str that the Agent uses. They share
a common shape so you can swap providers in run.py with one line:

    llm = make_claude_client()
    # llm = make_openai_client()
    # llm = make_gemini_client()
    # llm = make_ollama_client()

All clients implement:
  - Sensible defaults for the current generation of models (as of May 2026).
  - Configurable model, max_tokens, temperature, timeout.
  - Retries with exponential backoff on transient errors.
  - Clear error messages when API keys are missing or the server is down.
  - Reasonable timeouts so a hung provider doesn't freeze the agent loop.

Model strings are current as of this writing but providers update fast.
If a model returns 404 or "model not found," check the provider's docs:
  Anthropic:  https://docs.claude.com/en/docs/about-claude/models
  OpenAI:     https://platform.openai.com/docs/models
  Google:     https://ai.google.dev/gemini-api/docs/models
  Ollama:     ollama list   (locally installed models)
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable, Optional

LLMCall = Callable[..., str]
# A client is callable as:
#   llm(user_prompt)
#   llm(user_prompt, system=system_prompt)
#   llm(user_prompt, system=system_prompt, attachments=[...])
#
# attachments: optional list of dicts describing image/PDF blobs to send
#   alongside the text prompt. Each attachment looks like:
#     {"kind": "image" | "pdf", "media_type": "image/jpeg", "data": <bytes>,
#      "filename": "..."}
#   Clients that don't support multimodal input simply ignore attachments
#   (they're already represented as text in the prompt anyway).
#
# Each client also exposes llm.stream(...) — a generator with the same
# signature that yields text chunks as they arrive from the model. The
# webapp checks for the presence of .stream() to decide whether to use
# server-sent events. Retry semantics for streaming: only the initial
# connection is retried; once chunks start flowing, a mid-stream failure
# raises (because re-emitting already-seen chunks would corrupt output).


# ---------------------------------------------------------------------------
# Shared retry helper
# ---------------------------------------------------------------------------

def _retry_with_backoff(
    fn: Callable[[], str],
    *,
    max_attempts: int = 4,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
) -> str:
    """
    Call fn() with retries on transient errors. Doubles delay each attempt
    (1s, 2s, 4s, 8s by default). Re-raises the last exception if all attempts fail.
    """
    delay = initial_delay
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except retryable_exceptions as e:
            last_exc = e
            if attempt == max_attempts:
                raise
            sys.stderr.write(
                f"[llm] attempt {attempt}/{max_attempts} failed ({type(e).__name__}): {e}; "
                f"retrying in {delay:.1f}s\n"
            )
            time.sleep(delay)
            delay *= backoff_factor
    # Unreachable, but satisfies type checker
    if last_exc:
        raise last_exc
    raise RuntimeError("retry loop exited without result")


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------

def make_claude_client(
    model: str = "claude-opus-4-7",
    max_tokens: int = 1024,
    temperature: float = 1.0,
    timeout_s: float = 60.0,
) -> LLMCall:
    """
    Anthropic Claude client.

    Default model is claude-opus-4-7 (most capable as of May 2026). For
    cost-sensitive workloads use claude-sonnet-4-6 or claude-haiku-4-5.
    Set ANTHROPIC_API_KEY in the environment.
    """
    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("set ANTHROPIC_API_KEY")

    client = anthropic.Anthropic(timeout=timeout_s)

    # Retry on rate limits, server errors, and connection issues. NOT on 4xx
    # client errors (bad request, invalid model) — those need code changes.
    retryable = (
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
    )

    def call(prompt: str, system: Optional[str] = None,
             attachments: Optional[list[dict]] = None) -> str:
        def _do():
            # Build user message: optional attachments first, then the text.
            # Claude accepts image and document content blocks alongside text.
            content_blocks: list = []
            if attachments:
                import base64
                for att in attachments:
                    data = att.get("data", b"")
                    if not isinstance(data, (bytes, bytearray)):
                        continue
                    b64 = base64.standard_b64encode(data).decode("ascii")
                    if att.get("kind") == "image":
                        content_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": att.get("media_type", "image/png"),
                                "data": b64,
                            },
                        })
                    elif att.get("kind") == "pdf":
                        content_blocks.append({
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": b64,
                            },
                        })
            content_blocks.append({"type": "text", "text": prompt})

            kwargs = dict(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": content_blocks}],
            )
            if system:
                kwargs["system"] = system
            msg = client.messages.create(**kwargs)
            return "".join(b.text for b in msg.content if hasattr(b, "text"))

        return _retry_with_backoff(_do, retryable_exceptions=retryable)

    def _build_messages_kwargs(prompt, system, attachments):
        """Shared message construction for both call and stream paths."""
        content_blocks: list = []
        if attachments:
            import base64
            for att in attachments:
                data = att.get("data", b"")
                if not isinstance(data, (bytes, bytearray)):
                    continue
                b64 = base64.standard_b64encode(data).decode("ascii")
                if att.get("kind") == "image":
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": att.get("media_type", "image/png"),
                            "data": b64,
                        },
                    })
                elif att.get("kind") == "pdf":
                    content_blocks.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": b64,
                        },
                    })
        content_blocks.append({"type": "text", "text": prompt})
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": content_blocks}],
        )
        if system:
            kwargs["system"] = system
        return kwargs

    def stream(prompt: str, system: Optional[str] = None,
               attachments: Optional[list[dict]] = None):
        """
        Yield text chunks as they arrive from the model. Generator.

        Retries the INITIAL connection only — once the stream is open and
        chunks are flowing, mid-stream failures aren't retried (because we'd
        have to re-emit chunks the caller already saw, which would be wrong).
        If you need mid-stream resilience, the caller should handle it.
        """
        def _open_stream():
            kwargs = _build_messages_kwargs(prompt, system, attachments)
            return client.messages.stream(**kwargs)

        # Retry just the connection-open step. Once we're inside the `with`,
        # we're committed to whatever chunks come out.
        stream_ctx = _retry_with_backoff(_open_stream, retryable_exceptions=retryable)
        with stream_ctx as s:
            for text in s.text_stream:
                yield text

    call.stream = stream
    return call


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def make_openai_client(
    model: str = "gpt-5.5",
    max_tokens: int = 1024,
    temperature: float = 1.0,
    timeout_s: float = 60.0,
) -> LLMCall:
    """
    OpenAI client using the Chat Completions API.

    Default model is gpt-5.5 (current flagship as of May 2026). For
    cost-sensitive workloads use gpt-5.4-mini or gpt-5.4-nano. Reasoning
    models (o-series) may have different parameter requirements; check docs.
    Set OPENAI_API_KEY in the environment.
    """
    try:
        from openai import OpenAI
        import openai as openai_mod
    except ImportError:
        sys.exit("pip install openai")

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("set OPENAI_API_KEY")

    client = OpenAI(timeout=timeout_s)

    retryable = (
        openai_mod.APIConnectionError,
        openai_mod.APITimeoutError,
        openai_mod.RateLimitError,
        openai_mod.InternalServerError,
    )

    def _build_messages(prompt, system, attachments):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        user_content: list = []
        if attachments:
            import base64
            for att in attachments:
                if att.get("kind") != "image":
                    continue
                data = att.get("data", b"")
                if not isinstance(data, (bytes, bytearray)):
                    continue
                media_type = att.get("media_type", "image/png")
                b64 = base64.standard_b64encode(data).decode("ascii")
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"},
                })
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def call(prompt: str, system: Optional[str] = None,
             attachments: Optional[list[dict]] = None) -> str:
        def _do():
            messages = _build_messages(prompt, system, attachments)
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            content = resp.choices[0].message.content
            return content if content is not None else ""

        return _retry_with_backoff(_do, retryable_exceptions=retryable)

    def stream(prompt: str, system: Optional[str] = None,
               attachments: Optional[list[dict]] = None):
        """Yield text chunks as they arrive. See Claude's stream() for retry semantics."""
        def _open_stream():
            messages = _build_messages(prompt, system, attachments)
            return client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                stream=True,
            )

        completion = _retry_with_backoff(_open_stream, retryable_exceptions=retryable)
        for chunk in completion:
            # OpenAI chunks have choices[0].delta.content; some chunks (role
            # markers, finish reasons) have content=None — skip those.
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                yield text

    call.stream = stream
    return call


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

def make_gemini_client(
    model: str = "gemini-3.1-pro",
    max_output_tokens: int = 1024,
    temperature: float = 1.0,
    timeout_s: float = 60.0,
) -> LLMCall:
    """
    Google Gemini client using the google-genai SDK (the current one;
    google-generativeai is the older deprecated package).

    Default model is gemini-3.1-pro (current flagship as of May 2026).
    For cost-sensitive workloads use gemini-3.1-flash or gemini-3.1-flash-lite.
    Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.
    """
    try:
        from google import genai
        from google.genai import types as genai_types
        from google.genai import errors as genai_errors
    except ImportError:
        sys.exit("pip install google-genai")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("set GEMINI_API_KEY or GOOGLE_API_KEY")

    client = genai.Client(api_key=api_key)

    # google-genai surfaces errors as ServerError / ClientError / APIError.
    # Retry on transient ones, not on 4xx-class ClientError.
    retryable = (genai_errors.ServerError, genai_errors.APIError, ConnectionError, TimeoutError)

    def _build_contents_and_cfg(prompt, system, attachments):
        cfg = genai_types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            system_instruction=system if system else None,
        )
        parts: list = []
        if attachments:
            for att in attachments:
                data = att.get("data", b"")
                if not isinstance(data, (bytes, bytearray)):
                    continue
                media_type = att.get("media_type")
                if not media_type:
                    if att.get("kind") == "image":
                        media_type = "image/png"
                    elif att.get("kind") == "pdf":
                        media_type = "application/pdf"
                    else:
                        continue
                parts.append(genai_types.Part.from_bytes(
                    data=bytes(data), mime_type=media_type
                ))
        parts.append(prompt)
        contents = parts if attachments else prompt
        return contents, cfg

    def call(prompt: str, system: Optional[str] = None,
             attachments: Optional[list[dict]] = None) -> str:
        def _do():
            contents, cfg = _build_contents_and_cfg(prompt, system, attachments)
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=cfg,
            )
            return resp.text or ""

        return _retry_with_backoff(_do, retryable_exceptions=retryable)

    def stream(prompt: str, system: Optional[str] = None,
               attachments: Optional[list[dict]] = None):
        """Yield text chunks as they arrive. See Claude's stream() for retry semantics."""
        def _open_stream():
            contents, cfg = _build_contents_and_cfg(prompt, system, attachments)
            return client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=cfg,
            )

        chunks = _retry_with_backoff(_open_stream, retryable_exceptions=retryable)
        for chunk in chunks:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    call.stream = stream
    return call


# ---------------------------------------------------------------------------
# Local Ollama
# ---------------------------------------------------------------------------

def make_ollama_client(
    model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    max_tokens: int = 1024,
    temperature: float = 1.0,
    timeout_s: float = 120.0,
) -> LLMCall:
    """
    Local Ollama client. Ollama serves an HTTP API on port 11434 by default.

    Default model is llama3.1:8b (good general default, ~5GB RAM at Q4).
    Other strong picks as of 2026:
      - qwen3:8b           — fast, capable, good agent / tool use
      - qwen2.5-coder:7b   — best small coding model
      - gemma4:9b          — Google's open model, ~6GB RAM
      - llama3.3:70b       — much stronger, ~40GB RAM
      - deepseek-r1:7b     — chain-of-thought reasoning
    Run `ollama list` to see what's installed locally; `ollama pull <name>`
    to download a new one.

    The longer default timeout reflects that local models can be slow on
    CPU-only or under-provisioned hardware.
    """
    try:
        import requests
    except ImportError:
        sys.exit("pip install requests")

    url = f"{base_url.rstrip('/')}/api/generate"

    # Quick connectivity probe so we fail fast with a clear message
    try:
        requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
    except requests.exceptions.RequestException as e:
        sys.exit(
            f"cannot reach Ollama at {base_url}: {e}\n"
            "is the server running? try: `ollama serve` (or check the app)"
        )

    retryable = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    )

    def _build_payload(prompt, system, attachments, stream_flag: bool):
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream_flag,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            payload["system"] = system
        if attachments:
            import base64
            imgs = []
            for att in attachments:
                if att.get("kind") != "image":
                    continue
                data = att.get("data", b"")
                if not isinstance(data, (bytes, bytearray)):
                    continue
                imgs.append(base64.standard_b64encode(data).decode("ascii"))
            if imgs:
                payload["images"] = imgs
        return payload

    def call(prompt: str, system: Optional[str] = None,
             attachments: Optional[list[dict]] = None) -> str:
        def _do():
            payload = _build_payload(prompt, system, attachments, stream_flag=False)
            resp = requests.post(url, json=payload, timeout=timeout_s)
            if resp.status_code == 404:
                raise RuntimeError(
                    f"Ollama returned 404 for model '{model}'. "
                    f"Have you pulled it? Try: ollama pull {model}"
                )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")

        return _retry_with_backoff(_do, retryable_exceptions=retryable)

    def stream(prompt: str, system: Optional[str] = None,
               attachments: Optional[list[dict]] = None):
        """
        Yield text chunks as they arrive. Ollama streams JSON-per-line with
        a `response` field on each chunk. See Claude's stream() for retry
        semantics — only the connection-open is retried.
        """
        import json as _json

        def _open_stream():
            payload = _build_payload(prompt, system, attachments, stream_flag=True)
            r = requests.post(url, json=payload, timeout=timeout_s, stream=True)
            if r.status_code == 404:
                raise RuntimeError(
                    f"Ollama returned 404 for model '{model}'. "
                    f"Have you pulled it? Try: ollama pull {model}"
                )
            r.raise_for_status()
            return r

        resp = _retry_with_backoff(_open_stream, retryable_exceptions=retryable)
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except ValueError:
                    continue
                chunk = obj.get("response")
                if chunk:
                    yield chunk
                if obj.get("done"):
                    break
        finally:
            resp.close()

    call.stream = stream
    return call


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Run with one of: claude | openai | gemini | ollama
      python llm_clients.py claude
    Append --stream to test the streaming path:
      python llm_clients.py claude --stream
    """
    if len(sys.argv) < 2:
        print("usage: python llm_clients.py {claude|openai|gemini|ollama} [--stream]")
        sys.exit(1)

    provider = sys.argv[1].lower()
    use_stream = "--stream" in sys.argv[2:]
    builders = {
        "claude": make_claude_client,
        "openai": make_openai_client,
        "gemini": make_gemini_client,
        "ollama": make_ollama_client,
    }
    if provider not in builders:
        print(f"unknown provider: {provider}")
        sys.exit(1)

    print(f"building {provider} client...")
    llm = builders[provider]()
    if use_stream:
        print(f"streaming from {provider}...")
        chunks = []
        for piece in llm.stream("Count from one to five, one number per line."):
            sys.stdout.write(piece)
            sys.stdout.flush()
            chunks.append(piece)
        print(f"\n[stream complete: {len(chunks)} chunks, {sum(len(c) for c in chunks)} chars]")
    else:
        print(f"calling {provider}...")
        response = llm("Reply with a single word: 'pong'")
        print(f"response: {response!r}")
