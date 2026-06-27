"""
llm_clients — production-ready LLM client builders for the timechain agent.

Each builder returns a callable str -> str that the Agent uses. They share
a common shape so you can swap providers in run.py with one line:

    llm = make_claude_client()
    # llm = make_openai_client()
    # llm = make_openrouter_client()
    # llm = make_deepseek_client()
    # llm = make_gemini_client()
    # llm = make_ollama_client()

All clients implement:
  - Sensible defaults for the current generation of models (as of May 2026).
  - Configurable model, max_tokens, temperature, timeout.
  - Retries with exponential backoff on transient errors.
  - Clear error messages when API keys are missing or the server is down.
  - Reasonable timeouts so a hung provider doesn't freeze the agent loop.

OpenRouter and DeepSeek both expose OpenAI-compatible APIs, so their
builders reuse the `openai` SDK pointed at a different base URL — no extra
dependency beyond `openai`.

Model strings are current as of this writing but providers update fast.
If a model returns 404 or "model not found," check the provider's docs:
  Anthropic:  https://docs.claude.com/en/docs/about-claude/models
  OpenAI:     https://platform.openai.com/docs/models
  OpenRouter: https://openrouter.ai/models
  DeepSeek:   https://api-docs.deepseek.com/quick_start/pricing
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
#
# After each call (or stream), a client records why generation stopped on
# the attribute `llm.last_finish_reason`. Providers name this differently
# — OpenAI / OpenRouter / DeepSeek use finish_reason ("stop" | "length"),
# Anthropic uses stop_reason ("end_turn" | "max_tokens") — so the value is
# stored raw and `was_truncated()` below normalizes it. It is None before
# the first call, or when a provider didn't report a reason.


def was_truncated(llm) -> bool:
    """
    True if the LLM's most recent response was cut off at the max_tokens
    ceiling rather than finishing naturally.

    Reads `llm.last_finish_reason`, set by the client after each call.
    Recognizes the truncation signal across providers:
      - OpenAI / OpenRouter / DeepSeek : finish_reason == "length"
      - Anthropic                      : stop_reason  == "max_tokens"
    Returns False if the attribute is absent or None (e.g. a custom client
    that doesn't report a reason) — callers treat "unknown" as "complete"
    so the truncation marker only ever shows on a confirmed cut-off.
    """
    reason = getattr(llm, "last_finish_reason", None)
    return reason in ("length", "max_tokens")


# ---------------------------------------------------------------------------
# Shared retry helper
# ---------------------------------------------------------------------------

def _retry_with_backoff(
    fn: Callable[[], str],
    *,
    retryable_exceptions: tuple,
    max_attempts: int = 4,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> str:
    """
    Call fn() with retries on transient errors. Doubles delay each attempt
    (1s, 2s, 4s, 8s by default). Re-raises the last exception if all attempts fail.

    `retryable_exceptions` is REQUIRED — every caller in this module
    passes a provider-specific tuple of network/rate-limit errors. The
    previous default of `(Exception,)` was unused (every caller passed
    its own tuple) but would have been a footgun: it would retry
    authentication failures, bad-request errors, missing-model errors,
    and so on, four times before failing. Making the parameter required
    keeps the helper from accidentally swallowing those.
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
            # Record why generation stopped. Anthropic uses `stop_reason`
            # ("end_turn" | "max_tokens" | ...); was_truncated() maps
            # "max_tokens" to the truncation signal.
            call.last_finish_reason = getattr(msg, "stop_reason", None)
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

        The stop reason is recorded on `call.last_finish_reason` once the
        stream completes — the Anthropic streaming SDK exposes the final
        assembled message after the text stream is exhausted.
        """
        call.last_finish_reason = None

        def _open_stream():
            kwargs = _build_messages_kwargs(prompt, system, attachments)
            return client.messages.stream(**kwargs)

        # Retry just the connection-open step. Once we're inside the `with`,
        # we're committed to whatever chunks come out.
        stream_ctx = _retry_with_backoff(_open_stream, retryable_exceptions=retryable)
        with stream_ctx as s:
            for text in s.text_stream:
                yield text
            # After the text stream is exhausted, the SDK can assemble the
            # final message, which carries stop_reason. Guarded: if the SDK
            # shape changes, a missing reason just means "complete".
            try:
                final = s.get_final_message()
                call.last_finish_reason = getattr(final, "stop_reason", None)
            except Exception:
                pass

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
            choice = resp.choices[0]
            # Record why generation stopped ("stop" vs "length") so callers
            # can detect a max_tokens cut-off. See was_truncated().
            call.last_finish_reason = getattr(choice, "finish_reason", None)
            content = choice.message.content
            return content if content is not None else ""

        return _retry_with_backoff(_do, retryable_exceptions=retryable)

    def stream(prompt: str, system: Optional[str] = None,
               attachments: Optional[list[dict]] = None):
        """Yield text chunks as they arrive. See Claude's stream() for retry semantics."""
        call.last_finish_reason = None

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
            choice = chunk.choices[0]
            fr = getattr(choice, "finish_reason", None)
            if fr is not None:
                call.last_finish_reason = fr
            text = getattr(choice.delta, "content", None)
            if text:
                yield text

    call.stream = stream
    return call


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------

def make_openrouter_client(
    model: str = "anthropic/claude-opus-4.7",
    max_tokens: int = 1024,
    temperature: float = 1.0,
    timeout_s: float = 60.0,
) -> LLMCall:
    """
    OpenRouter client. OpenRouter is an aggregator that exposes an
    OpenAI-compatible API and routes to hundreds of models behind one
    endpoint, so this reuses the `openai` SDK pointed at OpenRouter's base
    URL — no separate dependency.

    The `model` string is namespaced by provider, e.g.
    "anthropic/claude-opus-4.7", "openai/gpt-5.5", "deepseek/deepseek-chat",
    "google/gemini-3.1-pro", "meta-llama/llama-3.3-70b-instruct". Browse the
    catalogue at https://openrouter.ai/models.

    Set OPENROUTER_API_KEY in the environment.
    """
    try:
        from openai import OpenAI
        import openai as openai_mod
    except ImportError:
        sys.exit("pip install openai")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("set OPENROUTER_API_KEY")

    # Same SDK as the OpenAI client, just a different base URL and key.
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=timeout_s,
    )

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
            choice = resp.choices[0]
            # Record why generation stopped ("stop" vs "length"). See
            # was_truncated().
            call.last_finish_reason = getattr(choice, "finish_reason", None)
            content = choice.message.content
            return content if content is not None else ""

        return _retry_with_backoff(_do, retryable_exceptions=retryable)

    def stream(prompt: str, system: Optional[str] = None,
               attachments: Optional[list[dict]] = None):
        """Yield text chunks as they arrive. See Claude's stream() for retry semantics."""
        call.last_finish_reason = None

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
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            fr = getattr(choice, "finish_reason", None)
            if fr is not None:
                call.last_finish_reason = fr
            text = getattr(choice.delta, "content", None)
            if text:
                yield text

    call.stream = stream
    return call


# ---------------------------------------------------------------------------
# DeepSeek
# ---------------------------------------------------------------------------

def _deepseek_answer_text(content, reasoning_content) -> str:
    """
    Pick the text to use from a DeepSeek response.

    Normally the answer is in `content`. But the V4 models can, for a thin
    prompt, route their entire output into the thinking trace
    (`reasoning_content`) and leave `content` empty — which would
    otherwise surface to the user as a blank "(no response)" turn. So when
    `content` is empty, fall back to `reasoning_content`. Both empty
    yields "". Pure function — unit-testable without an API.
    """
    if content:
        return content
    if reasoning_content:
        return reasoning_content
    return ""


def make_deepseek_client(
    model: str = "deepseek-v4-pro",
    max_tokens: int = 1024,
    temperature: float = 1.0,
    timeout_s: float = 60.0,
) -> LLMCall:
    """
    DeepSeek client. DeepSeek's API is OpenAI-compatible, so this reuses the
    `openai` SDK pointed at DeepSeek's base URL — no separate dependency.

    Models (DeepSeek V4 series):
      - deepseek-v4-pro    — 1.6T-param reasoning-heavy model (default)
      - deepseek-v4-flash  — 284B-param fast, economical model

    The legacy names deepseek-chat / deepseek-reasoner still work as of
    this writing but DeepSeek will discontinue them on 2026-07-24; during
    the grace period both route to deepseek-v4-flash. Use the v4 names.

    Note on reasoning: the V4 models support a thinking mode toggled by a
    `thinking` request parameter rather than by a separate model name.
    This client does not send that parameter, so it runs the model in its
    default (non-thinking) mode. If the API returns a separate
    `reasoning_content` trace, this client uses only the final answer
    (`message.content`) and discards the trace, so the agent treats the
    model like any other. The trace, when present, is on
    `resp.choices[0].message.reasoning_content`.

    Set DEEPSEEK_API_KEY in the environment.

    Privacy note: DeepSeek's API is operated from China. As with any hosted
    provider, prompt content — including retrieved memory records — is sent
    to the provider. Use the Ollama provider if nothing should leave the
    machine.
    """
    try:
        from openai import OpenAI
        import openai as openai_mod
    except ImportError:
        sys.exit("pip install openai")

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("set DEEPSEEK_API_KEY")

    # Same SDK as the OpenAI client, just a different base URL and key.
    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=api_key,
        timeout=timeout_s,
    )

    retryable = (
        openai_mod.APIConnectionError,
        openai_mod.APITimeoutError,
        openai_mod.RateLimitError,
        openai_mod.InternalServerError,
    )

    def _build_messages(prompt, system, attachments):
        # DeepSeek's chat models are text-only; image attachments are
        # ignored here (they're already represented as text in the prompt
        # by the agent's context builder anyway). Kept as a flat string
        # rather than the content-block list form for simplicity.
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
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
            choice = resp.choices[0]
            # Record why generation stopped so callers can tell a complete
            # answer ("stop") from one cut off at the token ceiling
            # ("length"). Stored on the callable as `last_finish_reason`;
            # the str return value is unchanged so the LLMCall contract
            # holds. See was_truncated() for the read side.
            call.last_finish_reason = getattr(choice, "finish_reason", None)
            # Normally use message.content — the final answer. But the V4
            # models can, for a thin prompt, route their entire output into
            # the thinking trace (reasoning_content) and leave content
            # empty. _deepseek_answer_text falls back to the trace in that
            # case so the turn isn't a blank "(no response)".
            msg = choice.message
            return _deepseek_answer_text(
                msg.content, getattr(msg, "reasoning_content", None))

        return _retry_with_backoff(_do, retryable_exceptions=retryable)

    def stream(prompt: str, system: Optional[str] = None,
               attachments: Optional[list[dict]] = None):
        """
        Yield text chunks as they arrive. See Claude's stream() for retry
        semantics.

        Normally yields the answer chunks (`delta.content`). The V4 models
        also stream a thinking trace (`delta.reasoning_content`), which is
        normally discarded. But for a thin prompt the model can put its
        whole output in the trace and emit no content at all — which would
        surface as "(no response)". So this accumulates the reasoning
        trace as it streams, and if the stream ends having yielded zero
        content, it yields the accumulated trace as a fallback. The normal
        case (content present) is unaffected — the trace is still dropped.

        As with the non-streaming path, the finish reason is recorded on
        `call.last_finish_reason` once the stream ends.
        """
        # Reset before the stream so a stale value from an earlier call
        # can't be misread if this stream ends without a finish reason.
        call.last_finish_reason = None

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
        yielded_content = False
        reasoning_parts: list[str] = []
        for chunk in completion:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            # The finish reason arrives on the last chunk; keep the most
            # recent non-null value seen.
            fr = getattr(choice, "finish_reason", None)
            if fr is not None:
                call.last_finish_reason = fr
            delta = choice.delta
            text = getattr(delta, "content", None)
            if text:
                yielded_content = True
                yield text
            else:
                # Accumulate the thinking trace in case content stays empty.
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    reasoning_parts.append(rc)
        # Fallback: the model produced only a thinking trace, no answer.
        # Yield the trace so the turn isn't a blank "(no response)".
        if not yielded_content and reasoning_parts:
            yield "".join(reasoning_parts)

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

    def _gemini_finish_reason(resp) -> Optional[str]:
        """
        Extract a normalized finish reason from a Gemini response.

        Gemini surfaces the stop reason on `resp.candidates[0].finish_reason`,
        as a `FinishReason` enum whose `.name` is e.g. "STOP", "MAX_TOKENS",
        "SAFETY", "RECITATION". was_truncated() recognizes "max_tokens" (the
        Anthropic stop_reason value), so when we see Gemini's MAX_TOKENS we
        normalize it to that. Other values are passed through lowercase so
        operators can still see them for diagnostics. Returns None when the
        candidate or reason is missing.
        """
        try:
            cand = resp.candidates[0]
        except (AttributeError, IndexError, TypeError):
            return None
        fr = getattr(cand, "finish_reason", None)
        if fr is None:
            return None
        # Newer SDKs expose an enum with `.name`; older ones a bare string.
        name = getattr(fr, "name", None) or str(fr)
        name = name.upper().split(".")[-1]  # "FinishReason.MAX_TOKENS" -> "MAX_TOKENS"
        if name == "MAX_TOKENS":
            return "max_tokens"  # matches was_truncated()'s recognized values
        return name.lower()

    def call(prompt: str, system: Optional[str] = None,
             attachments: Optional[list[dict]] = None) -> str:
        def _do():
            contents, cfg = _build_contents_and_cfg(prompt, system, attachments)
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=cfg,
            )
            # Record why generation stopped, normalized so was_truncated()
            # recognizes Gemini's MAX_TOKENS the same as the other providers.
            call.last_finish_reason = _gemini_finish_reason(resp)
            return resp.text or ""

        return _retry_with_backoff(_do, retryable_exceptions=retryable)

    def stream(prompt: str, system: Optional[str] = None,
               attachments: Optional[list[dict]] = None):
        """Yield text chunks as they arrive. See Claude's stream() for retry semantics."""
        call.last_finish_reason = None

        def _open_stream():
            contents, cfg = _build_contents_and_cfg(prompt, system, attachments)
            return client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=cfg,
            )

        chunks = _retry_with_backoff(_open_stream, retryable_exceptions=retryable)
        last_chunk = None
        for chunk in chunks:
            last_chunk = chunk
            text = getattr(chunk, "text", None)
            if text:
                yield text
        # Final chunk carries the finish reason on its candidate. Same
        # normalization as the non-stream path so the truncation marker
        # is consistent across both modes.
        if last_chunk is not None:
            call.last_finish_reason = _gemini_finish_reason(last_chunk)

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

    def _normalize_done_reason(value):
        """
        Map Ollama's `done_reason` to the values was_truncated() recognizes.
        Ollama uses "length" when the generation hit num_predict (per-call
        max_tokens) and "stop" when it ended naturally — same vocabulary as
        OpenAI's `finish_reason`, so "length" passes through unchanged and
        was_truncated() reports the cut-off correctly.
        """
        if not value:
            return None
        return str(value).lower()

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
            # Record why generation stopped. Ollama's `/api/generate` emits
            # `done_reason` on the final response — "length" when
            # num_predict is hit (cut off), "stop" on natural completion.
            # was_truncated() reads "length", so the truncation marker now
            # works for the local-model path same as hosted providers.
            call.last_finish_reason = _normalize_done_reason(data.get("done_reason"))
            return data.get("response", "")

        return _retry_with_backoff(_do, retryable_exceptions=retryable)

    def stream(prompt: str, system: Optional[str] = None,
               attachments: Optional[list[dict]] = None):
        """
        Yield text chunks as they arrive. Ollama streams JSON-per-line with
        a `response` field on each chunk. See Claude's stream() for retry
        semantics — only the connection-open is retried.

        The terminating chunk (the one with `done: true`) carries the
        `done_reason`. We record it on `call.last_finish_reason` so the
        streaming path reports truncation the same as the non-streaming
        call path.
        """
        import json as _json
        call.last_finish_reason = None

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
                    call.last_finish_reason = _normalize_done_reason(
                        obj.get("done_reason"))
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
    Run with one of: claude | openai | openrouter | deepseek | gemini | ollama
      python llm_clients.py claude
    Append --stream to test the streaming path:
      python llm_clients.py claude --stream
    """
    if len(sys.argv) < 2:
        print("usage: python llm_clients.py "
              "{claude|openai|openrouter|deepseek|gemini|ollama} [--stream]")
        sys.exit(1)

    provider = sys.argv[1].lower()
    use_stream = "--stream" in sys.argv[2:]
    builders = {
        "claude": make_claude_client,
        "openai": make_openai_client,
        "openrouter": make_openrouter_client,
        "deepseek": make_deepseek_client,
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
