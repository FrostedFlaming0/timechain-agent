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
# A client is callable as: llm(user_prompt) or llm(user_prompt, system=system_prompt)


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

    def call(prompt: str, system: Optional[str] = None) -> str:
        def _do():
            kwargs = dict(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if system:
                kwargs["system"] = system
            msg = client.messages.create(**kwargs)
            return "".join(b.text for b in msg.content if hasattr(b, "text"))

        return _retry_with_backoff(_do, retryable_exceptions=retryable)

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

    def call(prompt: str, system: Optional[str] = None) -> str:
        def _do():
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            content = resp.choices[0].message.content
            return content if content is not None else ""

        return _retry_with_backoff(_do, retryable_exceptions=retryable)

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

    def call(prompt: str, system: Optional[str] = None) -> str:
        def _do():
            cfg = genai_types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                system_instruction=system if system else None,
            )
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=cfg,
            )
            return resp.text or ""

        return _retry_with_backoff(_do, retryable_exceptions=retryable)

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

    def call(prompt: str, system: Optional[str] = None) -> str:
        def _do():
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            }
            if system:
                payload["system"] = system
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

    return call


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Run with one of: claude | openai | gemini | ollama
    e.g. `python llm_clients.py claude`
    """
    if len(sys.argv) < 2:
        print("usage: python llm_clients.py {claude|openai|gemini|ollama}")
        sys.exit(1)

    provider = sys.argv[1].lower()
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
    print(f"calling {provider}...")
    response = llm("Reply with a single word: 'pong'")
    print(f"response: {response!r}")
