"""OpenAI-compatible chat-completion client, cachetta-cached.

One module-level ``LLM`` singleton; no per-call construction. The
constructor takes no args and reads everything it needs from
``shared.config`` (cache, endpoint defaults).

A single instance owns three things callers shouldn't re-establish per
request:

- **One ``httpx.Client``**, reused across every ``send_chat_completion``
  call. Saves a TCP+TLS handshake per request.
- **Retry+backoff** on transient failures (429, 5xx, connection errors)
  via ``build_default_backend``.
- **Cachetta-backed disk cache** of the response content string, keyed
  by ``(model, prompt, schema_json)``. A repeat call returns the prior
  response from disk with no network hit. JSON validity is checked
  *inside* the cached function so a malformed body raises before
  cachetta sees a return value -- the cache never holds poison.

The endpoint shape is the de-facto cross-vendor standard: Ollama
(``/v1/chat/completions``), vLLM, llama.cpp's server, OpenAI itself,
LiteLLM gateways all accept the same request and constrain output via
``response_format.json_schema``.

"""

from __future__ import annotations

import json

import httpx

from ..build_default_backend import build_default_backend
from ..config import cache

llm_cache = cache / "llm"


class LLMError(Exception):
    """Anything wrong on the LLM round-trip: transport, HTTP status,
    malformed response shape, or non-JSON content. Schema validation
    against a Pydantic model happens *outside* the LLM and raises its
    own error at the call site."""


class LLM:
    """An OpenAI-compatible chat-completion client.

    Constructed once at module load; reads everything from
    ``shared.config``. Repeated ``(model, prompt, schema)`` tuples
    return from disk with no network call.

    Cron processes are short-lived; the httpx.Client is GC'd with the
    instance and no explicit close is needed.
    """

    def __init__(self) -> None:
        from ..config import (
            DEFAULT_CLASSIFY_BASE_URL,
            DEFAULT_CLASSIFY_TIMEOUT_S,
        )

        self.url = DEFAULT_CLASSIFY_BASE_URL.rstrip("/") + "/chat/completions"
        self.timeout_s = DEFAULT_CLASSIFY_TIMEOUT_S
        self.headers = {"content-type": "application/json"}

        client = httpx.Client(timeout=self.timeout_s)
        self._backend = build_default_backend(self.url, self.headers, client)

    @llm_cache(hashed=True)
    def send_chat_completion(self, model: str, prompt: str, schema_json: str) -> str:
        """POST one chat-completion request; return the content string.

        Pure inside-cachetta function: takes the three values that key the
        cache, returns the raw JSON content. Raising here means cachetta
        does **not** store anything, so a transient backend failure or
        malformed body never poisons the cache.
        """
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            # OpenAI's json_schema response_format; Ollama and others honor
            # the same shape via the /v1 compatibility layer.
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Output",
                    "schema": json.loads(schema_json),
                    "strict": True,
                },
            },
        }
        try:
            resp = self._backend(payload, self.timeout_s)
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        try:
            content = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected chat-completions response: {resp!r}") from exc
        # Pre-validate JSON so cachetta only stores parseable content.
        # Schema validation happens at the call site so that a future
        # schema_json change (which becomes a different cache key) gets a
        # fresh decode -- but anything *this* key returns is guaranteed to
        # be JSON. Belt-and-suspenders with the cache condition.
        try:
            json.loads(content)
        except (ValueError, TypeError) as exc:
            raise LLMError(f"invalid JSON in LLM response: {exc}") from exc
        return content
