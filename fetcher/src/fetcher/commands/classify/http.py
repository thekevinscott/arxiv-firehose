"""http_classifier: a Classifier backed by OpenAI-compatible chat completions.

The endpoint shape is the de-facto cross-vendor standard: Ollama
(``/v1/chat/completions``), vLLM, llama.cpp's server, OpenAI itself,
LiteLLM gateways and others all accept the same request and constrain
output via ``response_format.json_schema``. Switching backend is a config
edit, not a code edit.

Behaviours that are easy to forget if hand-rolled per call:

- **Connection pooling.** One ``httpx.Client`` per ``http_classifier``
  call, reused across every ``call(...)``. Saves a TCP+TLS handshake per
  paper × per flag.
- **Retry with backoff.** Transient 429 / 5xx / connection errors retry
  three times with exponential backoff; 4xx-other surface immediately as
  a real error (a bug, not a blip).
- **api_key from env.** ``FETCHER_LLM_API_KEY`` overrides an empty config
  value, so credentials stay out of ``data/config.toml``.
- **Cachetta-backed response cache.** When ``cache_dir`` is set, the
  exact same ``(model, prompt, schema)`` tuple returns the prior LLM
  response from disk with no network call. The decorator inlines a
  ``path=`` lambda that hashes the three inputs into
  ``{cache_dir}/llm/{sha}.pkl``. This is the only classify-side
  caching mechanism in fetcher; nothing else holds LLM output in
  memory or on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Callable

import httpx
from cachetta import Cachetta
from pydantic import BaseModel, ValidationError

from ...shared.config import DEFAULT_CLASSIFY_BASE_URL
from .types import Classifier, ClassifyError, HttpBackend

API_KEY_ENV = "FETCHER_LLM_API_KEY"
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
_RETRIES = 3
_BACKOFF_BASE_S = 0.5  # 0.5s, 1s, 2s
# A (model, prompt, schema) tuple is deterministic for the LLM's output;
# cache permanently. Matches the arxiv download cache duration in
# shared/download.py.
_CACHE_DURATION = timedelta(days=36500)


def _cache_key(model: str, prompt: str, schema_json: str) -> str:
    """sha256 over the three inputs joined by a NUL byte, truncated to 16
    hex chars. NUL prevents concatenation collisions (impossible inside a
    JSON serialization or a model tag); 16 chars = 64 bits = collision-free
    in the ~10^7-cell space a daily classify run touches over years."""
    body = b"\0".join(
        [model.encode("utf-8"), prompt.encode("utf-8"), schema_json.encode("utf-8")]
    )
    return hashlib.sha256(body).hexdigest()[:16]


def _resolve_api_key(api_key: str | None) -> str | None:
    """Empty config falls back to the env var; secrets stay out of toml."""
    if api_key:
        return api_key
    return os.environ.get(API_KEY_ENV) or None


def _build_default_backend(
    url: str, headers: dict[str, str], client: httpx.Client
) -> HttpBackend:
    """An httpx-backed backend with retry+backoff on transient failures.

    Retries only on the codes that mean "try again" -- 4xx-other propagates
    immediately. The client is owned by the enclosing ``http_classifier``
    call (one per Classifier), so connection pooling works across papers.
    """
    def send(payload: dict, timeout: float) -> dict:
        last_exc: Exception | None = None
        for attempt in range(_RETRIES):
            try:
                r = client.post(url, json=payload, headers=headers, timeout=timeout)
            except httpx.HTTPError as exc:
                last_exc = exc
            else:
                if r.status_code < 400:
                    return r.json()
                if r.status_code not in _RETRY_STATUS:
                    r.raise_for_status()  # non-retryable -> bubble up
                last_exc = httpx.HTTPStatusError(
                    f"{r.status_code} from LLM backend", request=r.request, response=r,
                )
            if attempt < _RETRIES - 1:
                time.sleep(_BACKOFF_BASE_S * (2 ** attempt))
        assert last_exc is not None
        raise last_exc

    return send


def http_classifier(
    model: str,
    *,
    base_url: str = DEFAULT_CLASSIFY_BASE_URL,
    api_key: str | None = None,
    timeout_s: float = 60.0,
    cache_dir: Path | None = None,
    backend: HttpBackend | None = None,
) -> Classifier:
    """Build a Classifier that POSTs OpenAI-compatible chat completions.

    *base_url* defaults to ``DEFAULT_CLASSIFY_BASE_URL`` (a local Ollama);
    ``ClassifyConfig.base_url`` is the runtime source of truth and shares
    the same constant. *backend* is the seam unit tests use to inject a
    fake HTTP send so they can assert on the exact request payload without
    touching the network. With *backend* unset the call builds a real
    httpx.Client with retries and pooling.

    *cache_dir* enables the cachetta-backed response cache: a repeat
    ``call(prompt, schema)`` with the same model returns the prior
    response from disk and never invokes *backend*. This is the only
    caching mechanism in classify -- there is no in-memory dict, no
    file-existence skip; cachetta does it transparently.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"content-type": "application/json"}
    resolved_key = _resolve_api_key(api_key)
    if resolved_key:
        headers["authorization"] = f"Bearer {resolved_key}"

    if backend is None:
        # The httpx.Client outlives this builder call -- it's captured in
        # the closure below, lives as long as the Classifier, and is GC'd
        # with it. Cron processes are short-lived, so no explicit close.
        client = httpx.Client(timeout=timeout_s)
        backend = _build_default_backend(url, headers, client)

    def send(model_: str, prompt: str, schema_json: str) -> str:
        """Send the chat-completion request and return the content string.

        Pure inside-cachetta function: takes the three values that key
        the cache, returns the raw JSON content. Raising here means
        cachetta does **not** store anything, so a transient backend
        failure or malformed body never poisons the cache.
        """
        payload = {
            "model": model_,
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
            resp = backend(payload, timeout_s)
        except httpx.HTTPError as exc:
            raise ClassifyError(f"LLM request failed: {exc}") from exc
        try:
            content = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ClassifyError(f"unexpected chat-completions response: {resp!r}") from exc
        # Pre-validate JSON so cachetta only stores parseable content.
        # Schema validation happens outside the cache so that a future
        # schema_json change (which becomes a different cache key) gets
        # a fresh decode -- but anything *this* key returns is guaranteed
        # to be JSON. Belt-and-suspenders with the cache condition.
        try:
            json.loads(content)
        except (ValueError, TypeError) as exc:
            raise ClassifyError(f"invalid JSON in LLM response: {exc}") from exc
        return content

    if cache_dir is not None:
        # Inline the cachetta decorator: lambda receives send's args, computes
        # the on-disk path. The condition predicate rejects empty content so
        # cachetta only persists usable JSON; the JSON-parse guard inside
        # send raises before return, so transient failures and malformed
        # bodies never reach disk either.
        cache = Cachetta(
            path=lambda model_, prompt, schema_json: (
                cache_dir / "llm" / f"{_cache_key(model_, prompt, schema_json)}.pkl"
            ),
            duration=_CACHE_DURATION,
            condition=lambda content: isinstance(content, str) and content.strip() != "",
        )
        send = cache(send)

    def call(prompt: str, schema: type[BaseModel]) -> BaseModel:
        # sort_keys keeps the cache key stable across runs even if
        # Pydantic shuffles dict order between releases.
        schema_json = json.dumps(schema.model_json_schema(), sort_keys=True)
        content = send(model, prompt, schema_json)
        try:
            return schema.model_validate_json(content)
        except (ValidationError, ValueError) as exc:
            raise ClassifyError(f"invalid LLM response: {exc}") from exc

    return Classifier(call=call)
