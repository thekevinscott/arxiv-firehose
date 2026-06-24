"""HTTP backend factory for the LLM client.

A backend is a ``(payload, timeout) -> response`` callable -- the seam
the LLM uses for byte-fetching. ``build_default_backend`` wraps an
``httpx.Client`` with retry+backoff on transient failures; tests inject
a fake to skip the network entirely.

The retry policy lives here so a test can patch ``_BACKOFF_BASE_S`` to
zero without touching the LLM module. Retry mechanics are shared with
``shared.http`` via ``shared.retry.with_retry``.
"""

from __future__ import annotations

from typing import Callable

import httpx

from .retry import with_retry

# (payload_json, timeout_s) -> response_json. Default backend is httpx-
# backed with retries; tests inject a fake.
HttpBackend = Callable[[dict, float], dict]

# Codes that mean "try again". 4xx-other is final; retrying a 401 only
# wastes the per-process timeout budget.
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
_RETRIES = 3
_BACKOFF_BASE_S = 0.5  # 0.5s, 1s, 2s


def _is_retryable(exc: Exception) -> bool:
    """Status in ``_RETRY_STATUS`` or any transport error."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRY_STATUS
    return isinstance(exc, httpx.TransportError)


def build_default_backend(
    url: str, headers: dict[str, str], client: httpx.Client
) -> HttpBackend:
    """An httpx-backed backend with retry+backoff on transient failures.

    The client is owned by the caller (typically an ``LLM`` instance),
    so connection pooling works across every call routed through this
    backend.
    """
    def send(payload: dict, timeout: float) -> dict:
        def attempt() -> dict:
            r = client.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()  # raises HTTPStatusError on >=400
            return r.json()

        return with_retry(
            attempt,
            is_retryable=_is_retryable,
            attempts=_RETRIES,
            base=_BACKOFF_BASE_S,
        )

    return send
