"""HTTP backend factory for the LLM client.

A backend is a ``(payload, timeout) -> response`` callable -- the seam
the LLM uses for byte-fetching. ``build_default_backend`` wraps an
``httpx.Client`` with retry+backoff on transient failures; tests inject
a fake to skip the network entirely.

The retry policy lives here so a test can monkeypatch ``_BACKOFF_BASE_S``
to zero without touching the LLM module.
"""

from __future__ import annotations

import time
from typing import Callable

import httpx

# (payload_json, timeout_s) -> response_json. Default backend is httpx-
# backed with retries; tests inject a fake.
HttpBackend = Callable[[dict, float], dict]

# Codes that mean "try again". 4xx-other is final; retrying a 401 only
# wastes the per-process timeout budget.
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
_RETRIES = 3
_BACKOFF_BASE_S = 0.5  # 0.5s, 1s, 2s


def build_default_backend(
    url: str, headers: dict[str, str], client: httpx.Client
) -> HttpBackend:
    """An httpx-backed backend with retry+backoff on transient failures.

    The client is owned by the caller (typically an ``LLM`` instance),
    so connection pooling works across every call routed through this
    backend.
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
