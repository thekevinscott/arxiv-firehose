"""Decide whether a failed GET is worth retrying."""

from __future__ import annotations

import httpx


def is_retryable(exc: Exception) -> bool:
    """True if *exc* is worth retrying: a transport error or a 5xx response.

    A 4xx (notably a 404 for a withdrawn paper) is final -- retrying it only
    wastes the rate-limit budget.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)
