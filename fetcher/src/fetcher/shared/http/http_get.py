"""The one byte-fetcher every network read goes through."""

from __future__ import annotations

import httpx

from ..retry import with_retry
from .is_retryable import is_retryable
from .rate_limit import rate_limit

USER_AGENT = "fetcher/0.1 (+https://github.com/thekevinscott/arxiv-firehose)"


def http_get(url: str, timeout: float) -> bytes:
    """One rate-limited httpx GET, retried on 5xx / transport errors.

    Tests patch this function with ``patch.object`` to stub the network
    out without touching cachetta."""
    def once() -> bytes:
        rate_limit()
        resp = httpx.get(
            url,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.content

    return with_retry(once, is_retryable=is_retryable)
