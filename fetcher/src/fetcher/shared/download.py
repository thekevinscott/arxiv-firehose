"""arxiv HTTP fetchers, cachetta-cached at the URL level.

Three sibling caches keyed by `hash(*args)` (the cachetta `hashed=True`
shape) sit on top of one rate-limited httpx getter. The on-disk layout is
``<cache>/feeds/<hash>``, ``<cache>/papers/<hash>``, ``<cache>/html/<hash>``.
Files are opaque; the seam between caller and bytes is a pure URL.

Conditions reject HTML error pages so a 200-with-junk body never poisons
the long-lived paper/html caches; the 1-day feed cache rejects HTML for
the same reason.

The rate limiter lives inside ``_http_get``, so it only sleeps on a real
network call -- a cachetta hit costs nothing.
"""

from __future__ import annotations

import time
from datetime import timedelta

import httpx

from .config import cache
from .retry import with_retry

USER_AGENT = "fetcher/0.1 (+https://github.com/thekevinscott/arxiv-firehose)"
REQUEST_SLEEP = 3.0  # arxiv politeness floor, seconds
RSS_URL = "https://rss.arxiv.org/rss/{category}"

# A paper's PDF/source/HTML never changes for a fixed version: cache forever.
PAPER_CACHE_DURATION = timedelta(days=36500)
# arxiv regenerates RSS once per day, so caching it a day means at most one
# feed request per category per day.
FEED_CACHE_DURATION = timedelta(days=1)

# Module-level: the last-request clock and the rate limiter that reads it are
# correct only at concurrency = 1 (the sole supported mode -- see config.py).
_last_request = 0.0

_PDF_MAGIC = b"%PDF"
_GZIP_MAGIC = b"\x1f\x8b"


def _looks_like_paper(body: bytes) -> bool:
    """True if *body* is a PDF or a gzip archive -- arxiv's two paper formats."""
    return body[:4] == _PDF_MAGIC or body[:2] == _GZIP_MAGIC


def _looks_like_feed(body: bytes) -> bool:
    """True if *body* looks like an RSS/XML feed rather than an error page."""
    head = body.lstrip()[:512]
    return head.startswith(b"<?xml") or b"<rss" in head


def _looks_like_html(body: bytes) -> bool:
    """True if *body* is an HTML document -- arxiv's native paper rendering."""
    head = body.lstrip()[:512].lower()
    return head.startswith(b"<!doctype html") or b"<html" in head


def _rate_limit() -> None:
    """Block until at least REQUEST_SLEEP has passed since the last call."""
    global _last_request
    delta = time.monotonic() - _last_request
    if _last_request and delta < REQUEST_SLEEP:
        time.sleep(REQUEST_SLEEP - delta)
    _last_request = time.monotonic()


def _is_retryable(exc: Exception) -> bool:
    """True if *exc* is worth retrying: a transport error or a 5xx response.

    A 4xx (notably a 404 for a withdrawn paper) is final -- retrying it only
    wastes the rate-limit budget.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


def _http_get(url: str, timeout: float) -> bytes:
    """One rate-limited httpx GET against the live arxiv, retried on 5xx /
    transport errors. Tests patch this function with ``patch.object`` to
    stub the network out without touching cachetta."""
    def once() -> bytes:
        _rate_limit()
        resp = httpx.get(
            url,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.content

    return with_retry(once, is_retryable=_is_retryable)


# Three sibling caches under the shared root. ``hashed=True`` treats each
# subfolder as a bucket and writes one file per arg-set under it. The
# subfolder also doubles as the namespace -- a paper URL and a feed url
# can never collide.
#
# Overrides are baked onto the sibling itself via ``.copy(...)`` (not via
# decorator kwargs) so that the decorated function holds a reference to
# *this* instance -- the one tests redirect with
# ``patch.object(_feed_cache, "path", tmp_path / ...)``. Passing kwargs
# at decoration time would call ``replace(self, **kwargs)`` and wrap a
# distinct copy, defeating the patch.
_feed_cache = (cache / "feeds").copy(
    hashed=True,
    duration=FEED_CACHE_DURATION,
    condition=lambda body: isinstance(body, bytes) and _looks_like_feed(body),
)
_paper_cache = (cache / "papers").copy(
    hashed=True,
    duration=PAPER_CACHE_DURATION,
    condition=lambda body: isinstance(body, bytes) and _looks_like_paper(body),
)
_html_cache = (cache / "html").copy(
    hashed=True,
    duration=PAPER_CACHE_DURATION,
    condition=lambda body: isinstance(body, bytes) and _looks_like_html(body),
)


@_feed_cache
def fetch_feed(category: str) -> bytes:
    """Fetch the arxiv RSS feed for *category*. Cached one day."""
    return _http_get(RSS_URL.format(category=category), 30.0)


@_paper_cache
def fetch_paper(url: str) -> bytes:
    """Fetch a paper body (PDF or gzip e-print archive). Cached ~forever."""
    return _http_get(url, 120.0)


@_html_cache
def fetch_html(url: str) -> bytes:
    """Fetch arxiv's native HTML rendering of a paper. Cached ~forever."""
    return _http_get(url, 120.0)
