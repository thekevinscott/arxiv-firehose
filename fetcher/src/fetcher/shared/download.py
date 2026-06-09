"""Network downloads, cached with cachetta.

Every HTTP GET for a PDF or e-print archive goes through a cachetta-wrapped
function. The cache lives in its own directory (default ~/.cache/arxiv-firehose),
separate from the arxiv data dir. arxiv content is immutable for a given paper
version, so cache entries effectively never expire.

The rate limiter sits *inside* the wrapped function, so it only sleeps on a
real network call -- a cache hit costs nothing.

The actual byte-fetching is a *transport*: a ``(url, timeout) -> bytes``
callable. The default transport is the real, rate-limited httpx GET. Tests
inject a fixture-backed fake transport through the public API instead of
monkeypatching httpx -- see AGENTS.md.
"""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path
from typing import Callable

import httpx
from cachetta import Cachetta

from .retry import with_retry

# A transport turns a URL into bytes. The cachetta layer sits above it; a
# transport is only ever called on a genuine cache miss.
Transport = Callable[[str, float], bytes]

USER_AGENT = "fetcher/0.1 (+https://github.com/thekevinscott/arxiv-firehose)"
REQUEST_SLEEP = 3.0  # arxiv politeness floor, seconds
RSS_URL = "https://rss.arxiv.org/rss/{category}"

# A paper's PDF/source never changes for a fixed version: cache permanently.
CACHE_DURATION = timedelta(days=36500)
# The RSS feed is regenerated once per day (after arxiv's daily announcement),
# so caching it for a day means at most one feed request per category per day.
FEED_CACHE_DURATION = timedelta(days=1)

# Module-level: the last-request clock and the rate limiter that reads it are
# correct only at concurrency = 1 (the sole supported mode -- see config.py).
_last_request = 0.0

_PDF_MAGIC = b"%PDF"
_GZIP_MAGIC = b"\x1f\x8b"


def _looks_like_paper(body: bytes) -> bool:
    """True if *body* is a PDF or a gzip archive -- arxiv's two paper formats.

    arxiv sometimes answers 200 with an HTML error page; that is neither, so
    it fails this check and the cachetta ``condition`` declines to cache it.
    """
    return body[:4] == _PDF_MAGIC or body[:2] == _GZIP_MAGIC


def _looks_like_feed(body: bytes) -> bool:
    """True if *body* looks like an RSS/XML feed rather than an error page."""
    head = body.lstrip()[:512]
    return head.startswith(b"<?xml") or b"<rss" in head


def _looks_like_html(body: bytes) -> bool:
    """True if *body* is an HTML document -- arxiv's native paper rendering.

    Guards the html cache: a plain-text or otherwise non-HTML error body must
    not be cached for the 100-year paper duration.
    """
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


def _get_once(url: str, timeout: float) -> bytes:
    """A single rate-limited httpx GET against the live arxiv."""
    _rate_limit()
    resp = httpx.get(
        url,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.content


def real_transport(url: str, timeout: float) -> bytes:
    """The default transport: a rate-limited httpx GET against the live arxiv.

    The 3-second arxiv politeness sleep happens inside ``_get_once``, so it is
    incurred only on a real network call -- never on a cachetta hit. A
    transient failure (5xx or a transport error) is retried with backoff.
    """
    return with_retry(lambda: _get_once(url, timeout), is_retryable=_is_retryable)


def _cache_path(cache_dir: Path, url: str) -> Path:
    """Map a download URL to an inspectable cache file under *cache_dir*.

    e.g. https://arxiv.org/pdf/2401.12345v1     -> {cache}/pdf/2401.12345v1.pkl
         https://arxiv.org/e-print/2401.12345   -> {cache}/eprint/2401.12345.pkl
         https://arxiv.org/html/2401.12345v1    -> {cache}/html/2401.12345v1.pkl
    """
    if "/pdf/" in url:
        kind, ident = "pdf", url.rsplit("/pdf/", 1)[1]
    elif "/e-print/" in url:
        kind, ident = "eprint", url.rsplit("/e-print/", 1)[1]
    elif "/html/" in url:
        kind, ident = "html", url.rsplit("/html/", 1)[1]
    else:
        # Unreachable in practice: PDFs and e-prints are the only URLs routed
        # here, and RSS uses its own path lambda. Kept as a defensive default.
        kind, ident = "other", url.rsplit("/", 1)[-1]
    slug = ident.replace("/", "_") or "index"
    return cache_dir / kind / f"{slug}.pkl"


def make_downloader(
    cache_dir: Path, transport: Transport | None = None
) -> Callable[[str], bytes]:
    """Return a ``download(url) -> bytes`` function backed by the cachetta cache.

    A cache hit returns the stored bytes with no network call and no sleep.
    A cache miss calls *transport* and caches the body. *transport* defaults
    to the real rate-limited httpx GET; tests pass a fake.
    """
    fetch = transport or real_transport
    cache = Cachetta(
        path=lambda url: _cache_path(cache_dir, url),
        duration=CACHE_DURATION,
        # Cache only a real paper body. An HTML error page served as a 200
        # must not poison the cache for the 100-year duration.
        condition=lambda body: isinstance(body, bytes) and _looks_like_paper(body),
    )

    @cache
    def download(url: str) -> bytes:
        return fetch(url, 120.0)

    return download


def make_html_fetcher(
    cache_dir: Path, transport: Transport | None = None
) -> Callable[[str], bytes]:
    """Return a ``fetch_html(url) -> bytes`` function backed by the cachetta cache.

    arxiv's native HTML for a fixed paper version never changes, so it is
    cached permanently, like the PDF/e-print bytes. An error body that is not
    HTML fails the ``condition`` and is not cached. *transport* defaults to
    the real rate-limited httpx GET; tests pass a fake.
    """
    fetch = transport or real_transport
    cache = Cachetta(
        path=lambda url: _cache_path(cache_dir, url),
        duration=CACHE_DURATION,
        condition=lambda body: isinstance(body, bytes) and _looks_like_html(body),
    )

    @cache
    def fetch_html(url: str) -> bytes:
        return fetch(url, 120.0)

    return fetch_html


def make_feed_fetcher(
    cache_dir: Path, transport: Transport | None = None
) -> Callable[[str], bytes]:
    """Return a ``fetch_feed(category) -> bytes`` function for arxiv RSS.

    Cached for one day: the feed only regenerates once per day, so repeated
    runs within a day reuse the cached feed and never touch arxiv. *transport*
    defaults to the real rate-limited httpx GET; tests pass a fake.
    """
    fetch = transport or real_transport
    cache = Cachetta(
        path=lambda category: cache_dir / "rss" / f"{category}.pkl",
        duration=FEED_CACHE_DURATION,
        # Cache only a real feed body, never an HTML error page.
        condition=lambda body: isinstance(body, bytes) and _looks_like_feed(body),
    )

    @cache
    def fetch_feed(category: str) -> bytes:
        return fetch(RSS_URL.format(category=category), 30.0)

    return fetch_feed
