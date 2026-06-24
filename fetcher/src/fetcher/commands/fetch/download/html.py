"""The arxiv HTML fetcher: paper URL -> native HTML bytes, cached forever."""

from __future__ import annotations

from datetime import timedelta

from ....shared import http
from ....shared.config import cache

# arxiv's HTML rendering never changes for a fixed version: cache forever.
HTML_CACHE_DURATION = timedelta(days=36500)

html_cache = cache / "html"


def looks_like_html(body: bytes) -> bool:
    """True if *body* is an HTML document -- arxiv's native paper rendering."""
    head = body.lstrip()[:512].lower()
    return head.startswith(b"<!doctype html") or b"<html" in head


@html_cache(
    hashed=True,
    duration=HTML_CACHE_DURATION,
    condition=lambda body: isinstance(body, bytes) and looks_like_html(body),
)
def fetch_html(url: str) -> bytes:
    """Fetch arxiv's native HTML rendering of a paper. Cached ~forever."""
    return http.http_get(url, 120.0)
