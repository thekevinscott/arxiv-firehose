"""The arxiv RSS feed fetcher: one category -> raw feed bytes, cached a day."""

from __future__ import annotations

from datetime import timedelta

from ....shared import http
from ....shared.config import cache

RSS_URL = "https://rss.arxiv.org/rss/{category}"

# arxiv regenerates RSS once per day, so caching it a day means at most one
# feed request per category per day.
FEED_CACHE_DURATION = timedelta(days=1)

feed_cache = cache / "feeds"


def looks_like_feed(body: bytes) -> bool:
    """True if *body* looks like an RSS/XML feed rather than an error page."""
    head = body.lstrip()[:512]
    return head.startswith(b"<?xml") or b"<rss" in head


@feed_cache(
    hashed=True,
    duration=FEED_CACHE_DURATION,
    condition=lambda body: isinstance(body, bytes) and looks_like_feed(body),
)
def fetch_feed(category: str) -> bytes:
    """Fetch the arxiv RSS feed for *category*. Cached one day."""
    return http.http_get(RSS_URL.format(category=category), 30.0)
