"""arxiv fetchers, cachetta-cached at the URL level.

One module per fetcher -- feed, paper, html -- each owning its cache
sibling, duration, and body-shape condition. The caches are keyed by
`hash(*args)` (the cachetta `hashed=True` shape) and sit on top of the
shared rate-limited getter (``shared.http``). The on-disk layout is
``<cache>/feeds/<hash>``, ``<cache>/papers/<hash>``, ``<cache>/html/<hash>``.
Files are opaque; the seam between caller and bytes is a pure URL.

Conditions reject HTML error pages so a 200-with-junk body never poisons
the long-lived paper/html caches; the 1-day feed cache rejects HTML for
the same reason.

The rate limiter lives inside ``http_get``, so it only sleeps on a real
network call -- a cachetta hit costs nothing.
"""

from .feed import fetch_feed
from .html import fetch_html
from .paper import fetch_paper

__all__ = ["fetch_feed", "fetch_html", "fetch_paper"]
