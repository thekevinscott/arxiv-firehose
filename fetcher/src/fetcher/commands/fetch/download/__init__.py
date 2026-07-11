"""arxiv fetchers, cachetta-cached at the URL level.

One module per fetcher -- api (day slices), paper, html -- each owning its
cache sibling, duration, and body-shape condition. The caches are keyed by
`hash(*args)` (the cachetta `hashed=True` shape) and sit on top of the
shared rate-limited getter (``shared.http``). The on-disk layout is
``<cache>/slices/<hash>``, ``<cache>/papers/<hash>``, ``<cache>/html/<hash>``.
Files are opaque; the seam between caller and bytes is a pure URL.

Conditions reject HTML error pages so a 200-with-junk body never poisons
the long-lived caches.

The rate limiter lives inside ``http_get``, so it only sleeps on a real
network call -- a cachetta hit costs nothing.
"""

from .api import fetch_day, fetch_id
from .html import fetch_html
from .paper import fetch_paper

__all__ = ["fetch_day", "fetch_html", "fetch_id", "fetch_paper"]
