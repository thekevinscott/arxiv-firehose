"""The arxiv export API fetcher: one (categories, day) slice -> Atom bytes.

Sync queries the export API in per-day submittedDate windows instead of
RSS. A day-slice is an immutable query: once a day is a few days in the
past its result set never changes, so it is cached ~forever and any later
run can replay it. This is what makes sync self-healing -- a missed cron
day costs nothing, because tomorrow's run covers the same window and pulls
the missed slice from arxiv exactly once.
"""

from __future__ import annotations

from datetime import date, timedelta
from urllib.parse import quote

from ....shared import http
from ....shared.config import cache

API_URL = (
    "https://export.arxiv.org/api/query"
    "?search_query={query}&start=0&max_results={max_results}"
)

# ~14 tracked categories currently announce ~500 papers/day combined; one
# page holds a comfortable multiple of that. parse logs a warning when a
# slice comes back exactly full (possible truncation).
MAX_RESULTS = 2000

# A slice this many days old is settled: arxiv's announcement pipeline has
# fully flushed and the result set is immutable. Younger slices are still
# accreting entries, so they only get a short cache life.
SETTLED_AFTER_DAYS = 3
SETTLED_CACHE_DURATION = timedelta(days=3650)
RECENT_CACHE_DURATION = timedelta(days=1)

# Separate cache namespaces: a slice cached while recent must not satisfy
# the settled lookup later with a possibly-partial body.
settled_cache = cache / "slices"
recent_cache = cache / "slices-recent"


def looks_like_feed(body: bytes) -> bool:
    """True if *body* looks like an Atom/XML feed rather than an error page."""
    head = body.lstrip()[:512]
    return head.startswith(b"<?xml") or b"<feed" in head


def query_url(categories: tuple[str, ...], day: date) -> str:
    cats = " OR ".join(f"cat:{c}" for c in categories)
    window = f"submittedDate:[{day:%Y%m%d}0000 TO {day:%Y%m%d}2359]"
    query = quote(f"({cats}) AND {window}", safe="")
    return API_URL.format(query=query, max_results=MAX_RESULTS)


def _get(categories: tuple[str, ...], day_iso: str) -> bytes:
    url = query_url(categories, date.fromisoformat(day_iso))
    return http.http_get(url, 30.0)


_cache_kwargs = dict(
    hashed=True,
    condition=lambda body: isinstance(body, bytes) and looks_like_feed(body),
)


@settled_cache(duration=SETTLED_CACHE_DURATION, **_cache_kwargs)
def _fetch_settled(categories: tuple[str, ...], day_iso: str) -> bytes:
    return _get(categories, day_iso)


@recent_cache(duration=RECENT_CACHE_DURATION, **_cache_kwargs)
def _fetch_recent(categories: tuple[str, ...], day_iso: str) -> bytes:
    return _get(categories, day_iso)


def is_settled(day: date, today: date) -> bool:
    """True when *day*'s result set can no longer change."""
    return day <= today - timedelta(days=SETTLED_AFTER_DAYS)


def fetch_day(categories: tuple[str, ...], day: date) -> bytes:
    """Fetch one day-slice of the tracked categories, cached by age tier."""
    fetch = _fetch_settled if is_settled(day, date.today()) else _fetch_recent
    return fetch(categories, day.isoformat())
