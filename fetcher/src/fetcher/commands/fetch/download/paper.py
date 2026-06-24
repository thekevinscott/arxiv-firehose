"""The arxiv paper-body fetcher: e-print URL -> PDF/gzip bytes, cached forever."""

from __future__ import annotations

from datetime import timedelta

from ....shared import http
from ....shared.config import cache

# A paper's PDF/source never changes for a fixed version: cache forever.
PAPER_CACHE_DURATION = timedelta(days=36500)

PDF_MAGIC = b"%PDF"
GZIP_MAGIC = b"\x1f\x8b"

paper_cache = cache / "papers"


def looks_like_paper(body: bytes) -> bool:
    """True if *body* is a PDF or a gzip archive -- arxiv's two paper formats."""
    return body[:4] == PDF_MAGIC or body[:2] == GZIP_MAGIC


@paper_cache(
    hashed=True,
    duration=PAPER_CACHE_DURATION,
    condition=lambda body: isinstance(body, bytes) and looks_like_paper(body),
)
def fetch_paper(url: str) -> bytes:
    """Fetch a paper body (PDF or gzip e-print archive). Cached ~forever."""
    return http.http_get(url, 120.0)
