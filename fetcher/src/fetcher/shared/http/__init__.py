"""Rate-limited HTTP GET with retry -- the network transport.

One generic byte-fetcher: ``http_get`` sleeps to honor a politeness
floor between real network calls, retries transient failures (5xx /
transport errors) via ``shared.retry.with_retry``, and returns raw
bytes. Callers own everything above bytes: URLs, caching, body-shape
validation.
"""

from .http_get import http_get

__all__ = ["http_get"]
