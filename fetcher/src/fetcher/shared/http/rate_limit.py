"""Politeness floor between real network calls."""

from __future__ import annotations

import time

REQUEST_SLEEP = 3.0  # arxiv politeness floor, seconds

# Module-level: the last-request clock and the rate limiter that reads it are
# correct only at concurrency = 1 (the sole supported mode -- see config.py).
_last_request = 0.0


def rate_limit() -> None:
    """Block until at least REQUEST_SLEEP has passed since the last call."""
    global _last_request
    delta = time.monotonic() - _last_request
    if _last_request and delta < REQUEST_SLEEP:
        time.sleep(REQUEST_SLEEP - delta)
    _last_request = time.monotonic()
