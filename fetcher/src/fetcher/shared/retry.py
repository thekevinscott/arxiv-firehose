"""Predicate-based retry with exponential backoff.

A thin shim over ``while True: try ... except``: takes a thunk and a
predicate that decides whether an exception is worth retrying. Backoff
doubles each attempt (``base``, ``base*2``, ``base*4``...). The final
attempt re-raises whatever it caught regardless of the predicate.

Lives here because ``shared.http`` (arxiv GETs) and
``shared.build_default_backend`` (LLM POSTs) both need the same shape;
the only thing that varies is *what counts as retryable*.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

_T = TypeVar("_T")


def with_retry(
    fn: Callable[[], _T],
    *,
    is_retryable: Callable[[Exception], bool],
    attempts: int = 3,
    base: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> _T:
    """Call *fn*; on a retryable exception, sleep and retry up to *attempts*.

    Backoff before the i-th retry is ``base * 2**i`` seconds. A non-retryable
    exception is re-raised immediately; a retryable one is re-raised after
    the final attempt.
    """
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if not is_retryable(exc) or i == attempts - 1:
                raise
            sleep(base * (2 ** i))
    raise AssertionError("unreachable")  # pragma: no cover
