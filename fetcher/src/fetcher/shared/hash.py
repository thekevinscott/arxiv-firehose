"""Deterministic cache-key hashing.

Python's built-in ``hash()`` is randomized per process (PYTHONHASHSEED),
so a disk cache keyed by it misses every cross-run lookup. ``hash``
here is a sha256-backed digest over positional and keyword arguments,
truncated to 16 hex chars (64 bits = collision-free in any realistic
cache space), stable forever.

Designed for the ``cachetta`` path-lambda pattern:

    from ..shared.config import cache
    from ..shared.hash import hash

    @cache(path=lambda *args, **kwargs:
        cache.path / "llm" / f"{hash(*args, **kwargs)}.pkl")
    def fn(...): ...
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def hash(*args: Any, **kwargs: Any) -> str:
    """Deterministic 16-hex-char sha256 over *args* and *kwargs*.

    Arguments are serialized via ``json.dumps(..., sort_keys=True,
    default=str)``: dict order is normalized, and any non-JSON-native
    value (Path, datetime, etc.) is coerced through ``str()``. The
    digest survives process restarts, interpreter versions, and any
    ``PYTHONHASHSEED`` value.

    Shadows the builtin ``hash`` in the importing module; that builtin
    is unsuitable for disk caches anyway (PYTHONHASHSEED-randomized).
    """
    payload = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
