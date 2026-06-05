"""Cachetta-backed cache for LLM calls.

Every classify request is a deterministic function of ``(model, prompt,
response-schema)``. Cache the response content on disk under
``{cache_dir}/llm/{sha}.pkl`` and a repeat call serves bytes from disk
with no network hit.

This is the LLM analogue of the arxiv download cache in ``download.py``:
same cachetta primitive, same "never expires" duration (content is
keyed by every input that could change the answer), same on-disk-only
strategy. No in-memory dict, no per-process state -- a fresh process
shares the cache with every other run.

The cache stores raw response content (a JSON string). Pydantic
validation happens *inside* the cached function so a malformed response
raises before cachetta writes -- which means the cache never holds a
poisoned value.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path

from cachetta import Cachetta

# A (model, prompt, schema) tuple uniquely determines the LLM's output for
# any temperature-0 decode and stays a useful key at higher temperatures
# (where caching skipped redo is the explicit goal). Match the arxiv
# download cache: keep entries forever; the key already encodes every
# input that could legitimately invalidate them.
CACHE_DURATION = timedelta(days=36500)


def cache_key(model: str, prompt: str, schema_json: str) -> str:
    """The on-disk cache key for one ``(model, prompt, schema)`` tuple.

    sha256 over the three inputs joined by a NUL byte (impossible inside
    a JSON serialization or a model tag), truncated to 16 hex chars. 16
    chars = 64 bits = collision-free in practice for the ~10^7-cell space
    a daily classify run touches over years.
    """
    body = b"\0".join(
        [model.encode("utf-8"), prompt.encode("utf-8"), schema_json.encode("utf-8")]
    )
    return hashlib.sha256(body).hexdigest()[:16]


def cache_path(cache_dir: Path, model: str, prompt: str, schema_json: str) -> Path:
    """Map a ``(model, prompt, schema)`` tuple to its on-disk pickle path."""
    return cache_dir / "llm" / f"{cache_key(model, prompt, schema_json)}.pkl"


def make_llm_cache(cache_dir: Path) -> Cachetta:
    """Return a cachetta decorator for ``(model, prompt, schema_json) -> str``.

    The decorated function returns the response *content string* -- the
    "message.content" field of an OpenAI-compatible chat-completions
    response. Validating JSON happens *inside* the decorated function so
    a malformed body raises before cachetta sees a return value and
    nothing poisoned ever reaches disk.

    Mirrors ``make_downloader`` / ``make_html_fetcher`` in download.py:
    same Cachetta primitive, separate kind directory (``llm/``), same
    "permanent" duration.
    """
    return Cachetta(
        path=lambda model, prompt, schema_json: cache_path(
            cache_dir, model, prompt, schema_json
        ),
        duration=CACHE_DURATION,
        # Only cache a non-empty string. cachetta also skips caching when
        # the wrapped function raises -- which is the path the http
        # classifier uses to refuse cache writes on transport errors,
        # 4xx/5xx responses, or malformed JSON.
        condition=lambda content: isinstance(content, str) and content.strip() != "",
    )
