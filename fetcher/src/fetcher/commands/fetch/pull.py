"""pull: bespoke single-paper retrieval by arxiv id.

The daily sync mirrors only new v1 papers in tracked categories; pull
mirrors whatever the caller asks for by id -- e.g. the citations of a
paper worth tracing. Metadata comes from the export API's ``id_list=``
query (the same Atom shape sync parses, with no version or category
filter).

Pull is metadata-only, like the daily ingest: search/classify/embed
need abstracts, not paper bodies. Markdown arrives only when render is
explicitly invoked (``fetcher render`` / ``POST /render``).

Idempotency lives on disk, not in a cache: a paper whose folder already
carries ``metadata.json`` is skipped before any network call, so
re-pulling a list converges to a no-op.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import httpx

from . import download
from .sync import _parse_entry
from ...shared.atomic_write import atomic_write_json
from ...shared.config import Config
from ...shared.paths import metadata_path, parse_id


def _status(exc: Exception) -> int | None:
    return getattr(getattr(exc, "response", None), "status_code", None)


def run(
    data_dir: Path,
    config: Config,
    log: logging.Logger,
    ids: list[str],
    dry_run: bool = False,
) -> dict[str, int]:
    """Execute pull. Returns a counts dict."""
    now = datetime.now(timezone.utc).isoformat()
    counts = {"pulled": 0, "existing": 0, "invalid": 0, "not_found": 0,
              "failed": 0}

    log.info("pull start: %d ids", len(ids))

    for raw in ids:
        try:
            arxiv_id = parse_id(raw).raw
        except ValueError:
            counts["invalid"] += 1
            log.error("pull %s: not an arxiv id", raw)
            continue

        meta_file = metadata_path(data_dir, arxiv_id)
        if meta_file.exists():
            counts["existing"] += 1
            log.info("pull %s: already mirrored", arxiv_id)
            continue

        if dry_run:
            log.info("[dry-run] would pull %s", arxiv_id)
            continue

        try:
            body = download.fetch_id(arxiv_id)
        except httpx.HTTPError as exc:
            if _status(exc) == 429:
                log.warning(
                    "arxiv rate limited (429) at %s; stopping pull for this run",
                    arxiv_id,
                )
                break
            if _status(exc) == 404:
                counts["not_found"] += 1
                log.warning("pull %s: not on arxiv (HTTP 404)", arxiv_id)
                continue
            counts["failed"] += 1
            log.error("pull %s: %s", arxiv_id, exc)
            continue
        feed = feedparser.parse(body)
        rec = _parse_entry(feed.entries[0], None) if feed.entries else None
        if rec is None:
            counts["not_found"] += 1
            log.warning("pull %s: no usable entry in API response", arxiv_id)
            continue
        atomic_write_json(meta_file, rec.to_metadata(now))
        counts["pulled"] += 1
        log.info("pull %s: wrote metadata.json", arxiv_id)

    log.info("pull done: %s", counts)
    return counts
