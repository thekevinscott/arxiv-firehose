"""fetch: the daily ingest cycle (sync metadata, then render markdown).

Two stages:
- ``sync``   pulls recent paper metadata from arxiv RSS feeds (one
  ``metadata.json`` per paper folder).
- ``render`` produces a markdown rendering for each known paper (HTML →
  LaTeX → PDF fallback chain).

The composite ``run`` here calls both in order and appends a record to
``runs.jsonl`` for durable history. ``api.fetch`` is the SDK entry-point.

Classify is a separate command (its own CLI + cron entry); fetch does not
trigger it. Keeping ingest and classify decoupled lets the fetch cron stay
quick and predictable while classify reruns cheaply when prompts change.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from ...shared.config import Config
from ...shared.convert import REAL_CONVERTER, Converter
from ...shared.download import Transport
from . import render, sync

__all__ = ["render", "run", "sync"]


def run(
    data_dir: Path,
    cache_dir: Path,
    config: Config,
    log: logging.Logger,
    limit: int | None = None,
    dry_run: bool = False,
    transport: Transport | None = None,
    converter: Converter = REAL_CONVERTER,
) -> dict[str, object]:
    """Run the ingest pipeline: sync-metadata then render-markdown.

    Returns ``{"added", "updated", "render"}``. Each non-dry run appends a
    record to ``data_dir/runs.jsonl`` -- a durable history (timing and
    counts) for investigating what a given run did.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    log.info("=== fetch: sync ===")
    added, updated = sync.run(
        data_dir, cache_dir, config, log,
        limit=limit, dry_run=dry_run, transport=transport,
    )
    log.info("=== fetch: render ===")
    counts = render.run(
        data_dir, cache_dir, config, log,
        limit=limit, dry_run=dry_run, transport=transport, converter=converter,
    )
    log.info("=== fetch: done ===")

    if not dry_run:
        record = {
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(time.monotonic() - t0, 3),
            "added": added,
            "updated": updated,
            "render": counts,
        }
        with (data_dir / "runs.jsonl").open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    return {"added": added, "updated": updated, "render": counts}
