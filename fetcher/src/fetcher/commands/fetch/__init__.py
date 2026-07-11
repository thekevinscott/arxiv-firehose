"""fetch: the daily ingest cycle (sync metadata, then embed abstracts).

Two stages:
- ``sync``  pulls recent paper metadata from the arxiv export API (one
  ``metadata.json`` per paper folder).
- ``embed`` populates ``embeddings.parquet`` from each paper's abstract.

The composite ``run`` here calls both in order and appends a record to
``runs.jsonl`` for durable history. ``api.fetch`` is the SDK entry-point.

Rendering markdown (``render``) is deliberately NOT a stage: it is the
heavy path (up to three paced downloads per paper) and search/classify
only need abstracts. It runs only when explicitly invoked
(``fetcher render`` / ``POST /render`` / ``api.render_markdown``).

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
from .. import embed
from . import pull, render, sync

__all__ = ["pull", "render", "run", "sync"]


def run(
    data_dir: Path,
    config: Config,
    log: logging.Logger,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Run the ingest pipeline: sync-metadata, then embed.

    Returns ``{"added", "updated", "embed"}``. Each non-dry run appends
    a record to ``data_dir/runs.jsonl`` -- a durable history (timing and
    counts) for investigating what a given run did.

    embed runs last and is self-healing: it embeds every paper missing
    from ``embeddings.parquet``, whether that paper was added today or
    slipped through some earlier run. A single-run failure (model load,
    duckdb, disk) is logged and the pipeline exits cleanly -- the next
    day's cron will retry, so /search converges eventually rather than
    failing loudly and blocking the whole ingest cycle.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    log.info("=== fetch: sync ===")
    added, updated = sync.run(data_dir, config, log, limit=limit, dry_run=dry_run)
    log.info("=== fetch: embed ===")
    try:
        embed_counts = embed.run(data_dir, log, dry_run=dry_run, limit=limit)
    except Exception as exc:  # noqa: BLE001 -- embed must not abort fetch
        log.error("embed failed (will retry next run): %s", exc)
        embed_counts = {"embedded": 0, "skipped": 0, "total": 0, "error": str(exc)}
    log.info("=== fetch: done ===")

    if not dry_run:
        record = {
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(time.monotonic() - t0, 3),
            "added": added,
            "updated": updated,
            "embed": embed_counts,
        }
        with (data_dir / "runs.jsonl").open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    return {"added": added, "updated": updated, "embed": embed_counts}
