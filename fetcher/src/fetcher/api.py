"""fetcher Python SDK.

Every command is a function here. The CLI (``cli.py``) is a thin typer
wrapper over these. Each function handles config loading and logger setup
itself, so a caller only supplies the data directory and options.

One cron-level command -- ``fetch`` (ingest: sync metadata, then embed
abstracts) -- plus ``status`` for read-only counts. The firehose mirrors
only paper metadata (abstracts included); it never fetches paper bodies.

Network I/O flows through ``commands.fetch.download.fetch_day`` /
``fetch_id``, each cachetta-cached. Tests stub them out with
``unittest.mock.patch.object`` -- no transport seam to thread. The cache
root is the process-wide ``shared.config.cache``; override its location
with the ``ARXIV_FIREHOSE_CACHE_DIR`` env var.
"""

from __future__ import annotations

from pathlib import Path

from . import serve as serve_mod
from .commands import embed as embed_mod
from .commands import fetch as fetch_mod
from .commands import status as status_mod
from .shared.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    load_config,
)
from .shared.dirsql_schema import query as _dirsql_query
from .shared.logsetup import get_logger

DEFAULT_SERVE_HOST = serve_mod.DEFAULT_HOST
DEFAULT_SERVE_PORT = serve_mod.DEFAULT_PORT

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_DATA_DIR",
    "DEFAULT_SERVE_HOST",
    "DEFAULT_SERVE_PORT",
    "embed",
    "fetch",
    "pull",
    "serve",
    "sql",
    "status",
    "sync_metadata",
]


def sync_metadata(
    data_dir: Path = DEFAULT_DATA_DIR,
    config_file: Path | None = None,
    verbose: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Fetch RSS metadata for tracked categories; write a folder per paper.

    Returns ``(added, updated)`` folder counts. A stage of ``fetch`` --
    callable on its own for tests or granular use, not exposed on the CLI.
    """
    log = get_logger(data_dir, "sync-metadata", verbose)
    cfg = load_config(data_dir, config_file)
    return fetch_mod.sync.run(data_dir, cfg, log, limit=limit, dry_run=dry_run)


def fetch(
    data_dir: Path = DEFAULT_DATA_DIR,
    config_file: Path | None = None,
    verbose: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Run the daily ingest cycle: sync-metadata, then embed abstracts.

    Returns ``{"added", "updated", "embed", "status"}``. Each non-dry run
    appends a record to ``data_dir/runs.jsonl`` -- a durable history for
    investigating what a given run did.

    The firehose mirrors metadata only -- it never fetches paper bodies.
    Abstracts arrive with the metadata and feed the embedding stage that
    powers /search.
    """
    log = get_logger(data_dir, "fetch", verbose)
    cfg = load_config(data_dir, config_file)
    result = fetch_mod.run(data_dir, cfg, log, limit=limit, dry_run=dry_run)
    result["status"] = "" if dry_run else status_mod.render(data_dir, config_file)
    return result


def pull(
    ids: list[str],
    data_dir: Path = DEFAULT_DATA_DIR,
    config_file: Path | None = None,
    verbose: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """Mirror specific papers by arxiv id -- the bespoke retrieval path.

    Use case: tracing a paper's citations. Unlike the daily sync, no
    category or version filter applies -- whatever is asked for by id
    gets its metadata mirrored. Metadata-only, like the daily ingest.

    Returns a counts dict (``pulled`` / ``existing`` / ``invalid`` /
    ``not_found`` / ``failed``). Idempotent: a paper already carrying
    metadata.json is skipped before any network call.
    """
    log = get_logger(data_dir, "pull", verbose)
    cfg = load_config(data_dir, config_file)
    return fetch_mod.pull.run(data_dir, cfg, log, ids, dry_run=dry_run)


def embed(
    data_dir: Path = DEFAULT_DATA_DIR,
    config_file: Path | None = None,
    verbose: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Embed every paper missing from ``embeddings.json``.

    A stage of ``fetch`` -- callable on its own for a manual backfill
    or a targeted rerun (e.g. after a metadata correction). Idempotent:
    a paper already in the file is skipped, so a re-run is a no-op
    once everything is embedded.
    """
    log = get_logger(data_dir, "embed", verbose)
    # config is unused today (model name is a constant) but kept in the
    # signature to match the other SDK stage functions; a future toggle
    # for model choice would land in [embed] without breaking callers.
    _ = load_config(data_dir, config_file)
    return embed_mod.run(data_dir, log, dry_run=dry_run, limit=limit)


def status(
    data_dir: Path = DEFAULT_DATA_DIR,
    config_file: Path | None = None,
) -> str:
    """Return the status report, computed by scanning the data dir."""
    return status_mod.render(data_dir, config_file)


def sql(
    statement: str,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> list[dict]:
    """Run one read-only SQL statement against the dirsql schema.

    Tables live in ``shared.dirsql_schema`` (``papers``, ``metadata``
    EAV, ``embeddings``). dirsql scans ``data_dir.parent`` -- the same
    root the schema globs are written against -- and its authorizer
    rejects any non-read statement, so this is read-only by construction.

    Returns the result rows as dicts. The programmatic twin of the
    ``POST /sql`` endpoint and the metadata counterpart to /search
    (which owns the DuckDB-over-embeddings surface).
    """
    return _dirsql_query(statement, data_dir.parent)


def serve(
    data_dir: Path = DEFAULT_DATA_DIR,
    config_file: Path | None = None,
    *,
    host: str = DEFAULT_SERVE_HOST,
    port: int = DEFAULT_SERVE_PORT,
) -> None:
    """Run the HTTP API. Blocks; use a systemd unit for the daemon.

    A tailnet-only counterpart to the CLI: ``status`` / ``fetch`` over
    HTTP so future ops don't require SSH. Bind to the tailscale IP in
    production; default ``127.0.0.1`` is for local dev.
    """
    serve_mod.serve(data_dir, config_file, host=host, port=port)
