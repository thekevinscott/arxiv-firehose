"""fetcher command-line interface.

A thin typer wrapper over the Python SDK in ``api.py``: each command parses
flags and delegates. No behavior lives here -- new behavior goes in the SDK.

One cron-level command -- ``fetch`` (daily ingest: sync metadata, then
embed abstracts) -- plus ``status`` for read-only counts, ``pull`` for
bespoke by-id retrieval, ``embed`` for a standalone embedding backfill,
and ``sql`` / ``serve`` for query and HTTP access. The ``sync_metadata``
stage is SDK-only; for granular debugging call it from a REPL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import api

app = typer.Typer(
    help="Maintain a local mirror of arxiv paper metadata (abstracts) with semantic search.",
    no_args_is_help=True,
    add_completion=False,
)

DataDir = typer.Option(api.DEFAULT_DATA_DIR, "--data-dir", help="Arxiv data directory.")
# Cache root is process-wide (shared.config.cache); override its location
# with the ARXIV_FIREHOSE_CACHE_DIR env var, not a flag. The cache is
# transparent -- no CLI surface should depend on its layout.
ConfigFile = typer.Option(None, "--config", help="Override config.toml path.")
Verbose = typer.Option(False, "--verbose", "-v", help="Debug logging to stderr.")
Limit = typer.Option(None, "--limit", help="Process at most N items.")
DryRun = typer.Option(False, "--dry-run", help="Plan only; no network or writes.")


@app.command("fetch")
def fetch(
    data_dir: Path = DataDir,
    config: Optional[Path] = ConfigFile,
    verbose: bool = Verbose,
    limit: Optional[int] = Limit,
    dry_run: bool = DryRun,
) -> None:
    """Run the daily ingest cycle: sync metadata, then embed abstracts."""
    result = api.fetch(
        data_dir, config,
        verbose=verbose, limit=limit, dry_run=dry_run,
    )
    if not dry_run:
        typer.echo("")
        typer.echo(result["status"])


@app.command("pull")
def pull(
    ids: list[str] = typer.Argument(
        ..., help="arxiv ids to mirror (e.g. 2401.12345 cs/0501001)."
    ),
    data_dir: Path = DataDir,
    config: Optional[Path] = ConfigFile,
    verbose: bool = Verbose,
    dry_run: bool = DryRun,
) -> None:
    """Mirror specific papers' metadata by id, e.g. citations."""
    result = api.pull(
        ids, data_dir, config,
        verbose=verbose, dry_run=dry_run,
    )
    typer.echo(
        f"pulled={result['pulled']} existing={result['existing']} "
        f"not_found={result['not_found']} invalid={result['invalid']} "
        f"failed={result['failed']}"
    )


@app.command("embed")
def embed(
    data_dir: Path = DataDir,
    config: Optional[Path] = ConfigFile,
    verbose: bool = Verbose,
    limit: Optional[int] = Limit,
    dry_run: bool = DryRun,
) -> None:
    """Populate embeddings.json for every paper missing one.

    Reads only ``metadata.json.abstract``.
    Runs to convergence: papers already in the file are skipped.
    Also runs as a stage inside ``fetch``; this entry point is for a
    standalone backfill / manual retrigger.
    """
    counts = api.embed(
        data_dir, config,
        verbose=verbose, limit=limit, dry_run=dry_run,
    )
    typer.echo(
        f"embedded={counts['embedded']} "
        f"skipped={counts['skipped']} "
        f"total={counts['total']}"
    )


@app.command("status")
def status(
    data_dir: Path = DataDir,
    config: Optional[Path] = ConfigFile,
) -> None:
    """Print counts: papers known, categories tracked, last sync, disk usage."""
    typer.echo(api.status(data_dir, config))


@app.command("sql")
def sql(
    statement: str = typer.Argument(
        ..., help="Read-only SQL against the dirsql schema."
    ),
    data_dir: Path = DataDir,
) -> None:
    """Run one read-only SQL query against the dirsql schema; print JSON.

    Tables: papers, metadata (EAV: paper_id/key/value), embeddings.
    Writes are rejected by dirsql's authorizer. Example:

        fetcher sql "SELECT primary_category, COUNT(*) n FROM papers GROUP BY 1"
    """
    rows = api.sql(statement, data_dir)
    typer.echo(json.dumps(rows, indent=2, default=str))


@app.command("serve")
def serve(
    data_dir: Path = DataDir,
    config: Optional[Path] = ConfigFile,
    host: str = typer.Option(
        api.DEFAULT_SERVE_HOST, "--host",
        help="Bind address. Use the tailscale IP for tailnet access; "
        "default 127.0.0.1 is local-dev only.",
    ),
    port: int = typer.Option(api.DEFAULT_SERVE_PORT, "--port", help="HTTP port."),
) -> None:
    """Run the HTTP API. Foregrounded; use systemd for the daemon."""
    api.serve(data_dir, config, host=host, port=port)


if __name__ == "__main__":
    app()
