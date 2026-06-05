"""fetcher command-line interface.

A thin typer wrapper over the Python SDK in ``api.py``: each command parses
flags and delegates. No behavior lives here -- new behavior goes in the SDK.

Two cron-level commands -- ``fetch`` (daily ingest) and ``classify``
(daily labeling) -- plus ``status`` for read-only counts. The fetch stages
(``sync_metadata`` and ``render_markdown``) are SDK-only; for granular
debugging call them from a REPL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import api

app = typer.Typer(
    help="Maintain a local mirror of arxiv papers as markdown (plus metadata).",
    no_args_is_help=True,
    add_completion=False,
)

DataDir = typer.Option(api.DEFAULT_DATA_DIR, "--data-dir", help="Arxiv data directory.")
CacheDir = typer.Option(
    api.DEFAULT_CACHE_DIR, "--cache-dir",
    help="cachetta download cache (kept separate from the data dir).",
)
ConfigFile = typer.Option(None, "--config", help="Override config.toml path.")
Verbose = typer.Option(False, "--verbose", "-v", help="Debug logging to stderr.")
Limit = typer.Option(None, "--limit", help="Process at most N items.")
DryRun = typer.Option(False, "--dry-run", help="Plan only; no network or writes.")


@app.command("fetch")
def fetch(
    data_dir: Path = DataDir,
    cache_dir: Path = CacheDir,
    config: Optional[Path] = ConfigFile,
    verbose: bool = Verbose,
    limit: Optional[int] = Limit,
    dry_run: bool = DryRun,
) -> None:
    """Run the daily ingest cycle: sync metadata, then render markdown."""
    result = api.fetch(
        data_dir, cache_dir, config,
        verbose=verbose, limit=limit, dry_run=dry_run,
    )
    if not dry_run:
        typer.echo("")
        typer.echo(result["status"])


@app.command("classify")
def classify(
    data_dir: Path = DataDir,
    config: Optional[Path] = ConfigFile,
    verbose: bool = Verbose,
    limit: Optional[int] = Limit,
    dry_run: bool = DryRun,
    force: bool = typer.Option(
        False, "--force",
        help="Reclassify every (paper, category) pair, ignoring existing files.",
    ),
) -> None:
    """Classify each paper's abstract into topic flags."""
    api.classify(
        data_dir, config,
        verbose=verbose, limit=limit, dry_run=dry_run, force=force,
    )


@app.command("status")
def status(data_dir: Path = DataDir) -> None:
    """Print counts: papers known, markdown on disk, classified."""
    typer.echo(api.status(data_dir))


if __name__ == "__main__":
    app()
