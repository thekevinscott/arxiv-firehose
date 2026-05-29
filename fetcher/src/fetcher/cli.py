"""fetcher command-line interface.

A thin typer wrapper over the Python SDK in ``api.py``: each command parses
flags and delegates. No behavior lives here -- new behavior goes in the SDK.
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


@app.command("sync-metadata")
def sync_metadata(
    data_dir: Path = DataDir,
    cache_dir: Path = CacheDir,
    config: Optional[Path] = ConfigFile,
    verbose: bool = Verbose,
    limit: Optional[int] = Limit,
    dry_run: bool = DryRun,
) -> None:
    """Fetch paper metadata for tracked categories; write a folder per paper."""
    api.sync_metadata(
        data_dir, cache_dir, config,
        verbose=verbose, limit=limit, dry_run=dry_run,
    )


@app.command("fetch")
def fetch(
    data_dir: Path = DataDir,
    cache_dir: Path = CacheDir,
    config: Optional[Path] = ConfigFile,
    verbose: bool = Verbose,
    limit: Optional[int] = Limit,
    dry_run: bool = DryRun,
) -> None:
    """Produce a markdown rendering for each known paper."""
    api.fetch(
        data_dir, cache_dir, config,
        verbose=verbose, limit=limit, dry_run=dry_run,
    )


@app.command("classify")
def classify(
    data_dir: Path = DataDir,
    config: Optional[Path] = ConfigFile,
    verbose: bool = Verbose,
    limit: Optional[int] = Limit,
    dry_run: bool = DryRun,
    force: bool = typer.Option(
        False, "--force",
        help="Reclassify even if classification.json already exists.",
    ),
) -> None:
    """Classify each paper's abstract into topic flags."""
    api.classify(
        data_dir, config,
        verbose=verbose, limit=limit, dry_run=dry_run, force=force,
    )


@app.command("status")
def status(data_dir: Path = DataDir) -> None:
    """Print counts: papers known, markdown on disk."""
    typer.echo(api.status(data_dir))


@app.command("run")
def run_all(
    data_dir: Path = DataDir,
    cache_dir: Path = CacheDir,
    config: Optional[Path] = ConfigFile,
    verbose: bool = Verbose,
    limit: Optional[int] = Limit,
    dry_run: bool = DryRun,
) -> None:
    """Run the full pipeline: sync-metadata then fetch."""
    result = api.run(
        data_dir, cache_dir, config,
        verbose=verbose, limit=limit, dry_run=dry_run,
    )
    if not dry_run:
        typer.echo("")
        typer.echo(result["status"])


if __name__ == "__main__":
    app()
