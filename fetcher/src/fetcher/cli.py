"""fetcher command-line interface.

A thin typer wrapper over the Python SDK in ``api.py``: each command parses
flags and delegates. No behavior lives here -- new behavior goes in the SDK.

Two cron-level commands -- ``fetch`` (daily ingest) and ``classify``
(daily labeling) -- ``status`` for read-only counts, and ``coax`` (a
developer command) for compiling a labels dir into a prompt artifact.
The fetch stages (``sync_metadata`` and ``render_markdown``) are SDK-only;
for granular debugging call them from a REPL.
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
def status(
    data_dir: Path = DataDir,
    config: Optional[Path] = ConfigFile,
) -> None:
    """Print counts: papers known, markdown on disk, classified."""
    typer.echo(api.status(data_dir, config))


@app.command("coax")
def coax(
    labels_dir: Path = typer.Argument(
        ..., help="Labels dir to compile (must contain _schema.json + examples).",
    ),
    out_dir: Path = typer.Option(
        ..., "--out", help="Where to write the compiled prompt artifact.",
    ),
    optimizer: Optional[str] = typer.Option(
        None, "--optimizer",
        help="Optimizer: 'gepa' (LLM-tuned) or omitted (raw template).",
    ),
    output_name: str = typer.Option(
        "output", "--output-name",
        help="Predicted field name in the rendered template (e.g. is_about_control).",
    ),
    model: Optional[str] = typer.Option(
        None, "--model",
        help="Model tag for --optimizer gepa (e.g. phi4:14b on Ollama).",
    ),
    base_url: str = typer.Option(
        api.DEFAULT_CLASSIFY_BASE_URL, "--base-url",
        help="OpenAI-compatible endpoint for --optimizer gepa.",
    ),
    data_dir: Path = DataDir,
    verbose: bool = Verbose,
) -> None:
    """Compile labels into a CoaxedPrompt artifact (content-cached)."""
    result = api.coax(
        labels_dir, out_dir, data_dir,
        optimizer=optimizer,
        output_name=output_name,
        model=model,
        base_url=base_url,
        verbose=verbose,
    )
    typer.echo(
        f"coax: {result['source']} (hash {result['hash']}) -> {result['out']}"
    )


if __name__ == "__main__":
    app()
