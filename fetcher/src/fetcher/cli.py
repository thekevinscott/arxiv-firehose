"""fetcher command-line interface.

A thin typer wrapper over the Python SDK in ``api.py``: each command parses
flags and delegates. No behavior lives here -- new behavior goes in the SDK.

Two cron-level commands -- ``fetch`` (daily ingest) and ``classify``
(daily labeling) -- ``status`` for read-only counts, and
``train-categories`` (a developer command) that walks a labels root and
compiles every category subdir into a prompt artifact. The fetch stages
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
    cache_dir: Path = CacheDir,
    config: Optional[Path] = ConfigFile,
    verbose: bool = Verbose,
    limit: Optional[int] = Limit,
    dry_run: bool = DryRun,
) -> None:
    """Classify each paper's abstract into topic flags.

    Idempotency lives in the dirsql missing-pairs query: a paper with
    a classifications/<cat>.json on disk is skipped server-side. To
    re-poll the LLM for a pair, delete its file; cachetta will serve
    the prior response from disk anyway.
    """
    api.classify(
        data_dir, cache_dir, config,
        verbose=verbose, limit=limit, dry_run=dry_run,
    )


@app.command("status")
def status(
    data_dir: Path = DataDir,
    config: Optional[Path] = ConfigFile,
) -> None:
    """Print counts: papers known, markdown on disk, classified."""
    typer.echo(api.status(data_dir, config))


@app.command("train-categories")
def train_categories(
    labels_root: Path = typer.Argument(
        Path("labels"),
        help="Labels root; each subdir with _schema.json is a category.",
    ),
    prompts_root: Path = typer.Option(
        Path("prompts"), "--prompts",
        help="Where to write the compiled prompt artifacts.",
    ),
    optimizer: Optional[str] = typer.Option(
        None, "--optimizer",
        help="Optimizer: 'gepa' (LLM-tuned) or omitted (raw template).",
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
    """Compile every category under labels/ into a prompt artifact."""
    results = api.train_categories(
        labels_root, prompts_root, data_dir,
        optimizer=optimizer,
        model=model,
        base_url=base_url,
        verbose=verbose,
    )
    if not results:
        typer.echo(f"train-categories: no categories under {labels_root}")
        return
    for name, info in results.items():
        typer.echo(
            f"train-categories: {info['source']:>5} {name} "
            f"(hash {info['hash']}) -> {info['out']}"
        )


if __name__ == "__main__":
    app()
