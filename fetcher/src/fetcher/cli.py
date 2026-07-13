"""fetcher command-line interface.

A thin typer wrapper over the Python SDK in ``api.py``: each command parses
flags and delegates. No behavior lives here -- new behavior goes in the SDK.

Two cron-level commands -- ``fetch`` (daily ingest) and ``classify``
(daily labeling) -- ``status`` for read-only counts, ``render`` (the
explicit-only markdown pass; no cron triggers it), and
``train-categories`` (a developer command) that walks a labels root and
compiles every category subdir into a prompt artifact. The remaining
fetch stage (``sync_metadata``) is SDK-only; for granular debugging
call it from a REPL.
"""

from __future__ import annotations

import json
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


@app.command("render")
def render(
    data_dir: Path = DataDir,
    config: Optional[Path] = ConfigFile,
    verbose: bool = Verbose,
    limit: Optional[int] = Limit,
    dry_run: bool = DryRun,
) -> None:
    """Render markdown for every known paper missing one.

    Explicit-only -- no cron or auto-executing script runs this. It is
    the heavy path (up to three paced downloads per paper); search and
    classify only need abstracts, which fetch already mirrors.
    """
    counts = api.render_markdown(
        data_dir, config,
        verbose=verbose, limit=limit, dry_run=dry_run,
    )
    typer.echo(
        f"html={counts['html']} latex={counts['latex']} "
        f"pdf={counts['pdf']} absent={counts['absent']} "
        f"failed={counts['failed']} skipped={counts['skipped']}"
    )


@app.command("classify")
def classify(
    data_dir: Path = DataDir,
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
        data_dir, config,
        verbose=verbose, limit=limit, dry_run=dry_run,
    )


@app.command("embed")
def embed(
    data_dir: Path = DataDir,
    config: Optional[Path] = ConfigFile,
    verbose: bool = Verbose,
    limit: Optional[int] = Limit,
    dry_run: bool = DryRun,
) -> None:
    """Populate embeddings.parquet for every paper missing one.

    Independent of ``render`` -- reads only ``metadata.json.abstract``.
    Runs to convergence: papers already in the parquet are skipped.
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
    """Print counts: papers known, markdown on disk, classified."""
    typer.echo(api.status(data_dir, config))


@app.command("sql")
def sql(
    statement: str = typer.Argument(
        ..., help="Read-only SQL against the dirsql schema."
    ),
    data_dir: Path = DataDir,
) -> None:
    """Run one read-only SQL query against the dirsql schema; print JSON.

    Tables: papers, metadata (EAV: paper_id/key/value), papers_categories,
    categories, markdown, no_markdown. Writes are rejected by dirsql's
    authorizer. Example:

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
        help="Model tag for --optimizer gepa (e.g. Qwen3.6-27B-Q4_K_M on llama.cpp).",
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
