"""fetcher Python SDK.

Every command is a function here. The CLI (``cli.py``) is a thin typer
wrapper over these. Each function handles config loading and logger setup
itself, so a caller only supplies the data directory and options.

Two cron-level commands -- ``fetch`` (ingest: sync metadata, then embed
abstracts) and ``classify`` (label each paper against a taxonomy) -- plus
``status`` for read-only counts. Rendering markdown is explicit-only
(``render_markdown`` / ``fetcher render`` / ``POST /render``); it is the
heavy path and no auto-executing script triggers it.

Network I/O flows through ``commands.fetch.download.fetch_day`` /
``fetch_paper`` / ``fetch_html``, each cachetta-cached. Tests stub them
out with ``unittest.mock.patch.object`` -- no transport seam to thread.
The cache root is the process-wide ``shared.config.cache``; override its
location with the ``ARXIV_FIREHOSE_CACHE_DIR`` env var.
"""

from __future__ import annotations

from pathlib import Path

from . import serve as serve_mod
from .commands import classify as classify_mod
from .commands import embed as embed_mod
from .commands import fetch as fetch_mod
from .commands import status as status_mod
from .commands import train_categories as train_categories_mod
from .commands.classify import Classifier
from .shared.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CLASSIFY_BASE_URL,
    DEFAULT_DATA_DIR,
    load_config,
)
from .shared.dirsql_schema import query as _dirsql_query
from .shared.convert import REAL_CONVERTER, Converter
from .shared.logsetup import get_logger

DEFAULT_SERVE_HOST = serve_mod.DEFAULT_HOST
DEFAULT_SERVE_PORT = serve_mod.DEFAULT_PORT

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_DATA_DIR",
    "DEFAULT_SERVE_HOST",
    "DEFAULT_SERVE_PORT",
    "classify",
    "embed",
    "fetch",
    "pull",
    "render_markdown",
    "serve",
    "sql",
    "status",
    "sync_metadata",
    "train_categories",
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


def render_markdown(
    data_dir: Path = DEFAULT_DATA_DIR,
    config_file: Path | None = None,
    verbose: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    converter: Converter = REAL_CONVERTER,
) -> dict[str, int]:
    """Produce a markdown rendering for each known paper.

    Returns a counts dict. Explicit-only: no cron or auto-executing
    script calls this -- rendering is heavy (up to three paced downloads
    per paper). Exposed as ``fetcher render`` and ``POST /render``.
    *converter* is a test seam; it defaults to the real
    arxiv2md/pypandoc converter.
    """
    log = get_logger(data_dir, "render", verbose)
    cfg = load_config(data_dir, config_file)
    return fetch_mod.render.run(
        data_dir, cfg, log,
        limit=limit, dry_run=dry_run, converter=converter,
    )


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

    Rendering markdown is NOT part of the cycle -- it is explicit-only
    (``render_markdown``), since search/classify need only abstracts.

    Classify is a separate command (``api.classify``) on its own schedule;
    fetch does not trigger it. Keeping ingest and classify decoupled lets
    the fetch cron stay quick and predictable while classify can be rerun
    cheaply when prompts change.
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
    gets its metadata mirrored. Metadata-only, like the daily ingest;
    markdown arrives only via an explicit ``render_markdown`` call.

    Returns a counts dict (``pulled`` / ``existing`` / ``invalid`` /
    ``not_found`` / ``failed``). Idempotent: a paper already carrying
    metadata.json is skipped before any network call.
    """
    log = get_logger(data_dir, "pull", verbose)
    cfg = load_config(data_dir, config_file)
    return fetch_mod.pull.run(data_dir, cfg, log, ids, dry_run=dry_run)


def classify(
    data_dir: Path = DEFAULT_DATA_DIR,
    config_file: Path | None = None,
    verbose: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    classifier: Classifier | None = None,
) -> dict[str, int]:
    """Classify every paper's abstract into topic flags.

    Returns a counts dict. *classifier* is a test seam (like *transport*);
    by default it talks to the local Ollama configured in ``[classify]``.
    With ``[classify] prompts_dirs = []`` the call no-ops cleanly so the
    cron stays green while labels are still being authored.

    Idempotency is dirsql-driven: the missing-pairs SQL query only
    surfaces (paper, category) pairs without a classification file on
    disk, so a re-run is naturally a no-op once everything is labeled.
    To re-poll the LLM for a pair, delete its
    ``classifications/<cat>.json``; cachetta will serve the prior
    response from disk anyway (no network).

    The cachetta-backed LLM response cache lives at
    ``shared.config.cache`` (``~/.cache/arxiv-firehose``); a repeat
    ``(model, prompt, schema)`` triple serves from disk with no network
    call. The cache is the only mechanism that makes a re-run cheap --
    there is no in-memory dict, no file-existence shortcut beyond the
    SQL filter, and no ``--force`` flag.
    """
    log = get_logger(data_dir, "classify", verbose)
    cfg = load_config(data_dir, config_file)
    return classify_mod.run(
        data_dir, cfg, log,
        limit=limit, dry_run=dry_run, classifier=classifier,
    )


def train_categories(
    labels_root: Path,
    prompts_root: Path,
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    optimizer: str | None = None,
    model: str | None = None,
    base_url: str = DEFAULT_CLASSIFY_BASE_URL,
    verbose: bool = False,
    cache_root: Path = train_categories_mod.CACHE_ROOT,
) -> dict[str, dict[str, str]]:
    """Compile every category under *labels_root* into *prompts_root*.

    A "category" is any subdir of ``labels_root`` that carries a
    ``_schema.json``. Its name (``is-about-control``) becomes the output
    dir (``prompts_root/is-about-control``) and -- with hyphens swapped to
    underscores -- the runtime flag key (``is_about_control``).

    Each per-category compile is content-cached under
    ``~/.cache/arxiv-firehose/classify/{hash}/``. With *optimizer* =
    "gepa" the labels also drive a DSPy fit against *model* / *base_url*.

    Returns ``{name: {"hash", "source", "out"}, ...}``. *data_dir* is
    used only for the log file location.
    """
    log = get_logger(data_dir, "train-categories", verbose)
    return train_categories_mod.run(
        labels_root, prompts_root, log,
        optimizer=optimizer,
        model=model,
        base_url=base_url,
        cache_root=cache_root,
    )


def embed(
    data_dir: Path = DEFAULT_DATA_DIR,
    config_file: Path | None = None,
    verbose: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Embed every paper missing from ``embeddings.parquet``.

    A stage of ``fetch`` -- callable on its own for a manual backfill
    or a targeted rerun (e.g. after a metadata correction). Idempotent:
    a paper already in the parquet is skipped, so a re-run is a no-op
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
    """Return the status report, computed by scanning the data dir.

    Reads ``[classify] prompts_dirs`` from config to know which categories
    a "fully classified" paper should carry.
    """
    return status_mod.render(data_dir, config_file)


def sql(
    statement: str,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> list[dict]:
    """Run one read-only SQL statement against the dirsql schema.

    Tables live in ``shared.dirsql_schema`` (``papers``, ``metadata``
    EAV, ``papers_categories``, ``categories``, ``markdown``,
    ``no_markdown``). dirsql scans ``data_dir.parent`` -- the same root
    the schema globs are written against -- and its authorizer rejects
    any non-read statement, so this is read-only by construction.

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

    A tailnet-only counterpart to the CLI: ``status`` / ``fetch`` /
    ``classify`` over HTTP so future ops don't require SSH. Bind to the
    tailscale IP in production; default ``127.0.0.1`` is for local dev.
    """
    serve_mod.serve(data_dir, config_file, host=host, port=port)
