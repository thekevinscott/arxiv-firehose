"""fetcher Python SDK.

Every command is a function here. The CLI (``cli.py``) is a thin typer
wrapper over these. Each function handles config loading and logger setup
itself, so a caller only supplies the data/cache directories and options.

Two cron-level commands -- ``fetch`` (ingest: sync metadata, then render
markdown) and ``classify`` (label each paper against a taxonomy) -- plus
``status`` for read-only counts. The fetch stages (``sync_metadata`` and
``render_markdown``) are also exported here for granular use and tests;
they are not exposed on the CLI.

Network I/O flows through an injectable *transport* (``(url, timeout) ->
bytes``). The default is the real rate-limited httpx GET; integration tests
pass a fixture-backed fake. See AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path

from .commands import classify as classify_mod
from .commands import fetch as fetch_mod
from .commands import status as status_mod
from .commands.classify import Classifier
from .shared.config import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, load_config
from .shared.convert import REAL_CONVERTER, Converter
from .shared.download import Transport
from .shared.logsetup import get_logger

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_DATA_DIR",
    "classify",
    "fetch",
    "render_markdown",
    "status",
    "sync_metadata",
]


def sync_metadata(
    data_dir: Path = DEFAULT_DATA_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    config_file: Path | None = None,
    verbose: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    transport: Transport | None = None,
) -> tuple[int, int]:
    """Fetch RSS metadata for tracked categories; write a folder per paper.

    Returns ``(added, updated)`` folder counts. A stage of ``fetch`` --
    callable on its own for tests or granular use, not exposed on the CLI.
    """
    log = get_logger(data_dir, "sync-metadata", verbose)
    cfg = load_config(data_dir, config_file)
    return fetch_mod.sync.run(
        data_dir, cache_dir, cfg, log,
        limit=limit, dry_run=dry_run, transport=transport,
    )


def render_markdown(
    data_dir: Path = DEFAULT_DATA_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    config_file: Path | None = None,
    verbose: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    transport: Transport | None = None,
    converter: Converter = REAL_CONVERTER,
) -> dict[str, int]:
    """Produce a markdown rendering for each known paper.

    Returns a counts dict. A stage of ``fetch`` -- callable on its own for
    tests or granular use, not exposed on the CLI. *converter* is a test
    seam (like *transport*); it defaults to the real arxiv2md/pypandoc
    converter.
    """
    log = get_logger(data_dir, "render", verbose)
    cfg = load_config(data_dir, config_file)
    return fetch_mod.render.run(
        data_dir, cache_dir, cfg, log,
        limit=limit, dry_run=dry_run, transport=transport, converter=converter,
    )


def fetch(
    data_dir: Path = DEFAULT_DATA_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    config_file: Path | None = None,
    verbose: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    transport: Transport | None = None,
    converter: Converter = REAL_CONVERTER,
) -> dict[str, object]:
    """Run the daily ingest cycle: sync-metadata, then render markdown.

    Returns ``{"added", "updated", "render", "status"}``. Each non-dry run
    appends a record to ``data_dir/runs.jsonl`` -- a durable history for
    investigating what a given run did.

    Classify is a separate command (``api.classify``) on its own schedule;
    fetch does not trigger it. Keeping ingest and classify decoupled lets
    the fetch cron stay quick and predictable while classify can be rerun
    cheaply when prompts change.
    """
    log = get_logger(data_dir, "fetch", verbose)
    cfg = load_config(data_dir, config_file)
    result = fetch_mod.run(
        data_dir, cache_dir, cfg, log,
        limit=limit, dry_run=dry_run, transport=transport, converter=converter,
    )
    result["status"] = "" if dry_run else status_mod.render(data_dir, config_file)
    return result


def classify(
    data_dir: Path = DEFAULT_DATA_DIR,
    config_file: Path | None = None,
    verbose: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    classifier: Classifier | None = None,
) -> dict[str, int]:
    """Classify every paper's abstract into topic flags.

    Returns a counts dict. *classifier* is a test seam (like *transport*);
    by default it talks to the local Ollama configured in ``[classify]``.
    With ``[classify] prompts_dirs = []`` the call no-ops cleanly so the
    cron stays green while labels are still being authored.
    """
    log = get_logger(data_dir, "classify", verbose)
    cfg = load_config(data_dir, config_file)
    return classify_mod.run(
        data_dir, cfg, log,
        limit=limit, dry_run=dry_run, force=force, classifier=classifier,
    )


def status(
    data_dir: Path = DEFAULT_DATA_DIR,
    config_file: Path | None = None,
) -> str:
    """Return the status report, computed by scanning the data dir.

    Reads ``[classify] prompts_dirs`` from config to know which categories
    a "fully classified" paper should carry.
    """
    return status_mod.render(data_dir, config_file)
