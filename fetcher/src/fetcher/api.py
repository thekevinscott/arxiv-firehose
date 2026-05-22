"""fetcher Python SDK.

Every command is a function here. The CLI (``cli.py``) is a thin typer
wrapper over these. Each function handles config loading and logger setup
itself, so a caller only supplies the data/cache directories and options.

Network I/O flows through an injectable *transport* (``(url, timeout) ->
bytes``). The default is the real rate-limited httpx GET; integration tests
pass a fixture-backed fake. See AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path

from . import fetch as fetch_mod
from . import status as status_mod
from . import sync as sync_mod
from .config import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, load_config
from .convert import REAL_CONVERTER, Converter
from .download import Transport
from .logsetup import get_logger

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_DATA_DIR",
    "sync_metadata",
    "fetch",
    "status",
    "run",
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

    Returns ``(added, updated)`` folder counts.
    """
    log = get_logger(data_dir, "sync-metadata", verbose)
    cfg = load_config(data_dir, config_file)
    return sync_mod.run(
        data_dir, cache_dir, cfg, log,
        limit=limit, dry_run=dry_run, transport=transport,
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
) -> dict[str, int]:
    """Produce a markdown rendering for each known paper.

    Returns a counts dict. *converter* is a test seam (like *transport*); it
    defaults to the real arxiv2md/pypandoc converter.
    """
    log = get_logger(data_dir, "fetch", verbose)
    cfg = load_config(data_dir, config_file)
    return fetch_mod.run(
        data_dir, cache_dir, cfg, log,
        limit=limit, dry_run=dry_run, transport=transport, converter=converter,
    )


def status(data_dir: Path = DEFAULT_DATA_DIR) -> str:
    """Return the status report, computed by scanning the data dir."""
    return status_mod.render(data_dir)


def run(
    data_dir: Path = DEFAULT_DATA_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    config_file: Path | None = None,
    verbose: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    transport: Transport | None = None,
    converter: Converter = REAL_CONVERTER,
) -> dict[str, object]:
    """Run the full pipeline: sync-metadata, then fetch.

    Returns ``{"added", "updated", "fetch", "status"}``.
    """
    log = get_logger(data_dir, "run", verbose)
    cfg = load_config(data_dir, config_file)

    log.info("=== run: sync-metadata ===")
    added, updated = sync_mod.run(
        data_dir, cache_dir, cfg, log,
        limit=limit, dry_run=dry_run, transport=transport,
    )
    log.info("=== run: fetch ===")
    counts = fetch_mod.run(
        data_dir, cache_dir, cfg, log,
        limit=limit, dry_run=dry_run, transport=transport, converter=converter,
    )
    log.info("=== run: done ===")

    return {
        "added": added,
        "updated": updated,
        "fetch": counts,
        "status": "" if dry_run else status_mod.render(data_dir),
    }
