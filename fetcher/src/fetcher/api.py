"""fetcher Python SDK.

Every command is a function here. The CLI (``cli.py``) is a thin typer
wrapper over these. Each function handles config loading and logger setup
itself, so a caller only supplies the data/cache directories and options.

Network I/O flows through an injectable *transport* (``(url, timeout) ->
bytes``). The default is the real rate-limited httpx GET; integration tests
pass a fixture-backed fake. See AGENTS.md.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from . import classify as classify_mod
from . import fetch as fetch_mod
from . import status as status_mod
from . import sync as sync_mod
from .classify import Classifier
from .config import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, load_config
from .convert import REAL_CONVERTER, Converter
from .download import Transport
from .logsetup import get_logger

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_DATA_DIR",
    "sync_metadata",
    "fetch",
    "classify",
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
    """Run the ingest pipeline: sync-metadata then fetch.

    Returns ``{"added", "updated", "fetch", "status"}``. Each non-dry run
    appends a record to ``data_dir/runs.jsonl`` -- a durable history
    (timing and counts) for investigating what a given run did.

    Classify is a separate process; run it via ``fetcher classify`` (or
    ``api.classify``) on its own schedule. Keeping ingest and classify
    decoupled lets the fetch cron stay quick and predictable while
    classify can be rerun cheaply when prompts change.
    """
    log = get_logger(data_dir, "run", verbose)
    cfg = load_config(data_dir, config_file)

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

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

    if not dry_run:
        record = {
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(time.monotonic() - t0, 3),
            "added": added,
            "updated": updated,
            "fetch": counts,
        }
        with (data_dir / "runs.jsonl").open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    return {
        "added": added,
        "updated": updated,
        "fetch": counts,
        "status": "" if dry_run else status_mod.render(data_dir),
    }
