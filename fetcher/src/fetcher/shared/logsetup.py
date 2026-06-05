"""Per-subcommand logging: a rotating file under logs/, plus stderr."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FMT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def get_logger(data_dir: Path, command: str, verbose: bool) -> logging.Logger:
    """Return a logger writing to logs/{command}.log and stderr."""
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # cachetta logs its own ERROR (carrying httpx's MDN link) for every failed
    # download and then re-raises -- the caller logs a concise line of its
    # own, so silence the duplicate.
    logging.getLogger("cachetta").setLevel(logging.CRITICAL)

    logger = logging.getLogger(f"fetcher.{command}")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    fh = RotatingFileHandler(
        logs_dir / f"{command}.log", maxBytes=50 * 1024 * 1024, backupCount=3
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(sh)

    logger.propagate = False
    return logger
