"""Atomic write of ``classification.json``."""

from __future__ import annotations

import json
from pathlib import Path


def write_classification(payload: dict, dest: Path) -> None:
    """Atomic write: ``.part`` + rename, matching the paper.md pattern."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.rename(dest)
