"""Atomic write of ``classification.json``."""

from __future__ import annotations

from pathlib import Path

from ...shared.atomic_write import atomic_write_json


def write_classification(payload: dict, dest: Path) -> None:
    """Atomic write: ``.part`` + rename, matching the paper.md pattern."""
    atomic_write_json(dest, payload)
