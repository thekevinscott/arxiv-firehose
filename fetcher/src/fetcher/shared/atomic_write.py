"""Atomic file writes: ``.part`` + ``rename``.

Every persistent artifact (markdown, classification JSON, categories
index) is written via this helper. The .part suffix + rename guarantees
a partial write never appears at the final path: dirsql watchers and
re-runs see either the prior version or the new one, never a truncated
file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def atomic_write_text(dest: Path, content: str) -> None:
    """Write *content* to *dest* atomically.

    Creates parent dirs as needed; writes to ``<dest>.part`` first, then
    renames into place. The rename is atomic on POSIX within a filesystem.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(dest)


def atomic_write_json(dest: Path, payload: Any) -> None:
    """Write *payload* as indented, sort_keys JSON, atomically.

    sort_keys keeps the file byte-stable across runs -- dirsql watchers
    and ``git diff`` both stay quiet when the data hasn't changed.
    """
    atomic_write_text(dest, json.dumps(payload, indent=2, sort_keys=True))
