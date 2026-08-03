"""status: a one-screen summary, computed by scanning the filesystem."""

from __future__ import annotations

import json
from pathlib import Path

from ..shared.paths import iter_paper_dirs


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def render(data_dir: Path, config_file: Path | None = None) -> str:
    """Build the status report by walking the paper folders.

    Metadata-only: counts papers known, the primary categories seen, the
    last sync, and total disk usage. *config_file* is accepted for a
    uniform command signature but is not read."""
    papers = 0
    cats: set[str] = set()
    total_bytes = 0

    for pd in iter_paper_dirs(data_dir):
        papers += 1
        try:
            meta = json.loads((pd / "metadata.json").read_text())
            cats.add(meta.get("primary_category", "?"))
        except (OSError, json.JSONDecodeError):
            pass

        for f in pd.rglob("*"):
            if f.is_file():
                total_bytes += f.stat().st_size

    lines = [
        f"Categories tracked: {', '.join(sorted(cats)) or '(none)'}",
        f"Papers known:       {papers:,}",
    ]

    last = data_dir / "last_sync.json"
    if last.exists():
        try:
            s = json.loads(last.read_text())
            lines.append(
                f"Last sync:          {s.get('finished_at', '?')} "
                f"(added {s.get('papers_added', 0)}, "
                f"updated {s.get('papers_updated', 0)})"
            )
        except (OSError, json.JSONDecodeError):
            pass
    else:
        lines.append("Last sync:          (never)")

    lines.append(f"Disk usage:         {_human(total_bytes)}")
    return "\n".join(lines)
