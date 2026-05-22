"""status: a one-screen summary, computed by scanning the filesystem."""

from __future__ import annotations

import json
from pathlib import Path

from .paths import iter_paper_dirs


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def render(data_dir: Path) -> str:
    """Build the status report by walking the paper folders."""
    papers = 0
    have_md = 0
    no_md = 0
    cats: set[str] = set()
    md_bytes = 0
    total_bytes = 0

    for pd in iter_paper_dirs(data_dir):
        papers += 1
        try:
            meta = json.loads((pd / "metadata.json").read_text())
            cats.add(meta.get("primary_category", "?"))
        except (OSError, json.JSONDecodeError):
            pass

        md = pd / "paper.md"
        if md.exists() and md.stat().st_size > 0:
            have_md += 1
            md_bytes += md.stat().st_size
        if (pd / ".no_markdown").exists():
            no_md += 1

        for f in pd.rglob("*"):
            if f.is_file():
                total_bytes += f.stat().st_size

    lines = [
        f"Categories tracked: {', '.join(sorted(cats)) or '(none)'}",
        f"Papers known:       {papers:,}",
        f"Markdown on disk:   {have_md:,}  "
        f"({no_md:,} have none available, "
        f"{papers - have_md - no_md:,} not yet fetched)",
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

    lines.append(
        f"Disk usage:         {_human(total_bytes)} "
        f"(markdown: {_human(md_bytes)})"
    )
    return "\n".join(lines)
