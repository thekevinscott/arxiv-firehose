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

    Metadata + markdown coverage only. Classification counts were removed
    from the report -- the classify stage is parked on a wip branch, not
    on main. *config_file* is accepted for a uniform command signature but
    is no longer read.
    """
    papers = 0
    have_md = 0
    no_md = 0
    not_yet_fetched = 0
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

        # Mutually exclusive: paper.md is the durable truth and wins over
        # a stale .no_markdown marker (which can be left behind when a
        # paper that initially had no markdown gets re-rendered later).
        # Counting both inflates the totals and pushes the residual
        # ``papers - have_md - no_md`` negative for "not yet fetched".
        md = pd / "paper.md"
        if md.exists() and md.stat().st_size > 0:
            have_md += 1
            md_bytes += md.stat().st_size
        elif (pd / ".no_markdown").exists():
            no_md += 1
        else:
            not_yet_fetched += 1

        for f in pd.rglob("*"):
            if f.is_file():
                total_bytes += f.stat().st_size

    lines = [
        f"Categories tracked: {', '.join(sorted(cats)) or '(none)'}",
        f"Papers known:       {papers:,}",
        f"Markdown on disk:   {have_md:,}  "
        f"({no_md:,} have none available, "
        f"{not_yet_fetched:,} not yet fetched)",
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
