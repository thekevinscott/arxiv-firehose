"""status: a one-screen summary, computed by scanning the filesystem."""

from __future__ import annotations

import json
from pathlib import Path

from .paths import iter_paper_dirs


def _read_expected_categories(root: Path) -> set[str]:
    """Set of category ids from ``categories.json`` at the dirsql ROOT
    (= ``data_dir.parent``). Empty set if the file is missing or unreadable
    -- callers treat that as "taxonomy not configured yet"."""
    cats_path = root / "categories.json"
    if not cats_path.exists():
        return set()
    try:
        return {c["id"] for c in json.loads(cats_path.read_text())}
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return set()


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def render(data_dir: Path) -> str:
    """Build the status report by walking the paper folders."""
    # "Fully classified" = the paper has a classifications/<cat>.json for
    # every category in the taxonomy. If categories.json is missing we
    # fall back to "has at least one classification" (best the data
    # allows) -- the cron is healthy when classification has begun.
    expected_cats = _read_expected_categories(data_dir.parent)
    papers = 0
    have_md = 0
    no_md = 0
    fully_classified = 0
    partially_classified = 0
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
        labels = {f.stem for f in (pd / "classifications").glob("*.json")} \
            if (pd / "classifications").is_dir() else set()
        if expected_cats and labels >= expected_cats:
            fully_classified += 1
        elif labels:
            partially_classified += 1
        elif not expected_cats and (pd / "classification.json").exists():
            fully_classified += 1  # legacy single-file layout

        for f in pd.rglob("*"):
            if f.is_file():
                total_bytes += f.stat().st_size

    lines = [
        f"Categories tracked: {', '.join(sorted(cats)) or '(none)'}",
        f"Papers known:       {papers:,}",
        f"Markdown on disk:   {have_md:,}  "
        f"({no_md:,} have none available, "
        f"{papers - have_md - no_md:,} not yet fetched)",
        f"Classified:         {fully_classified:,}  "
        f"({partially_classified:,} partial, "
        f"{papers - fully_classified - partially_classified:,} not yet classified)",
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
