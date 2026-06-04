"""dirsql schema + factory for the arxiv-firehose data dir.

Until dirsql PR #220 (native-language config files) lands, the dirsql
Rust CLI only accepts ``.dirsql.toml`` -- which can't parse the JSON
inside ``metadata.json``/``classifications/*.json`` to expose fields as
columns. So fetcher uses dirsql **in-process** through this Python
factory.

Tables exposed:

    papers              -- one row per paper folder (from metadata.json)
    categories          -- one row per known classifier (from categories.json)
    papers_categories   -- one row per (paper, category) outcome (from
                           data/<id>/classifications/<cat>.json)

``ROOT`` is the directory dirsql scans (defaults to the production
location on tower; override with ``ARXIV_FIREHOSE_ROOT`` for local
tests). The ``data_dir`` that the rest of fetcher passes around lives
**inside** ROOT at ``ROOT/data`` -- pass ``data_dir.parent`` here.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dirsql import DirSQL, Table

DEFAULT_ROOT = "/mnt/bertha/data/arxiv-firehose"


def _arxiv_id_to_iso(arxiv_id: str) -> str | None:
    """Parse a post-2007 arxiv id (``YYMM.NNNNN``) into a month-first ISO
    timestamp. Returns None for legacy ids (e.g. ``hep-th/9901001``) which
    use a different scheme this firehose doesn't currently ingest."""
    head = arxiv_id.split(".", 1)[0]
    if len(head) != 4 or not head.isdigit():
        return None
    yy, mm = int(head[:2]), int(head[2:])
    year = 2000 + yy if yy < 90 else 1900 + yy
    return datetime(year, mm, 1, tzinfo=timezone.utc).isoformat()


def _read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_paper(path: str) -> list[dict]:
    meta = _read_json(path)
    arxiv_id = Path(path).parent.name
    return [{
        "arxiv_id": arxiv_id,
        "abstract": meta.get("abstract", ""),
        "announced_at": _arxiv_id_to_iso(arxiv_id),
    }]


def _extract_categories(path: str) -> list[dict]:
    return _read_json(path)


def _extract_paper_category(path: str) -> list[dict]:
    obj = _read_json(path)
    parts = Path(path).parts
    # .../data/<arxiv_id>/classifications/<cat>.json
    arxiv_id = parts[-3]
    cat_id = Path(parts[-1]).stem
    return [{
        "paper_id": arxiv_id,
        "category_id": cat_id,
        "output": bool(obj.get("output", False)),
    }]


def build_app(root: str | os.PathLike | None = None) -> DirSQL:
    """Build a DirSQL app rooted at *root* (or ``ARXIV_FIREHOSE_ROOT``,
    or the production default). Must be called from inside an async
    context -- DirSQL schedules its initial scan with
    ``asyncio.ensure_future`` at construction time."""
    root = str(root or os.environ.get("ARXIV_FIREHOSE_ROOT", DEFAULT_ROOT))
    return DirSQL(
        root,
        tables=[
            Table(
                ddl="""CREATE TABLE papers (
                    arxiv_id     TEXT PRIMARY KEY,
                    abstract     TEXT,
                    announced_at TEXT
                )""",
                glob="data/*/metadata.json",
                extract=_extract_paper,
            ),
            Table(
                ddl="""CREATE TABLE categories (
                    id   TEXT PRIMARY KEY,
                    name TEXT
                )""",
                glob="categories.json",
                extract=_extract_categories,
            ),
            # No composite PRIMARY KEY -- dirsql appends synthetic
            # columns after the DDL, and SQLite forbids columns after a
            # table-level constraint. Uniqueness comes from the
            # one-file-per-(paper, category) layout on disk.
            Table(
                ddl="""CREATE TABLE papers_categories (
                    paper_id    TEXT NOT NULL,
                    category_id TEXT NOT NULL,
                    output      INTEGER NOT NULL
                )""",
                glob="data/*/classifications/*.json",
                extract=_extract_paper_category,
            ),
        ],
    )


# SQL: every (paper, category) pair that has no classification file yet.
# Used by classify.run as the work queue -- "only classify what's
# missing." Drives idempotency for free (an existing file means a row in
# papers_categories means the LEFT JOIN drops the pair).
MISSING_PAIRS_SQL = """
    SELECT p.arxiv_id AS paper_id, c.id AS category_id
    FROM papers p
    CROSS JOIN categories c
    LEFT JOIN papers_categories pc
      ON pc.paper_id = p.arxiv_id AND pc.category_id = c.id
    WHERE pc.paper_id IS NULL
    ORDER BY p.arxiv_id, c.id
"""

# Same shape, every pair (for --force reruns).
ALL_PAIRS_SQL = """
    SELECT p.arxiv_id AS paper_id, c.id AS category_id
    FROM papers p CROSS JOIN categories c
    ORDER BY p.arxiv_id, c.id
"""
