"""dirsql schema + factory for the arxiv-firehose data dir.

Until dirsql PR #220 (native-language config files) lands, the dirsql
Rust CLI only accepts ``.dirsql.toml`` -- which can't parse the JSON
inside ``metadata.json``/``classifications/*.json`` to expose fields as
columns. So fetcher uses dirsql **in-process** through this Python
factory.

Tables exposed:

    papers              -- one row per paper folder (from metadata.json)
    papers_categories   -- one row per (paper, category) outcome (from
                           data/<id>/classifications/<cat>.json)
    categories          -- one row per active category (from
                           ROOT/categories/<cat>.json, materialized by
                           classify.run from config.classify.prompts_dirs)

The taxonomy of category ids lives in **two** mirrored places: the
authored config (``[classify] prompts_dirs``) and the per-cat index
files under ``ROOT/categories/`` that classify.run rewrites at the
start of every run. Authoring stays in config; the index files exist
solely so dirsql has a table to ``SELECT category_id FROM`` and
``CROSS JOIN`` against ``papers``. Same pattern as ``papers`` (rows
from per-paper ``metadata.json`` files written by sync) -- materialize
derived state as files dirsql can scan.

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


def _extract_category(path: str) -> list[dict]:
    """One categories row per file under ``categories/``. The cat id is
    the file's stem; ``prompts_dir`` is whatever classify.run wrote there
    (kept around as a pointer for debugging -- not joined against)."""
    obj = _read_json(path)
    return [{
        "category_id": Path(path).stem,
        "prompts_dir": str(obj.get("prompts_dir", "")),
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
            Table(
                ddl="""CREATE TABLE categories (
                    category_id TEXT PRIMARY KEY,
                    prompts_dir TEXT
                )""",
                glob="categories/*.json",
                extract=_extract_category,
            ),
        ],
    )


# Every (paper, category) pair that has no classification file yet. The
# CROSS JOIN materializes the full grid; the LEFT JOIN + WHERE-NULL keeps
# only the missing cells. Result: one row per LLM call this run needs to
# make. Re-runs return only the *new* missing pairs; cachetta makes any
# accidental repeat call free.
MISSING_PAIRS_SQL = """
SELECT p.arxiv_id AS paper_id, c.category_id AS category_id
FROM papers p
CROSS JOIN categories c
LEFT JOIN papers_categories pc
    ON pc.paper_id = p.arxiv_id AND pc.category_id = c.category_id
WHERE pc.paper_id IS NULL
ORDER BY p.arxiv_id, c.category_id
""".strip()
