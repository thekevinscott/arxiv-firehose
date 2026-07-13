"""dirsql schema + factory for the arxiv-firehose data dir.

fetcher uses dirsql **in-process** through this Python factory: the
per-table ``extract`` callbacks parse the JSON inside
``metadata.json`` / ``classifications/*.json`` and turn it into columns.
The equivalent ``.dirsql.toml`` ``on-file`` hooks would shell out once
per file; the in-process path is a plain function call, so we keep it.

Tables exposed (no paper *content* -- no abstract text, no ``paper.md``
body -- only metadata and presence markers):

    papers              -- one row per paper folder (from metadata.json)
    metadata            -- EAV rows: (id, paper_id, key, value), one per
                           metadata.json field except the abstract
    papers_categories   -- one row per (paper, category) outcome (from
                           data/<id>/classifications/<cat>.json)
    categories          -- one row per active category (from
                           ROOT/categories/<cat>.json, materialized by
                           classify.run from config.classify.prompts_dirs)
    markdown            -- one row per data/<id>/paper.md (paper_id + bytes)
    no_markdown         -- one row per data/<id>/.no_markdown marker

The taxonomy of category ids lives in **two** mirrored places: the
authored config (``[classify] prompts_dirs``) and the per-cat index
files under ``ROOT/categories/`` that classify.run rewrites at the start
of every run. Same pattern as ``papers`` -- materialize derived state as
files dirsql can scan.

``ROOT`` is the directory dirsql scans (defaults to the production
location on tower; override with ``ARXIV_FIREHOSE_ROOT`` for local
tests). The ``data_dir`` that the rest of fetcher passes around lives
**inside** ROOT at ``ROOT/data`` -- pass ``data_dir.parent`` here.

Persistence: ``build_app`` defaults to ``persist=True`` (SQLite cache at
``<ROOT>/.dirsql/cache.db``) so a warm process re-scans only changed
files. dirsql does **not** backfill a newly-added table from unchanged
files, so a schema change would silently under-populate a warm cache.
``build_app`` guards against that: it fingerprints the table set and
wipes ``<ROOT>/.dirsql`` whenever the fingerprint changes, forcing one
cold rescan after a deploy that alters the schema.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from dirsql import DirSQL, Table

DEFAULT_ROOT = "/mnt/bertha/data/arxiv-firehose"

# sqlite-vec supplies vec_distance_cosine() for /search. Loaded onto every
# dirsql connection (harmless for the metadata-only /sql path). The pip
# package's importable name is ``sqlite_vec``; its loadable's init symbol
# is ``sqlite3_vec_init`` (doesn't match the filename-derived default).
_EXTENSIONS = [{"path": "sqlite_vec", "entrypoint": "sqlite3_vec_init"}]


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


def _normalize_announced_at(raw, arxiv_id: str) -> str | None:
    """Parse metadata.json's ``announced_at`` into an ISO-8601 UTC string.

    The corpus stores announced_at in RFC-2822 shape
    (``"Tue, 14 Apr 2026 15:37:30 +0000"``, offsets vary by era). Parsing
    it to ISO *here*, once at index time, is the fix for the old footgun:
    ISO sorts lexically, so ``WHERE announced_at >= '2026-06-13'`` and
    SQLite ``datetime()`` both work without a query-time strptime. Empty
    or unparseable values fall back to the coarse month-from-id ISO."""
    if raw:
        try:
            dt = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            dt = None
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
    return _arxiv_id_to_iso(arxiv_id)


def _read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_paper(path: str) -> list[dict]:
    meta = _read_json(path)
    arxiv_id = Path(path).parent.name
    return [{
        "arxiv_id": arxiv_id,
        "announced_at": _normalize_announced_at(
            meta.get("announced_at"), arxiv_id
        ),
        "primary_category": meta.get("primary_category"),
    }]


def _extract_embeddings(path: str) -> list[dict]:
    """One row per paper in ``embeddings.json`` (a single JSON array).

    The embedding is stored as a JSON-array TEXT string -- sqlite-vec's
    ``vec_distance_cosine`` reads JSON vectors directly, no binary
    encoding needed. Rows missing an id or vector are dropped."""
    rows = _read_json(path)
    return [
        {"paper_id": r["arxiv_id"], "embedding": json.dumps(r["embedding"])}
        for r in rows
        if r.get("arxiv_id") and r.get("embedding") is not None
    ]


# metadata.json keys never mirrored into the EAV ``metadata`` table:
# ``arxiv_id`` is already the papers PK, and ``abstract`` is paper
# *content* we deliberately keep out of the DB.
_METADATA_SKIP_KEYS = frozenset({"arxiv_id", "abstract"})


def _extract_metadata_kv(path: str) -> list[dict]:
    """One EAV row per metadata.json field (except id/abstract).

    List/dict values are JSON-encoded so the ``value`` column stays TEXT;
    every scalar is stringified. ``id`` is omitted from each row so
    dirsql's SQLite AUTOINCREMENT assigns it."""
    meta = _read_json(path)
    paper_id = Path(path).parent.name
    rows = []
    for key, value in meta.items():
        if key in _METADATA_SKIP_KEYS:
            continue
        if isinstance(value, (list, dict)):
            value = json.dumps(value, sort_keys=True)
        elif value is not None:
            value = str(value)
        rows.append({"paper_id": paper_id, "key": key, "value": value})
    return rows


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


def _extract_markdown(path: str) -> list[dict]:
    """One row per paper.md: the paper id and the file size in bytes.

    Reads no content -- ``status`` only needs presence and byte totals, so
    the size comes straight from ``stat`` (empty files land as 0 and are
    filtered in SQL)."""
    return [{
        "paper_id": Path(path).parent.name,
        "size_bytes": os.path.getsize(path),
    }]


def _extract_marker(path: str) -> list[dict]:
    return [{"paper_id": Path(path).parent.name}]


# (ddl, glob, extract) for every table. The order is fingerprinted, so
# keep it stable; append new tables at the end.
_TABLE_SPECS = (
    (
        """CREATE TABLE papers (
            arxiv_id         TEXT PRIMARY KEY,
            announced_at     TEXT,
            primary_category TEXT
        )""",
        "data/*/metadata.json",
        _extract_paper,
    ),
    (
        """CREATE TABLE metadata (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT NOT NULL,
            key      TEXT NOT NULL,
            value    TEXT
        )""",
        "data/*/metadata.json",
        _extract_metadata_kv,
    ),
    # No composite PRIMARY KEY -- dirsql appends synthetic columns after
    # the DDL, and SQLite forbids columns after a table-level constraint.
    # Uniqueness comes from the one-file-per-(paper, category) layout.
    (
        """CREATE TABLE papers_categories (
            paper_id    TEXT NOT NULL,
            category_id TEXT NOT NULL,
            output      INTEGER NOT NULL
        )""",
        "data/*/classifications/*.json",
        _extract_paper_category,
    ),
    (
        """CREATE TABLE categories (
            category_id TEXT PRIMARY KEY,
            prompts_dir TEXT
        )""",
        "categories/*.json",
        _extract_category,
    ),
    (
        """CREATE TABLE markdown (
            paper_id   TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL
        )""",
        "data/*/paper.md",
        _extract_markdown,
    ),
    (
        "CREATE TABLE no_markdown (paper_id TEXT PRIMARY KEY)",
        "data/*/.no_markdown",
        _extract_marker,
    ),
    # Vector search surface: one row per paper, embedding as JSON-array
    # TEXT for sqlite-vec. Populated from the single embeddings.json at
    # the data-dir root (see commands/embed.py).
    (
        """CREATE TABLE embeddings (
            paper_id  TEXT PRIMARY KEY,
            embedding TEXT NOT NULL
        )""",
        "data/embeddings.json",
        _extract_embeddings,
    ),
)

# Bump on any change to an ``extract`` callback's output shape that the
# DDL alone doesn't capture (the fingerprint already covers ddl + glob).
# v2: papers.announced_at switched from month-from-id to parsed ISO, and
# the embeddings table was added.
_SCHEMA_VERSION = 2


def _fingerprint() -> str:
    """Stable hash of the table set + schema version.

    Changes whenever a DDL, glob, or the version constant changes -- the
    signal ``build_app`` uses to decide the warm persist cache is stale."""
    h = hashlib.sha256()
    h.update(str(_SCHEMA_VERSION).encode())
    for ddl, glob, _ in _TABLE_SPECS:
        h.update(b"\0")
        h.update(" ".join(ddl.split()).encode())
        h.update(b"\0")
        h.update(glob.encode())
    return h.hexdigest()


def _reconcile_persist_cache(root: Path) -> None:
    """Wipe ``<root>/.dirsql`` when the schema fingerprint has changed.

    dirsql only re-extracts files whose mtime moved, so a table added
    since the cache was built would stay empty for every pre-existing
    file. Clearing the cache on a fingerprint change forces one cold
    rescan, repopulating every table from scratch."""
    cache_dir = root / ".dirsql"
    stamp = cache_dir / "schema_version"
    want = _fingerprint()
    if stamp.exists() and stamp.read_text().strip() == want:
        return
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp.write_text(want)


def build_app(
    root: str | os.PathLike | None = None, *, persist: bool = True
) -> DirSQL:
    """Build a DirSQL app rooted at *root* (or ``ARXIV_FIREHOSE_ROOT``, or
    the production default). Must be called from inside an async context --
    DirSQL schedules its initial scan with ``asyncio.ensure_future`` at
    construction time.

    With *persist* true (the default) the SQLite cache lives at
    ``<root>/.dirsql/cache.db`` and is auto-invalidated on any schema
    change (see :func:`_reconcile_persist_cache`)."""
    root = Path(str(root or os.environ.get("ARXIV_FIREHOSE_ROOT", DEFAULT_ROOT)))
    if persist:
        _reconcile_persist_cache(root)
    return DirSQL(
        str(root),
        tables=[
            Table(ddl=ddl, glob=glob, extract=extract)
            for ddl, glob, extract in _TABLE_SPECS
        ],
        extensions=_EXTENSIONS,
        persist=persist,
    )


def query(
    sql: str, root: str | os.PathLike | None = None, *, persist: bool = True
) -> list[dict]:
    """Run *sql* against a fresh in-process dirsql app and return the rows.

    The synchronous bridge every sync caller (status, embed, the CLI, the
    HTTP endpoint) uses: it builds the app, waits for the initial scan,
    runs one read-only query, and tears the app down -- all inside a
    private event loop via ``asyncio.run``. dirsql's authorizer rejects
    writes, so *sql* is effectively read-only."""
    async def _run() -> list[dict]:
        db = build_app(root, persist=persist)
        await db.ready()
        return await db.query(sql)

    return asyncio.run(_run())


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
