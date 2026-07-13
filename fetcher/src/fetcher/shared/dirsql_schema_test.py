"""Unit tests for the dirsql schema factory.

Exercises the real dirsql engine against a tiny on-disk fixture tree --
no mocks. Each test builds a root with a couple of paper folders and
asserts the tables the rest of fetcher queries against.
"""
from __future__ import annotations

import json
from pathlib import Path

from .dirsql_schema import (
    MISSING_PAIRS_SQL,
    _fingerprint,
    _reconcile_persist_cache,
    query,
)


def _write_paper(
    root: Path,
    arxiv_id: str,
    *,
    meta: dict | None = None,
    markdown: str | None = None,
    no_markdown: bool = False,
    classifications: dict[str, bool] | None = None,
) -> None:
    pd = root / "data" / arxiv_id
    pd.mkdir(parents=True)
    base = {"arxiv_id": arxiv_id, "primary_category": "cs.LG"}
    base.update(meta or {})
    (pd / "metadata.json").write_text(json.dumps(base))
    if markdown is not None:
        (pd / "paper.md").write_text(markdown)
    if no_markdown:
        (pd / ".no_markdown").write_text("")
    for cat, output in (classifications or {}).items():
        cdir = pd / "classifications"
        cdir.mkdir(exist_ok=True)
        (cdir / f"{cat}.json").write_text(json.dumps({"output": output}))


def _write_category(root: Path, cat_id: str) -> None:
    cats = root / "categories"
    cats.mkdir(parents=True, exist_ok=True)
    (cats / f"{cat_id}.json").write_text(json.dumps({"prompts_dir": "/x"}))


def describe_papers_table():
    def it_has_one_row_per_paper_with_derived_announced_at(tmp_path):
        _write_paper(tmp_path, "2401.00001")
        _write_paper(tmp_path, "2402.09999", meta={"primary_category": "cs.AI"})

        rows = query("SELECT * FROM papers ORDER BY arxiv_id", tmp_path)

        assert [r["arxiv_id"] for r in rows] == ["2401.00001", "2402.09999"]
        assert rows[0]["announced_at"].startswith("2024-01-01")
        assert rows[1]["primary_category"] == "cs.AI"


def describe_metadata_eav():
    def it_emits_a_row_per_field_but_never_the_abstract(tmp_path):
        _write_paper(
            tmp_path,
            "2401.00001",
            meta={"title": "T", "abstract": "long body", "categories": ["cs.LG"]},
        )

        rows = query(
            "SELECT key, value FROM metadata ORDER BY key", tmp_path
        )
        keys = [r["key"] for r in rows]

        assert "abstract" not in keys
        assert "arxiv_id" not in keys
        assert "title" in keys
        # list values are JSON-encoded to keep value TEXT
        cats = next(r["value"] for r in rows if r["key"] == "categories")
        assert json.loads(cats) == ["cs.LG"]

    def it_autoincrements_the_id_column(tmp_path):
        _write_paper(tmp_path, "2401.00001", meta={"title": "T", "doi": "d"})

        ids = [r["id"] for r in query("SELECT id FROM metadata", tmp_path)]

        assert len(ids) == len(set(ids))  # unique
        assert all(isinstance(i, int) for i in ids)


def describe_presence_tables():
    def it_records_markdown_size_and_no_markdown_marker(tmp_path):
        _write_paper(tmp_path, "2401.00001", markdown="# body\n")
        _write_paper(tmp_path, "2401.00002", no_markdown=True)

        md = query("SELECT * FROM markdown", tmp_path)
        nomd = query("SELECT paper_id FROM no_markdown", tmp_path)

        assert md == [{"paper_id": "2401.00001", "size_bytes": 7}]
        assert nomd == [{"paper_id": "2401.00002"}]


def describe_missing_pairs_sql():
    def it_returns_only_uncovered_paper_category_cells(tmp_path):
        _write_category(tmp_path, "is_ml")
        _write_category(tmp_path, "is_md")
        _write_paper(tmp_path, "2401.00001", classifications={"is_ml": True})
        _write_paper(tmp_path, "2401.00002")

        pairs = query(MISSING_PAIRS_SQL, tmp_path)
        got = {(r["paper_id"], r["category_id"]) for r in pairs}

        assert got == {
            ("2401.00001", "is_md"),
            ("2401.00002", "is_ml"),
            ("2401.00002", "is_md"),
        }


def describe_persist_cache():
    def it_writes_a_schema_stamp_and_reuses_the_cache(tmp_path):
        _write_paper(tmp_path, "2401.00001")

        query("SELECT 1 AS x FROM papers", tmp_path)

        stamp = tmp_path / ".dirsql" / "schema_version"
        assert stamp.exists()
        assert stamp.read_text().strip() == _fingerprint()
        assert (tmp_path / ".dirsql" / "cache.db").exists()

    def it_wipes_a_stale_cache_on_fingerprint_mismatch(tmp_path):
        _write_paper(tmp_path, "2401.00001")
        query("SELECT 1 AS x FROM papers", tmp_path)
        stamp = tmp_path / ".dirsql" / "schema_version"
        stamp.write_text("stale-fingerprint")

        # A new build must reconcile: stamp rewritten to the real value.
        _reconcile_persist_cache(tmp_path)

        assert stamp.read_text().strip() == _fingerprint()
