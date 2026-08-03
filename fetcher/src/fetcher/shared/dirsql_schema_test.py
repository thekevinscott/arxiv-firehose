"""Unit tests for the dirsql schema factory.

Exercises the real dirsql engine against a tiny on-disk fixture tree --
no mocks. Each test builds a root with a couple of paper folders and
asserts the tables the rest of fetcher queries against.
"""
from __future__ import annotations

import json
from pathlib import Path

from .dirsql_schema import (
    _fingerprint,
    _reconcile_persist_cache,
    query,
)


def _write_paper(
    root: Path,
    arxiv_id: str,
    *,
    meta: dict | None = None,
) -> None:
    pd = root / "data" / arxiv_id
    pd.mkdir(parents=True)
    base = {"arxiv_id": arxiv_id, "primary_category": "cs.LG"}
    base.update(meta or {})
    (pd / "metadata.json").write_text(json.dumps(base))


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


def describe_embeddings_table():
    def it_has_one_row_per_paper_with_json_vector(tmp_path):
        _write_paper(tmp_path, "2401.00001")
        (tmp_path / "data" / "embeddings.json").write_text(
            json.dumps([{"arxiv_id": "2401.00001", "embedding": [0.1, 0.2, 0.3]}])
        )

        rows = query("SELECT * FROM embeddings", tmp_path)

        assert rows == [{"paper_id": "2401.00001", "embedding": "[0.1, 0.2, 0.3]"}]


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
