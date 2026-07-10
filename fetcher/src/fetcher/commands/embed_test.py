"""Unit tests for the embed command.

The real model2vec is replaced with a deterministic fake -- these tests
assert the file-walking / diffing / parquet-merge logic, not the
correctness of the embedding model.

Layout mirrors what sync-metadata writes: one folder per arxiv_id
under the data dir, each carrying a metadata.json with an ``abstract``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from fetcher.commands import embed

DIM = embed.EMBED_DIM


class FakeModel:
    """Deterministic stand-in for model2vec's ``StaticModel``.

    Encodes each text as a vector whose first component is a rolling
    counter -- lets a test assert exact vector content and prove the
    right text reached ``encode``.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        # Every vector begins with a distinctive component and pads with
        # zeros -- enough to round-trip through DuckDB's FLOAT[256] type.
        arr = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, _ in enumerate(texts):
            arr[i, 0] = float(i + 1)
        return arr


def _write_paper(data_dir: Path, arxiv_id: str, abstract: str, **extra) -> None:
    pd = data_dir / arxiv_id
    pd.mkdir(parents=True, exist_ok=True)
    payload = {"arxiv_id": arxiv_id, "abstract": abstract, **extra}
    (pd / "metadata.json").write_text(json.dumps(payload))


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def log() -> logging.Logger:
    # Silent logger -- test assertions cover the behavior; log noise is
    # captured by pytest's caplog when a test opts in. Named *outside*
    # the ``fetcher.embed`` hierarchy on purpose: api.embed configures
    # ``fetcher.embed`` with ``propagate=False`` at runtime, so a child
    # logger under that name would drop records before caplog's root
    # handler ever saw them if any test triggered api.embed first.
    return logging.getLogger("embed_test")


def describe_embed_run():
    def it_embeds_every_paper_on_a_first_run(data_dir: Path, log):
        _write_paper(data_dir, "2401.00001", "abstract one")
        _write_paper(data_dir, "2401.00002", "abstract two")
        model = FakeModel()

        counts = embed.run(data_dir, log, model=model)

        assert counts == {"embedded": 2, "skipped": 0, "total": 2}
        assert embed.embeddings_path(data_dir).exists()
        # Fake model saw the abstracts in id-order (paths are sorted).
        assert model.calls == [["abstract one", "abstract two"]]

    def it_is_a_noop_when_every_paper_is_already_embedded(data_dir: Path, log):
        _write_paper(data_dir, "2401.00001", "abstract one")
        model = FakeModel()
        embed.run(data_dir, log, model=model)
        model.calls.clear()

        second = embed.run(data_dir, log, model=model)

        assert second == {"embedded": 0, "skipped": 1, "total": 1}
        assert model.calls == []  # encode was not called at all

    def it_embeds_only_papers_missing_from_the_parquet(data_dir: Path, log):
        _write_paper(data_dir, "2401.00001", "abstract one")
        model = FakeModel()
        embed.run(data_dir, log, model=model)
        # Now add a second paper; only it should be encoded on rerun.
        _write_paper(data_dir, "2401.00002", "abstract two")
        model.calls.clear()

        second = embed.run(data_dir, log, model=model)

        assert second == {"embedded": 1, "skipped": 1, "total": 2}
        assert model.calls == [["abstract two"]]

    def it_skips_a_paper_with_bad_metadata(data_dir: Path, log, caplog):
        _write_paper(data_dir, "2401.00001", "abstract one")
        # Corrupt metadata: not valid JSON.
        (data_dir / "2401.00002").mkdir()
        (data_dir / "2401.00002" / "metadata.json").write_text("{not json")

        with caplog.at_level(logging.WARNING, logger="embed_test"):
            counts = embed.run(data_dir, log, model=FakeModel())

        assert counts == {"embedded": 1, "skipped": 0, "total": 1}
        assert any("bad metadata.json" in r.message for r in caplog.records)

    def it_skips_a_paper_with_an_empty_abstract(data_dir: Path, log, caplog):
        _write_paper(data_dir, "2401.00001", "abstract one")
        _write_paper(data_dir, "2401.00002", "   ")  # blank

        with caplog.at_level(logging.WARNING, logger="embed_test"):
            counts = embed.run(data_dir, log, model=FakeModel())

        assert counts == {"embedded": 1, "skipped": 0, "total": 1}
        assert any("empty abstract" in r.message for r in caplog.records)

    def it_honors_limit(data_dir: Path, log):
        for i in range(1, 6):
            _write_paper(data_dir, f"2401.0000{i}", f"abstract {i}")
        model = FakeModel()

        counts = embed.run(data_dir, log, model=model, limit=2)

        assert counts == {"embedded": 2, "skipped": 0, "total": 2}
        assert model.calls == [["abstract 1", "abstract 2"]]

    def it_writes_nothing_on_a_dry_run(data_dir: Path, log):
        _write_paper(data_dir, "2401.00001", "abstract one")

        counts = embed.run(data_dir, log, model=FakeModel(), dry_run=True)

        assert counts == {"embedded": 0, "skipped": 0, "total": 0}
        assert not embed.embeddings_path(data_dir).exists()

    def it_returns_correct_totals_when_pending_is_empty_with_prior_rows(
        data_dir: Path, log
    ):
        _write_paper(data_dir, "2401.00001", "abstract one")
        embed.run(data_dir, log, model=FakeModel())

        # No new papers; rerun.
        counts = embed.run(data_dir, log, model=FakeModel())
        assert counts == {"embedded": 0, "skipped": 1, "total": 1}


def describe_embed_parquet():
    def it_produces_a_parquet_that_supports_array_cosine_distance(
        data_dir: Path, log
    ):
        # After a real write, DuckDB must be able to read the embedding
        # column back and use it in array_cosine_distance -- the fixed
        # size (FLOAT[256]) has to survive the parquet round-trip.
        import duckdb

        _write_paper(data_dir, "2401.00001", "a")
        _write_paper(data_dir, "2401.00002", "b")
        embed.run(data_dir, log, model=FakeModel())

        con = duckdb.connect()
        rows = con.execute(
            f"""
            SELECT arxiv_id,
                   array_cosine_distance(
                       embedding::FLOAT[{DIM}],
                       (SELECT embedding::FLOAT[{DIM}] FROM read_parquet(?)
                        WHERE arxiv_id = '2401.00001')
                   ) AS d
            FROM read_parquet(?) ORDER BY d
            """,
            [str(embed.embeddings_path(data_dir))] * 2,
        ).fetchall()
        con.close()

        # 00001 vs itself = distance 0.
        assert rows[0] == ("2401.00001", pytest.approx(0.0, abs=1e-6))
