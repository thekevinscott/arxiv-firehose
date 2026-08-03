"""Unit tests for the status report.

A fixture builds a tiny data dir on disk (real files, no mocks); each
test asserts a specific line of the rendered report. Keeps these
end-to-end against the filesystem because that *is* the contract --
status reads what's on disk and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fetcher.commands.status import render


def _paper(
    data_dir: Path,
    arxiv_id: str,
    *,
    primary_category: str = "cs.LG",
) -> Path:
    """Materialize one paper folder; only the bits a test cares about."""
    pd = data_dir / arxiv_id
    pd.mkdir(parents=True, exist_ok=True)
    (pd / "metadata.json").write_text(json.dumps({
        "id": arxiv_id, "primary_category": primary_category,
    }))
    return pd


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


def describe_render():
    def it_reports_paper_counts(data_dir: Path):
        _paper(data_dir, "2601.00001")
        _paper(data_dir, "2601.00002")
        _paper(data_dir, "2601.00003", primary_category="cs.AI")
        out = render(data_dir)
        assert "Papers known:       3" in out
        assert "Categories tracked: cs.AI, cs.LG" in out

    def it_counts_empty_data_dir_as_zeros(data_dir: Path):
        out = render(data_dir)
        assert "Papers known:       0" in out
        assert "Categories tracked: (none)" in out
        assert "Last sync:          (never)" in out
