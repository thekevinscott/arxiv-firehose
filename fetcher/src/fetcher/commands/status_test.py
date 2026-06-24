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

CONFIG_NO_CLASSIFY = """\
[categories]
include = ["cs.LG"]
"""


def _paper(
    data_dir: Path,
    arxiv_id: str,
    *,
    primary_category: str = "cs.LG",
    paper_md: str | None = None,
    no_markdown: bool = False,
) -> Path:
    """Materialize one paper folder; only the bits a test cares about."""
    pd = data_dir / arxiv_id
    pd.mkdir(parents=True, exist_ok=True)
    (pd / "metadata.json").write_text(json.dumps({
        "id": arxiv_id, "primary_category": primary_category,
    }))
    if paper_md is not None:
        (pd / "paper.md").write_text(paper_md)
    if no_markdown:
        (pd / ".no_markdown").touch()
    return pd


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    (d / "config.toml").write_text(CONFIG_NO_CLASSIFY)
    return d


def describe_render():
    def it_reports_paper_counts(data_dir: Path):
        _paper(data_dir, "2601.00001", paper_md="hi")
        _paper(data_dir, "2601.00002", paper_md="hi")
        _paper(data_dir, "2601.00003", no_markdown=True)
        _paper(data_dir, "2601.00004")  # neither: not yet fetched
        out = render(data_dir)
        assert "Papers known:       4" in out
        assert "Markdown on disk:   2" in out
        assert "1 have none available" in out
        assert "1 not yet fetched" in out

    def it_does_not_double_count_when_paper_md_and_marker_coexist(
        data_dir: Path,
    ):
        """Regression: a paper with BOTH paper.md AND .no_markdown was
        incrementing two counters, pushing "not yet fetched" negative
        because the math assumed disjoint categories.
        """
        # All three papers have markdown; one also carries a stale marker
        # from a prior run when no markdown was available. The marker
        # must not contribute to the "have none available" tally.
        _paper(data_dir, "2601.00001", paper_md="hi")
        _paper(data_dir, "2601.00002", paper_md="hi", no_markdown=True)
        _paper(data_dir, "2601.00003", paper_md="hi")
        out = render(data_dir)
        assert "Papers known:       3" in out
        assert "Markdown on disk:   3" in out
        assert "0 have none available" in out
        # The smoking gun: must not be negative.
        assert "-1 not yet fetched" not in out
        assert "0 not yet fetched" in out

    def it_counts_empty_data_dir_as_zeros(data_dir: Path):
        out = render(data_dir)
        assert "Papers known:       0" in out
        assert "0 not yet fetched" in out
        assert "Last sync:          (never)" in out
