"""Integration tests for the ``run`` (full pipeline) and ``status`` SDK functions.

The two external converters (arxiv2md, pypandoc) are never called -- a fake
``Converter`` is injected (see conftest), so the suite is hermetic.

Fixture papers:
  2401.00001 -- arxiv HTML available  -> markdown via the HTML path
  2401.00002 -- no HTML, PDF-only e-print -> .no_markdown
  2401.00003 -- no HTML, LaTeX e-print -> markdown via the LaTeX fallback
"""

from fetcher import run, status


def describe_run():
    def it_syncs_then_fetches_and_returns_a_summary(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        result = run(data_dir, cache_dir, transport=fake_transport,
                     converter=fake_converter)

        assert result["added"] == 3
        assert result["updated"] == 0
        assert result["fetch"]["html"] == 1
        assert result["fetch"]["latex"] == 1
        assert result["fetch"]["absent"] == 1
        assert "Papers known:       3" in result["status"]

    def it_leaves_a_markdown_mirror_on_disk(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        run(data_dir, cache_dir, transport=fake_transport,
            converter=fake_converter)

        assert (data_dir / "2401.00001" / "metadata.json").exists()
        assert (data_dir / "2401.00001" / "paper.md").exists()
        assert (data_dir / "2401.00003" / "paper.md").exists()
        assert (data_dir / "2401.00002" / ".no_markdown").exists()
        # No PDF and no extracted LaTeX source ever reach the data dir.
        assert list(data_dir.rglob("*.pdf")) == []
        assert [p for p in data_dir.rglob("source") if p.is_dir()] == []

    def it_makes_only_uncacheable_404s_on_a_same_day_rerun(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        run(data_dir, cache_dir, transport=fake_transport,
            converter=fake_converter)
        after_first = list(fake_transport.calls)

        result = run(data_dir, cache_dir, transport=fake_transport,
                     converter=fake_converter)

        # The feed (cached a day), the HTML and the e-print archives are all
        # served from disk. Only the two papers with no arxiv HTML repeat a
        # request -- a 404 is uncacheable -- and only their /html/ URL.
        new_calls = fake_transport.calls[len(after_first):]
        assert all(c.startswith("https://arxiv.org/html/") for c in new_calls)
        assert result["fetch"]["html"] == 1
        assert result["fetch"]["latex"] == 1


def describe_status():
    def it_reports_zero_papers_for_an_empty_data_dir(data_dir):
        report = status(data_dir)

        assert "Papers known:       0" in report
        assert "Last sync:          (never)" in report

    def it_reports_counts_after_a_run(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        run(data_dir, cache_dir, transport=fake_transport,
            converter=fake_converter)

        report = status(data_dir)

        assert "Papers known:       3" in report
        assert "Markdown on disk:   2" in report
