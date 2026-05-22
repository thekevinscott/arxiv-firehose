"""Integration tests for the ``run`` (full pipeline) and ``status`` SDK functions.

The two external converters (arxiv2md, pypandoc) are never called -- a fake
``Converter`` is injected (see conftest), so the suite is hermetic.

Fixture papers:
  2401.00001 -- arxiv HTML available  -> markdown via the HTML path
  2401.00002 -- no HTML, PDF-only e-print -> markdown via the PDF fallback
  2401.00003 -- no HTML, LaTeX e-print -> markdown via the LaTeX fallback
  2401.00004 -- no HTML, no e-print, no PDF -> .no_markdown
"""

import json

from fetcher import run, status


def describe_run():
    def it_syncs_then_fetches_and_returns_a_summary(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        result = run(data_dir, cache_dir, transport=fake_transport,
                     converter=fake_converter)

        assert result["added"] == 4
        assert result["updated"] == 0
        assert result["fetch"]["html"] == 1
        assert result["fetch"]["latex"] == 1
        assert result["fetch"]["pdf"] == 1
        assert result["fetch"]["absent"] == 1
        assert "Papers known:       4" in result["status"]

    def it_leaves_a_markdown_mirror_on_disk(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        run(data_dir, cache_dir, transport=fake_transport,
            converter=fake_converter)

        assert (data_dir / "2401.00001" / "metadata.json").exists()
        assert (data_dir / "2401.00001" / "paper.md").exists()
        assert (data_dir / "2401.00002" / "paper.md").exists()
        assert (data_dir / "2401.00003" / "paper.md").exists()
        assert (data_dir / "2401.00004" / ".no_markdown").exists()
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

        # The feed (cached a day), the HTML and the e-print/PDF archives that
        # returned 200 are all served from disk. Only the uncacheable 404s
        # repeat: the /html/ miss for the three HTML-less papers, plus the
        # e-print and PDF misses for 2401.00004 (no representation at all).
        new_calls = fake_transport.calls[len(after_first):]
        assert set(new_calls) == {
            "https://arxiv.org/html/2401.00002v1",
            "https://arxiv.org/html/2401.00003v1",
            "https://arxiv.org/html/2401.00004v1",
            "https://arxiv.org/e-print/2401.00004",
            "https://arxiv.org/pdf/2401.00004v1",
        }
        assert result["fetch"]["html"] == 1
        assert result["fetch"]["latex"] == 1
        assert result["fetch"]["pdf"] == 1


def describe_run_tracking():
    def it_appends_one_record_per_run_to_runs_jsonl(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        run(data_dir, cache_dir, transport=fake_transport, converter=fake_converter)
        run(data_dir, cache_dir, transport=fake_transport, converter=fake_converter)

        lines = (data_dir / "runs.jsonl").read_text().splitlines()
        # One JSON object appended per run -- a durable history to investigate.
        assert len(lines) == 2
        assert all(json.loads(line) for line in lines)

    def it_records_timing_and_counts_for_the_run(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        run(data_dir, cache_dir, transport=fake_transport, converter=fake_converter)

        rec = json.loads((data_dir / "runs.jsonl").read_text().splitlines()[0])
        assert rec["started_at"] <= rec["finished_at"]
        assert rec["duration_s"] >= 0
        assert rec["added"] == 4
        assert rec["updated"] == 0
        assert rec["fetch"] == {"html": 1, "latex": 1, "pdf": 1, "absent": 1,
                                "failed": 0, "skipped": 0}

    def it_does_not_record_a_dry_run(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        run(data_dir, cache_dir, transport=fake_transport,
            converter=fake_converter, dry_run=True)

        # A dry run touches nothing on disk -- no run history either.
        assert not (data_dir / "runs.jsonl").exists()


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

        assert "Papers known:       4" in report
        assert "Markdown on disk:   3" in report
