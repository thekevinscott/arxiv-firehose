"""Integration tests for the ``render_markdown`` SDK function.

``render_markdown`` always processes every paper: cachetta serves the bytes
from disk or the network transparently, so from the SDK's perspective every
paper is a fresh fetch-and-rewrite. The two external converters (arxiv2md,
pypandoc) are never called -- a fake ``Converter`` is injected (see conftest),
so the suite is hermetic.

This is the markdown-render stage of ``fetch`` (the composite command),
called on its own through the SDK -- it is not exposed on the CLI.

Fixture papers:
  2401.00001 -- arxiv HTML available  -> markdown via the HTML path
  2401.00002 -- no HTML, PDF-only e-print -> markdown via the PDF fallback
  2401.00003 -- no HTML, LaTeX e-print -> markdown via the LaTeX fallback
  2401.00004 -- no HTML, no e-print, no PDF -> .no_markdown
"""

import dataclasses

import pytest

from fetcher import render_markdown, sync_metadata


def _read_render_log(data_dir):
    return (data_dir / "logs" / "render.log").read_text()


def describe_render_markdown():
    def it_writes_markdown_from_arxiv_html(data_dir, cache_dir, fake_transport,
                                           fake_converter):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        counts = render_markdown(data_dir, cache_dir, transport=fake_transport,
                                 converter=fake_converter)

        assert counts["html"] == 1
        md = (data_dir / "2401.00001" / "paper.md").read_text()
        assert md.startswith("# Markdown from HTML")

    def it_falls_back_to_latex_when_a_paper_has_no_html(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        counts = render_markdown(data_dir, cache_dir, transport=fake_transport,
                                 converter=fake_converter)

        assert counts["latex"] == 1
        md = (data_dir / "2401.00003" / "paper.md").read_text()
        assert md.startswith("# Markdown from LaTeX")

    def it_falls_back_to_pdf_when_a_paper_has_no_html_or_latex(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        counts = render_markdown(data_dir, cache_dir, transport=fake_transport,
                                 converter=fake_converter)

        # 2401.00002 has no arxiv HTML and a PDF-only e-print -- the LaTeX
        # path raises, so the PDF fallback is the path that yields markdown.
        assert counts["pdf"] == 1
        md = (data_dir / "2401.00002" / "paper.md").read_text()
        assert md.startswith("# Markdown from PDF")

    def it_marks_a_paper_with_no_markdown_available(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        counts = render_markdown(data_dir, cache_dir, transport=fake_transport,
                                 converter=fake_converter)

        # 2401.00004 has no HTML, no e-print and no PDF -- every conversion
        # path 404s, so it is the one paper left with no markdown.
        assert counts["absent"] == 1
        assert (data_dir / "2401.00004" / ".no_markdown").exists()
        assert not (data_dir / "2401.00004" / "paper.md").exists()

    def it_keeps_only_markdown_and_metadata_on_disk(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        render_markdown(data_dir, cache_dir, transport=fake_transport,
                        converter=fake_converter)

        for pid in ("2401.00001", "2401.00002", "2401.00003"):
            on_disk = {p.name for p in (data_dir / pid).iterdir()}
            assert on_disk == {"metadata.json", "paper.md"}
        # The PDF and the raw LaTeX source are never written to the data dir.
        assert list(data_dir.rglob("*.pdf")) == []
        assert [p for p in data_dir.rglob("source") if p.is_dir()] == []


def describe_render_markdown_always_rewrites():
    def it_overwrites_a_corrupted_markdown_file_already_on_disk(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        render_markdown(data_dir, cache_dir, transport=fake_transport,
                        converter=fake_converter)

        md = data_dir / "2401.00001" / "paper.md"
        md.write_text("corrupted")  # must not skip a paper that "exists"

        render_markdown(data_dir, cache_dir, transport=fake_transport,
                        converter=fake_converter)

        assert md.read_text().startswith("# Markdown from HTML")

    def it_reuses_cached_content_on_a_rerun(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        first = render_markdown(data_dir, cache_dir, transport=fake_transport,
                                converter=fake_converter)
        after_first = list(fake_transport.calls)

        again = render_markdown(data_dir, cache_dir, transport=fake_transport,
                                converter=fake_converter)

        # The feed, the converted HTML and the e-print/PDF archives that
        # returned 200 are all cached. Only the uncacheable 404s repeat: the
        # /html/ miss for the three HTML-less papers, plus 2401.00004's
        # e-print and PDF misses (it has no representation at all).
        new_calls = fake_transport.calls[len(after_first):]
        assert set(new_calls) == {
            "https://arxiv.org/html/2401.00002v1",
            "https://arxiv.org/html/2401.00003v1",
            "https://arxiv.org/html/2401.00004v1",
            "https://arxiv.org/e-print/2401.00004",
            "https://arxiv.org/pdf/2401.00004v1",
        }
        assert again == first

    def it_makes_no_request_on_a_dry_run(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        before = list(fake_transport.calls)

        counts = render_markdown(data_dir, cache_dir, dry_run=True,
                                 transport=fake_transport,
                                 converter=fake_converter)

        assert counts["html"] == 0 and counts["latex"] == 0
        assert counts["pdf"] == 0
        assert fake_transport.calls == before
        assert not (data_dir / "2401.00001" / "paper.md").exists()


def describe_render_markdown_resilience():
    def it_skips_a_paper_with_corrupt_metadata_and_processes_the_rest(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        # A folder with an unreadable metadata.json must not abort the run.
        bad = data_dir / "2401.09998"
        bad.mkdir()
        (bad / "metadata.json").write_text("{ not valid json")

        counts = render_markdown(data_dir, cache_dir, transport=fake_transport,
                                 converter=fake_converter)

        assert counts["html"] == 1  # the valid HTML paper still converted
        assert counts["latex"] == 1
        assert counts["skipped"] == 1
        assert "skip 2401.09998: bad metadata.json" in _read_render_log(data_dir)


def describe_latex_failures_fall_through_to_pdf():
    # Regression coverage for the 3-tier chain. The LaTeX path collapses
    # every unconvertible-source outcome into one ValueError: a PDF-only
    # e-print, a pandoc timeout, or a pandoc parse rejection (exit 64).
    # Whatever the cause, render must route the paper to the PDF fallback --
    # never abort it or mark it absent while a PDF is still reachable.
    @pytest.mark.parametrize("reason", [
        "e-print archive has no LaTeX source",
        "LaTeX conversion exceeded the 120s pandoc timeout",
        "pandoc rejected the LaTeX source (exit 64)",
    ])
    def it_routes_a_latex_valueerror_to_the_pdf_path(
        data_dir, cache_dir, fake_transport, fake_converter, reason
    ):
        def failing_latex(eprint: bytes) -> str:
            raise ValueError(reason)

        converter = dataclasses.replace(fake_converter, latex=failing_latex)
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        counts = render_markdown(data_dir, cache_dir, transport=fake_transport,
                                 converter=converter)

        # 2401.00003 has a LaTeX e-print -- normally the LaTeX path. With the
        # LaTeX converter rejecting it, the paper falls through to its PDF,
        # joining 2401.00002 (already PDF-only) on the PDF path.
        assert counts["latex"] == 0
        assert counts["pdf"] == 2
        md = (data_dir / "2401.00003" / "paper.md").read_text()
        assert md.startswith("# Markdown from PDF")


def describe_render_markdown_logging():
    def it_logs_a_200_for_a_converted_paper(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        render_markdown(data_dir, cache_dir, transport=fake_transport,
                        converter=fake_converter)

        log_text = _read_render_log(data_dir)
        assert "html 2401.00001: HTTP 200" in log_text
        assert "tex  2401.00003: HTTP 200" in log_text

    def it_logs_a_404_concisely_without_the_mdn_link(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        render_markdown(data_dir, cache_dir, verbose=True,
                        transport=fake_transport, converter=fake_converter)

        # 2401.00002 / 2401.00003 have no arxiv HTML -> a 404 is logged, but
        # without httpx's multi-line MDN documentation link.
        log_text = _read_render_log(data_dir)
        assert "html 2401.00002: HTTP 404" in log_text
        assert "developer.mozilla" not in log_text
