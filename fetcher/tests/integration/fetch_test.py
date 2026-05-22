"""Integration tests for the ``fetch`` SDK function.

``fetch`` always processes every paper: cachetta serves the bytes from disk
or the network transparently, so from the SDK's perspective every paper is a
fresh fetch-and-rewrite. The two external converters (arxiv2md, pypandoc) are
never called -- a fake ``Converter`` is injected (see conftest), so the suite
is hermetic.

Fixture papers:
  2401.00001 -- arxiv HTML available  -> markdown via the HTML path
  2401.00002 -- no HTML, PDF-only e-print -> .no_markdown
  2401.00003 -- no HTML, LaTeX e-print -> markdown via the LaTeX fallback
"""

import json

from fetcher import fetch, sync_metadata


def _read_fetch_log(data_dir):
    return (data_dir / "logs" / "fetch.log").read_text()


def describe_fetch():
    def it_writes_markdown_from_arxiv_html(data_dir, cache_dir, fake_transport,
                                           fake_converter):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        counts = fetch(data_dir, cache_dir, transport=fake_transport,
                       converter=fake_converter)

        assert counts["html"] == 1
        md = (data_dir / "2401.00001" / "paper.md").read_text()
        assert md.startswith("# Markdown from HTML")

    def it_falls_back_to_latex_when_a_paper_has_no_html(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        counts = fetch(data_dir, cache_dir, transport=fake_transport,
                       converter=fake_converter)

        assert counts["latex"] == 1
        md = (data_dir / "2401.00003" / "paper.md").read_text()
        assert md.startswith("# Markdown from LaTeX")

    def it_marks_a_paper_with_no_markdown_available(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        counts = fetch(data_dir, cache_dir, transport=fake_transport,
                       converter=fake_converter)

        assert counts["absent"] == 1
        assert (data_dir / "2401.00002" / ".no_markdown").exists()
        assert not (data_dir / "2401.00002" / "paper.md").exists()

    def it_keeps_only_markdown_and_metadata_on_disk(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        fetch(data_dir, cache_dir, transport=fake_transport,
              converter=fake_converter)

        for pid in ("2401.00001", "2401.00003"):
            on_disk = {p.name for p in (data_dir / pid).iterdir()}
            assert on_disk == {"metadata.json", "paper.md"}
        # The PDF and the raw LaTeX source are never written to the data dir.
        assert list(data_dir.rglob("*.pdf")) == []
        assert [p for p in data_dir.rglob("source") if p.is_dir()] == []


def describe_fetch_always_rewrites():
    def it_overwrites_a_corrupted_markdown_file_already_on_disk(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        fetch(data_dir, cache_dir, transport=fake_transport,
              converter=fake_converter)

        md = data_dir / "2401.00001" / "paper.md"
        md.write_text("corrupted")  # fetch must not skip a paper that "exists"

        fetch(data_dir, cache_dir, transport=fake_transport,
              converter=fake_converter)

        assert md.read_text().startswith("# Markdown from HTML")

    def it_reuses_cached_content_on_a_rerun(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        first = fetch(data_dir, cache_dir, transport=fake_transport,
                      converter=fake_converter)
        after_first = list(fake_transport.calls)

        again = fetch(data_dir, cache_dir, transport=fake_transport,
                      converter=fake_converter)

        # The feed, the converted HTML and the e-print archives are all
        # cached. Only the two papers with no arxiv HTML repeat a request --
        # a 404 is uncacheable -- and only their /html/ URL.
        new_calls = fake_transport.calls[len(after_first):]
        assert all(c.startswith("https://arxiv.org/html/") for c in new_calls)
        assert again == first

    def it_makes_no_request_on_a_dry_run(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        before = list(fake_transport.calls)

        counts = fetch(data_dir, cache_dir, dry_run=True, transport=fake_transport,
                       converter=fake_converter)

        assert counts["html"] == 0 and counts["latex"] == 0
        assert fake_transport.calls == before
        assert not (data_dir / "2401.00001" / "paper.md").exists()


def describe_fetch_resilience():
    def it_skips_a_paper_with_corrupt_metadata_and_processes_the_rest(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        # A folder with an unreadable metadata.json must not abort the run.
        bad = data_dir / "2401.09998"
        bad.mkdir()
        (bad / "metadata.json").write_text("{ not valid json")

        counts = fetch(data_dir, cache_dir, transport=fake_transport,
                       converter=fake_converter)

        assert counts["html"] == 1  # the valid HTML paper still converted
        assert counts["latex"] == 1
        assert counts["skipped"] == 1
        assert "skip 2401.09998: bad metadata.json" in _read_fetch_log(data_dir)


def describe_fetch_logging():
    def it_logs_a_200_for_a_converted_paper(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        fetch(data_dir, cache_dir, transport=fake_transport,
              converter=fake_converter)

        log_text = _read_fetch_log(data_dir)
        assert "html 2401.00001: HTTP 200" in log_text
        assert "tex  2401.00003: HTTP 200" in log_text

    def it_logs_a_404_concisely_without_the_mdn_link(
        data_dir, cache_dir, fake_transport, fake_converter
    ):
        sync_metadata(data_dir, cache_dir, transport=fake_transport)
        fetch(data_dir, cache_dir, verbose=True, transport=fake_transport,
              converter=fake_converter)

        # 2401.00002 / 2401.00003 have no arxiv HTML -> a 404 is logged, but
        # without httpx's multi-line MDN documentation link.
        log_text = _read_fetch_log(data_dir)
        assert "html 2401.00002: HTTP 404" in log_text
        assert "developer.mozilla" not in log_text
