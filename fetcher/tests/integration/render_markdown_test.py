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
  2607.02140 -- pathological PDF (heatmap images) -> memory bomb regression
"""

import dataclasses
import json
import multiprocessing as mp
import resource
from pathlib import Path

import pytest

from fetcher import render_markdown, sync_metadata

# The pathological paper from the 2026-07-03/04 tower OOM: an arxiv HTML 404,
# a LaTeX e-print that pandoc rejects, and a PDF with embedded heatmaps that
# pymupdf4llm rasterizes into ~70 MiB numpy arrays per image. The RED test
# below runs render on this paper with the REAL converter under a memory cap
# and asserts the process peak stays bounded.
PATHOLOGICAL_PID = "2607.02140"
# 4 GB is the smallest RLIMIT_AS that lets both pandoc's Haskell RTS arena
# and enough of pymupdf's virtual reservation start; below that pymupdf gets
# stuck in an internal alloc-retry loop rather than raising cleanly.
_PATHOLOGICAL_MEMCAP_MB = 4096
_PATHOLOGICAL_PEAK_BUDGET_MB = 500
# Wall covers: the LaTeX path invoking pandoc under RLIMIT_AS (fast fail on
# this paper, but with headroom for cold-cache pandoc startup under a full
# pytest suite's memory pressure) *and* the PDF grandchild's own 120 s cap.
_PATHOLOGICAL_WALL_S = 240


def _read_render_log(data_dir):
    return (data_dir / "logs" / "render.log").read_text()


def describe_render_markdown():
    def it_writes_markdown_from_arxiv_html(data_dir, arxiv,
                                           fake_converter):
        sync_metadata(data_dir)
        counts = render_markdown(data_dir,
                                 converter=fake_converter)

        assert counts["html"] == 1
        md = (data_dir / "2401.00001" / "paper.md").read_text()
        assert md.startswith("# Markdown from HTML")

    def it_falls_back_to_latex_when_a_paper_has_no_html(
        data_dir, arxiv, fake_converter
    ):
        sync_metadata(data_dir)
        counts = render_markdown(data_dir,
                                 converter=fake_converter)

        assert counts["latex"] == 1
        md = (data_dir / "2401.00003" / "paper.md").read_text()
        assert md.startswith("# Markdown from LaTeX")

    def it_falls_back_to_pdf_when_a_paper_has_no_html_or_latex(
        data_dir, arxiv, fake_converter
    ):
        sync_metadata(data_dir)
        counts = render_markdown(data_dir,
                                 converter=fake_converter)

        # 2401.00002 has no arxiv HTML and a PDF-only e-print -- the LaTeX
        # path raises, so the PDF fallback is the path that yields markdown.
        assert counts["pdf"] == 1
        md = (data_dir / "2401.00002" / "paper.md").read_text()
        assert md.startswith("# Markdown from PDF")

    def it_marks_a_paper_with_no_markdown_available(
        data_dir, arxiv, fake_converter
    ):
        sync_metadata(data_dir)
        counts = render_markdown(data_dir,
                                 converter=fake_converter)

        # 2401.00004 has no HTML, no e-print and no PDF -- every conversion
        # path 404s, so it is the one paper left with no markdown.
        assert counts["absent"] == 1
        assert (data_dir / "2401.00004" / ".no_markdown").exists()
        assert not (data_dir / "2401.00004" / "paper.md").exists()

    def it_keeps_only_markdown_and_metadata_on_disk(
        data_dir, arxiv, fake_converter
    ):
        sync_metadata(data_dir)
        render_markdown(data_dir,
                        converter=fake_converter)

        for pid in ("2401.00001", "2401.00002", "2401.00003"):
            on_disk = {p.name for p in (data_dir / pid).iterdir()}
            assert on_disk == {"metadata.json", "paper.md"}
        # The PDF and the raw LaTeX source are never written to the data dir.
        assert list(data_dir.rglob("*.pdf")) == []
        assert [p for p in data_dir.rglob("source") if p.is_dir()] == []


def describe_render_markdown_always_rewrites():
    def it_overwrites_a_corrupted_markdown_file_already_on_disk(
        data_dir, arxiv, fake_converter
    ):
        sync_metadata(data_dir)
        render_markdown(data_dir,
                        converter=fake_converter)

        md = data_dir / "2401.00001" / "paper.md"
        md.write_text("corrupted")  # must not skip a paper that "exists"

        render_markdown(data_dir,
                        converter=fake_converter)

        assert md.read_text().startswith("# Markdown from HTML")

    def it_returns_the_same_counts_on_a_rerun(
        data_dir, arxiv, fake_converter
    ):
        sync_metadata(data_dir)
        first = render_markdown(data_dir, converter=fake_converter)

        again = render_markdown(data_dir, converter=fake_converter)

        # A rerun is deterministic given the same inputs: every paper
        # classifies into the same tier, so the counts must match. The
        # cachetta layer is bypassed in tests (see conftest) and tested
        # in its own suite -- not here.
        assert again == first

    def it_makes_no_request_on_a_dry_run(
        data_dir, arxiv, fake_converter
    ):
        sync_metadata(data_dir)
        before = list(arxiv.calls)

        counts = render_markdown(data_dir, dry_run=True,
                                 converter=fake_converter)

        assert counts["html"] == 0 and counts["latex"] == 0
        assert counts["pdf"] == 0
        assert arxiv.calls == before
        assert not (data_dir / "2401.00001" / "paper.md").exists()


def describe_render_markdown_resilience():
    def it_skips_a_paper_with_corrupt_metadata_and_processes_the_rest(
        data_dir, arxiv, fake_converter
    ):
        sync_metadata(data_dir)
        # A folder with an unreadable metadata.json must not abort the run.
        bad = data_dir / "2401.09998"
        bad.mkdir()
        (bad / "metadata.json").write_text("{ not valid json")

        counts = render_markdown(data_dir,
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
        data_dir, arxiv, fake_converter, reason
    ):
        def failing_latex(eprint: bytes) -> str:
            raise ValueError(reason)

        converter = dataclasses.replace(fake_converter, latex=failing_latex)
        sync_metadata(data_dir)
        counts = render_markdown(data_dir,
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
        data_dir, arxiv, fake_converter
    ):
        sync_metadata(data_dir)
        render_markdown(data_dir,
                        converter=fake_converter)

        log_text = _read_render_log(data_dir)
        assert "html 2401.00001: HTTP 200" in log_text
        assert "tex  2401.00003: HTTP 200" in log_text

    def it_logs_a_404_concisely_without_the_mdn_link(
        data_dir, arxiv, fake_converter
    ):
        sync_metadata(data_dir)
        render_markdown(data_dir, verbose=True,
                        converter=fake_converter)

        # 2401.00002 / 2401.00003 have no arxiv HTML -> a 404 is logged, but
        # without httpx's multi-line MDN documentation link.
        log_text = _read_render_log(data_dir)
        assert "html 2401.00002: HTTP 404" in log_text
        assert "developer.mozilla" not in log_text


# ---------------------------------------------------------------------------
# Pathological paper regression: 2607.02140
# ---------------------------------------------------------------------------
#
# 2026-07-03 and 2026-07-04 the tower fetch cron was OOM-killed at ~30 GB
# anon-rss while rendering 2607.02140. The paper has no arxiv HTML, a LaTeX
# e-print pandoc rejects (exit 64), and a PDF whose embedded heatmap images
# make pymupdf4llm rasterize into ~70 MiB float32 numpy arrays. The
# ``_markdown_from_pdf`` guard catches the eventual failure but the peak
# allocation still balloons and glibc's arena retains it, so per-paper
# leftovers accumulate across the ~15k-paper daily loop.
#
# The regression contract: the render *loop process* peak stays bounded
# regardless of what a single paper's PDF demands. The fix subprocess-
# isolates the PDF converter so a hostile paper eats its own grandchild's
# memory, not the fetch loop's.


def _run_render_under_memcap(data_dir_str: str, memcap_mb: int,
                             fixtures_dir_str: str, q: "mp.Queue") -> None:
    """Child entrypoint: run ``render_markdown`` under RLIMIT_AS.

    Reports ``(outcome, peak_mb, counts_or_detail)`` on *q*. Must be
    module-level so the ``spawn`` mp context can pickle it.
    """
    resource.setrlimit(
        resource.RLIMIT_AS,
        (memcap_mb * 1024 * 1024, memcap_mb * 1024 * 1024),
    )

    # Reproduce conftest's ``arxiv`` + ``no_cachetta`` fixtures in-process
    # -- ``spawn`` gives us a fresh interpreter with no pytest state.
    import httpx
    from unittest.mock import patch
    from cachetta.utils.cache_fn import _Cached

    from fetcher import render_markdown as _render_markdown
    from fetcher.shared import http

    fixtures = Path(fixtures_dir_str)

    def _resolve(url: str) -> Path | None:
        if "/html/" in url:
            ident = url.rsplit("/html/", 1)[1].replace("/", "_")
            return fixtures / f"html_{ident}.html"
        if "/pdf/" in url:
            ident = url.rsplit("/pdf/", 1)[1].replace("/", "_")
            return fixtures / f"pdf_{ident}.pdf"
        if "/e-print/" in url:
            ident = url.rsplit("/e-print/", 1)[1].replace("/", "_")
            targz = fixtures / f"eprint_{ident}.tar.gz"
            return targz if targz.exists() else fixtures / f"eprint_{ident}.pdf"
        return None

    def fake_http_get(url: str, timeout: float) -> bytes:
        path = _resolve(url)
        if path is None or not path.exists():
            req = httpx.Request("GET", url)
            resp = httpx.Response(404, request=req)
            raise httpx.HTTPStatusError(
                f"404 Not Found for {url}", request=req, response=resp,
            )
        return path.read_bytes()

    def bypass_cachetta(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    try:
        with patch.object(_Cached, "__call__", bypass_cachetta), \
             patch.object(http, "http_get", fake_http_get):
            counts = _render_markdown(Path(data_dir_str))
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        q.put(("OK", peak_kb // 1024, counts))
    except MemoryError as exc:
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        q.put(("MEMORY", peak_kb // 1024, str(exc)))
    except Exception as exc:  # noqa: BLE001
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        q.put(("EXCEPTION", peak_kb // 1024, f"{type(exc).__name__}: {exc}"))


def _seed_pathological_paper(data_dir: Path) -> None:
    """Write a metadata.json for ``PATHOLOGICAL_PID`` directly.

    ``sync_metadata`` would need this id in the RSS fixture; seeding the
    folder skips that -- render only reads arxiv_id and the three URLs.
    """
    pd = data_dir / PATHOLOGICAL_PID
    pd.mkdir()
    (pd / "metadata.json").write_text(json.dumps({
        "arxiv_id": PATHOLOGICAL_PID,
        "html_url": f"https://arxiv.org/html/{PATHOLOGICAL_PID}v1",
        "source_url": f"https://arxiv.org/e-print/{PATHOLOGICAL_PID}",
        "pdf_url": f"https://arxiv.org/pdf/{PATHOLOGICAL_PID}v1",
    }))


def describe_render_markdown_pathological_paper():
    def it_keeps_the_render_process_peak_under_500mb_on_a_pdf_memory_bomb(
        data_dir,
    ):
        _seed_pathological_paper(data_dir)
        fixtures_dir = Path(__file__).parent / "__fixtures__"

        ctx = mp.get_context("spawn")
        q: "mp.Queue" = ctx.Queue()
        p = ctx.Process(
            target=_run_render_under_memcap,
            args=(str(data_dir), _PATHOLOGICAL_MEMCAP_MB,
                  str(fixtures_dir), q),
        )
        p.start()
        p.join(timeout=_PATHOLOGICAL_WALL_S)

        if p.is_alive():
            p.terminate()
            p.join(5)
            if p.is_alive():
                p.kill()
            pytest.fail(
                f"render_markdown hung > {_PATHOLOGICAL_WALL_S}s "
                f"on {PATHOLOGICAL_PID}"
            )

        assert not q.empty(), (
            f"child exited {p.exitcode} without reporting -- "
            "likely killed by RLIMIT_AS before it could send"
        )
        outcome, peak_mb, detail = q.get()

        # The fetch loop's contract: no single paper causes the render
        # process to hold half a gigabyte of memory. The subprocess-isolated
        # PDF converter satisfies this by pushing pymupdf's allocations
        # into a grandchild that is killed on return.
        assert peak_mb < _PATHOLOGICAL_PEAK_BUDGET_MB, (
            f"render peak {peak_mb} MB exceeds {_PATHOLOGICAL_PEAK_BUDGET_MB} "
            f"MB budget (outcome={outcome}, detail={detail!r})"
        )
        assert outcome == "OK", (
            f"render did not complete cleanly: {outcome}: {detail!r} "
            f"(peak={peak_mb} MB)"
        )
