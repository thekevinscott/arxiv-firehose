"""Unit tests for the markdown conversion module.

The two external libraries -- arxiv2md and pypandoc -- are never called: each
conversion function takes the underlying callable as an injection seam, and
these tests pass a fake (no monkeypatching -- see AGENTS.md). What is tested
here is fetcher's own glue: byte decoding, archive extraction, main-tex
selection, delegation.
"""

import gzip
import io
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

from fetcher.shared.convert import (
    _is_substantial,
    _main_tex,
    _run_with_timeout,
    _safe_extract_tar,
    html_to_markdown,
    latex_to_markdown,
    pdf_to_markdown,
)


def _make_tar(members: dict[str, bytes]) -> bytes:
    """Build an in-memory tarball from {member name: bytes}."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def describe_html_to_markdown():
    def it_decodes_bytes_and_delegates_to_the_converter():
        seen = []

        def fake_convert(html: str) -> str:
            seen.append(html)
            return "# Converted"

        out = html_to_markdown(b"<h1>Hi</h1>", convert=fake_convert)

        assert out == "# Converted"
        assert seen == ["<h1>Hi</h1>"]

    def it_tolerates_undecodable_bytes():
        # A stray non-UTF-8 byte must not abort the conversion.
        html_to_markdown(b"<p>\xff bad byte</p>", convert=lambda s: s)


def describe_latex_to_markdown():
    def it_extracts_the_archive_picks_the_main_tex_and_calls_pandoc(tmp_path):
        calls = []

        def fake_pandoc(path, to, *, format, extra_args=None, cworkdir=None):
            calls.append((path, to, format))
            return "converted markdown body"

        body = _make_tar({
            "intro.tex": b"\\input{main}",
            "main.tex": b"\\documentclass{article}\\begin{document}x\\end{document}",
        })

        out = latex_to_markdown(body, pandoc=fake_pandoc)

        assert out == "converted markdown body"
        assert len(calls) == 1
        path, to, fmt = calls[0]
        assert path.endswith("main.tex")  # the \documentclass file
        assert (to, fmt) == ("gfm", "latex")

    def it_runs_pandoc_from_the_archive_root_so_includes_resolve(tmp_path):
        # pandoc resolves \input{}/\include{} relative to its process working
        # directory, not --resource-path. A multi-file paper whose main.tex
        # does \input{contents/...} loses every include unless pandoc runs
        # with cwd at the extraction root.
        seen = {}

        def fake_pandoc(path, to, *, format, extra_args=None, cworkdir=None):
            seen["cworkdir"] = cworkdir
            seen["main_under_cwd"] = (
                cworkdir is not None and Path(path).parent == Path(cworkdir)
            )
            seen["include_reachable"] = (
                cworkdir is not None
                and (Path(cworkdir) / "contents" / "intro.tex").exists()
            )
            return "converted markdown body"

        body = _make_tar({
            "main.tex": b"\\documentclass{article}\\input{contents/intro}",
            "contents/intro.tex": b"the actual body text",
        })

        latex_to_markdown(body, pandoc=fake_pandoc)

        assert seen["cworkdir"] is not None
        assert seen["main_under_cwd"]
        assert seen["include_reachable"]

    def it_handles_a_single_gzipped_tex(tmp_path):
        def fake_pandoc(path, to, *, format, extra_args=None, cworkdir=None):
            return "md"

        body = gzip.compress(b"\\documentclass{article}\\begin{document}y\\end{document}")
        assert latex_to_markdown(body, pandoc=fake_pandoc) == "md"

    def it_raises_when_the_eprint_is_pdf_only():
        with pytest.raises(ValueError):
            latex_to_markdown(b"%PDF-1.7\nnot latex", pandoc=lambda *a, **k: "")

    def it_raises_valueerror_when_pandoc_rejects_the_latex():
        # pandoc exits 64 on LaTeX it cannot parse. That is an expected
        # outcome for an adversarial paper, not a converter blow-up, so it
        # must surface as ValueError -- the same clean signal as a timeout --
        # not leak CalledProcessError to fetch's generic catch-all.
        def failing_pandoc(path, to, *, format, extra_args=None, cworkdir=None):
            raise subprocess.CalledProcessError(returncode=64, cmd=["pandoc"])

        body = _make_tar({
            "main.tex": b"\\documentclass{article}\\begin{document}x\\end{document}",
        })
        with pytest.raises(ValueError):
            latex_to_markdown(body, pandoc=failing_pandoc)

    def it_raises_valueerror_when_pandoc_times_out():
        # pandoc parses a custom verbatim env as ordinary LaTeX; an adversarial
        # paper can spin its parser for minutes. A timeout must surface as a
        # plain ValueError so fetch routes the paper to .no_markdown like any
        # other conversion miss -- never as an unbounded hang.
        def slow_pandoc(path, to, *, format, extra_args=None, cworkdir=None):
            raise subprocess.TimeoutExpired(cmd=["pandoc"], timeout=120)

        body = _make_tar({
            "main.tex": b"\\documentclass{article}\\begin{document}x\\end{document}",
        })
        with pytest.raises(ValueError):
            latex_to_markdown(body, pandoc=slow_pandoc)


def describe_pdf_to_markdown():
    def it_hands_the_pdf_bytes_to_the_converter():
        seen = []

        def fake_convert(pdf: bytes) -> str:
            seen.append(pdf)
            return "# Converted from PDF"

        out = pdf_to_markdown(b"%PDF-1.7\nbody", convert=fake_convert)

        assert out == "# Converted from PDF"
        assert seen == [b"%PDF-1.7\nbody"]


def describe__run_with_timeout():
    def it_returns_stdout_for_a_quick_command():
        out = _run_with_timeout(
            [sys.executable, "-c", "import sys; sys.stdout.write('hello')"],
            cwd=None, timeout=10,
        )
        assert out == "hello"

    def it_kills_a_command_that_exceeds_the_timeout():
        # pandoc has no wall-clock limit of its own; the wrapper must kill an
        # overrunning process, not wait it out.
        start = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            _run_with_timeout(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                cwd=None, timeout=0.3,
            )
        assert time.monotonic() - start < 3  # killed near the timeout, not at 10s


def describe__safe_extract_tar():
    def it_extracts_a_benign_member(tmp_path):
        dest = tmp_path / "src"
        written = _safe_extract_tar(_make_tar({"paper.tex": b"\\documentclass"}), dest)

        assert written == 1
        assert (dest / "paper.tex").read_bytes() == b"\\documentclass"

    def it_rejects_a_sibling_path_escape(tmp_path):
        # A member resolving to a sibling that shares dest's name as a string
        # prefix (dest "src", target "srchack"): a startswith() guard admits
        # it; the containment check rejects it.
        dest = tmp_path / "src"
        escape = f"x/../../{dest.name}hack/escape.tex"
        written = _safe_extract_tar(_make_tar({"ok.tex": b"ok", escape: b"PWNED"}), dest)

        assert (dest / "ok.tex").read_bytes() == b"ok"
        assert not (tmp_path / f"{dest.name}hack" / "escape.tex").exists()
        assert written == 1


def describe__main_tex():
    def it_prefers_the_file_with_documentclass(tmp_path):
        (tmp_path / "a.tex").write_bytes(b"\\section{x}")
        (tmp_path / "b.tex").write_bytes(b"\\documentclass{article}")
        assert _main_tex(tmp_path).name == "b.tex"

    def it_falls_back_to_a_file_named_main_tex(tmp_path):
        (tmp_path / "a.tex").write_bytes(b"\\section{x}")
        (tmp_path / "main.tex").write_bytes(b"\\section{y}")
        assert _main_tex(tmp_path).name == "main.tex"

    def it_returns_none_when_no_tex_is_present(tmp_path):
        (tmp_path / "fig.png").write_bytes(b"\x89PNG")
        assert _main_tex(tmp_path) is None


def describe__is_substantial():
    def it_accepts_a_real_document():
        assert _is_substantial("word " * 100) is True

    def it_rejects_a_short_stub():
        assert _is_substantial("HTML is not available for this paper.") is False

    def it_rejects_whitespace_only():
        assert _is_substantial("   \n\n  ") is False
