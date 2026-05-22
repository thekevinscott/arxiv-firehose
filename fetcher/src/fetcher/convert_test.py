"""Unit tests for the markdown conversion module.

The two external libraries -- arxiv2md and pypandoc -- are never called: each
conversion function takes the underlying callable as an injection seam, and
these tests pass a fake (no monkeypatching -- see AGENTS.md). What is tested
here is fetcher's own glue: byte decoding, archive extraction, main-tex
selection, delegation.
"""

import gzip
import io
import tarfile

import pytest

from fetcher.convert import (
    _is_substantial,
    _main_tex,
    _safe_extract_tar,
    html_to_markdown,
    latex_to_markdown,
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

        def fake_pandoc(path, to, *, format, extra_args=None):
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

    def it_handles_a_single_gzipped_tex(tmp_path):
        def fake_pandoc(path, to, *, format, extra_args=None):
            return "md"

        body = gzip.compress(b"\\documentclass{article}\\begin{document}y\\end{document}")
        assert latex_to_markdown(body, pandoc=fake_pandoc) == "md"

    def it_raises_when_the_eprint_is_pdf_only():
        with pytest.raises(ValueError):
            latex_to_markdown(b"%PDF-1.7\nnot latex", pandoc=lambda *a, **k: "")


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
