"""Convert a paper into markdown.

fetcher keeps a markdown rendering of every paper -- never the PDF or the raw
LaTeX. Two paths produce it:

  - arxiv native HTML (``https://arxiv.org/html/{id}``, LaTeXML-rendered) is
    parsed by arxiv2md. This is the primary path: source-derived, no ML.
  - for the ~3% of papers with no arxiv HTML, the LaTeX e-print archive is
    extracted to a temp dir and converted by pandoc (via pypandoc).

Both external libraries are *injection seams*: each function takes the
underlying callable as a keyword argument, defaulting to a lazy import of the
real library. Tests pass a fake through that seam -- the same
dependency-injection discipline download.py uses for the network transport
(see AGENTS.md). Nothing here touches the network.
"""

from __future__ import annotations

import gzip
import io
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_PDF_MAGIC = b"%PDF"
# Below this many non-whitespace characters a "conversion" is really an arxiv
# "HTML not available" stub or an empty render -- treat it as no markdown.
_MIN_MARKDOWN_CHARS = 200
# pandoc parses a custom verbatim environment (\DefineVerbatimEnvironment) as
# ordinary LaTeX; on a code-heavy appendix its parser backtracks for minutes.
# pandoc has no wall-clock limit of its own, so fetcher imposes one -- any
# legitimate paper converts in seconds, well under this.
_PANDOC_TIMEOUT_S = 120


def html_to_markdown(html: bytes, *, convert: Callable[[str], str] | None = None) -> str:
    """Convert arxiv HTML bytes to markdown.

    *convert* is the conversion seam; it defaults to arxiv2md's pure,
    network-free ``convert_html_to_markdown``.
    """
    if convert is None:
        from arxiv2md.markdown import convert_html_to_markdown as convert
    return convert(html.decode("utf-8", errors="replace"))


def _safe_extract_tar(body: bytes, dest: Path) -> int:
    """Extract a tarball into *dest*, rejecting path-traversal members.

    Returns the number of files written.
    """
    written = 0
    dest_resolved = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(body)) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name.lstrip("./")
            target = (dest / name).resolve()
            # True containment, not a string prefix: a startswith() guard
            # admits a sibling sharing dest's name (src vs srchack).
            if target != dest_resolved and dest_resolved not in target.parents:
                continue  # path traversal attempt, skip
            fh = tar.extractfile(member)
            if fh is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(fh.read())
            written += 1
    return written


def _extract_eprint(body: bytes, dest: Path) -> int:
    """Extract a LaTeX e-print archive into *dest*. Returns files written.

    The e-print endpoint returns a tarball, a single gzipped file, or (for
    some old/withdrawn papers) just a PDF -- only the first two yield LaTeX.
    """
    if body[:4] == _PDF_MAGIC:
        return 0
    try:
        return _safe_extract_tar(body, dest)
    except tarfile.ReadError:
        try:
            raw = gzip.decompress(body)
        except OSError:
            return 0
        if raw[:4] == _PDF_MAGIC:
            return 0
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "main.tex").write_bytes(raw)
        return 1


def _main_tex(directory: Path) -> Path | None:
    """Pick the entry-point .tex in an extracted archive.

    Prefer the file declaring ``\\documentclass``; then one literally named
    ``main.tex``; then the largest .tex. None if the archive has no .tex.
    """
    texs = sorted(directory.rglob("*.tex"))
    if not texs:
        return None
    for t in texs:
        if rb"\documentclass" in t.read_bytes():
            return t
    for t in texs:
        if t.name == "main.tex":
            return t
    return max(texs, key=lambda p: p.stat().st_size)


def _run_with_timeout(cmd: list[str], cwd: str | None, timeout: float) -> str:
    """Run *cmd*, capture its stdout, and kill it if it overruns *timeout*.

    ``subprocess.run(timeout=)`` SIGKILLs and reaps the child on expiry,
    raising ``TimeoutExpired`` -- so an overrunning pandoc cannot spin
    unbounded. A non-zero exit raises ``CalledProcessError``.
    """
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
        timeout=timeout, check=True,
    )
    return proc.stdout


def _default_pandoc(
    path: str, to: str, *, format: str,
    extra_args: list[str] | None = None, cworkdir: str | None = None,
) -> str:
    """The production pandoc seam: the bundled pandoc binary, time-bounded.

    ``pypandoc.convert_file`` offers no timeout, so fetcher invokes the binary
    directly -- pypandoc still locates it -- through ``_run_with_timeout``.
    """
    import pypandoc

    cmd = [
        pypandoc.get_pandoc_path(),
        "--from", format, "--to", to, path,
        *(extra_args or []),
    ]
    return _run_with_timeout(cmd, cworkdir, _PANDOC_TIMEOUT_S)


def latex_to_markdown(eprint: bytes, *, pandoc: Callable[..., str] | None = None) -> str:
    """Convert a LaTeX e-print archive to markdown via pandoc.

    The archive is extracted to a temporary directory, the main .tex located,
    and pandoc run against it; the temp dir is discarded. *pandoc* is the
    conversion seam, defaulting to the time-bounded ``_default_pandoc``.
    Raises ``ValueError`` when the archive carries no usable LaTeX source,
    when pandoc overruns its timeout, or when pandoc rejects the LaTeX
    (a non-zero exit) -- all expected outcomes for an adversarial paper.
    """
    if pandoc is None:
        pandoc = _default_pandoc
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        if _extract_eprint(eprint, work) == 0:
            raise ValueError("e-print archive has no LaTeX source")
        main = _main_tex(work)
        if main is None:
            raise ValueError("e-print archive contains no .tex file")
        # pandoc resolves \input{} / \include{} relative to its process
        # working directory, not --resource-path -- so a multi-file paper
        # needs cworkdir at the extraction root or every include is dropped.
        # --resource-path additionally covers \includegraphics and resources.
        try:
            return pandoc(
                str(main), "gfm", format="latex",
                extra_args=["--resource-path", str(work)],
                cworkdir=str(work),
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(
                f"LaTeX conversion exceeded the {_PANDOC_TIMEOUT_S}s pandoc timeout"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                f"pandoc rejected the LaTeX source (exit {exc.returncode})"
            ) from exc


def _default_pdf_convert(pdf: bytes) -> str:
    """Convert PDF bytes to markdown with pymupdf4llm (lazily imported)."""
    import pymupdf
    import pymupdf4llm

    doc = pymupdf.open(stream=pdf, filetype="pdf")
    try:
        return pymupdf4llm.to_markdown(doc, show_progress=False)
    finally:
        doc.close()


def pdf_to_markdown(pdf: bytes, *, convert: Callable[[bytes], str] | None = None) -> str:
    """Convert a paper's PDF to markdown -- the last-resort conversion path.

    Reached only for a paper with neither arxiv HTML nor a usable LaTeX
    e-print. arxiv PDFs are LaTeX-generated (a real text layer, never
    scanned), so a text extractor recovers prose well; math and tables
    degrade to plain text. *convert* is the seam, defaulting to pymupdf4llm.
    """
    if convert is None:
        convert = _default_pdf_convert
    return convert(pdf)


def _is_substantial(md: str) -> bool:
    """True if *md* carries real body text, not an empty or stub render."""
    return len(md.strip()) >= _MIN_MARKDOWN_CHARS


@dataclass(frozen=True)
class Converter:
    """The conversion seam ``fetch`` injects.

    Bundles the three byte->markdown callables so an integration test can swap
    in a fake without arxiv2md, pypandoc or pymupdf4llm ever being called.
    """

    html: Callable[[bytes], str]
    latex: Callable[[bytes], str]
    pdf: Callable[[bytes], str]


# The production converter: the real libraries, lazily imported on first call.
REAL_CONVERTER = Converter(
    html=html_to_markdown, latex=latex_to_markdown, pdf=pdf_to_markdown
)
