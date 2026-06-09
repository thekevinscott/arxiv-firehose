"""Shared fixtures for the SDK integration tests.

The real network is replaced two ways:

- ``arxiv``: a fixture that swaps ``shared.download._http_get`` for a
  fixture-backed fake, and redirects the three cachetta caches
  (feeds / papers / html) onto ``tmp_path``. The on-disk cache is
  real (cachetta is hit, populated, re-read), so tests can prove a
  rerun avoided the network by inspecting ``arxiv.calls``.
- ``fake_classifier``: a Classifier wired through the SDK's public
  ``classifier=`` parameter -- never patched.

No monkeypatching. All redirection uses ``unittest.mock.patch.object``
through context managers.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from fetcher.commands.classify import Classifier
from fetcher.shared import download
from fetcher.shared.convert import Converter

FIXTURES = Path(__file__).parent / "__fixtures__"

# Sentinel bodies the fake converter returns. Long enough to clear
# convert._is_substantial (>= 200 non-whitespace chars), so a test can assert
# paper.md holds exactly what the converter produced.
FAKE_HTML_MARKDOWN = "# Markdown from HTML\n\n" + "converted body text. " * 20
FAKE_LATEX_MARKDOWN = "# Markdown from LaTeX\n\n" + "converted body text. " * 20
FAKE_PDF_MARKDOWN = "# Markdown from PDF\n\n" + "converted body text. " * 20

# A data dir is bootstrapped with this config so a run touches exactly one
# feed (the fixture only provides cs.LG); the SDK would otherwise write a
# 3-category default.
CONFIG_TOML = """\
[categories]
include = ["cs.LG"]

[fetch]
source = "arxiv"
concurrency = 1
latex_fallback = true
pdf_fallback = true

[ingest]
backfill_days = 0
"""


def _resolve_fixture(url: str) -> Path | None:
    if "rss.arxiv.org/rss/" in url:
        category = url.rsplit("/rss/", 1)[1]
        return FIXTURES / f"rss_{category}.xml"
    if "/html/" in url:
        ident = url.rsplit("/html/", 1)[1].replace("/", "_")
        return FIXTURES / f"html_{ident}.html"
    if "/pdf/" in url:
        ident = url.rsplit("/pdf/", 1)[1].replace("/", "_")
        return FIXTURES / f"pdf_{ident}.pdf"
    if "/e-print/" in url:
        ident = url.rsplit("/e-print/", 1)[1].replace("/", "_")
        targz = FIXTURES / f"eprint_{ident}.tar.gz"
        return targz if targz.exists() else FIXTURES / f"eprint_{ident}.pdf"
    return None


def _raise_404(url: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", url)
    response = httpx.Response(404, request=request)
    return httpx.HTTPStatusError(
        f"Client error '404 Not Found' for url '{url}'\n"
        "For more information check: "
        "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404",
        request=request,
        response=response,
    )


@pytest.fixture
def arxiv(tmp_path):
    """Stub the network + isolate the cachetta caches in one fixture.

    Yields a namespace with a ``calls`` list -- every URL the production
    code actually sent through ``_http_get`` (so a cachetta hit doesn't
    register). Tests use that list to prove a rerun was served from disk.

    Cache redirection mutates the path on the three sibling Cachetta
    instances declared at module load in ``shared.download``. The
    ``patch.object`` context restores the original paths on exit so a
    test can't leak state into another.
    """
    calls: list[str] = []

    def fake_http_get(url: str, timeout: float) -> bytes:
        calls.append(url)
        path = _resolve_fixture(url)
        if path is None or not path.exists():
            raise _raise_404(url)
        return path.read_bytes()

    with patch.object(download, "_http_get", fake_http_get), \
         patch.object(download._feed_cache, "path", tmp_path / "feeds"), \
         patch.object(download._paper_cache, "path", tmp_path / "papers"), \
         patch.object(download._html_cache, "path", tmp_path / "html"):
        yield SimpleNamespace(calls=calls)


def _fake_latex(eprint: bytes) -> str:
    """Stand-in for convert.latex_to_markdown: mirrors its contract of raising
    ValueError when the e-print archive carries no LaTeX (a PDF-only body)."""
    if eprint[:4] == b"%PDF":
        raise ValueError("e-print archive has no LaTeX source")
    return FAKE_LATEX_MARKDOWN


@pytest.fixture
def fake_converter() -> Converter:
    """A Converter that never calls arxiv2md, pypandoc or pymupdf4llm.

    It returns deterministic sentinel markdown so an integration test can
    assert what fetch wrote without depending on the real libraries.
    """
    return Converter(
        html=lambda html: FAKE_HTML_MARKDOWN,
        latex=_fake_latex,
        pdf=lambda pdf: FAKE_PDF_MARKDOWN,
    )


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    (d / "config.toml").write_text(CONFIG_TOML)
    return d


def _write_prompts_artifact(folder: Path, output_name: str) -> Path:
    """A minimal CoaxedPrompt artifact: prompt.jinja + meta.json.

    Mirrors what ``coax`` would produce -- one Jinja template and the schema
    for a single boolean output. Lets integration tests exercise the real
    ``CoaxedPrompt`` without depending on the compile step.
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "prompt.jinja").write_text("Question: {{ abstract }}\n")
    meta = {
        "output_name": output_name,
        "fields": {
            "inputs": {"abstract": {"type": "str"}},
            "output": {"type": "bool"},
        },
    }
    (folder / "meta.json").write_text(json.dumps(meta))
    return folder


@pytest.fixture
def prompts_dirs(tmp_path: Path) -> list[Path]:
    """Two compiled prompts artifacts the classify tests share.

    ``is_about_ml`` and ``is_about_markdown`` -- chosen so the fake
    classifier can deterministically light up one but not the other based
    on the fixture papers' abstracts.
    """
    ml = _write_prompts_artifact(tmp_path / "prompts" / "is-about-ml", "is_about_ml")
    md = _write_prompts_artifact(
        tmp_path / "prompts" / "is-about-markdown", "is_about_markdown"
    )
    return [ml, md]


@pytest.fixture
def data_dir_classify(data_dir: Path, prompts_dirs: list[Path]) -> Path:
    """A ``data_dir`` wired for classify: ``[classify].prompts_dirs`` in
    config.toml points at the fixture prompts artifacts. The taxonomy
    (the set of category ids) is derived from those prompts at runtime --
    no separate categories file to keep in sync.
    """
    paths = ", ".join(f'"{p}"' for p in prompts_dirs)
    cfg = (data_dir / "config.toml").read_text()
    cfg += (
        "\n[classify]\n"
        f"prompts_dirs = [{paths}]\n"
        'model = "test-model"\n'
    )
    (data_dir / "config.toml").write_text(cfg)
    return data_dir


@pytest.fixture
def fake_classifier() -> Classifier:
    """A Classifier that decides each flag from the prompt's text.

    No LLM -- the fake reads the rendered prompt and lights up the schema's
    single field. The fixture abstracts contain the word "markdown" for
    papers 00001-00003 but not 00004, so ``is_about_markdown`` separates
    the four-paper fixture cleanly. ``is_about_ml`` stays False (the word
    "ml" is not in any abstract verbatim) so a test can assert both
    branches of the schema field.
    """
    def call(prompt, schema):
        field = next(iter(schema.model_json_schema()["properties"]))
        if field == "is_about_markdown":
            value = "markdown" in prompt.lower()
        else:
            value = False
        return schema(**{field: value})
    return Classifier(call=call)
