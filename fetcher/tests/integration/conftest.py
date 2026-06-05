"""Shared fixtures for the SDK integration tests.

The real httpx network transport is replaced by a fixture-backed fake,
injected through the SDK's public ``transport=`` parameter -- never by
monkeypatching a production module (see AGENTS.md).
"""

import json
from pathlib import Path

import httpx
import pytest

from fetcher.commands.classify import Classifier
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


class FakeTransport:
    """A fixture-backed stand-in for the real, rate-limited httpx transport.

    Serves bytes from ``tests/integration/__fixtures__/`` and records every
    URL requested, so a test can assert exactly which arxiv requests a run
    would have made -- and, by the absence of a call, that cachetta served a
    request from disk instead.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, url: str, timeout: float) -> bytes:
        self.calls.append(url)
        path = self._resolve(url)
        if path is None or not path.exists():
            # Mirror arxiv: an unknown URL is a 404, raised the way the real
            # httpx transport raises it (raise_for_status -> HTTPStatusError).
            request = httpx.Request("GET", url)
            response = httpx.Response(404, request=request)
            raise httpx.HTTPStatusError(
                f"Client error '404 Not Found' for url '{url}'\n"
                "For more information check: "
                "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404",
                request=request,
                response=response,
            )
        return path.read_bytes()

    @staticmethod
    def _resolve(url: str) -> Path | None:
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


@pytest.fixture
def fake_transport() -> FakeTransport:
    return FakeTransport()


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


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    # Deliberately separate from the data dir, and created lazily by cachetta.
    return tmp_path / "cache"


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
        'base_url = "http://localhost:11434/v1"\n'
        "timeout_s = 60.0\n"
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
