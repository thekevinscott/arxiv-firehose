"""Shared fixtures for the SDK integration tests.

The real network is replaced two ways:

- ``no_cachetta`` (autouse): cachetta is inert for every integration
  test. Any cachetta-decorated function dispatches straight to its bare
  original -- no disk reads, no disk writes, no cache state anywhere.
  Cachetta has its own test suite; these tests assert fetcher behavior.
- ``arxiv``: swaps ``shared.http.http_get`` for a fixture-backed
  fake. ``arxiv.calls`` records exactly the URLs the production code
  requested (with cachetta inert, every request is observable).

No monkeypatching. All redirection uses ``unittest.mock.patch.object``
through context managers.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from cachetta.utils.cache_fn import _Cached

from fetcher.commands import embed as embed_mod
from fetcher.shared import http

FIXTURES = Path(__file__).parent / "__fixtures__"


class _FakeEmbedder:
    """Offline stand-in for the tower HTTP embedder.

    ``embed.run`` only needs ``encode(list[str]) -> EMBED_DIM-length
    vectors``; these tests assert counts and file presence, not vector
    content, so a constant non-zero vector is enough."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        vec = [1.0] + [0.0] * (embed_mod.EMBED_DIM - 1)
        return [list(vec) for _ in texts]

# A data dir is bootstrapped with this config so a run tracks exactly one
# category (the fixture answers any day-slice query with the same Atom
# body); the SDK would otherwise write a 14-category default. backfill_days
# = 2 keeps the day window small: today plus the two days before it, so a
# sync makes exactly three API calls.
CONFIG_TOML = """\
[categories]
include = ["cs.LG"]

[fetch]
source = "arxiv"
concurrency = 1

[ingest]
backfill_days = 2
"""


def _resolve_fixture(url: str) -> Path | None:
    # Before the day-slice branch: an id_list query is also an
    # export.arxiv.org/api/query URL and would be shadowed by it.
    if "id_list=" in url:
        ident = url.split("id_list=", 1)[1].split("&", 1)[0].replace("/", "_")
        return FIXTURES / f"api_id_{ident}.xml"
    if "export.arxiv.org/api/query" in url:
        # Every day-slice query gets the same Atom body; sync's dedupe
        # collapses the repeats, mirroring how overlapping real slices
        # re-serve already-known papers.
        return FIXTURES / "api_query.xml"
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


@pytest.fixture(autouse=True)
def no_cachetta():
    """Make cachetta inert for every integration test.

    ``_Cached`` is the wrapper cachetta puts around every decorated
    function; its ``__call__`` is the single dispatch point for the
    whole library at runtime. Patching it to forward to ``self._fn``
    (the bare original, stored at decoration time) turns every cache --
    feeds, papers, future ones -- into a passthrough. No disk reads, no
    disk writes, no cache state anywhere.

    Cachetta's own behavior is covered by its own test suite.
    """
    def bypass(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    with patch.object(_Cached, "__call__", bypass):
        yield


@pytest.fixture(autouse=True)
def fake_embedder():
    """Keep the embed stage offline for every integration test.

    ``embed.run`` builds its embedder via ``embed.load_embedder`` when no
    ``model=`` is passed (the SDK path fetch uses). Patch that factory to
    the offline fake so ``fetch``/``embed`` never call tower's network."""
    with patch.object(embed_mod, "load_embedder", lambda: _FakeEmbedder()):
        yield


@pytest.fixture
def arxiv():
    """Stub the network: ``shared.http.http_get`` answers from the
    fixture files. Yields a namespace with a ``calls`` list -- every URL
    the production code requested (cachetta is inert, so every request
    is observable).
    """
    calls: list[str] = []

    def fake_http_get(url: str, timeout: float) -> bytes:
        calls.append(url)
        path = _resolve_fixture(url)
        if path is None or not path.exists():
            raise _raise_404(url)
        return path.read_bytes()

    with patch.object(http, "http_get", fake_http_get):
        yield SimpleNamespace(calls=calls)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    (d / "config.toml").write_text(CONFIG_TOML)
    return d
