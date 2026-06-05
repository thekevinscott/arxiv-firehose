"""Unit tests for cache-path derivation and transport-injected downloading.

These never touch the network: the downloader's transport is a plain
in-test callable, injected through ``make_downloader``'s public parameter --
no monkeypatching (see AGENTS.md).
"""

import pickle
from pathlib import Path

import httpx
import pytest

from fetcher.shared.download import (
    _cache_path,
    _is_retryable,
    _looks_like_feed,
    _looks_like_html,
    _looks_like_paper,
    _with_retry,
    make_downloader,
    make_feed_fetcher,
    make_html_fetcher,
)


def _status_error(code: int) -> httpx.HTTPStatusError:
    """Build an HTTPStatusError carrying *code*, as raise_for_status would."""
    request = httpx.Request("GET", "https://arxiv.org/x")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError(f"HTTP {code}", request=request, response=response)

CACHE = Path("/cache")

_GZIP_MAGIC = b"\x1f\x8b"
_HTML = b"<!DOCTYPE html><html><body>503 Service Unavailable</body></html>"


def describe_cache_path():
    def it_maps_a_pdf_url():
        assert _cache_path(CACHE, "https://arxiv.org/pdf/2401.12345v1") == (
            CACHE / "pdf" / "2401.12345v1.pkl"
        )

    def it_maps_an_eprint_url():
        assert _cache_path(CACHE, "https://arxiv.org/e-print/2401.12345") == (
            CACHE / "eprint" / "2401.12345.pkl"
        )

    def it_maps_an_html_url():
        assert _cache_path(CACHE, "https://arxiv.org/html/2401.12345v1") == (
            CACHE / "html" / "2401.12345v1.pkl"
        )

    def it_slugifies_a_legacy_id():
        assert _cache_path(CACHE, "https://arxiv.org/e-print/cs/0501001") == (
            CACHE / "eprint" / "cs_0501001.pkl"
        )


def describe_make_downloader():
    def it_calls_the_injected_transport_on_a_miss(tmp_path):
        calls = []

        def transport(url, timeout):
            calls.append(url)
            return b"%PDF-bytes"

        download = make_downloader(tmp_path / "cache", transport)
        assert download("https://arxiv.org/pdf/2401.00001v1") == b"%PDF-bytes"
        assert calls == ["https://arxiv.org/pdf/2401.00001v1"]

    def it_serves_a_repeat_call_from_cache_without_the_transport(tmp_path):
        calls = []

        def transport(url, timeout):
            calls.append(url)
            return b"%PDF-bytes"

        download = make_downloader(tmp_path / "cache", transport)
        url = "https://arxiv.org/pdf/2401.00002v1"
        download(url)
        download(url)  # second call must be a cache hit
        assert calls == [url]  # transport invoked exactly once

    def it_reads_a_prewarmed_cache_file_without_the_transport(tmp_path):
        cache_dir = tmp_path / "cache"
        url = "https://arxiv.org/pdf/2401.99999v1"
        target = _cache_path(cache_dir, url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pickle.dumps(b"%PDF-cached"))

        def exploding_transport(url, timeout):
            raise AssertionError("transport must not be called on a cache hit")

        download = make_downloader(cache_dir, exploding_transport)
        assert download(url) == b"%PDF-cached"


def describe_make_feed_fetcher():
    def it_requests_the_category_rss_url(tmp_path):
        calls = []

        def transport(url, timeout):
            calls.append(url)
            return b"<rss/>"

        fetch_feed = make_feed_fetcher(tmp_path / "cache", transport)
        assert fetch_feed("cs.LG") == b"<rss/>"
        assert calls == ["https://rss.arxiv.org/rss/cs.LG"]


def describe__looks_like_paper():
    def it_accepts_a_pdf():
        assert _looks_like_paper(b"%PDF-1.7\n...") is True

    def it_accepts_a_gzip_archive():
        assert _looks_like_paper(_GZIP_MAGIC + b"rest of tarball") is True

    def it_rejects_an_html_error_page():
        assert _looks_like_paper(_HTML) is False

    def it_rejects_empty_bytes():
        assert _looks_like_paper(b"") is False


def describe__looks_like_feed():
    def it_accepts_an_xml_declaration():
        assert _looks_like_feed(b'<?xml version="1.0"?><rss></rss>') is True

    def it_accepts_a_bare_rss_element():
        assert _looks_like_feed(b"<rss/>") is True

    def it_accepts_leading_whitespace():
        assert _looks_like_feed(b"\n  <?xml version='1.0'?>") is True

    def it_rejects_an_html_error_page():
        assert _looks_like_feed(_HTML) is False


def describe_make_downloader_cache_validation():
    def it_does_not_cache_a_non_paper_body(tmp_path):
        # arxiv occasionally answers 200 with an HTML error page. That junk
        # must not be cached, or _write_pdf rejects it forever.
        calls = []

        def transport(url, timeout):
            calls.append(url)
            return _HTML

        download = make_downloader(tmp_path / "cache", transport)
        url = "https://arxiv.org/pdf/2401.00003v1"
        download(url)
        download(url)  # not cached -> transport runs again
        assert calls == [url, url]


def describe_make_feed_fetcher_cache_validation():
    def it_does_not_cache_a_non_feed_body(tmp_path):
        calls = []

        def transport(url, timeout):
            calls.append(url)
            return _HTML

        fetch_feed = make_feed_fetcher(tmp_path / "cache", transport)
        fetch_feed("cs.LG")
        fetch_feed("cs.LG")  # not cached -> transport runs again
        assert len(calls) == 2


def describe__looks_like_html():
    def it_accepts_a_doctype_declaration():
        assert _looks_like_html(b"<!DOCTYPE html><html><body>x</body></html>") is True

    def it_accepts_a_bare_html_element():
        assert _looks_like_html(b"<html lang='en'><body/></html>") is True

    def it_accepts_leading_whitespace_and_mixed_case():
        assert _looks_like_html(b"\n  <!doctype HTML>\n<HTML>") is True

    def it_rejects_a_feed():
        assert _looks_like_html(b'<?xml version="1.0"?><rss></rss>') is False

    def it_rejects_a_pdf():
        assert _looks_like_html(b"%PDF-1.7\n...") is False

    def it_rejects_empty_bytes():
        assert _looks_like_html(b"") is False


def describe_make_html_fetcher():
    def it_calls_the_transport_then_serves_a_repeat_from_cache(tmp_path):
        calls = []

        def transport(url, timeout):
            calls.append(url)
            return b"<!DOCTYPE html><html><body>paper</body></html>"

        fetch_html = make_html_fetcher(tmp_path / "cache", transport)
        url = "https://arxiv.org/html/2401.00001v1"
        assert fetch_html(url).startswith(b"<!DOCTYPE html>")
        fetch_html(url)  # second call must be a cache hit
        assert calls == [url]  # transport invoked exactly once

    def it_does_not_cache_a_non_html_body(tmp_path):
        calls = []

        def transport(url, timeout):
            calls.append(url)
            return b"503 Service Unavailable - upstream down"  # plain text, not html

        # An error body that is not HTML must not poison the 100-year cache.
        fetch_html = make_html_fetcher(tmp_path / "cache", transport)
        url = "https://arxiv.org/html/2401.00009v1"
        fetch_html(url)
        fetch_html(url)
        assert calls == [url, url]


def describe__is_retryable():
    def it_retries_a_503():
        assert _is_retryable(_status_error(503)) is True

    def it_does_not_retry_a_404():
        assert _is_retryable(_status_error(404)) is False

    def it_retries_a_transport_error():
        exc = httpx.ConnectError("connection refused")
        assert _is_retryable(exc) is True

    def it_does_not_retry_an_unrelated_exception():
        assert _is_retryable(ValueError("nope")) is False


def describe__with_retry():
    def it_returns_on_the_first_success():
        assert _with_retry(lambda: "ok", sleep=lambda _s: None) == "ok"

    def it_succeeds_after_two_retryable_failures():
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise _status_error(503)
            return "ok"

        slept = []
        assert _with_retry(flaky, sleep=slept.append) == "ok"
        assert len(attempts) == 3
        assert slept == [1, 2]  # backoff before retries 2 and 3

    def it_gives_up_after_the_attempt_limit():
        attempts = []

        def always_503():
            attempts.append(1)
            raise _status_error(503)

        with pytest.raises(httpx.HTTPStatusError):
            _with_retry(always_503, attempts=3, sleep=lambda _s: None)
        assert len(attempts) == 3

    def it_reraises_a_non_retryable_immediately():
        attempts = []

        def always_404():
            attempts.append(1)
            raise _status_error(404)

        with pytest.raises(httpx.HTTPStatusError):
            _with_retry(always_404, sleep=lambda _s: None)
        assert len(attempts) == 1  # no retry on a 404
