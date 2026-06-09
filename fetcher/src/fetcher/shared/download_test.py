"""Unit tests for the body-shape predicates and the retry classifier.

The cachetta-backed ``fetch_feed`` / ``fetch_paper`` / ``fetch_html``
functions are exercised through their integration callers (sync, render)
with ``unittest.mock.patch.object`` on ``_http_get`` -- cachetta itself
has its own test suite, no point duplicating it here.
"""

import httpx

from fetcher.shared.download import (
    _is_retryable,
    _looks_like_feed,
    _looks_like_html,
    _looks_like_paper,
)


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://arxiv.org/x")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError(f"HTTP {code}", request=request, response=response)


_GZIP_MAGIC = b"\x1f\x8b"
_HTML = b"<!DOCTYPE html><html><body>503 Service Unavailable</body></html>"


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
