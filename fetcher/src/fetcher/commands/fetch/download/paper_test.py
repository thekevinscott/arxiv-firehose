"""Unit tests for the paper body-shape predicate.

``fetch_paper`` itself is exercised through its integration callers with
``unittest.mock.patch.object`` on ``shared.http.http_get``.
"""

from fetcher.commands.fetch.download.paper import GZIP_MAGIC, looks_like_paper

_HTML = b"<!DOCTYPE html><html><body>503 Service Unavailable</body></html>"


def describe_looks_like_paper():
    def it_accepts_a_pdf():
        assert looks_like_paper(b"%PDF-1.7\n...") is True

    def it_accepts_a_gzip_archive():
        assert looks_like_paper(GZIP_MAGIC + b"rest of tarball") is True

    def it_rejects_an_html_error_page():
        assert looks_like_paper(_HTML) is False

    def it_rejects_empty_bytes():
        assert looks_like_paper(b"") is False
