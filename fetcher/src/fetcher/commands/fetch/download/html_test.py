"""Unit tests for the html body-shape predicate.

``fetch_html`` itself is exercised through its integration callers with
``unittest.mock.patch.object`` on ``shared.http.http_get``.
"""

from fetcher.commands.fetch.download.html import looks_like_html


def describe_looks_like_html():
    def it_accepts_a_doctype_declaration():
        assert looks_like_html(b"<!DOCTYPE html><html><body>x</body></html>") is True

    def it_accepts_a_bare_html_element():
        assert looks_like_html(b"<html lang='en'><body/></html>") is True

    def it_accepts_leading_whitespace_and_mixed_case():
        assert looks_like_html(b"\n  <!doctype HTML>\n<HTML>") is True

    def it_rejects_a_feed():
        assert looks_like_html(b'<?xml version="1.0"?><rss></rss>') is False

    def it_rejects_a_pdf():
        assert looks_like_html(b"%PDF-1.7\n...") is False

    def it_rejects_empty_bytes():
        assert looks_like_html(b"") is False
