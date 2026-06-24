"""Unit tests for the feed body-shape predicate.

``fetch_feed`` itself is exercised through its integration callers with
``unittest.mock.patch.object`` on ``shared.http.http_get``.
"""

from fetcher.commands.fetch.download.feed import looks_like_feed

_HTML = b"<!DOCTYPE html><html><body>503 Service Unavailable</body></html>"


def describe_looks_like_feed():
    def it_accepts_an_xml_declaration():
        assert looks_like_feed(b'<?xml version="1.0"?><rss></rss>') is True

    def it_accepts_a_bare_rss_element():
        assert looks_like_feed(b"<rss/>") is True

    def it_accepts_leading_whitespace():
        assert looks_like_feed(b"\n  <?xml version='1.0'?>") is True

    def it_rejects_an_html_error_page():
        assert looks_like_feed(_HTML) is False
