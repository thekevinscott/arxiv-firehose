"""Unit tests for the export-API day-slice fetcher."""

from datetime import date

from fetcher.commands.fetch.download.api import (
    id_query_url,
    is_settled,
    looks_like_feed,
    query_url,
)


def describe_query_url():
    def it_builds_a_day_windowed_category_query():
        url = query_url(("cs.LG", "cs.AI"), date(2026, 7, 8))
        assert url.startswith("https://export.arxiv.org/api/query?search_query=")
        assert "cat%3Acs.LG%20OR%20cat%3Acs.AI" in url
        assert "submittedDate%3A%5B202607080000%20TO%20202607082359%5D" in url
        assert url.endswith("&start=0&max_results=2000")


def describe_id_query_url():
    def it_builds_a_single_id_query():
        assert id_query_url("2401.00001") == (
            "https://export.arxiv.org/api/query"
            "?id_list=2401.00001&start=0&max_results=1"
        )

    def it_keeps_a_legacy_id_slash_literal():
        # The export API accepts legacy ids verbatim: id_list=cs/0501001.
        assert "id_list=cs/0501001" in id_query_url("cs/0501001")


def describe_is_settled():
    def it_settles_a_day_older_than_the_threshold():
        assert is_settled(date(2026, 7, 4), today=date(2026, 7, 10))

    def it_settles_a_day_exactly_at_the_threshold():
        assert is_settled(date(2026, 7, 7), today=date(2026, 7, 10))

    def it_keeps_the_trailing_days_recent():
        assert not is_settled(date(2026, 7, 8), today=date(2026, 7, 10))
        assert not is_settled(date(2026, 7, 10), today=date(2026, 7, 10))


def describe_looks_like_feed():
    def it_accepts_an_xml_prologue():
        assert looks_like_feed(b'<?xml version="1.0"?><feed></feed>')

    def it_accepts_a_bare_atom_feed():
        assert looks_like_feed(b'<feed xmlns="http://www.w3.org/2005/Atom">')

    def it_rejects_an_html_error_page():
        assert not looks_like_feed(b"<html><body>503</body></html>")
