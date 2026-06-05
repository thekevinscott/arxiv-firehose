"""Unit tests for sync's RSS entry parsing.

arxiv RSS items carry an 'Announce Type' (new / cross / replace / replace-cross)
in the description header. fetcher mirrors only papers first announced this
week, so a non-'new' item -- a cross-list or a revision of an old paper -- must
be dropped before it becomes a metadata folder.
"""

from fetcher.commands.sync import _announce_type, _parse_entry


def describe__announce_type():
    def it_extracts_a_new_announcement():
        assert _announce_type("arXiv:2401.00001 Announce Type: new\nAbstract: x") == "new"

    def it_extracts_a_hyphenated_type():
        summary = "arXiv:2401.00001 Announce Type: replace-cross\nAbstract: x"
        assert _announce_type(summary) == "replace-cross"

    def it_is_empty_when_the_header_is_absent():
        assert _announce_type("just an abstract, no header") == ""


def _entry(announce: str) -> dict:
    """An RSS entry dict the way feedparser hands it to _parse_entry."""
    return {
        "id": "oai:arXiv.org:2401.00001v1",
        "title": "A Sample Paper",
        "summary": f"arXiv:2401.00001 Announce Type: {announce}\nAbstract: hello",
        "author": "Ada Lovelace",
        "tags": [{"term": "cs.LG"}],
        "published": "Mon, 01 Jan 2024 00:00:00 -0500",
    }


def describe__parse_entry():
    def it_keeps_a_new_announcement():
        rec = _parse_entry(_entry("new"))
        assert rec is not None
        assert rec.arxiv_id == "2401.00001"

    def it_drops_a_replacement():
        # A revised version of an existing (often years-old) paper.
        assert _parse_entry(_entry("replace")) is None

    def it_drops_a_replace_cross():
        assert _parse_entry(_entry("replace-cross")) is None

    def it_drops_a_cross_listing():
        assert _parse_entry(_entry("cross")) is None
