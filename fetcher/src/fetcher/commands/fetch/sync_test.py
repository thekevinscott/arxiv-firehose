"""Unit tests for sync's export-API entry parsing.

The API id carries the paper's latest version, and a cat: query matches
cross-lists; the mirror keeps only v1 papers whose primary category is
tracked. Timestamps are re-rendered in the corpus's RFC-2822 shape.
"""

import time

from fetcher.commands.fetch.sync import _parse_entry, _rfc2822

TRACKED = {"cs.LG"}


def _entry(**overrides) -> dict:
    """An Atom entry dict the way feedparser hands it to _parse_entry."""
    entry = {
        "id": "http://arxiv.org/abs/2401.00001v1",
        "title": "A  Sample\n Paper",
        "summary": "hello",
        "authors": [{"name": "Ada Lovelace"}, {"name": "Alan Turing"}],
        "tags": [{"term": "cs.LG"}, {"term": "cs.AI"}],
        "arxiv_primary_category": {"term": "cs.LG"},
        "published_parsed": time.struct_time((2024, 1, 1, 12, 0, 0, 0, 1, 0)),
    }
    entry.update(overrides)
    return entry


def describe__parse_entry():
    def it_keeps_a_v1_paper_in_a_tracked_category():
        rec = _parse_entry(_entry(), TRACKED)
        assert rec is not None
        assert rec.arxiv_id == "2401.00001"
        assert rec.title == "A Sample Paper"
        assert rec.authors == ["Ada Lovelace", "Alan Turing"]
        assert rec.categories == {"cs.LG", "cs.AI"}

    def it_drops_a_revision():
        entry = _entry(id="http://arxiv.org/abs/2012.09999v3")
        assert _parse_entry(entry, TRACKED) is None

    def it_drops_a_cross_list_whose_primary_is_untracked():
        entry = _entry(
            arxiv_primary_category={"term": "math.OC"},
            tags=[{"term": "math.OC"}, {"term": "cs.LG"}],
        )
        assert _parse_entry(entry, TRACKED) is None

    def it_falls_back_to_the_first_tag_for_the_primary():
        entry = _entry()
        del entry["arxiv_primary_category"]
        rec = _parse_entry(entry, TRACKED)
        assert rec is not None
        assert rec.primary_category == "cs.LG"

    def it_drops_garbage_ids():
        assert _parse_entry(_entry(id="nonsense"), TRACKED) is None

    def it_renders_published_in_rfc_2822():
        rec = _parse_entry(_entry(), TRACKED)
        assert rec is not None
        assert rec.announced_at == "Mon, 01 Jan 2024 12:00:00 +0000"


def describe__rfc2822():
    def it_formats_a_struct_time_as_utc_rfc_2822():
        parsed = time.struct_time((2026, 7, 8, 9, 0, 3, 2, 189, 0))
        assert _rfc2822(parsed) == "Wed, 08 Jul 2026 09:00:03 +0000"

    def it_is_empty_for_a_missing_timestamp():
        assert _rfc2822(None) == ""
