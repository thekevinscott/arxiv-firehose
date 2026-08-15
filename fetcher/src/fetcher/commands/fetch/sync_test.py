"""Unit tests for sync's export-API entry parsing.

The API id carries the paper's latest version, and a cat: query matches
cross-lists; the mirror keeps papers *published* (v1-submitted) inside
the sync window whose primary category is tracked -- whatever version
the id carries. Timestamps are re-rendered in the corpus's RFC-2822
shape.
"""

import logging
import time
from datetime import date
from unittest.mock import patch

import httpx

from fetcher.commands.fetch import sync
from fetcher.commands.fetch.sync import _parse_entry, _rfc2822
from fetcher.shared.config import CategoriesConfig, Config, IngestConfig

TRACKED = {"cs.LG"}

EMPTY_FEED = b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://export.arxiv.org/api/query")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError(f"{code}", request=request, response=response)


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

    def it_drops_an_entry_published_before_the_window():
        # An old paper resurfaces in a day slice via a revision; its
        # published (v1) date is what places it outside the window.
        entry = _entry(
            id="http://arxiv.org/abs/2012.09999v3",
            published_parsed=time.struct_time((2020, 12, 18, 12, 0, 0, 4, 353, 0)),
        )
        assert _parse_entry(entry, TRACKED, since=date(2024, 1, 1)) is None

    def it_keeps_a_revised_paper_published_in_the_window():
        # A long lookback sees already-revised papers: the id carries
        # v3 but the v1 submission date sits inside the window. Keep it,
        # pointing at the version the id names.
        entry = _entry(id="http://arxiv.org/abs/2401.00001v3")
        rec = _parse_entry(entry, TRACKED, since=date(2024, 1, 1))
        assert rec is not None
        assert rec.version == 3
        assert rec.pdf_url == "https://arxiv.org/pdf/2401.00001v3"

    def it_keeps_an_entry_with_no_published_date():
        # Without a <published> element the window test can't run; err
        # toward keeping the paper rather than silently dropping it.
        entry = _entry(published_parsed=None)
        rec = _parse_entry(entry, TRACKED, since=date(2024, 1, 1))
        assert rec is not None

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

    # tracked=None is the bespoke-pull mode: any paper the user asks for
    # by id is accepted, whatever its version or primary category.
    def it_keeps_any_version_for_a_bespoke_pull():
        entry = _entry(id="http://arxiv.org/abs/2012.09999v3")
        rec = _parse_entry(entry, None)
        assert rec is not None
        assert rec.version == 3
        assert rec.pdf_url == "https://arxiv.org/pdf/2012.09999v3"

    def it_keeps_an_untracked_primary_for_a_bespoke_pull():
        entry = _entry(
            arxiv_primary_category={"term": "math.OC"},
            tags=[{"term": "math.OC"}],
        )
        rec = _parse_entry(entry, None)
        assert rec is not None
        assert rec.primary_category == "math.OC"


def describe_collect_records():
    def it_filters_entries_published_before_the_window():
        cfg = Config(categories=CategoriesConfig(include=["cs.LG"]),
                     ingest=IngestConfig(backfill_days=0))
        today = date.today()
        feed = f"""\
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v2</id>
    <published>{today.isoformat()}T12:00:00Z</published>
    <title>In Window</title>
    <summary>kept</summary>
    <author><name>Ada</name></author>
    <arxiv:primary_category term="cs.LG"/>
    <category term="cs.LG"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2012.09999v3</id>
    <published>2020-12-18T12:00:00Z</published>
    <title>Out of Window</title>
    <summary>dropped</summary>
    <author><name>Margaret</name></author>
    <arxiv:primary_category term="cs.LG"/>
    <category term="cs.LG"/>
  </entry>
</feed>
""".encode()

        with patch.object(sync.download, "fetch_day", return_value=feed):
            records = sync.collect_records(cfg, logging.getLogger("test"))

        # The window start is the oldest day walked (days[-1]); the
        # revised-but-in-window paper survives with its version, the old
        # paper resurfacing via a revision does not.
        assert set(records) == {"2401.00001"}
        assert records["2401.00001"].version == 2

    def it_defers_older_days_to_the_next_run_on_a_429():
        cfg = Config(ingest=IngestConfig(backfill_days=5))
        calls: list[str] = []

        def fake(categories, day):
            calls.append(day.isoformat())
            if len(calls) >= 2:
                raise _status_error(429)
            return EMPTY_FEED

        with patch.object(sync.download, "fetch_day", side_effect=fake):
            sync.collect_records(cfg, logging.getLogger("test"))
        assert len(calls) == 2

    def it_keeps_walking_past_a_transient_failure():
        cfg = Config(ingest=IngestConfig(backfill_days=5))
        calls: list[str] = []

        def fake(categories, day):
            calls.append(day.isoformat())
            if len(calls) == 2:
                raise _status_error(503)
            return EMPTY_FEED

        with patch.object(sync.download, "fetch_day", side_effect=fake):
            sync.collect_records(cfg, logging.getLogger("test"))
        assert len(calls) == 6


def describe__rfc2822():
    def it_formats_a_struct_time_as_utc_rfc_2822():
        parsed = time.struct_time((2026, 7, 8, 9, 0, 3, 2, 189, 0))
        assert _rfc2822(parsed) == "Wed, 08 Jul 2026 09:00:03 +0000"

    def it_is_empty_for_a_missing_timestamp():
        assert _rfc2822(None) == ""
