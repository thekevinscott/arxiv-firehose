"""Integration tests for the ``sync_metadata`` SDK function."""

import json

from fetcher import sync_metadata


def describe_sync_metadata():
    def it_writes_a_metadata_folder_per_feed_entry(data_dir, arxiv):
        added, updated = sync_metadata(data_dir)

        assert (added, updated) == (4, 0)
        meta = json.loads((data_dir / "2401.00001" / "metadata.json").read_text())
        assert meta["title"] == "A Sample Paper on Gradient Methods"
        assert meta["authors"] == ["Ada Lovelace", "Alan Turing"]
        assert meta["arxiv_id"] == "2401.00001"
        assert "cs.AI" in meta["categories"]
        assert meta["abstract"].startswith("We present a sample paper")

    def it_drops_a_replacement_of_an_old_paper(data_dir, arxiv):
        added, _ = sync_metadata(data_dir)

        # The feed carries a 'replace' item (arXiv:2012.09999, first announced
        # in 2020). fetcher mirrors only first announcements, so it never
        # becomes a metadata folder -- the count stays 4, not 5.
        assert added == 4
        assert not (data_dir / "2012.09999").exists()

    def it_records_the_announcement_date_under_announced_at(
        data_dir, arxiv
    ):
        sync_metadata(data_dir)

        meta = json.loads((data_dir / "2401.00001" / "metadata.json").read_text())
        # arxiv RSS pubDate is the announcement date, not a submission date;
        # the field is named for what it actually carries.
        assert "announced_at" in meta
        assert "submitted_at" not in meta

    def it_records_the_arxiv_html_url(data_dir, arxiv):
        sync_metadata(data_dir)

        meta = json.loads((data_dir / "2401.00001" / "metadata.json").read_text())
        # fetch's primary path converts arxiv's native HTML to markdown.
        assert meta["html_url"] == "https://arxiv.org/html/2401.00001v1"

    def it_fetches_each_tracked_feed_exactly_once(data_dir, arxiv):
        sync_metadata(data_dir)

        assert arxiv.calls == ["https://rss.arxiv.org/rss/cs.LG"]

    def it_reuses_the_cached_feed_on_a_same_day_rerun(data_dir, arxiv):
        sync_metadata(data_dir)
        after_first = list(arxiv.calls)

        added, updated = sync_metadata(data_dir)

        # cachetta caches the feed for a day: no second arxiv request.
        assert arxiv.calls == after_first
        assert (added, updated) == (0, 4)

    def it_writes_a_last_sync_summary(data_dir, arxiv):
        sync_metadata(data_dir)

        summary = json.loads((data_dir / "last_sync.json").read_text())
        assert summary["papers_added"] == 4
        assert summary["categories"] == ["cs.LG"]

    def it_makes_no_request_on_a_dry_run(data_dir, arxiv):
        added, updated = sync_metadata(
            data_dir, dry_run=True
        )

        assert (added, updated) == (0, 0)
        assert arxiv.calls == []
        assert not (data_dir / "2401.00001").exists()

    def it_honors_the_limit(data_dir, arxiv):
        added, _ = sync_metadata(
            data_dir, limit=1
        )

        assert added == 1
