"""Integration tests for the ``sync_metadata`` SDK function.

sync pulls from the arxiv export API in per-day slices (submittedDate
windows), not RSS: each slice is an immutable query whose response can be
cached ~forever, so a missed day is re-fetched by any later run instead of
being lost when the RSS window rolls over.
"""

import json

from fetcher import sync_metadata


def describe_sync_metadata():
    def it_writes_a_metadata_folder_per_api_entry(data_dir, arxiv):
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

        # The response carries a v3 entry (arXiv:2012.09999v3, first
        # announced in 2020). fetcher mirrors only first versions, so it
        # never becomes a metadata folder.
        assert added == 4
        assert not (data_dir / "2012.09999").exists()

    def it_drops_a_cross_list_whose_primary_is_untracked(data_dir, arxiv):
        sync_metadata(data_dir)

        # 2401.00005 matches the cs.LG query via a cross-list, but its
        # primary category is math.OC -- outside the tracked set. The
        # mirror keeps only papers that *live* in a tracked category.
        assert not (data_dir / "2401.00005").exists()

    def it_records_the_publication_date_under_announced_at(data_dir, arxiv):
        sync_metadata(data_dir)

        meta = json.loads((data_dir / "2401.00001" / "metadata.json").read_text())
        # The API's <published> timestamp, rendered in the same RFC-2822
        # format the RSS era wrote, so the existing corpus and the papers
        # view's strptime keep working unchanged.
        assert meta["announced_at"] == "Mon, 01 Jan 2024 12:00:00 +0000"

    def it_records_the_arxiv_html_url(data_dir, arxiv):
        sync_metadata(data_dir)

        meta = json.loads((data_dir / "2401.00001" / "metadata.json").read_text())
        assert meta["html_url"] == "https://arxiv.org/html/2401.00001v1"

    def it_queries_one_day_slice_per_day_in_the_window(data_dir, arxiv):
        sync_metadata(data_dir)

        # backfill_days = 2 in the fixture config: today plus two days
        # back, one API call each, every one a distinct submittedDate
        # window over the tracked categories.
        assert len(arxiv.calls) == 3
        assert all("export.arxiv.org/api/query" in url for url in arxiv.calls)
        assert all("submittedDate" in url for url in arxiv.calls)
        assert len(set(arxiv.calls)) == 3

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

    def it_does_not_rewrite_existing_metadata(data_dir, arxiv):
        sync_metadata(data_dir)
        before = (data_dir / "2401.00001" / "metadata.json").read_text()

        added, updated = sync_metadata(data_dir)

        # A daily run re-reads a ~90-day window; papers already mirrored
        # must not be rewritten (no churn on synced_at, no needless I/O).
        assert (added, updated) == (0, 4)
        assert (data_dir / "2401.00001" / "metadata.json").read_text() == before
