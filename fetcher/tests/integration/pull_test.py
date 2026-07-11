"""Integration tests for the ``pull`` SDK function.

``pull`` is the bespoke single-paper path: given arxiv ids (e.g. the
citations of a paper worth tracing), it fetches each paper's metadata via
an ``id_list=`` export-API query and renders markdown through the exact
render code the daily ingest uses. Unlike sync, it applies no category or
version filter -- whatever the user asked for by id is mirrored.

Fixture papers:
  2401.00001 -- tracked, v1, arxiv HTML available -> markdown via HTML
  2012.09999 -- untracked primary (math.OC), v3, no HTML/e-print/PDF
                -> metadata still written, .no_markdown marker
  2401.99999 -- no fixture at all -> the id_list query 404s -> not_found
"""

import json

from fetcher import pull


def describe_pull():
    def it_writes_metadata_and_markdown_for_a_requested_id(
        data_dir, arxiv, fake_converter
    ):
        result = pull(["2401.00001"], data_dir, converter=fake_converter)

        assert result["pulled"] == 1
        meta = json.loads((data_dir / "2401.00001" / "metadata.json").read_text())
        assert meta["title"] == "A Sample Paper on Gradient Methods"
        assert meta["authors"] == ["Ada Lovelace", "Alan Turing"]
        md = (data_dir / "2401.00001" / "paper.md").read_text()
        assert md.startswith("# Markdown from HTML")

    def it_pulls_a_revised_untracked_paper_that_sync_would_drop(
        data_dir, arxiv, fake_converter
    ):
        result = pull(["2012.09999"], data_dir, converter=fake_converter)

        meta = json.loads((data_dir / "2012.09999" / "metadata.json").read_text())
        assert meta["version"] == 3
        assert meta["primary_category"] == "math.OC"
        # Every render path 404s for this paper -- same marker contract
        # as the daily render stage.
        assert (data_dir / "2012.09999" / ".no_markdown").exists()
        assert result["pulled"] == 0
        assert result["render"]["absent"] == 1

    def it_is_idempotent_on_a_second_pull(data_dir, arxiv, fake_converter):
        pull(["2401.00001"], data_dir, converter=fake_converter)
        before = len(arxiv.calls)

        again = pull(["2401.00001"], data_dir, converter=fake_converter)

        assert again["existing"] == 1
        assert again["pulled"] == 0
        # A re-pull of a mirrored paper costs zero network calls.
        assert len(arxiv.calls) == before

    def it_counts_an_unknown_paper_as_not_found(
        data_dir, arxiv, fake_converter
    ):
        result = pull(["2401.99999"], data_dir, converter=fake_converter)

        assert result["not_found"] == 1
        assert not (data_dir / "2401.99999").exists()

    def it_leaves_the_pulled_paper_visible_to_the_daily_render(
        data_dir, arxiv, fake_converter
    ):
        # A pulled paper is a first-class citizen of the data dir: its
        # folder carries metadata.json, so iter_paper_dirs (and therefore
        # render, status, embed) all see it.
        from fetcher.shared.paths import iter_paper_dirs

        pull(["2401.00001"], data_dir, converter=fake_converter)

        assert [d.name for d in iter_paper_dirs(data_dir)] == ["2401.00001"]
