"""Integration tests for the ``pull`` SDK function.

``pull`` is the bespoke single-paper path: given arxiv ids (e.g. the
citations of a paper worth tracing), it fetches each paper's metadata via
an ``id_list=`` export-API query. Unlike sync, it applies no category or
version filter -- whatever the user asked for by id is mirrored.

Pull is metadata-only, like the daily ingest: search/classify/embed need
abstracts, not paper bodies. Markdown for pulled papers arrives when
render is explicitly invoked (``fetcher render`` / ``POST /render``).

Fixture papers:
  2401.00001 -- tracked, v1
  2012.09999 -- untracked primary (math.OC), v3: sync would drop it twice
  2401.99999 -- no fixture at all -> the id_list query 404s -> not_found
"""

import json

from fetcher import pull


def describe_pull():
    def it_writes_metadata_for_a_requested_id(data_dir, arxiv):
        result = pull(["2401.00001"], data_dir)

        assert result["pulled"] == 1
        meta = json.loads((data_dir / "2401.00001" / "metadata.json").read_text())
        assert meta["title"] == "A Sample Paper on Gradient Methods"
        assert meta["authors"] == ["Ada Lovelace", "Alan Turing"]

    def it_does_not_render_markdown(data_dir, arxiv):
        pull(["2401.00001"], data_dir)

        assert not (data_dir / "2401.00001" / "paper.md").exists()
        assert not (data_dir / "2401.00001" / ".no_markdown").exists()
        # Exactly one network call: the id_list metadata query. No paper
        # bodies (html / e-print / pdf) are touched.
        assert len(arxiv.calls) == 1
        assert "id_list=2401.00001" in arxiv.calls[0]

    def it_pulls_a_revised_untracked_paper_that_sync_would_drop(
        data_dir, arxiv
    ):
        result = pull(["2012.09999"], data_dir)

        assert result["pulled"] == 1
        meta = json.loads((data_dir / "2012.09999" / "metadata.json").read_text())
        assert meta["version"] == 3
        assert meta["primary_category"] == "math.OC"

    def it_is_idempotent_on_a_second_pull(data_dir, arxiv):
        pull(["2401.00001"], data_dir)
        before = len(arxiv.calls)

        again = pull(["2401.00001"], data_dir)

        assert again["existing"] == 1
        assert again["pulled"] == 0
        # A re-pull of a mirrored paper costs zero network calls.
        assert len(arxiv.calls) == before

    def it_counts_an_unknown_paper_as_not_found(data_dir, arxiv):
        result = pull(["2401.99999"], data_dir)

        assert result["not_found"] == 1
        assert not (data_dir / "2401.99999").exists()

    def it_leaves_the_pulled_paper_visible_to_the_rest_of_the_pipeline(
        data_dir, arxiv
    ):
        # A pulled paper is a first-class citizen of the data dir: its
        # folder carries metadata.json, so iter_paper_dirs (and therefore
        # render, status, embed) all see it.
        from fetcher.shared.paths import iter_paper_dirs

        pull(["2401.00001"], data_dir)

        assert [d.name for d in iter_paper_dirs(data_dir)] == ["2401.00001"]
