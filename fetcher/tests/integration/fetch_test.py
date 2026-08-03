"""Integration tests for the ``fetch`` (full pipeline) and ``status`` SDK functions.

``fetch`` is the daily ingest cycle: sync metadata, then embed abstracts.
Paper bodies (PDF/HTML/LaTeX) are never downloaded -- the pipeline pulls
only metadata (which carries the abstract), and the search path only needs
abstracts.

Fixture papers: 2401.00001 .. 2401.00004 (see api_query.xml).
"""

import json
from unittest.mock import patch

from fetcher import embed, fetch, status
from fetcher.commands import embed as embed_mod
from fetcher.commands import fetch as fetch_mod


def describe_fetch():
    def it_syncs_then_embeds_and_returns_a_summary(data_dir, arxiv):
        result = fetch(data_dir)

        assert result["added"] == 4
        assert result["updated"] == 0
        assert result["embed"]["embedded"] == 4
        assert "Papers known:       4" in result["status"]

    def it_downloads_only_metadata(data_dir, arxiv):
        # The ingest cycle must never write paper bodies and must only
        # touch the export API's /api/query endpoint.
        result = fetch(data_dir)

        assert "render" not in result
        assert list(data_dir.rglob("paper.md")) == []
        assert all("/api/query" in url for url in arxiv.calls)

    def it_leaves_a_metadata_mirror_on_disk(data_dir, arxiv):
        fetch(data_dir)

        for n in range(1, 5):
            assert (data_dir / f"2401.0000{n}" / "metadata.json").exists()


def describe_fetch_tracking():
    def it_appends_one_record_per_run_to_runs_jsonl(data_dir, arxiv):
        fetch(data_dir)
        fetch(data_dir)

        lines = (data_dir / "runs.jsonl").read_text().splitlines()
        # One JSON object appended per run -- a durable history to investigate.
        assert len(lines) == 2
        assert all(json.loads(line) for line in lines)

    def it_records_timing_and_counts_for_the_run(data_dir, arxiv):
        fetch(data_dir)

        rec = json.loads((data_dir / "runs.jsonl").read_text().splitlines()[0])
        assert rec["started_at"] <= rec["finished_at"]
        assert rec["duration_s"] >= 0
        assert rec["added"] == 4
        assert rec["updated"] == 0

    def it_does_not_record_a_dry_run(data_dir, arxiv):
        fetch(data_dir, dry_run=True)

        # A dry run touches nothing on disk -- no run history either.
        assert not (data_dir / "runs.jsonl").exists()


def describe_fetch_embed_stage():
    # embed is a stage of fetch, not a separate cron command. These tests
    # pin the wiring: fetch's runs.jsonl carries embed counts, embed
    # populates embeddings.json, and an embed failure logs but does not
    # propagate so the ingest cycle stays green.
    def it_populates_embeddings_file_as_part_of_fetch(data_dir, arxiv):
        fetch(data_dir)

        assert embed_mod.embeddings_path(data_dir).exists()

    def it_records_embed_counts_in_runs_jsonl(data_dir, arxiv):
        fetch(data_dir)

        rec = json.loads((data_dir / "runs.jsonl").read_text().splitlines()[0])
        assert "embed" in rec
        assert rec["embed"]["embedded"] >= 1

    def it_survives_an_embed_stage_failure(data_dir, arxiv):
        # A model-load or disk error in embed must NOT abort the fetch.
        # The pipeline logs, records an error marker in the embed counts,
        # and returns normally so runs.jsonl still gets written and the
        # daily cron stays green. Next run tries again.
        with patch.object(fetch_mod.embed, "run", side_effect=RuntimeError("boom")):
            result = fetch(data_dir)

        assert result["added"] == 4  # sync still ran
        assert result["embed"]["error"] == "boom"
        assert result["embed"]["embedded"] == 0
        # runs.jsonl is still appended.
        assert (data_dir / "runs.jsonl").exists()


def describe_embed_sdk():
    # api.embed is the SDK entry-point for a manual backfill / rerun.
    # Unit tests cover the file-walking logic; here we prove the SDK
    # wrapper wires config-loading and logger setup correctly.
    def it_is_idempotent_across_calls(data_dir, arxiv):
        fetch(data_dir)

        # First embed already ran inside fetch; api.embed again is a no-op.
        counts = embed(data_dir)

        assert counts["embedded"] == 0
        assert counts["total"] >= 1


def describe_status():
    def it_reports_zero_papers_for_an_empty_data_dir(data_dir):
        report = status(data_dir)

        assert "Papers known:       0" in report
        assert "Last sync:          (never)" in report

    def it_reports_counts_after_a_fetch(data_dir, arxiv):
        fetch(data_dir)

        report = status(data_dir)

        assert "Papers known:       4" in report
