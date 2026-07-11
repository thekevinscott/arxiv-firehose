"""Unit tests for the bespoke single-paper pull command.

The network seam (``download.fetch_id``) is patched; the happy path
through real feed parsing is covered by the integration suite. Pull is
metadata-only -- rendering never happens here.
"""

import json
import logging
from unittest.mock import patch

import httpx

from fetcher.commands.fetch import pull
from fetcher.shared.config import Config

LOG = logging.getLogger("test")


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://export.arxiv.org/api/query")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError(f"{code}", request=request, response=response)


def describe_run():
    def it_skips_a_paper_whose_metadata_is_already_on_disk(tmp_path):
        d = tmp_path / "2401.00001"
        d.mkdir()
        (d / "metadata.json").write_text(json.dumps({"arxiv_id": "2401.00001"}))

        with patch.object(pull.download, "fetch_id") as fake:
            counts = pull.run(tmp_path, Config(), LOG, ["2401.00001"])

        fake.assert_not_called()
        assert counts["existing"] == 1
        assert counts["pulled"] == 0

    def it_counts_an_invalid_id(tmp_path):
        with patch.object(pull.download, "fetch_id") as fake:
            counts = pull.run(tmp_path, Config(), LOG, ["not-an-id"])

        fake.assert_not_called()
        assert counts["invalid"] == 1

    def it_counts_a_404_as_not_found(tmp_path):
        with patch.object(
            pull.download, "fetch_id", side_effect=_status_error(404)
        ):
            counts = pull.run(tmp_path, Config(), LOG, ["2401.00001"])

        assert counts["not_found"] == 1
        assert not (tmp_path / "2401.00001").exists()

    def it_stops_at_the_first_429(tmp_path):
        calls: list[str] = []

        def fake(arxiv_id):
            calls.append(arxiv_id)
            raise _status_error(429)

        with patch.object(pull.download, "fetch_id", side_effect=fake):
            pull.run(tmp_path, Config(), LOG, ["2401.00001", "2401.00002"])

        assert calls == ["2401.00001"]

    def it_plans_only_on_a_dry_run(tmp_path):
        with patch.object(pull.download, "fetch_id") as fake:
            counts = pull.run(
                tmp_path, Config(), LOG, ["2401.00001"], dry_run=True
            )

        fake.assert_not_called()
        assert counts["pulled"] == 0
        assert not (tmp_path / "2401.00001").exists()
