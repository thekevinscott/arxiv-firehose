"""Unit tests for arxiv ID parsing and on-disk path derivation."""

from pathlib import Path

import pytest

from fetcher.shared.paths import (
    id_from_entry_id,
    iter_paper_dirs,
    markdown_path,
    metadata_path,
    paper_dir,
    parse_id,
    version_from_entry_id,
)

DD = Path("/data")


def describe_parse_id():
    def it_parses_modern_ids():
        a = parse_id("2401.12345")
        assert a.raw == "2401.12345"
        assert a.yymm == "2401"
        assert not a.is_legacy
        assert a.slug == "2401.12345"

    def it_strips_a_version_suffix():
        assert parse_id("2401.12345v3").raw == "2401.12345"

    def it_parses_legacy_ids():
        a = parse_id("cs/0501001")
        assert a.raw == "cs/0501001"
        assert a.yymm == "0501"
        assert a.is_legacy
        assert a.slug == "cs_0501001"  # filesystem-safe folder name

    def it_parses_four_digit_modern_ids():
        assert parse_id("0704.0001").yymm == "0704"

    def it_rejects_garbage():
        with pytest.raises(ValueError):
            parse_id("not-an-id")


def describe_paper_paths():
    def it_derives_a_modern_paper_dir():
        assert paper_dir(DD, "2401.12345") == DD / "2401.12345"

    def it_slugifies_a_legacy_paper_dir():
        # the legacy id's '/' is slugified so it stays a single folder
        assert paper_dir(DD, "cs/0501001") == DD / "cs_0501001"

    def it_derives_the_markdown_path():
        assert markdown_path(DD, "2401.12345") == DD / "2401.12345" / "paper.md"

    def it_derives_the_metadata_path():
        assert metadata_path(DD, "2401.12345") == DD / "2401.12345" / "metadata.json"

    def it_slugifies_a_legacy_markdown_path():
        assert markdown_path(DD, "cs/0501001") == DD / "cs_0501001" / "paper.md"


def describe_entry_id_extraction():
    @pytest.mark.parametrize(
        "entry,expected_id,expected_ver",
        [
            ("oai:arXiv.org:2401.12345v2", "2401.12345", 2),
            ("http://arxiv.org/abs/2401.12345v1", "2401.12345", 1),
            ("http://arxiv.org/abs/cs/0501001v1", "cs/0501001", 1),
            ("oai:arXiv.org:2401.12345", "2401.12345", 1),  # version defaults to 1
        ],
    )
    def it_extracts_id_and_version(entry, expected_id, expected_ver):
        assert id_from_entry_id(entry) == expected_id
        assert version_from_entry_id(entry) == expected_ver


def describe_iter_paper_dirs():
    def it_yields_only_folders_with_metadata(tmp_path):
        good = tmp_path / "2401.00001"
        good.mkdir()
        (good / "metadata.json").write_text("{}")
        (tmp_path / "2401.00002").mkdir()  # no metadata.json -> skipped
        (tmp_path / "logs").mkdir()  # reserved -> skipped
        assert list(iter_paper_dirs(tmp_path)) == [good]

    def it_is_empty_for_a_missing_dir(tmp_path):
        assert list(iter_paper_dirs(tmp_path / "nope")) == []
