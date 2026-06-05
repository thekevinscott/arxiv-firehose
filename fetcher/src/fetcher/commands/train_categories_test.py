"""Unit tests for ``train_categories``: hash, single-compile, iterator.

The hash function is the cache's correctness contract: same content +
same knobs -> same hash; different content OR different knobs -> different
hash. ``compile_one`` is exercised against the real
``coaxer.compiler.distill`` (pure: Jinja2 + Pydantic, no network) with
optimizer=None. ``run`` is the iterator that walks a labels root and
calls ``compile_one`` per category subdir.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from fetcher.commands import train_categories as tc


def _make_labels(root: Path) -> Path:
    """Tiny labels dir: one positive + one negative example, plus a schema.

    Mirrors the layout fetcher/labels/is-*/{example}/record.json carries,
    so the real coaxer.compiler.distill accepts it.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "_schema.json").write_text(json.dumps({
        "inputs": {"abstract": {"type": "str", "desc": "Paper abstract"}},
        "output": {"type": "bool", "desc": "True iff about ML"},
    }))
    pos = root / "yes"
    pos.mkdir()
    (pos / "abstract.txt").write_text("A paper about machine learning.")
    (pos / "record.json").write_text(json.dumps({
        "id": "yes",
        "inputs": {"abstract": "abstract.txt"},
        "output": True,
    }))
    neg = root / "no"
    neg.mkdir()
    (neg / "abstract.txt").write_text("A paper about gardening.")
    (neg / "record.json").write_text(json.dumps({
        "id": "no",
        "inputs": {"abstract": "abstract.txt"},
        "output": False,
    }))
    return root


@pytest.fixture
def labels_dir(tmp_path: Path) -> Path:
    return _make_labels(tmp_path / "labels")


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "out"


@pytest.fixture
def log() -> logging.Logger:
    return logging.getLogger("train_categories_test")


def describe_output_name_for():
    def it_swaps_hyphens_for_underscores():
        assert tc.output_name_for("is-about-control") == "is_about_control"

    def it_passes_through_a_name_with_no_hyphens():
        assert tc.output_name_for("survey") == "survey"


def describe_hash_labels():
    def it_is_deterministic(labels_dir):
        a = tc.hash_labels(labels_dir, optimizer=None, output_name="is_ml")
        b = tc.hash_labels(labels_dir, optimizer=None, output_name="is_ml")
        assert a == b

    def it_changes_when_a_file_changes(labels_dir):
        before = tc.hash_labels(labels_dir, optimizer=None, output_name="is_ml")
        (labels_dir / "yes" / "abstract.txt").write_text("Different content.")
        after = tc.hash_labels(labels_dir, optimizer=None, output_name="is_ml")
        assert before != after

    def it_changes_when_a_file_is_added(labels_dir):
        before = tc.hash_labels(labels_dir, optimizer=None, output_name="is_ml")
        extra = labels_dir / "maybe"
        extra.mkdir()
        (extra / "abstract.txt").write_text("An ambiguous paper.")
        (extra / "record.json").write_text("{}")
        after = tc.hash_labels(labels_dir, optimizer=None, output_name="is_ml")
        assert before != after

    def it_changes_when_output_name_changes(labels_dir):
        a = tc.hash_labels(labels_dir, optimizer=None, output_name="is_ml")
        b = tc.hash_labels(labels_dir, optimizer=None, output_name="is_about_ml")
        assert a != b

    def it_changes_when_optimizer_changes(labels_dir):
        none = tc.hash_labels(labels_dir, optimizer=None, output_name="is_ml")
        gepa = tc.hash_labels(labels_dir, optimizer="gepa", output_name="is_ml")
        assert none != gepa


def describe_compile_one():
    def it_compiles_on_a_cache_miss(labels_dir, cache_root, out_dir, log):
        result = tc.compile_one(
            labels_dir, out_dir, log,
            output_name="is_ml", cache_root=cache_root,
        )

        assert result["source"] == "fresh"
        # The artifact ended up in both the cache (under its hash) and in
        # out_dir -- callers only ever read from out_dir.
        assert (cache_root / result["hash"] / "prompt.jinja").is_file()
        assert (out_dir / "prompt.jinja").is_file()
        assert (out_dir / "meta.json").is_file()

    def it_serves_a_second_run_from_the_cache(
        labels_dir, cache_root, out_dir, log
    ):
        # First run populates the cache.
        first = tc.compile_one(
            labels_dir, out_dir, log,
            output_name="is_ml", cache_root=cache_root,
        )
        meta_path = cache_root / first["hash"] / "meta.json"
        first_meta = json.loads(meta_path.read_text())

        # Second run, fresh out_dir -- cache key matches, so distill is
        # NOT re-invoked. Proof: the cache's meta.json compiled_at must
        # be unchanged (a fresh distill would rewrite it with `now`).
        out2 = out_dir.parent / "out2"
        second = tc.compile_one(
            labels_dir, out2, log,
            output_name="is_ml", cache_root=cache_root,
        )

        assert second["source"] == "cache"
        assert second["hash"] == first["hash"]
        assert json.loads(meta_path.read_text())["compiled_at"] == first_meta["compiled_at"]
        # And the artifact reached out2 too, copied from cache.
        assert (out2 / "prompt.jinja").is_file()

    def it_recompiles_when_a_label_changes(labels_dir, cache_root, out_dir, log):
        first = tc.compile_one(
            labels_dir, out_dir, log,
            output_name="is_ml", cache_root=cache_root,
        )
        # Mutate a label; the cache key flips and a fresh compile runs.
        (labels_dir / "yes" / "abstract.txt").write_text("New content here.")

        second = tc.compile_one(
            labels_dir, out_dir, log,
            output_name="is_ml", cache_root=cache_root,
        )

        assert second["source"] == "fresh"
        assert second["hash"] != first["hash"]

    def it_rejects_gepa_without_a_model(labels_dir, cache_root, out_dir, log):
        # --optimizer gepa requires --model; misconfiguration must error
        # loudly rather than silently fall back.
        with pytest.raises(ValueError, match="--model is required"):
            tc.compile_one(
                labels_dir, out_dir, log,
                optimizer="gepa", output_name="is_ml", cache_root=cache_root,
            )


def describe_discover_categories():
    def it_finds_subdirs_with_a_schema(tmp_path):
        labels = tmp_path / "labels"
        _make_labels(labels / "is-about-control")
        _make_labels(labels / "is-survey")
        # A non-category sibling: no _schema.json. README.md, history.jsonl
        # and similar live here and must not be treated as categories.
        (labels / "README.md").write_text("# labels")
        misc = labels / "notes"
        misc.mkdir()
        (misc / "draft.md").write_text("scratch")

        found = tc.discover_categories(labels)

        assert [p.name for p in found] == ["is-about-control", "is-survey"]

    def it_returns_empty_for_a_missing_dir(tmp_path):
        assert tc.discover_categories(tmp_path / "missing") == []


def describe_run():
    def it_compiles_every_category_under_labels(tmp_path, cache_root, log):
        labels = tmp_path / "labels"
        _make_labels(labels / "is-about-control")
        _make_labels(labels / "is-survey")
        prompts = tmp_path / "prompts"

        results = tc.run(labels, prompts, log, cache_root=cache_root)

        assert sorted(results) == ["is-about-control", "is-survey"]
        assert (prompts / "is-about-control" / "prompt.jinja").is_file()
        assert (prompts / "is-survey" / "prompt.jinja").is_file()
        # Each category's output_name is derived from its dirname:
        # is-about-control -> is_about_control.
        meta = json.loads((prompts / "is-about-control" / "meta.json").read_text())
        assert meta["output_name"] == "is_about_control"

    def it_serves_a_second_run_from_the_cache(tmp_path, cache_root, log):
        labels = tmp_path / "labels"
        _make_labels(labels / "is-survey")
        prompts = tmp_path / "prompts"
        tc.run(labels, prompts, log, cache_root=cache_root)

        # Re-run into a fresh prompts root; the cache hit means no
        # recompile. Source = "cache" proves it.
        prompts2 = tmp_path / "prompts2"
        results = tc.run(labels, prompts2, log, cache_root=cache_root)

        assert results["is-survey"]["source"] == "cache"

    def it_returns_empty_when_labels_root_has_no_categories(tmp_path, cache_root, log):
        empty = tmp_path / "empty-labels"
        empty.mkdir()
        results = tc.run(empty, tmp_path / "prompts", log, cache_root=cache_root)
        assert results == {}
