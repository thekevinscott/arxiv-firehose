"""Unit tests for ``coax.hash_labels`` + cache behavior of ``coax.run``.

The hash function is the cache's correctness contract: same content +
same knobs -> same hash; different content OR different knobs -> different
hash. ``run`` is exercised against the real ``coaxer.compiler.distill``
(it's pure: Jinja2 + Pydantic, no network) with optimizer=None.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from fetcher.commands import coax


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
    return logging.getLogger("coax_test")


def describe_hash_labels():
    def it_is_deterministic(labels_dir):
        a = coax.hash_labels(labels_dir, optimizer=None, output_name="is_ml")
        b = coax.hash_labels(labels_dir, optimizer=None, output_name="is_ml")
        assert a == b

    def it_changes_when_a_file_changes(labels_dir):
        before = coax.hash_labels(labels_dir, optimizer=None, output_name="is_ml")
        (labels_dir / "yes" / "abstract.txt").write_text("Different content.")
        after = coax.hash_labels(labels_dir, optimizer=None, output_name="is_ml")
        assert before != after

    def it_changes_when_a_file_is_added(labels_dir):
        before = coax.hash_labels(labels_dir, optimizer=None, output_name="is_ml")
        extra = labels_dir / "maybe"
        extra.mkdir()
        (extra / "abstract.txt").write_text("An ambiguous paper.")
        (extra / "record.json").write_text("{}")
        after = coax.hash_labels(labels_dir, optimizer=None, output_name="is_ml")
        assert before != after

    def it_changes_when_output_name_changes(labels_dir):
        a = coax.hash_labels(labels_dir, optimizer=None, output_name="is_ml")
        b = coax.hash_labels(labels_dir, optimizer=None, output_name="is_about_ml")
        assert a != b

    def it_changes_when_optimizer_changes(labels_dir):
        none = coax.hash_labels(labels_dir, optimizer=None, output_name="is_ml")
        gepa = coax.hash_labels(labels_dir, optimizer="gepa", output_name="is_ml")
        assert none != gepa


def describe_run():
    def it_compiles_on_a_cache_miss(labels_dir, cache_root, out_dir, log):
        result = coax.run(
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
        first = coax.run(
            labels_dir, out_dir, log,
            output_name="is_ml", cache_root=cache_root,
        )
        meta_path = cache_root / first["hash"] / "meta.json"
        first_meta = json.loads(meta_path.read_text())

        # Second run, fresh out_dir -- cache key matches, so distill is
        # NOT re-invoked. Proof: the cache's meta.json compiled_at must
        # be unchanged (a fresh distill would rewrite it with `now`).
        out2 = out_dir.parent / "out2"
        second = coax.run(
            labels_dir, out2, log,
            output_name="is_ml", cache_root=cache_root,
        )

        assert second["source"] == "cache"
        assert second["hash"] == first["hash"]
        assert json.loads(meta_path.read_text())["compiled_at"] == first_meta["compiled_at"]
        # And the artifact reached out2 too, copied from cache.
        assert (out2 / "prompt.jinja").is_file()

    def it_recompiles_when_a_label_changes(labels_dir, cache_root, out_dir, log):
        first = coax.run(
            labels_dir, out_dir, log,
            output_name="is_ml", cache_root=cache_root,
        )
        # Mutate a label; the cache key flips and a fresh compile runs.
        (labels_dir / "yes" / "abstract.txt").write_text("New content here.")

        second = coax.run(
            labels_dir, out_dir, log,
            output_name="is_ml", cache_root=cache_root,
        )

        assert second["source"] == "fresh"
        assert second["hash"] != first["hash"]

    def it_rejects_gepa_without_a_model(labels_dir, cache_root, out_dir, log):
        # --optimizer gepa requires --model; misconfiguration must error
        # loudly rather than silently fall back.
        with pytest.raises(ValueError, match="--model is required"):
            coax.run(
                labels_dir, out_dir, log,
                optimizer="gepa", output_name="is_ml", cache_root=cache_root,
            )
