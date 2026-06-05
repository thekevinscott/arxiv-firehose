"""Integration tests for the ``classify`` SDK function.

classify is **dirsql-query-driven**: it asks the in-process dirsql
database for every (paper, category) pair that lacks a classification
file, then runs the matching coaxed prompt against each. Idempotency is
free -- already-classified pairs do not appear in the query result.

Output layout: one JSON file per (paper, category) at
``<data_dir>/<arxiv_id>/classifications/<category_id>.json``, with shape
``{"output": <bool>, "model": ..., "classified_at": ...}``. The
combined-per-paper ``classification.json`` from the v1 layout is gone.

A fake Ollama-style classifier is injected through the SDK's
``classifier=`` parameter -- never patched. The real ``CoaxedPrompt`` *is*
exercised (it's pure: Jinja2 + Pydantic, no network) using tiny compiled
artifacts written into ``tmp_path``.

Fixture papers (from sync_metadata): 2401.00001 / 00002 / 00003 carry
the word "markdown" in their abstracts; 2401.00004 does not. The
``fake_classifier`` keys off that word so an assertion can show the
per-paper flag decisions all the way through the wiring.
"""

import json
from pathlib import Path

import pytest

from fetcher import classify, sync_metadata

PAPERS = ("2401.00001", "2401.00002", "2401.00003", "2401.00004")
CATS = ("is_about_ml", "is_about_markdown")


def _read_classify_log(data_dir: Path) -> str:
    return (data_dir / "logs" / "classify.log").read_text()


def _classification_path(data_dir: Path, arxiv_id: str, cat: str) -> Path:
    return data_dir / arxiv_id / "classifications" / f"{cat}.json"


def _classification(data_dir: Path, arxiv_id: str, cat: str) -> dict:
    return json.loads(_classification_path(data_dir, arxiv_id, cat).read_text())


def describe_classify():
    def it_writes_one_file_per_paper_per_category(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)

        counts = classify(data_dir_classify, classifier=fake_classifier)

        # 4 papers × 2 categories = 8 missing pairs -> 8 files.
        assert counts["classified"] == 8
        assert counts["failed"] == 0
        for pid in PAPERS:
            for cat in CATS:
                payload = _classification(data_dir_classify, pid, cat)
                assert isinstance(payload["output"], bool)
                assert payload["model"] == "test-model"
                assert "classified_at" in payload

    def it_passes_the_abstract_through_to_the_classifier(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)

        classify(data_dir_classify, classifier=fake_classifier)

        # 00001-00003 abstracts contain "markdown"; 00004's does not.
        # Proves abstract reached prompt and prompt reached classifier.
        assert _classification(data_dir_classify, "2401.00001", "is_about_markdown")["output"] is True
        assert _classification(data_dir_classify, "2401.00002", "is_about_markdown")["output"] is True
        assert _classification(data_dir_classify, "2401.00003", "is_about_markdown")["output"] is True
        assert _classification(data_dir_classify, "2401.00004", "is_about_markdown")["output"] is False
        # is_about_ml stays False everywhere (the fake's other branch).
        for pid in PAPERS:
            assert _classification(data_dir_classify, pid, "is_about_ml")["output"] is False

    def it_only_classifies_missing_pairs(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)
        # Pre-seed one pair to look "already done" -- it must NOT appear
        # in the dirsql work queue, so the classifier sees one fewer call.
        seeded = _classification_path(data_dir_classify, "2401.00001", "is_about_ml")
        seeded.parent.mkdir(parents=True, exist_ok=True)
        seeded.write_text(json.dumps({"output": True, "model": "seeded"}))

        counts = classify(data_dir_classify, classifier=fake_classifier)

        # 7 missing pairs (8 - 1 pre-seeded). The seeded file is left
        # untouched (still says model=seeded).
        assert counts["classified"] == 7
        assert counts["cached"] == 1
        assert _classification(data_dir_classify, "2401.00001", "is_about_ml")["model"] == "seeded"

    def it_no_ops_when_every_pair_is_already_classified(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)
        classify(data_dir_classify, classifier=fake_classifier)

        again = classify(data_dir_classify, classifier=fake_classifier)

        # Every (paper, cat) pair now has a file -> the missing-pairs
        # query returns nothing -> no LLM calls.
        assert again["classified"] == 0
        assert again["cached"] == 8

    def it_reclassifies_every_pair_with_force(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)
        classify(data_dir_classify, classifier=fake_classifier)

        again = classify(data_dir_classify, classifier=fake_classifier, force=True)

        # --force ignores the missing-pairs filter and reruns every pair.
        assert again["classified"] == 8
        assert again["cached"] == 0

    def it_no_ops_when_prompts_dirs_is_empty(
        data_dir, cache_dir, fake_transport, fake_classifier
    ):
        # Default config.toml from conftest carries no [classify] block,
        # so prompts_dirs defaults to []. classify must return clean
        # zeros and write nothing -- the daily cron stays green while
        # labels are still being authored.
        sync_metadata(data_dir, cache_dir, transport=fake_transport)

        counts = classify(data_dir, classifier=fake_classifier)

        assert counts == {"classified": 0, "cached": 0, "skipped": 0, "failed": 0}
        for pid in PAPERS:
            assert not (data_dir / pid / "classifications").exists()
        assert "classify: disabled" in _read_classify_log(data_dir)

    def it_no_ops_when_categories_json_is_missing(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)
        # Remove the fixture-written categories.json; classify must
        # treat "no taxonomy" the same as "no prompts" -- log + zero.
        (data_dir_classify.parent / "categories.json").unlink()

        counts = classify(data_dir_classify, classifier=fake_classifier)

        assert counts == {"classified": 0, "cached": 0, "skipped": 0, "failed": 0}
        for pid in PAPERS:
            assert not (data_dir_classify / pid / "classifications").exists()
        assert "categories.json" in _read_classify_log(data_dir_classify)

    def it_skips_a_category_with_no_matching_prompts_dir(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)
        # Add a third category to the taxonomy without adding a matching
        # prompts_dir. Those (paper, cat) pairs have no classifier, so
        # they are logged and counted as skipped, never written.
        cats_path = data_dir_classify.parent / "categories.json"
        cats = json.loads(cats_path.read_text())
        cats.append({"id": "is_about_dragons", "name": "About dragons"})
        cats_path.write_text(json.dumps(cats))

        counts = classify(data_dir_classify, classifier=fake_classifier)

        # 4 papers × 2 cats with prompts = 8 classified.
        # 4 papers × 1 cat without prompts = 4 skipped.
        assert counts["classified"] == 8
        assert counts["skipped"] == 4
        for pid in PAPERS:
            assert not _classification_path(data_dir_classify, pid, "is_about_dragons").exists()
        assert "no prompt for category is_about_dragons" in _read_classify_log(data_dir_classify)

    def it_makes_no_writes_on_a_dry_run(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)

        counts = classify(data_dir_classify, classifier=fake_classifier, dry_run=True)

        assert counts["classified"] == 0
        for pid in PAPERS:
            assert not (data_dir_classify / pid / "classifications").exists()

    def it_honors_limit(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)

        # limit caps the number of (paper, cat) pairs processed, not
        # papers -- the query already orders by (paper, cat), so limit=3
        # takes the first three missing pairs deterministically.
        counts = classify(data_dir_classify, classifier=fake_classifier, limit=3)

        assert counts["classified"] == 3
        # Determinism: 00001/is_about_markdown + 00001/is_about_ml +
        # 00002/is_about_markdown (lexicographic on category id).
        written = sorted(
            (p.parent.parent.name, p.stem)
            for p in data_dir_classify.glob("*/classifications/*.json")
        )
        assert written == [
            ("2401.00001", "is_about_markdown"),
            ("2401.00001", "is_about_ml"),
            ("2401.00002", "is_about_markdown"),
        ]


def describe_classify_resilience():
    def it_counts_a_failed_pair_without_aborting(
        data_dir_classify, cache_dir, fake_transport
    ):
        # An LLM that raises on the first call must mark that pair
        # failed and continue -- never abort the run partway through.
        from fetcher.commands.classify import Classifier

        calls = {"n": 0}

        def call(prompt, schema):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("ollama exploded")
            field = next(iter(schema.model_json_schema()["properties"]))
            return schema(**{field: False})

        flaky = Classifier(call=call)
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)

        counts = classify(data_dir_classify, classifier=flaky)

        # First pair raises -> failed=1; the other 7 pairs go through.
        assert counts["failed"] == 1
        assert counts["classified"] == 7


def describe_classify_logging():
    def it_logs_each_classification_with_its_output(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)
        classify(data_dir_classify, classifier=fake_classifier)

        log_text = _read_classify_log(data_dir_classify)
        assert "class 2401.00001/is_about_markdown" in log_text
        assert "classify start" in log_text
        assert "classify done" in log_text
