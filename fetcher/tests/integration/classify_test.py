"""Integration tests for the ``classify`` SDK function.

A fake Ollama-style classifier is injected through the SDK's ``classifier=``
parameter -- never patched. The real ``CoaxedPrompt`` *is* exercised
(it's pure: Jinja2 + Pydantic, no network) using a tiny compiled artifact
written into ``tmp_path``.

Fixture papers (from sync_metadata): 2401.00001 / 00002 / 00003 carry the
word "markdown" in their abstracts; 2401.00004 does not. The shared
``fake_classifier`` keys off that word so an assertion shows the per-paper
flag decisions all the way through the wiring.
"""

import json
from pathlib import Path

import pytest

from fetcher import classify, sync_metadata


def _read_classify_log(data_dir: Path) -> str:
    return (data_dir / "logs" / "classify.log").read_text()


def _classification(data_dir: Path, arxiv_id: str) -> dict:
    return json.loads((data_dir / arxiv_id / "classification.json").read_text())


def describe_classify():
    def it_writes_classification_json_per_paper(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)

        counts = classify(data_dir_classify, classifier=fake_classifier)

        assert counts["classified"] == 4
        assert counts["failed"] == 0
        # Each paper carries every configured flag, named exactly as the
        # CoaxedPrompt's response_format Pydantic field.
        for pid in ("2401.00001", "2401.00002", "2401.00003", "2401.00004"):
            payload = _classification(data_dir_classify, pid)
            assert payload["arxiv_id"] == pid
            assert set(payload["flags"]) == {"is_about_ml", "is_about_markdown"}
            assert payload["model"] == "test-model"
            assert "classified_at" in payload

    def it_passes_the_abstract_through_to_the_classifier(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)

        classify(data_dir_classify, classifier=fake_classifier)

        # 00001-00003 abstracts contain "markdown"; 00004's does not. The
        # fake classifier turns that into is_about_markdown true/false per
        # paper, proving the abstract reached the prompt and the prompt
        # reached the classifier.
        assert _classification(data_dir_classify, "2401.00001")["flags"]["is_about_markdown"] is True
        assert _classification(data_dir_classify, "2401.00002")["flags"]["is_about_markdown"] is True
        assert _classification(data_dir_classify, "2401.00003")["flags"]["is_about_markdown"] is True
        assert _classification(data_dir_classify, "2401.00004")["flags"]["is_about_markdown"] is False
        # is_about_ml stays False everywhere (the fake's other branch).
        for pid in ("2401.00001", "2401.00002", "2401.00003", "2401.00004"):
            assert _classification(data_dir_classify, pid)["flags"]["is_about_ml"] is False

    def it_skips_a_paper_that_already_has_a_classification(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)
        first = classify(data_dir_classify, classifier=fake_classifier)

        again = classify(data_dir_classify, classifier=fake_classifier)

        # First pass classifies every paper; second pass finds them all on
        # disk and treats them as cached. Idempotent reruns keep the cron
        # cheap once a paper has been classified.
        assert first["classified"] == 4
        assert again["classified"] == 0
        assert again["cached"] == 4

    def it_reclassifies_with_force(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)
        classify(data_dir_classify, classifier=fake_classifier)

        again = classify(data_dir_classify, classifier=fake_classifier, force=True)

        # --force re-runs the classifier against every paper, overwriting
        # the existing classification.json.
        assert again["classified"] == 4
        assert again["cached"] == 0

    def it_no_ops_when_prompts_dirs_is_empty(
        data_dir, cache_dir, fake_transport, fake_classifier
    ):
        # Default config.toml from conftest carries no [classify] block, so
        # prompts_dirs defaults to []. classify must return clean zeros and
        # write nothing -- the daily cron stays green while labels are
        # still being authored.
        sync_metadata(data_dir, cache_dir, transport=fake_transport)

        counts = classify(data_dir, classifier=fake_classifier)

        assert counts == {"classified": 0, "cached": 0, "skipped": 0, "failed": 0}
        for pid in ("2401.00001", "2401.00002", "2401.00003", "2401.00004"):
            assert not (data_dir / pid / "classification.json").exists()
        assert "classify: disabled" in _read_classify_log(data_dir)

    def it_skips_a_paper_with_corrupt_metadata(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)
        bad = data_dir_classify / "2401.09998"
        bad.mkdir()
        (bad / "metadata.json").write_text("{ not valid json")

        counts = classify(data_dir_classify, classifier=fake_classifier)

        # A single unreadable metadata.json must not abort the run -- log
        # it, count it, classify the rest.
        assert counts["classified"] == 4
        assert counts["skipped"] == 1
        assert "skip 2401.09998: bad metadata.json" in _read_classify_log(data_dir_classify)

    def it_makes_no_writes_on_a_dry_run(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)

        counts = classify(data_dir_classify, classifier=fake_classifier, dry_run=True)

        assert counts["classified"] == 0
        for pid in ("2401.00001", "2401.00002", "2401.00003", "2401.00004"):
            assert not (data_dir_classify / pid / "classification.json").exists()

    def it_honors_limit(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)

        counts = classify(data_dir_classify, classifier=fake_classifier, limit=2)

        # iter_paper_dirs yields by sorted arxiv id, so limit=2 takes the
        # first two papers deterministically.
        assert counts["classified"] == 2
        assert (data_dir_classify / "2401.00001" / "classification.json").exists()
        assert (data_dir_classify / "2401.00002" / "classification.json").exists()
        assert not (data_dir_classify / "2401.00003" / "classification.json").exists()


def describe_classify_resilience():
    def it_counts_a_failed_paper_without_aborting(
        data_dir_classify, cache_dir, fake_transport
    ):
        # An LLM that raises must mark the paper failed and continue --
        # never abort the run partway through.
        from fetcher.classify import Classifier

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

        # First paper × first classifier raises -> that paper counts as
        # failed and writes no classification.json. The remaining three
        # papers go through both classifiers cleanly.
        assert counts["failed"] == 1
        assert counts["classified"] == 3
        assert not (data_dir_classify / "2401.00001" / "classification.json").exists()
        for pid in ("2401.00002", "2401.00003", "2401.00004"):
            assert (data_dir_classify / pid / "classification.json").exists()


def describe_classify_logging():
    def it_logs_each_paper_with_its_flags(
        data_dir_classify, cache_dir, fake_transport, fake_classifier
    ):
        sync_metadata(data_dir_classify, cache_dir, transport=fake_transport)
        classify(data_dir_classify, classifier=fake_classifier)

        log_text = _read_classify_log(data_dir_classify)
        assert "class 2401.00001:" in log_text
        assert "is_about_markdown" in log_text
        assert "classify start" in log_text
        assert "classify done" in log_text
