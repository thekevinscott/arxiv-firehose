"""Integration tests for the ``classify`` SDK function.

classify is **dirsql-query-driven**: it materializes a categories index
under ``<ROOT>/categories/`` from ``[classify] prompts_dirs``, then asks
dirsql for every (paper, category) pair that lacks a classification
file (one SQL ``CROSS JOIN ... LEFT JOIN ... WHERE NULL``). Idempotency
is free -- already-classified pairs never appear in the query result.
Re-running the LLM for an existing pair is also free at the network
layer: ``http_classifier`` sits behind cachetta keyed by
``(model, prompt, schema)``.

Output layout: one JSON file per (paper, category) at
``<data_dir>/<arxiv_id>/classifications/<category_id>.json``, with shape
``{"output": <bool>, "model": ..., "classified_at": ...}``.

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
        data_dir_classify, arxiv, fake_classifier
    ):
        sync_metadata(data_dir_classify)

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
        data_dir_classify, arxiv, fake_classifier
    ):
        sync_metadata(data_dir_classify)

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
        data_dir_classify, arxiv, fake_classifier
    ):
        sync_metadata(data_dir_classify)
        # Pre-seed one pair to look "already done" -- it must NOT appear
        # in the dirsql missing-pairs query, so the classifier sees one
        # fewer call and the seeded file is left untouched.
        seeded = _classification_path(data_dir_classify, "2401.00001", "is_about_ml")
        seeded.parent.mkdir(parents=True, exist_ok=True)
        seeded.write_text(json.dumps({"output": True, "model": "seeded"}))

        counts = classify(data_dir_classify, classifier=fake_classifier)

        # 7 missing pairs (8 - 1 pre-seeded). The seeded file's model
        # field still says "seeded" -- classify never re-wrote it.
        assert counts["classified"] == 7
        assert _classification(data_dir_classify, "2401.00001", "is_about_ml")["model"] == "seeded"

    def it_no_ops_when_every_pair_is_already_classified(
        data_dir_classify, arxiv, fake_classifier
    ):
        sync_metadata(data_dir_classify)
        classify(data_dir_classify, classifier=fake_classifier)

        again = classify(data_dir_classify, classifier=fake_classifier)

        # Every (paper, cat) pair now has a file -> the missing-pairs
        # query returns nothing -> no classifier calls. No "cached"
        # counter -- the SQL filter is the only idempotency layer.
        assert again == {"classified": 0, "skipped": 0, "failed": 0}

    def it_reclassifies_a_pair_after_its_file_is_deleted(
        data_dir_classify, arxiv, fake_classifier
    ):
        # The way to "re-run" classify for a pair is to delete its
        # output file. The SQL query then surfaces it again. cachetta
        # serves the LLM response from disk so the re-classify costs
        # nothing at the network layer.
        sync_metadata(data_dir_classify)
        classify(data_dir_classify, classifier=fake_classifier)

        target = _classification_path(data_dir_classify, "2401.00001", "is_about_markdown")
        target.unlink()
        counts = classify(data_dir_classify, classifier=fake_classifier)

        assert counts["classified"] == 1
        assert target.exists()
        payload = json.loads(target.read_text())
        assert payload["output"] is True  # 00001 still says "markdown"

    def it_no_ops_when_prompts_dirs_is_empty(
        data_dir, arxiv, fake_classifier
    ):
        # Default config.toml from conftest carries no [classify] block,
        # so prompts_dirs defaults to []. classify must return clean
        # zeros and write nothing -- the daily cron stays green while
        # labels are still being authored.
        sync_metadata(data_dir)

        counts = classify(data_dir, classifier=fake_classifier)

        assert counts == {"classified": 0, "skipped": 0, "failed": 0}
        for pid in PAPERS:
            assert not (data_dir / pid / "classifications").exists()
        assert "classify: disabled" in _read_classify_log(data_dir)

    def it_no_ops_when_every_prompts_dir_is_unbuilt(
        data_dir, arxiv, fake_classifier, tmp_path
    ):
        # Point prompts_dirs at a path with no compiled artifact --
        # classify must log + zero, same as the empty-prompts_dirs case.
        # Keeps the daily cron green while the user is still labeling
        # and compiling prompts.
        sync_metadata(data_dir)
        cfg = (data_dir / "config.toml").read_text()
        cfg += (
            "\n[classify]\n"
            f'prompts_dirs = ["{tmp_path / "no-such-prompt"}"]\n'
            'model = "test-model"\n'
        )
        (data_dir / "config.toml").write_text(cfg)

        counts = classify(data_dir, classifier=fake_classifier)

        assert counts == {"classified": 0, "skipped": 0, "failed": 0}
        for pid in PAPERS:
            assert not (data_dir / pid / "classifications").exists()
        assert "classify: disabled" in _read_classify_log(data_dir)

    def it_leaves_orphan_classification_files_untouched(
        data_dir_classify, arxiv, fake_classifier
    ):
        # A category that's no longer in prompts_dirs (e.g. dropped from
        # config) may still have a classifications/<old>.json on disk
        # from a previous run. The materialized <ROOT>/categories/
        # index will NOT carry an entry for it, so the CROSS JOIN does
        # not pair it with any paper; the orphan classification stays
        # untouched and uncounted.
        sync_metadata(data_dir_classify)
        orphan = _classification_path(data_dir_classify, "2401.00001", "is_about_dragons")
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text(json.dumps({"output": True, "model": "orphan"}))

        counts = classify(data_dir_classify, classifier=fake_classifier)

        # 8 in-config pairs run; the orphan is neither counted nor touched.
        assert counts["classified"] == 8
        assert orphan.exists()
        assert json.loads(orphan.read_text())["model"] == "orphan"

    def it_writes_a_categories_index_under_root(
        data_dir_classify, arxiv, fake_classifier
    ):
        # The dirsql `categories` table is fed by per-cat index files
        # under <ROOT>/categories/. classify materializes them at the
        # start of each run so dirsql has a table to CROSS JOIN against.
        sync_metadata(data_dir_classify)

        classify(data_dir_classify, classifier=fake_classifier)

        cats_dir = data_dir_classify.parent / "categories"
        assert (cats_dir / "is_about_ml.json").exists()
        assert (cats_dir / "is_about_markdown.json").exists()
        payload = json.loads((cats_dir / "is_about_ml.json").read_text())
        assert payload["category_id"] == "is_about_ml"
        assert payload["prompts_dir"]  # non-empty pointer back to source

    def it_removes_a_category_index_when_its_prompts_dir_is_dropped(
        data_dir_classify, arxiv, fake_classifier, tmp_path
    ):
        # Run once with two categories -- index has both files. Then
        # drop one from config and re-run; the stale index file must
        # be removed so dirsql does not surface (paper, stale-cat)
        # pairs and trigger pointless classifier calls.
        sync_metadata(data_dir_classify)
        classify(data_dir_classify, classifier=fake_classifier)

        cats_dir = data_dir_classify.parent / "categories"
        assert (cats_dir / "is_about_markdown.json").exists()

        # Rewrite config to keep only is-about-ml. The base config in
        # the data_dir fixture is the conftest CONFIG_TOML plus the
        # [classify] block appended by data_dir_classify -- safe to
        # truncate everything from "[classify]" onward and re-append
        # a single-prompt block.
        full = (data_dir_classify / "config.toml").read_text()
        ml_path = tmp_path / "prompts" / "is-about-ml"
        trimmed = full.split("[classify]", 1)[0].rstrip()
        new_cfg = (
            f"{trimmed}\n\n[classify]\n"
            f'prompts_dirs = ["{ml_path}"]\n'
            'model = "test-model"\n'
        )
        (data_dir_classify / "config.toml").write_text(new_cfg)

        classify(data_dir_classify, classifier=fake_classifier)

        assert (cats_dir / "is_about_ml.json").exists()
        assert not (cats_dir / "is_about_markdown.json").exists()

    def it_makes_no_writes_on_a_dry_run(
        data_dir_classify, arxiv, fake_classifier
    ):
        sync_metadata(data_dir_classify)

        counts = classify(data_dir_classify, classifier=fake_classifier, dry_run=True)

        assert counts["classified"] == 0
        for pid in PAPERS:
            assert not (data_dir_classify / pid / "classifications").exists()

    def it_honors_limit(
        data_dir_classify, arxiv, fake_classifier
    ):
        sync_metadata(data_dir_classify)

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
        data_dir_classify, arxiv
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
        sync_metadata(data_dir_classify)

        counts = classify(data_dir_classify, classifier=flaky)

        # First pair raises -> failed=1; the other 7 pairs go through.
        assert counts["failed"] == 1
        assert counts["classified"] == 7


def describe_classify_logging():
    def it_logs_each_classification_with_its_output(
        data_dir_classify, arxiv, fake_classifier
    ):
        sync_metadata(data_dir_classify)
        classify(data_dir_classify, classifier=fake_classifier)

        log_text = _read_classify_log(data_dir_classify)
        assert "class 2401.00001/is_about_markdown" in log_text
        assert "classify start" in log_text
        assert "classify done" in log_text
