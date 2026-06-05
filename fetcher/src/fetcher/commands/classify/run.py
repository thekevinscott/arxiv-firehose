"""classify.run: for every (paper, category) pair that lacks a
classification file, call the matching coaxed prompt against the paper's
abstract.

One file per (paper, category) at ``<data_dir>/<arxiv_id>/classifications/
<category_id>.json``, shape ``{"output": <bool>, "model": ..., "classified_at": ...}``.
Idempotency is free -- a pair is "missing" only if its file is not on
disk, so an already-classified pair is never reclassified (unless
``--force`` is set).

The taxonomy of category ids is derived from ``[classify] prompts_dirs``
in config: each compiled coaxer artifact's output field name *is* its
category id. There is no separate ``categories.json`` to keep in sync.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from coaxer import CoaxedPrompt

from ...shared.config import Config
from ...shared.dirsql_schema import ALL_PAPERS_SQL, EXISTING_PAIRS_SQL, build_app
from .coaxed import flag_name, load_coaxed, render_inputs
from .http import http_classifier
from .store import write_classification
from .types import Classifier


def run(
    data_dir: Path,
    cache_dir: Path,
    config: Config,
    log: logging.Logger,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    classifier: Classifier | None = None,
) -> dict[str, int]:
    """Classify every missing (paper, category) pair. Returns a counts dict.

    counts: ``{"classified", "cached", "skipped", "failed"}``.
    - classified -- pair the classifier successfully labeled this run.
    - cached     -- pair already had a classification file (didn't run).
    - skipped    -- paper whose ``metadata.json`` was unreadable.
    - failed     -- pair the classifier raised on; no file written.

    When ``[classify] prompts_dirs`` is empty (or none of them are
    compiled), ``run`` logs one "disabled" line and returns zeros -- the
    daily cron stays green while the taxonomy is still being authored.

    *cache_dir* is the cachetta location the LLM HTTP client writes to;
    a repeat ``(model, prompt, schema)`` triple serves from disk with
    no network call. This is the only caching mechanism in classify.
    """
    counts = {"classified": 0, "cached": 0, "skipped": 0, "failed": 0}

    prompts_dirs = list(config.classify.prompts_dirs)
    if not prompts_dirs:
        log.info("classify: disabled (no prompts_dirs configured)")
        return counts

    coaxed_by_cat = _load_coaxed_by_category(prompts_dirs, log)
    if not coaxed_by_cat:
        log.info("classify: disabled (no usable prompts dirs)")
        return counts

    cats = sorted(coaxed_by_cat)
    papers, existing = asyncio.run(_query_state(data_dir.parent))
    pairs = _work_queue(papers, cats, existing, force=force)
    if not force:
        counts["cached"] = len(papers) * len(cats) - len(pairs)

    if classifier is None:
        classifier = http_classifier(
            config.classify.model,
            base_url=config.classify.base_url,
            api_key=config.classify.api_key or None,
            timeout_s=config.classify.timeout_s,
            cache_dir=cache_dir,
        )

    log.info(
        "classify start: %d pairs to process (cached=%d), %d classifiers, model=%s",
        len(pairs), counts["cached"], len(coaxed_by_cat), config.classify.model,
    )

    processed = 0
    for paper_id, cat_id in pairs:
        if limit is not None and processed >= limit:
            break
        processed += 1
        _classify_one_pair(
            data_dir, paper_id, cat_id, coaxed_by_cat, classifier, config,
            counts, log, dry_run=dry_run,
        )

    log.info("classify done: %s", counts)
    return counts


async def _query_state(root: Path) -> tuple[list[str], set[tuple[str, str]]]:
    """Return (paper_ids, existing_pairs).

    ``existing_pairs`` is the set of (paper, category) that already have a
    classification file on disk -- subtracted from the full cross product
    to build the missing-work queue.
    """
    db = build_app(root)
    await db.ready()
    paper_rows = await db.query(ALL_PAPERS_SQL)
    pc_rows = await db.query(EXISTING_PAIRS_SQL)
    papers = [r["paper_id"] for r in paper_rows]
    existing = {(r["paper_id"], r["category_id"]) for r in pc_rows}
    return papers, existing


def _work_queue(
    papers: list[str],
    cats: list[str],
    existing: set[tuple[str, str]],
    *,
    force: bool,
) -> list[tuple[str, str]]:
    """Cross-product papers × cats, ordered (paper, cat). With force=False
    skips pairs already in *existing* (the idempotent path)."""
    pairs = [(p, c) for p in papers for c in cats]
    if force:
        return pairs
    return [pair for pair in pairs if pair not in existing]


def _load_coaxed_by_category(
    prompts_dirs: list[str], log: logging.Logger
) -> dict[str, CoaxedPrompt]:
    """Map ``category_id`` -> CoaxedPrompt. A prompt's category_id is the
    name of its (sole) output field -- the value the user put in
    ``output_name`` at coax compile time. This *is* the taxonomy.
    """
    loaded: dict[str, CoaxedPrompt] = {}
    for raw in prompts_dirs:
        path = Path(raw)
        cp = load_coaxed(path)
        if cp is None:
            log.warning("classify: missing or unbuilt prompts dir %s", path)
            continue
        loaded[flag_name(cp)] = cp
    return loaded


def _classify_one_pair(
    data_dir: Path,
    paper_id: str,
    cat_id: str,
    coaxed_by_cat: dict[str, CoaxedPrompt],
    classifier: Classifier,
    config: Config,
    counts: dict[str, int],
    log: logging.Logger,
    *,
    dry_run: bool,
) -> None:
    """Render the prompt, call the classifier, write the per-pair file."""
    coaxed = coaxed_by_cat[cat_id]  # cats came from coaxed_by_cat -- never absent
    meta = _read_metadata(data_dir / paper_id, counts, log)
    if meta is None:
        return

    if dry_run:
        log.info("[dry-run] would classify %s/%s", paper_id, cat_id)
        return

    dest = data_dir / paper_id / "classifications" / f"{cat_id}.json"
    try:
        prompt = coaxed(**render_inputs(coaxed, meta))
        result = classifier.call(prompt, coaxed.response_format)
        output = bool(getattr(result, cat_id))
        payload: dict[str, Any] = {
            "output": output,
            "model": config.classify.model,
            "classified_at": datetime.now(timezone.utc).isoformat(),
        }
        write_classification(payload, dest)
        counts["classified"] += 1
        log.info("class %s/%s: %s", paper_id, cat_id, output)
    except Exception as exc:  # noqa: BLE001 -- one pair's LLM failure
        counts["failed"] += 1                #     must not abort the run
        log.error("class %s/%s: %s", paper_id, cat_id, exc)


def _read_metadata(
    pd: Path, counts: dict[str, int], log: logging.Logger
) -> dict | None:
    """Read metadata.json or count+log the skip. A single bad paper folder
    must not abort the run."""
    try:
        return json.loads((pd / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        counts["skipped"] += 1
        log.error("skip %s: bad metadata.json", pd.name)
        return None
