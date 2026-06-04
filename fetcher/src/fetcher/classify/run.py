"""classify.run: query dirsql for (paper, category) pairs that lack a
classification file, then call the matching coaxed prompt against each.

One file per (paper, category) at ``<data_dir>/<arxiv_id>/classifications/
<category_id>.json``, shape ``{"output": <bool>, "model": ..., "classified_at": ...}``.
Idempotency is free -- the dirsql query joins ``categories`` against the
``papers_categories`` view (which mirrors the files on disk), so an
already-classified pair never appears in the work queue.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from coaxer import CoaxedPrompt

from ..config import Config
from ..dirsql_schema import ALL_PAIRS_SQL, MISSING_PAIRS_SQL, build_app
from .coaxed import flag_name, load_coaxed, render_inputs
from .http import http_classifier
from .store import write_classification
from .types import Classifier


def run(
    data_dir: Path,
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
    - skipped    -- pair's category has no matching prompts_dir configured.
    - failed     -- pair the classifier raised on; no file written.

    When ``[classify] prompts_dirs`` is empty or ``categories.json`` is
    missing, ``run`` logs one "disabled" line and returns zeros -- the
    daily cron stays green while the taxonomy is still being authored.
    """
    counts = {"classified": 0, "cached": 0, "skipped": 0, "failed": 0}

    prompts_dirs = list(config.classify.prompts_dirs)
    if not prompts_dirs:
        log.info("classify: disabled (no prompts_dirs configured)")
        return counts

    root = data_dir.parent
    if not (root / "categories.json").exists():
        log.info("classify: disabled (categories.json missing at %s)", root)
        return counts

    coaxed_by_cat = _load_coaxed_by_category(prompts_dirs, log)
    if not coaxed_by_cat:
        log.info("classify: no usable prompts dirs")
        return counts

    pairs, total_pairs = asyncio.run(_query_pairs(root, force=force))
    if not force:
        # Pairs already on disk are absent from the missing query --
        # count them so the run summary reports the full picture.
        counts["cached"] = total_pairs - len(pairs)

    if classifier is None:
        classifier = http_classifier(
            config.classify.model,
            base_url=config.classify.base_url,
            api_key=config.classify.api_key or None,
            timeout_s=config.classify.timeout_s,
        )

    log.info(
        "classify start: %d pairs to process (cached=%d), %d classifiers, model=%s",
        len(pairs), counts["cached"], len(coaxed_by_cat), config.classify.model,
    )

    processed = 0
    for row in pairs:
        if limit is not None and processed >= limit:
            break
        processed += 1
        _classify_one_pair(
            data_dir, row, coaxed_by_cat, classifier, config, counts, log,
            dry_run=dry_run,
        )

    log.info("classify done: %s", counts)
    return counts


async def _query_pairs(root: Path, *, force: bool) -> tuple[list[dict], int]:
    """Return (work_queue, total_pairs).

    ``work_queue`` is the rows to process this run -- every (paper, cat)
    pair when force=True, only the missing ones otherwise. ``total_pairs``
    is always the full cross-join (papers × categories) so callers can
    compute "cached" as total - missing.
    """
    db = build_app(root)
    await db.ready()
    work_queue = await db.query(ALL_PAIRS_SQL if force else MISSING_PAIRS_SQL)
    total = await db.query("SELECT COUNT(*) AS n FROM papers, categories")
    return list(work_queue), int(total[0]["n"])


def _load_coaxed_by_category(
    prompts_dirs: list[str], log: logging.Logger
) -> dict[str, CoaxedPrompt]:
    """Map ``category_id`` -> CoaxedPrompt. A prompt's category_id is the
    name of its (sole) output field -- which is what the user puts in
    ``categories.json`` as the canonical id.
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
    row: dict,
    coaxed_by_cat: dict[str, CoaxedPrompt],
    classifier: Classifier,
    config: Config,
    counts: dict[str, int],
    log: logging.Logger,
    *,
    dry_run: bool,
) -> None:
    """Render the prompt, call the classifier, write the per-pair file."""
    paper_id = row["paper_id"]
    cat_id = row["category_id"]
    coaxed = coaxed_by_cat.get(cat_id)
    if coaxed is None:
        counts["skipped"] += 1
        log.warning("classify: no prompt for category %s (paper %s)", cat_id, paper_id)
        return

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
