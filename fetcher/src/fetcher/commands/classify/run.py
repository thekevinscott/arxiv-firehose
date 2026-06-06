"""classify.run: dirsql-driven, cachetta-cached LLM classification.

Pipeline (no in-process caching anywhere):

1. Load each compiled prompt named in ``[classify] prompts_dirs`` -- in
   memory only, for the duration of the run.
2. Materialize one file per active category at
   ``<ROOT>/categories/<cat>.json`` and remove any orphan from a
   dropped prompts_dir. This is the index dirsql ``CROSS JOIN``s
   against ``papers`` -- the only on-disk reflection of the configured
   taxonomy.
3. Ask dirsql for every ``(paper, category)`` pair that lacks a
   classification file. One query, one ``LEFT JOIN ... WHERE NULL``.
4. For each pair, render the prompt and hand it to the classifier.
   The default HTTP classifier sits behind cachetta keyed by
   ``(model, prompt, schema_json)`` -- a repeat call serves bytes from
   disk with no network hit.
5. Atomic-write the per-pair result file. dirsql's watcher picks it up
   for the next run; the missing-pairs query then excludes it.

Output: one JSON file per (paper, category) at
``<data_dir>/<arxiv_id>/classifications/<cat_id>.json``, shape
``{"output": <bool>, "model": ..., "classified_at": ...}``.
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
from ...shared.dirsql_schema import MISSING_PAIRS_SQL, build_app
from ...shared.llm import llm
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
    classifier: Classifier | None = None,
) -> dict[str, int]:
    """Classify every missing (paper, category) pair.

    counts: ``{"classified", "skipped", "failed"}``.
    - classified -- pair the classifier successfully labeled this run.
    - skipped    -- paper whose ``metadata.json`` was unreadable.
    - failed     -- pair the classifier raised on; no file written.

    There is **no** "cached" counter. The missing-pairs SQL query is
    the only idempotency layer (pairs with a file on disk simply do
    not appear in the result). The only caching is the cachetta layer
    inside the HTTP classifier; it is transparent.

    When ``[classify] prompts_dirs`` is empty (or none of them are
    compiled), ``run`` logs one "disabled" line and returns zeros --
    the daily cron stays green while the taxonomy is still being
    authored.

    The LLM HTTP client caches responses at ``shared.config.cache``;
    a repeat ``(model, prompt, schema)`` triple serves from disk with
    no network call.
    """
    counts = {"classified": 0, "skipped": 0, "failed": 0}

    prompts_dirs = list(config.classify.prompts_dirs)
    if not prompts_dirs:
        log.info("classify: disabled (no prompts_dirs configured)")
        return counts

    coaxed_by_cat = _load_coaxed_by_category(prompts_dirs, log)
    if not coaxed_by_cat:
        log.info("classify: disabled (no usable prompts dirs)")
        return counts

    root = data_dir.parent
    _materialize_categories(root, coaxed_by_cat)

    pairs = asyncio.run(_query_missing_pairs(root))

    if classifier is None:
        classifier = http_classifier(config.classify.model, llm)

    log.info(
        "classify start: %d missing pairs, %d categories, model=%s",
        len(pairs), len(coaxed_by_cat), config.classify.model,
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


def _materialize_categories(
    root: Path, coaxed_by_cat: dict[str, CoaxedPrompt]
) -> None:
    """Mirror the active prompts as ``<root>/categories/<cat>.json``.

    Each run rewrites the dir to match the loaded prompts exactly. A
    file for a category that's been dropped from config is removed so
    dirsql's ``categories`` table stays in lockstep with config -- the
    SQL missing-pairs query will not surface a (paper, dropped-cat)
    pair, and any prior classification.json for the dropped cat is
    left untouched on disk (an orphan, intentionally).
    """
    cats_dir = root / "categories"
    cats_dir.mkdir(parents=True, exist_ok=True)
    active = set(coaxed_by_cat)

    for existing in cats_dir.glob("*.json"):
        if existing.stem not in active:
            existing.unlink()

    for cat_id, coaxed in coaxed_by_cat.items():
        payload = {
            "category_id": cat_id,
            "prompts_dir": str(Path(coaxed._path).resolve()),
        }
        dest = cats_dir / f"{cat_id}.json"
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        tmp.rename(dest)


async def _query_missing_pairs(root: Path) -> list[tuple[str, str]]:
    """Run the missing-pairs SQL against an in-process dirsql app.

    Returns ``(paper_id, category_id)`` tuples for every pair without a
    classification file on disk. The order is the SQL's
    ``ORDER BY (paper_id, category_id)`` -- deterministic so
    ``--limit`` is reproducible.
    """
    db = build_app(root)
    await db.ready()
    rows = await db.query(MISSING_PAIRS_SQL)
    return [(r["paper_id"], r["category_id"]) for r in rows]


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
