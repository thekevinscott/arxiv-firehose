"""classify.run: walk paper folders, classify each abstract.

For each compiled CoaxedPrompt artifact listed in ``[classify] prompts_dirs``,
fetcher renders the prompt with the paper's metadata, calls the LLM, and
combines every flag into one ``classification.json`` beside ``metadata.json``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from coaxer import CoaxedPrompt

from ..config import Config
from ..paths import iter_paper_dirs
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
    """Classify every paper. Returns a counts dict.

    counts: ``{"classified", "cached", "skipped", "failed"}``. When
    ``[classify] prompts_dirs`` is empty the run logs a single "disabled"
    line and returns zeros -- the cron stays green while labels are still
    being authored.
    """
    counts = {"classified": 0, "cached": 0, "skipped": 0, "failed": 0}

    prompts_dirs = list(config.classify.prompts_dirs)
    if not prompts_dirs:
        log.info("classify: disabled (no prompts_dirs configured)")
        return counts

    coaxed_list = _load_all_coaxed(prompts_dirs, log)
    if not coaxed_list:
        log.info("classify: no usable prompts dirs")
        return counts

    if classifier is None:
        classifier = http_classifier(
            config.classify.model,
            base_url=config.classify.base_url,
            api_key=config.classify.api_key or None,
            timeout_s=config.classify.timeout_s,
        )

    paper_dirs = list(iter_paper_dirs(data_dir))
    log.info("classify start: %d papers, %d classifiers, model=%s",
             len(paper_dirs), len(coaxed_list), config.classify.model)

    processed = 0
    for pd in paper_dirs:
        if limit is not None and processed >= limit:
            break
        meta = _read_metadata(pd, counts, log)
        if meta is None:
            continue
        processed += 1
        _classify_one(
            pd, meta, coaxed_list, classifier, config, counts, log,
            dry_run=dry_run, force=force,
        )

    log.info("classify done: %s", counts)
    return counts


def _load_all_coaxed(
    prompts_dirs: list[str], log: logging.Logger
) -> list[tuple[CoaxedPrompt, str]]:
    """One coaxed prompt + its flag name per config entry; warn-and-skip
    missing dirs (the user may still be compiling one of several flags)."""
    loaded: list[tuple[CoaxedPrompt, str]] = []
    for raw in prompts_dirs:
        path = Path(raw)
        cp = load_coaxed(path)
        if cp is None:
            log.warning("classify: missing or unbuilt prompts dir %s", path)
            continue
        loaded.append((cp, flag_name(cp)))
    return loaded


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


def _classify_one(
    pd: Path,
    meta: dict,
    coaxed_list: list[tuple[CoaxedPrompt, str]],
    classifier: Classifier,
    config: Config,
    counts: dict[str, int],
    log: logging.Logger,
    *,
    dry_run: bool,
    force: bool,
) -> None:
    """Run every classifier against one paper, write the combined result."""
    arxiv_id = meta.get("arxiv_id", pd.name)
    dest = pd / "classification.json"
    if dest.exists() and not force:
        counts["cached"] += 1
        return
    if dry_run:
        log.info("[dry-run] would classify %s", arxiv_id)
        return
    try:
        flags: dict[str, Any] = {}
        for coaxed, fname in coaxed_list:
            prompt = coaxed(**render_inputs(coaxed, meta))
            result = classifier.call(prompt, coaxed.response_format)
            flags[fname] = getattr(result, fname)
        payload = {
            "arxiv_id": arxiv_id,
            "flags": flags,
            "model": config.classify.model,
            "classified_at": datetime.now(timezone.utc).isoformat(),
        }
        write_classification(payload, dest)
        counts["classified"] += 1
        log.info("class %s: %s", arxiv_id, flags)
    except Exception as exc:  # noqa: BLE001 -- one paper's LLM failure
        counts["failed"] += 1                #     must not abort the run
        log.error("class %s: %s", arxiv_id, exc)
