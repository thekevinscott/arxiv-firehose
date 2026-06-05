"""train-categories: compile every labels subdir into a prompt artifact.

One ``labels/<category>/`` dir per binary flag. ``train-categories`` walks
``labels/``, treats each subdir as a category, and compiles it into
``prompts/<category>/``. The category name is the subdir basename
(``is-about-control``); the output field name is the same with hyphens
swapped to underscores (``is_about_control``), which is also the flag
key the runtime ``classify`` writes per paper.

Each compile is wrapped in a content-addressed cache at
``~/.cache/arxiv-firehose/classify/{hash}/``. Unchanged labels copy from
the cache; the expensive path (``--optimizer gepa``, one LLM round-trip
per example) only runs on a real label change.

The cache key folds in (a) every file under the labels subdir, (b) the
optimizer name, (c) the output_name -- so changing labels OR a compile
knob invalidates cleanly, while matching content matching knobs hits.

``train-categories`` is a developer command, not a cron one. It runs at
"I just edited labels" time, not on a schedule.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any

from coaxer.compiler import distill

CACHE_ROOT = Path.home() / ".cache" / "arxiv-firehose" / "classify"


def output_name_for(labels_subdir_name: str) -> str:
    """Turn a labels subdir name into the flag key the runtime expects.

    ``is-about-control`` -> ``is_about_control``. Python attribute names
    (which the Pydantic response_format builds from the output field)
    can't carry hyphens, so the convention is hyphenated dirs on disk +
    underscored field names in code. Single source of truth: every
    category id is derived here.
    """
    return labels_subdir_name.replace("-", "_")


def hash_labels(
    labels_dir: Path, *, optimizer: str | None, output_name: str
) -> str:
    """16-hex-char content hash over a labels dir plus compile knobs.

    Mirrors coaxer's own ``_hash_labels`` so the cache key matches the
    ``label_hash`` coaxer records in meta.json -- a future reader who
    sees the cache dir and the artifact's meta.json gets the same value.
    Optimizer + output_name are folded in *after* the file content so a
    different compile knob doesn't pull a stale artifact for the same
    labels (and vice versa).
    """
    h = hashlib.sha256()
    for p in sorted(labels_dir.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(labels_dir).as_posix().encode())
            h.update(b"\0")
            h.update(p.read_bytes())
    h.update(b"\0optimizer=")
    h.update(str(optimizer or "none").encode())
    h.update(b"\0output_name=")
    h.update(output_name.encode())
    return h.hexdigest()[:16]


def _copy_artifact(src: Path, dst: Path) -> None:
    """Copy *src*'s files into *dst*. Overwrites; subdirs not expected.

    coaxer's distill outputs are all flat files (prompt.jinja, meta.json,
    history.jsonl, dspy.json). A flat copy keeps fetcher's I/O auditable.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.is_file():
            shutil.copy2(entry, dst / entry.name)


def _build_lm(
    optimizer: str | None, model: str | None, base_url: str | None
) -> Any:
    """Build the DSPy LM coaxer's optimizer drives.

    Returns None for ``optimizer=None`` (raw template, no LM needed). For
    ``optimizer="gepa"`` builds an OpenAILM pointed at the configured
    model + base_url -- same shape as the runtime classify backend, so
    labels are compiled against the same model that will later score
    real abstracts.
    """
    if optimizer is None:
        return None
    if optimizer == "gepa":
        from coaxer import OpenAILM
        if model is None:
            raise ValueError("--model is required with --optimizer gepa")
        kwargs: dict[str, Any] = {"model": model}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAILM(**kwargs)
    raise ValueError(f"Unknown optimizer: {optimizer!r}")


def compile_one(
    labels_dir: Path,
    out_dir: Path,
    log: logging.Logger,
    *,
    optimizer: str | None = None,
    output_name: str = "output",
    model: str | None = None,
    base_url: str | None = None,
    cache_root: Path = CACHE_ROOT,
) -> dict[str, str]:
    """Compile a single labels dir into a CoaxedPrompt artifact at *out_dir*.

    Returns ``{"hash", "source", "out"}`` where ``source`` is "cache" or
    "fresh". The cache lives at ``cache_root/{hash}/``; on a hit the
    artifact is copied straight into *out_dir*, on a miss
    ``coaxer.compiler.distill`` compiles into the cache and then copies.

    *model* + *base_url* only matter when *optimizer == "gepa"* -- with
    ``optimizer=None`` distill is a pure local render (Jinja2 + Pydantic)
    and the LM args are ignored.
    """
    h = hash_labels(labels_dir, optimizer=optimizer, output_name=output_name)
    cache_path = cache_root / h
    fresh = not (cache_path / "prompt.jinja").is_file()

    if fresh:
        log.info("train: cache miss for %s, compiling -> %s", labels_dir, cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        lm = _build_lm(optimizer, model, base_url)
        distill(
            labels_dir, cache_path,
            lm=lm, optimizer=optimizer, output_name=output_name,
        )
    else:
        log.info("train: cache hit for %s (%s)", labels_dir, h)

    _copy_artifact(cache_path, out_dir)
    return {
        "hash": h,
        "source": "fresh" if fresh else "cache",
        "out": str(out_dir),
    }


def discover_categories(labels_root: Path) -> list[Path]:
    """Every subdir of *labels_root* that looks like a category.

    A category dir is identified by a ``_schema.json`` at its top --
    same signal coaxer uses to recognise a compilable labels folder.
    README.md, history.jsonl, and other non-category siblings are
    skipped silently.
    """
    if not labels_root.is_dir():
        return []
    return sorted(
        sub for sub in labels_root.iterdir()
        if sub.is_dir() and (sub / "_schema.json").is_file()
    )


def run(
    labels_root: Path,
    prompts_root: Path,
    log: logging.Logger,
    *,
    optimizer: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    cache_root: Path = CACHE_ROOT,
) -> dict[str, dict[str, str]]:
    """Compile every category under *labels_root* into *prompts_root*.

    Returns ``{category_dirname: {"hash", "source", "out"}, ...}`` --
    one entry per labels subdir that carried a ``_schema.json``. Each
    category's output_name is derived from its dirname
    (``is-about-control`` -> ``is_about_control``), keeping the runtime
    flag key in lockstep with the on-disk taxonomy.
    """
    categories = discover_categories(labels_root)
    if not categories:
        log.warning("train: no categories found under %s", labels_root)
        return {}

    log.info("train: %d categories under %s", len(categories), labels_root)
    results: dict[str, dict[str, str]] = {}
    for labels_dir in categories:
        name = labels_dir.name
        out_dir = prompts_root / name
        results[name] = compile_one(
            labels_dir, out_dir, log,
            optimizer=optimizer,
            output_name=output_name_for(name),
            model=model,
            base_url=base_url,
            cache_root=cache_root,
        )
    return results
