"""coax: compile a labels dir into a CoaxedPrompt artifact, content-cached.

Wraps ``coaxer.compiler.distill`` so a labels dir's compiled output --
``prompt.jinja`` + ``meta.json`` + optional ``dspy.json`` -- lives at
``~/.cache/arxiv-firehose/classify/{hash}/``, keyed by a hash of the
labels' content plus the compile knobs. A re-run with unchanged labels
copies from the cache instead of recompiling -- the win that matters is
``--optimizer gepa``, where each compile is an LLM round-trip per example.

The cache is content-addressed: the hash covers every file under the
labels dir (sorted, with relpath separators), plus the optimizer name
and output_name -- so changing labels OR a compile knob invalidates the
cache cleanly.

Coax is a developer command, not a cron-level one. It runs at "I just
finished labeling" time, not on a schedule, so its log goes to stdout
via the CLI's typer.echo rather than to ``logs/coax.log``.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any

from coaxer.compiler import distill

CACHE_ROOT = Path.home() / ".cache" / "arxiv-firehose" / "classify"


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
    history.jsonl, dspy.json). A flat copy keeps fetcher's I/O auditable
    -- shutil.copytree would refuse if dst exists, and dirs_exist_ok
    only landed in 3.8 with a confusing semantics; an explicit per-file
    copy is clearer.
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


def run(
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
    """Compile *labels_dir* into a CoaxedPrompt artifact at *out_dir*.

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
        log.info("coax: cache miss for %s, compiling -> %s", labels_dir, cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        lm = _build_lm(optimizer, model, base_url)
        distill(
            labels_dir, cache_path,
            lm=lm, optimizer=optimizer, output_name=output_name,
        )
    else:
        log.info("coax: cache hit for %s (%s)", labels_dir, h)

    _copy_artifact(cache_path, out_dir)
    return {
        "hash": h,
        "source": "fresh" if fresh else "cache",
        "out": str(out_dir),
    }
