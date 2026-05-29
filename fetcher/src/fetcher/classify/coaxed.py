"""Thin wrappers around ``coaxer.CoaxedPrompt``.

Each compiled coaxer artifact is one binary flag. fetcher loads it, reads
the single output field name from ``response_format``, and renders the
prompt with whatever input fields the user declared at compile time.
"""

from __future__ import annotations

import json
from pathlib import Path

from coaxer import CoaxedPrompt


def load_coaxed(prompts_dir: Path | str) -> CoaxedPrompt | None:
    """Return the CoaxedPrompt or None if the artifact is missing.

    classify treats a missing or unbuilt prompts dir as "feature not yet
    configured" -- the daily cron must not red-flag while the user is still
    labeling examples and compiling prompts.
    """
    if not prompts_dir:
        return None
    path = Path(prompts_dir)
    if not (path / "prompt.jinja").is_file():
        return None
    return CoaxedPrompt(str(path))


def flag_name(coaxed: CoaxedPrompt) -> str:
    """The single output field name from a coaxer artifact's response_format.

    coaxer 0.3.x compiles one classifier per labels dir, with exactly one
    output field. Reading the name from ``response_format`` keeps fetcher's
    flag keys in lockstep with whatever the user named ``output_name``
    during ``coax`` compile.
    """
    props = coaxed.response_format.model_json_schema()["properties"]
    return next(iter(props))


def input_names(coaxed: CoaxedPrompt) -> list[str]:
    """Names of the Jinja input variables the compiled prompt expects."""
    meta_path = Path(coaxed._path) / "meta.json"
    info = json.loads(meta_path.read_text())
    return list(info.get("fields", {}).get("inputs", {}).keys())


def render_inputs(coaxed: CoaxedPrompt, meta: dict) -> dict[str, str]:
    """Pluck the Jinja inputs the template needs from a paper's metadata.

    Each declared input is stringified; missing inputs default to "" so a
    paper missing a field (e.g. an empty abstract) does not abort the whole
    classify pass.
    """
    return {name: str(meta.get(name, "")) for name in input_names(coaxed)}
