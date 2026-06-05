"""Unit tests for the coaxer artifact wrappers.

The real ``CoaxedPrompt`` is exercised here: it's a pure, network-free
Jinja2 + Pydantic-schema wrapper that needs only a small on-disk artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

from fetcher.commands.classify import load_coaxed


def _write_artifact(folder: Path, body: str = "Classify: {{ abstract }}",
                    *, output_type: str = "bool",
                    output_name: str = "is_about_ml",
                    inputs: dict | None = None) -> Path:
    """Write a minimal prompts artifact: prompt.jinja + meta.json.

    Mirrors coaxer's own test helper so we can exercise the real
    ``CoaxedPrompt`` without invoking the ``coax`` compile step.
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "prompt.jinja").write_text(body)
    meta = {
        "output_name": output_name,
        "fields": {
            "inputs": inputs or {"abstract": {"type": "str"}},
            "output": {"type": output_type},
        },
    }
    (folder / "meta.json").write_text(json.dumps(meta))
    return folder


def describe_load_coaxed():
    def it_returns_none_when_path_is_empty():
        # An unconfigured prompts_dir ("") must no-op cleanly -- the daily
        # cron stays green while the user is still labeling.
        assert load_coaxed("") is None

    def it_returns_none_when_dir_is_missing(tmp_path: Path):
        assert load_coaxed(tmp_path / "does-not-exist") is None

    def it_returns_none_when_prompt_jinja_is_absent(tmp_path: Path):
        # A bare directory with no compiled artifact: treat as "not yet
        # compiled", not as a hard error.
        (tmp_path / "halfbuilt").mkdir()
        assert load_coaxed(tmp_path / "halfbuilt") is None

    def it_loads_a_compiled_prompt(tmp_path: Path):
        folder = _write_artifact(tmp_path / "is-about-ml")
        cp = load_coaxed(folder)

        assert cp is not None
        # CoaxedPrompt is a str subclass; the raw template text is the str
        # value, and rendering binds Jinja variables.
        assert "{{ abstract }}" in str(cp)
        assert cp(abstract="hello world") == "Classify: hello world"

    def it_loads_when_passed_a_string_path(tmp_path: Path):
        folder = _write_artifact(tmp_path / "is-about-ml")
        assert load_coaxed(str(folder)) is not None
