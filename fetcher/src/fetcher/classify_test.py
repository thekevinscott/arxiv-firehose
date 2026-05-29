"""Unit tests for ``classify``: the injection seam and helpers.

The real Ollama HTTP client is never reached -- a fake client is injected,
mirroring the ``Transport`` / ``Converter`` pattern (see AGENTS.md). The
real ``CoaxedPrompt`` *is* used: it's a pure, network-free Jinja2 +
Pydantic-schema wrapper that needs only a small on-disk artifact to load.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from fetcher.classify import (
    ClassifyError,
    Classifier,
    load_coaxed,
    ollama_classifier,
)


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


class _FakeOllama:
    """Records every chat() call and returns a scripted JSON content body."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def chat(self, *, model, messages, format):
        self.calls.append({"model": model, "messages": messages, "format": format})
        return {"message": {"content": self.content}}


def describe_ollama_classifier():
    def it_passes_prompt_and_schema_to_the_client():
        # The seam: the Ollama client is injected, not patched. The classifier
        # must hand the rendered prompt and the Pydantic JSON schema through
        # to client.chat() exactly once per call.
        fake = _FakeOllama('{"is_about_ml": true}')
        clf = ollama_classifier("qwen3:8b", client=fake)

        class Out(BaseModel):
            is_about_ml: bool

        result = clf.call("Is this ML? abstract here.", Out)

        assert isinstance(result, Out)
        assert result.is_about_ml is True
        assert len(fake.calls) == 1
        assert fake.calls[0]["model"] == "qwen3:8b"
        assert fake.calls[0]["messages"] == [
            {"role": "user", "content": "Is this ML? abstract here."}
        ]
        # Ollama's `format` parameter takes the JSON schema dict so the model
        # is constrained to produce valid JSON for the response_format.
        assert fake.calls[0]["format"] == Out.model_json_schema()

    def it_validates_the_response_into_the_pydantic_model():
        fake = _FakeOllama('{"is_about_safety": false}')
        clf = ollama_classifier("qwen3:8b", client=fake)

        class Out(BaseModel):
            is_about_safety: bool

        result = clf.call("any prompt", Out)

        assert result.is_about_safety is False

    def it_raises_classifyerror_on_invalid_json():
        # A clean Pydantic ValidationError must surface as the package's own
        # ClassifyError so callers don't depend on Pydantic internals.
        fake = _FakeOllama("not json at all")
        clf = ollama_classifier("qwen3:8b", client=fake)

        class Out(BaseModel):
            is_about_ml: bool

        with pytest.raises(ClassifyError):
            clf.call("any prompt", Out)

    def it_raises_classifyerror_when_required_field_is_missing():
        fake = _FakeOllama('{"some_other_field": true}')
        clf = ollama_classifier("qwen3:8b", client=fake)

        class Out(BaseModel):
            is_about_ml: bool

        with pytest.raises(ClassifyError):
            clf.call("any prompt", Out)


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


def describe_Classifier():
    def it_is_a_simple_callable_seam():
        # Classifier itself is just a wrapper around a callable -- the test
        # asserts the dataclass holds the function and forwards arguments
        # untouched. The real wiring is exercised through ollama_classifier
        # and through the integration suite.
        def fake_call(prompt, schema):
            return schema(is_about_ml=("ml" in prompt.lower()))

        class Out(BaseModel):
            is_about_ml: bool

        clf = Classifier(call=fake_call)
        result = clf.call("ML paper abstract", Out)
        assert isinstance(result, Out)
        assert result.is_about_ml is True
