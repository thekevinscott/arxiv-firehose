"""Unit tests for the thin http_classifier adapter.

The substantive HTTP, retry, cache, and credential behaviours all live
in ``shared.llm.LLM`` and are covered by ``shared/llm/llm_test.py``.
These tests cover only what ``http_classifier`` adds on top:

- binding a model name to a Classifier;
- translating the Pydantic schema into the cache-key/payload JSON;
- converting ``LLMError`` (generic) and Pydantic ``ValidationError``
  into ``ClassifyError`` so callers don't depend on either internals.
"""

from __future__ import annotations

import sys
from datetime import timedelta

import pytest
from cachetta import Cachetta
from pydantic import BaseModel

from fetcher.commands.classify import (
    ClassifyError,
    http_classifier,
)
from fetcher.shared import config as config_module
from fetcher.shared.llm import LLM

# `fetcher.shared.llm.llm` as an attribute is shadowed by the
# `llm = LLM()` singleton in shared/llm/__init__.py; reach the actual
# submodule via sys.modules so monkeypatch can rebind names in its
# namespace.
llm_module = sys.modules["fetcher.shared.llm.llm"]


class _FakeBackend:
    """Records every POST and returns a scripted chat-completions response.

    Wire shape: ``{"choices":[{"message":{"content": "<json>"}}]}``.
    ``content`` is whatever JSON the test scripts; the classifier must
    json-validate it into the schema.
    """

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def __call__(self, payload: dict, timeout: float) -> dict:
        self.calls.append({"payload": payload, "timeout": timeout})
        return {"choices": [{"message": {"content": self.content}}]}


class _Out(BaseModel):
    is_about_ml: bool


def _llm(monkeypatch, tmp_path, backend) -> LLM:
    """A fresh LLM with isolated cache + injected backend.

    Patches ``shared.config.cache`` and ``shared.llm.llm.build_default_backend``
    *before* constructing the LLM so the real network/cache is never
    touched.
    """
    monkeypatch.setattr(
        config_module, "cache",
        Cachetta(path=tmp_path, duration=timedelta(days=1)),
    )
    monkeypatch.setattr(llm_module, "build_default_backend", lambda *_a, **_kw: backend)
    return LLM()


def describe_http_classifier():
    def it_posts_an_openai_compatible_chat_completion(monkeypatch, tmp_path):
        fake = _FakeBackend('{"is_about_ml": true}')
        clf = http_classifier("qwen3:8b", _llm(monkeypatch, tmp_path, fake))

        result = clf.call("Is this ML? abstract here.", _Out)

        assert isinstance(result, _Out)
        assert result.is_about_ml is True
        assert len(fake.calls) == 1
        payload = fake.calls[0]["payload"]
        assert payload["model"] == "qwen3:8b"
        assert payload["messages"] == [
            {"role": "user", "content": "Is this ML? abstract here."}
        ]
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["schema"] == _Out.model_json_schema()
        assert payload["response_format"]["json_schema"]["strict"] is True

    def it_validates_the_response_into_the_pydantic_model(monkeypatch, tmp_path):
        fake = _FakeBackend('{"is_about_safety": false}')

        class Out(BaseModel):
            is_about_safety: bool

        clf = http_classifier("qwen3:8b", _llm(monkeypatch, tmp_path, fake))

        assert clf.call("any prompt", Out).is_about_safety is False

    def it_raises_classifyerror_on_invalid_json(monkeypatch, tmp_path):
        # An LLMError (raised by LLM on bad JSON) must surface as
        # ClassifyError so callers don't depend on shared.llm internals.
        fake = _FakeBackend("not json at all")
        clf = http_classifier("qwen3:8b", _llm(monkeypatch, tmp_path, fake))

        with pytest.raises(ClassifyError):
            clf.call("any prompt", _Out)

    def it_raises_classifyerror_when_required_field_is_missing(monkeypatch, tmp_path):
        # Schema validation happens inside http_classifier (outside LLM).
        # A Pydantic ValidationError must surface as ClassifyError.
        fake = _FakeBackend('{"some_other_field": true}')
        clf = http_classifier("qwen3:8b", _llm(monkeypatch, tmp_path, fake))

        with pytest.raises(ClassifyError):
            clf.call("any prompt", _Out)

    def it_raises_classifyerror_on_an_http_error(monkeypatch, tmp_path):
        import httpx
        def boom(payload, timeout):
            raise httpx.ConnectError("backend down")

        clf = http_classifier("qwen3:8b", _llm(monkeypatch, tmp_path, boom))

        with pytest.raises(ClassifyError):
            clf.call("any prompt", _Out)

    def it_raises_classifyerror_on_a_malformed_response_shape(monkeypatch, tmp_path):
        def odd_shape(payload, timeout):
            return {"error": "rate limited"}

        clf = http_classifier("qwen3:8b", _llm(monkeypatch, tmp_path, odd_shape))

        with pytest.raises(ClassifyError):
            clf.call("any prompt", _Out)
