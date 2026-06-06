"""Unit tests for shared.llm.LLM.

Covers credentials/Bearer header, retry/backoff, cachetta-backed disk
cache, LLMError shape. ``http_classifier``'s tests cover only the thin
schema/ClassifyError adapter on top.

``LLM()`` takes no args -- everything comes from ``shared.config``. Each
test isolates state by monkeypatching ``shared.config.cache`` to a
``Cachetta(path=tmp_path, ...)`` and (where the backend is faked)
patching ``shared.llm.llm.build_default_backend`` *before* constructing
the LLM. Nothing ever writes to ``~/.cache/arxiv-firehose``.
"""

from __future__ import annotations

from datetime import timedelta

import httpx
import pytest
from cachetta import Cachetta

import sys

import fetcher.shared.build_default_backend as bdb_module
from fetcher.shared import config as config_module
from fetcher.shared.llm import API_KEY_ENV, LLM, LLMError

# `fetcher.shared.llm.llm` as an attribute is shadowed by the
# `llm = LLM()` singleton in shared/llm/__init__.py; reach the actual
# submodule via sys.modules so monkeypatch can rebind names in its
# namespace.
llm_module = sys.modules["fetcher.shared.llm.llm"]


class _FakeBackend:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def __call__(self, payload: dict, timeout: float) -> dict:
        self.calls.append({"payload": payload, "timeout": timeout})
        return {"choices": [{"message": {"content": self.content}}]}


_SCHEMA_JSON = '{"type":"object","properties":{"is_ml":{"type":"boolean"}},"required":["is_ml"]}'


def _isolate_cache(monkeypatch, tmp_path) -> None:
    """Point shared.config.cache at tmp_path so the LLM about to be
    constructed wraps an isolated Cachetta."""
    monkeypatch.setattr(
        config_module, "cache",
        Cachetta(path=tmp_path, duration=timedelta(days=1)),
    )


def _llm_with_backend(monkeypatch, tmp_path, backend) -> LLM:
    """Construct an LLM whose ``_backend`` is *backend*.

    Patches ``build_default_backend`` in the llm module so the
    constructor's call returns *backend* directly. Also isolates the
    cache to tmp_path.
    """
    _isolate_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(llm_module, "build_default_backend", lambda *_a, **_kw: backend)
    return LLM()


def describe_send_chat_completion():
    def it_returns_the_content_string(monkeypatch, tmp_path):
        fake = _FakeBackend('{"is_ml": true}')
        llm = _llm_with_backend(monkeypatch, tmp_path, fake)
        assert llm.send_chat_completion("m", "p", _SCHEMA_JSON) == '{"is_ml": true}'

    def it_posts_the_openai_chat_completion_payload(monkeypatch, tmp_path):
        fake = _FakeBackend('{"is_ml": true}')
        llm = _llm_with_backend(monkeypatch, tmp_path, fake)
        llm.send_chat_completion("qwen3:8b", "abstract", _SCHEMA_JSON)
        payload = fake.calls[0]["payload"]
        assert payload["model"] == "qwen3:8b"
        assert payload["messages"] == [{"role": "user", "content": "abstract"}]
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["strict"] is True

    def it_raises_llmerror_on_invalid_json(monkeypatch, tmp_path):
        fake = _FakeBackend("not json at all")
        llm = _llm_with_backend(monkeypatch, tmp_path, fake)
        with pytest.raises(LLMError):
            llm.send_chat_completion("m", "p", _SCHEMA_JSON)

    def it_raises_llmerror_on_a_malformed_response_shape(monkeypatch, tmp_path):
        def odd(payload, timeout):
            return {"error": "boom"}
        llm = _llm_with_backend(monkeypatch, tmp_path, odd)
        with pytest.raises(LLMError):
            llm.send_chat_completion("m", "p", _SCHEMA_JSON)

    def it_raises_llmerror_on_an_http_error(monkeypatch, tmp_path):
        def boom(payload, timeout):
            raise httpx.ConnectError("backend down")
        llm = _llm_with_backend(monkeypatch, tmp_path, boom)
        with pytest.raises(LLMError):
            llm.send_chat_completion("m", "p", _SCHEMA_JSON)


def describe_llm_credentials():
    def it_reads_api_key_from_env_when_unset(monkeypatch, tmp_path):
        monkeypatch.setenv(API_KEY_ENV, "env-secret")
        _isolate_cache(monkeypatch, tmp_path)
        captured: dict = {}

        def mock_post(self, url, *, json, headers, timeout):
            captured["auth"] = headers.get("authorization")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"is_ml": true}'}}]},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.Client, "post", mock_post)
        llm = LLM()
        llm.send_chat_completion("m", "p", _SCHEMA_JSON)

        assert captured["auth"] == "Bearer env-secret"

    def it_omits_authorization_when_no_key_is_present(monkeypatch, tmp_path):
        monkeypatch.delenv(API_KEY_ENV, raising=False)
        _isolate_cache(monkeypatch, tmp_path)
        captured: dict = {}

        def mock_post(self, url, *, json, headers, timeout):
            captured["headers"] = dict(headers)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"is_ml": true}'}}]},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.Client, "post", mock_post)
        llm = LLM()
        llm.send_chat_completion("m", "p", _SCHEMA_JSON)

        assert "authorization" not in captured["headers"]


def describe_llm_cache():
    def it_serves_a_repeat_call_from_disk(monkeypatch, tmp_path):
        fake = _FakeBackend('{"is_ml": true}')
        llm = _llm_with_backend(monkeypatch, tmp_path, fake)

        a = llm.send_chat_completion("m", "p", _SCHEMA_JSON)
        b = llm.send_chat_completion("m", "p", _SCHEMA_JSON)

        assert a == b == '{"is_ml": true}'
        assert len(fake.calls) == 1

    def it_keys_separately_per_prompt(monkeypatch, tmp_path):
        fake = _FakeBackend('{"is_ml": true}')
        llm = _llm_with_backend(monkeypatch, tmp_path, fake)

        llm.send_chat_completion("m", "p1", _SCHEMA_JSON)
        llm.send_chat_completion("m", "p2", _SCHEMA_JSON)

        assert len(fake.calls) == 2

    def it_keys_separately_per_schema(monkeypatch, tmp_path):
        fake = _FakeBackend('{"is_ml": true}')
        llm = _llm_with_backend(monkeypatch, tmp_path, fake)

        llm.send_chat_completion("m", "p", _SCHEMA_JSON)
        llm.send_chat_completion("m", "p", _SCHEMA_JSON + " ")  # different json text

        assert len(fake.calls) == 2

    def it_keys_separately_per_model(monkeypatch, tmp_path):
        fake = _FakeBackend('{"is_ml": true}')
        llm = _llm_with_backend(monkeypatch, tmp_path, fake)

        llm.send_chat_completion("m1", "p", _SCHEMA_JSON)
        llm.send_chat_completion("m2", "p", _SCHEMA_JSON)

        assert len(fake.calls) == 2

    def it_does_not_cache_a_malformed_response(monkeypatch, tmp_path):
        attempts = {"n": 0}

        def flaky(payload, timeout):
            attempts["n"] += 1
            content = "not json" if attempts["n"] == 1 else '{"is_ml": true}'
            return {"choices": [{"message": {"content": content}}]}

        llm = _llm_with_backend(monkeypatch, tmp_path, flaky)

        with pytest.raises(LLMError):
            llm.send_chat_completion("m", "p", _SCHEMA_JSON)
        result = llm.send_chat_completion("m", "p", _SCHEMA_JSON)

        assert result == '{"is_ml": true}'
        assert attempts["n"] == 2  # the bad response was not cached

    def it_does_not_cache_a_transient_backend_failure(monkeypatch, tmp_path):
        attempts = {"n": 0}

        def flaky(payload, timeout):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise httpx.ConnectError("backend down")
            return {"choices": [{"message": {"content": '{"is_ml": true}'}}]}

        llm = _llm_with_backend(monkeypatch, tmp_path, flaky)

        with pytest.raises(LLMError):
            llm.send_chat_completion("m", "p", _SCHEMA_JSON)
        result = llm.send_chat_completion("m", "p", _SCHEMA_JSON)

        assert result == '{"is_ml": true}'
        assert attempts["n"] == 2

    def it_persists_cache_across_separate_llm_instances(monkeypatch, tmp_path):
        # A fresh process (modeled by a new LLM) reading the same cache
        # reuses what the first wrote -- proving the cache is on disk,
        # not in process memory.
        _isolate_cache(monkeypatch, tmp_path)

        fake_one = _FakeBackend('{"is_ml": true}')
        monkeypatch.setattr(llm_module, "build_default_backend", lambda *_a, **_kw: fake_one)
        llm_one = LLM()
        llm_one.send_chat_completion("m", "p", _SCHEMA_JSON)

        fake_two = _FakeBackend('{"is_ml": true}')
        monkeypatch.setattr(llm_module, "build_default_backend", lambda *_a, **_kw: fake_two)
        llm_two = LLM()
        result = llm_two.send_chat_completion("m", "p", _SCHEMA_JSON)

        assert result == '{"is_ml": true}'
        assert len(fake_one.calls) == 1
        assert len(fake_two.calls) == 0  # served from disk


def describe_llm_retry():
    def it_retries_on_a_transient_5xx_then_succeeds(monkeypatch, tmp_path):
        monkeypatch.setattr(bdb_module, "_BACKOFF_BASE_S", 0)
        _isolate_cache(monkeypatch, tmp_path)
        calls = {"n": 0}

        def mock_post(self, url, *, json, headers, timeout):
            calls["n"] += 1
            req = httpx.Request("POST", url)
            if calls["n"] == 1:
                return httpx.Response(503, text="overloaded", request=req)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"is_ml": true}'}}]},
                request=req,
            )

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        llm = LLM()
        result = llm.send_chat_completion("m", "p", _SCHEMA_JSON)

        assert calls["n"] == 2
        assert result == '{"is_ml": true}'

    def it_does_not_retry_on_a_non_retryable_4xx(monkeypatch, tmp_path):
        monkeypatch.setattr(bdb_module, "_BACKOFF_BASE_S", 0)
        _isolate_cache(monkeypatch, tmp_path)
        calls = {"n": 0}

        def mock_post(self, url, *, json, headers, timeout):
            calls["n"] += 1
            return httpx.Response(
                401, text="bad token", request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        llm = LLM()
        with pytest.raises(LLMError):
            llm.send_chat_completion("m", "p", _SCHEMA_JSON)
        assert calls["n"] == 1

    def it_gives_up_after_three_attempts(monkeypatch, tmp_path):
        monkeypatch.setattr(bdb_module, "_BACKOFF_BASE_S", 0)
        _isolate_cache(monkeypatch, tmp_path)
        calls = {"n": 0}

        def mock_post(self, url, *, json, headers, timeout):
            calls["n"] += 1
            return httpx.Response(
                503, text="still overloaded",
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        llm = LLM()
        with pytest.raises(LLMError):
            llm.send_chat_completion("m", "p", _SCHEMA_JSON)
        assert calls["n"] == 3
