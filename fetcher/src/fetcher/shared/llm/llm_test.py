"""Unit tests for shared.llm.LLM.

Covers payload shape, error mapping, retry/backoff, and the "don't cache
poison" guarantee. Generic cache-behavior tests (keys per arg, repeat hit,
cross-instance persistence) belong to cachetta and are not duplicated here.

Patch strategy: ``unittest.mock.patch.object`` only -- never ``monkeypatch``.

- Tests that don't care about caching replace ``LLM.send_chat_completion``
  with its ``__wrapped__`` (the raw, uncached function functools preserves).
- Cache-poison tests need a real on-disk cache to observe non-storage; they
  rewrap the raw fn with a tmp-path Cachetta and patch the class attribute
  with that. No private ``_cache`` access in either case.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import httpx
import pytest
from cachetta import Cachetta

import fetcher.shared.build_default_backend as bdb_module
from fetcher.shared.llm import LLM, LLMError

# `fetcher.shared.llm.llm` as an attribute is shadowed by the singleton
# in shared/llm/__init__.py; reach the submodule via sys.modules to
# patch names in its namespace (e.g. build_default_backend).
llm_module = sys.modules["fetcher.shared.llm.llm"]


class _FakeBackend:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def __call__(self, payload: dict, timeout: float) -> dict:
        self.calls.append({"payload": payload, "timeout": timeout})
        return {"choices": [{"message": {"content": self.content}}]}


_SCHEMA_JSON = '{"type":"object","properties":{"is_ml":{"type":"boolean"}},"required":["is_ml"]}'


@pytest.fixture
def uncached():
    """Replace ``send_chat_completion`` with its uncached raw form.

    Cache behavior is cachetta's concern; LLM's own tests assert payload
    shape and error mapping, both of which are observable without a cache.
    """
    raw = LLM.send_chat_completion.__wrapped__
    with patch.object(LLM, "send_chat_completion", raw):
        yield


@pytest.fixture
def fresh_cache(tmp_path):
    """Rewrap the raw fn with a scratch Cachetta under tmp_path.

    Used by cache-poison tests: they need a real on-disk cache to observe
    that a failed call was *not* stored. Same ``hashed=True`` / dir-per-arg
    layout as the production decorator.
    """
    raw = LLM.send_chat_completion.__wrapped__
    rewrapped = (Cachetta(path=tmp_path) / "llm")(hashed=True)(raw)
    with patch.object(LLM, "send_chat_completion", rewrapped):
        yield


def _llm_with_backend(backend) -> LLM:
    """Construct an LLM whose ``_backend`` is *backend*.

    Patches ``build_default_backend`` in the llm namespace so the
    constructor wires the fake in directly. The patch only needs to be
    live during construction -- the resulting LLM stores ``backend`` on
    ``self._backend`` and uses it for the rest of its life.
    """
    with patch.object(llm_module, "build_default_backend", lambda *_a, **_kw: backend):
        return LLM()


def describe_send_chat_completion():
    def it_returns_the_content_string(uncached):
        fake = _FakeBackend('{"is_ml": true}')
        llm = _llm_with_backend(fake)
        assert llm.send_chat_completion("m", "p", _SCHEMA_JSON) == '{"is_ml": true}'

    def it_posts_the_openai_chat_completion_payload(uncached):
        fake = _FakeBackend('{"is_ml": true}')
        llm = _llm_with_backend(fake)
        llm.send_chat_completion("qwen3:8b", "abstract", _SCHEMA_JSON)
        payload = fake.calls[0]["payload"]
        assert payload["model"] == "qwen3:8b"
        assert payload["messages"] == [{"role": "user", "content": "abstract"}]
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["strict"] is True

    def it_raises_llmerror_on_invalid_json(uncached):
        fake = _FakeBackend("not json at all")
        llm = _llm_with_backend(fake)
        with pytest.raises(LLMError):
            llm.send_chat_completion("m", "p", _SCHEMA_JSON)

    def it_raises_llmerror_on_a_malformed_response_shape(uncached):
        def odd(payload, timeout):
            return {"error": "boom"}
        llm = _llm_with_backend(odd)
        with pytest.raises(LLMError):
            llm.send_chat_completion("m", "p", _SCHEMA_JSON)

    def it_raises_llmerror_on_an_http_error(uncached):
        def boom(payload, timeout):
            raise httpx.ConnectError("backend down")
        llm = _llm_with_backend(boom)
        with pytest.raises(LLMError):
            llm.send_chat_completion("m", "p", _SCHEMA_JSON)


def describe_llm_cache():
    def it_does_not_cache_a_malformed_response(fresh_cache):
        attempts = {"n": 0}

        def flaky(payload, timeout):
            attempts["n"] += 1
            content = "not json" if attempts["n"] == 1 else '{"is_ml": true}'
            return {"choices": [{"message": {"content": content}}]}

        llm = _llm_with_backend(flaky)

        with pytest.raises(LLMError):
            llm.send_chat_completion("m", "p", _SCHEMA_JSON)
        result = llm.send_chat_completion("m", "p", _SCHEMA_JSON)

        assert result == '{"is_ml": true}'
        assert attempts["n"] == 2  # the bad response was not cached

    def it_does_not_cache_a_transient_backend_failure(fresh_cache):
        attempts = {"n": 0}

        def flaky(payload, timeout):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise httpx.ConnectError("backend down")
            return {"choices": [{"message": {"content": '{"is_ml": true}'}}]}

        llm = _llm_with_backend(flaky)

        with pytest.raises(LLMError):
            llm.send_chat_completion("m", "p", _SCHEMA_JSON)
        result = llm.send_chat_completion("m", "p", _SCHEMA_JSON)

        assert result == '{"is_ml": true}'
        assert attempts["n"] == 2


def describe_llm_retry():
    def it_retries_on_a_transient_5xx_then_succeeds(uncached):
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

        with patch.object(bdb_module, "_BACKOFF_BASE_S", 0), \
             patch.object(httpx.Client, "post", mock_post):
            llm = LLM()
            result = llm.send_chat_completion("m", "p", _SCHEMA_JSON)

        assert calls["n"] == 2
        assert result == '{"is_ml": true}'

    def it_does_not_retry_on_a_non_retryable_4xx(uncached):
        calls = {"n": 0}

        def mock_post(self, url, *, json, headers, timeout):
            calls["n"] += 1
            return httpx.Response(
                401, text="bad token", request=httpx.Request("POST", url),
            )

        with patch.object(bdb_module, "_BACKOFF_BASE_S", 0), \
             patch.object(httpx.Client, "post", mock_post):
            llm = LLM()
            with pytest.raises(LLMError):
                llm.send_chat_completion("m", "p", _SCHEMA_JSON)
        assert calls["n"] == 1

    def it_gives_up_after_three_attempts(uncached):
        calls = {"n": 0}

        def mock_post(self, url, *, json, headers, timeout):
            calls["n"] += 1
            return httpx.Response(
                503, text="still overloaded",
                request=httpx.Request("POST", url),
            )

        with patch.object(bdb_module, "_BACKOFF_BASE_S", 0), \
             patch.object(httpx.Client, "post", mock_post):
            llm = LLM()
            with pytest.raises(LLMError):
                llm.send_chat_completion("m", "p", _SCHEMA_JSON)
        assert calls["n"] == 3
