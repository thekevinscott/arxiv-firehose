"""Unit tests for the OpenAI-compatible http_classifier.

The real HTTP send is never made -- a fake backend is injected, mirroring
the ``Transport`` / ``Converter`` pattern (see AGENTS.md).
"""

from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel

from fetcher.commands.classify import (
    API_KEY_ENV,
    Classifier,
    ClassifyError,
    http_classifier,
)


class _FakeBackend:
    """Records every POST and returns a scripted chat-completions response.

    Mirrors the OpenAI/Ollama wire shape: ``{"choices":[{"message":
    {"content": "<json>"}}]}``. The ``content`` is whatever JSON string the
    test scripts; the classifier must json-validate it into the schema.
    """

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def __call__(self, payload: dict, timeout: float) -> dict:
        self.calls.append({"payload": payload, "timeout": timeout})
        return {"choices": [{"message": {"content": self.content}}]}


class _Out(BaseModel):
    is_about_ml: bool


def describe_http_classifier():
    def it_posts_an_openai_compatible_chat_completion():
        fake = _FakeBackend('{"is_about_ml": true}')
        clf = http_classifier("qwen3:8b", backend=fake)

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

    def it_validates_the_response_into_the_pydantic_model():
        fake = _FakeBackend('{"is_about_safety": false}')

        class Out(BaseModel):
            is_about_safety: bool

        clf = http_classifier("qwen3:8b", backend=fake)

        assert clf.call("any prompt", Out).is_about_safety is False

    def it_raises_classifyerror_on_invalid_json():
        # A Pydantic ValidationError must surface as our own error type so
        # callers don't depend on Pydantic internals.
        fake = _FakeBackend("not json at all")
        clf = http_classifier("qwen3:8b", backend=fake)

        with pytest.raises(ClassifyError):
            clf.call("any prompt", _Out)

    def it_raises_classifyerror_when_required_field_is_missing():
        fake = _FakeBackend('{"some_other_field": true}')
        clf = http_classifier("qwen3:8b", backend=fake)

        with pytest.raises(ClassifyError):
            clf.call("any prompt", _Out)

    def it_raises_classifyerror_on_http_error():
        # A connection failure surfaces as ClassifyError, not an httpx
        # exception that leaks through.
        def boom(payload, timeout):
            raise httpx.ConnectError("backend down")

        clf = http_classifier("qwen3:8b", backend=boom)

        with pytest.raises(ClassifyError):
            clf.call("any prompt", _Out)

    def it_raises_classifyerror_on_malformed_response_shape():
        # Some gateways return errors as 200s with no `choices` -- catch
        # that cleanly rather than KeyError.
        def odd_shape(payload, timeout):
            return {"error": "rate limited"}

        clf = http_classifier("qwen3:8b", backend=odd_shape)

        with pytest.raises(ClassifyError):
            clf.call("any prompt", _Out)


def describe_http_classifier_credentials():
    def it_reads_api_key_from_env_when_config_is_empty(monkeypatch):
        # api_key="" in config.toml must fall back to FETCHER_LLM_API_KEY so
        # the secret stays out of files that get backed up and snapshotted.
        # Assert via the default httpx-backed path: build the real client,
        # then short-circuit the send through a fake httpx transport.
        monkeypatch.setenv(API_KEY_ENV, "env-secret")
        captured: dict = {}

        def fake_handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"is_about_ml": true}'}}]},
            )

        # Replace the underlying httpx.Client transport before http_classifier
        # opens it. Easiest: monkeypatch httpx.Client to use MockTransport.
        original_post = httpx.Client.post

        def mock_post(self, url, *, json, headers, timeout):
            req = httpx.Request("POST", url, json=json, headers=headers)
            return fake_handler(req)

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        clf = http_classifier("qwen3:8b", api_key="")
        clf.call("any prompt", _Out)

        assert captured["auth"] == "Bearer env-secret"

    def it_prefers_explicit_api_key_over_env(monkeypatch):
        monkeypatch.setenv(API_KEY_ENV, "env-secret")
        captured: dict = {}

        def mock_post(self, url, *, json, headers, timeout):
            captured["auth"] = headers.get("authorization")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"is_about_ml": true}'}}]},
            )

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        clf = http_classifier("qwen3:8b", api_key="explicit-key")
        clf.call("any prompt", _Out)

        assert captured["auth"] == "Bearer explicit-key"

    def it_omits_authorization_when_no_key_is_present(monkeypatch):
        monkeypatch.delenv(API_KEY_ENV, raising=False)
        captured: dict = {}

        def mock_post(self, url, *, json, headers, timeout):
            captured["headers"] = dict(headers)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"is_about_ml": true}'}}]},
            )

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        clf = http_classifier("qwen3:8b")
        clf.call("any prompt", _Out)

        assert "authorization" not in captured["headers"]


def describe_http_classifier_cache():
    def it_serves_a_repeat_call_from_the_disk_cache(tmp_path):
        # With cache_dir set, the same (model, prompt, schema) hits cachetta
        # on the second call -- the fake backend is never invoked again.
        # This is the only allowed caching mechanism in classify.
        fake = _FakeBackend('{"is_about_ml": true}')
        clf = http_classifier("qwen3:8b", backend=fake, cache_dir=tmp_path)

        a = clf.call("abstract X", _Out)
        b = clf.call("abstract X", _Out)

        assert isinstance(a, _Out) and isinstance(b, _Out)
        assert a.is_about_ml is True and b.is_about_ml is True
        assert len(fake.calls) == 1

    def it_keys_separately_per_prompt(tmp_path):
        # Different prompt -> different cache file -> backend reinvoked.
        fake = _FakeBackend('{"is_about_ml": true}')
        clf = http_classifier("qwen3:8b", backend=fake, cache_dir=tmp_path)

        clf.call("abstract X", _Out)
        clf.call("abstract Y", _Out)

        assert len(fake.calls) == 2

    def it_keys_separately_per_schema(tmp_path):
        # Same prompt, different schema -> different cache file. Two
        # categories with the same abstract still each get an LLM call.
        from pydantic import BaseModel as _BM

        class Other(_BM):
            is_about_safety: bool

        # The fake must answer correctly for whichever schema is asked.
        # Two backends because _FakeBackend returns one canned content.
        ml_fake = _FakeBackend('{"is_about_ml": true}')
        clf_ml = http_classifier("qwen3:8b", backend=ml_fake, cache_dir=tmp_path)
        safety_fake = _FakeBackend('{"is_about_safety": false}')
        clf_safety = http_classifier("qwen3:8b", backend=safety_fake, cache_dir=tmp_path)

        clf_ml.call("abstract X", _Out)
        clf_safety.call("abstract X", Other)

        assert len(ml_fake.calls) == 1
        assert len(safety_fake.calls) == 1

    def it_does_not_cache_a_malformed_response(tmp_path):
        # A non-JSON content raises ClassifyError before cachetta sees a
        # return value -- the next call retries the backend.
        attempts = {"n": 0}

        def flaky(payload, timeout):
            attempts["n"] += 1
            content = "not json" if attempts["n"] == 1 else '{"is_about_ml": true}'
            return {"choices": [{"message": {"content": content}}]}

        clf = http_classifier("qwen3:8b", backend=flaky, cache_dir=tmp_path)

        with pytest.raises(ClassifyError):
            clf.call("abstract X", _Out)
        result = clf.call("abstract X", _Out)

        assert result.is_about_ml is True
        assert attempts["n"] == 2  # the bad response was not cached

    def it_does_not_cache_a_transient_backend_failure(tmp_path):
        # Same key, exception first, success second -> backend called
        # twice because exceptions are never cached.
        attempts = {"n": 0}

        def flaky(payload, timeout):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise httpx.ConnectError("backend down")
            return {"choices": [{"message": {"content": '{"is_about_ml": true}'}}]}

        clf = http_classifier("qwen3:8b", backend=flaky, cache_dir=tmp_path)

        with pytest.raises(ClassifyError):
            clf.call("abstract X", _Out)
        result = clf.call("abstract X", _Out)

        assert result.is_about_ml is True
        assert attempts["n"] == 2

    def it_persists_cache_across_separate_classifier_instances(tmp_path):
        # A fresh process (modeled by a new classifier) reading the same
        # cache_dir reuses what the first wrote -- proving the cache is
        # on disk, not in process memory.
        fake_one = _FakeBackend('{"is_about_ml": true}')
        clf_one = http_classifier("qwen3:8b", backend=fake_one, cache_dir=tmp_path)
        clf_one.call("abstract X", _Out)

        fake_two = _FakeBackend('{"is_about_ml": true}')
        clf_two = http_classifier("qwen3:8b", backend=fake_two, cache_dir=tmp_path)
        result = clf_two.call("abstract X", _Out)

        assert result.is_about_ml is True
        assert len(fake_one.calls) == 1
        assert len(fake_two.calls) == 0  # served from disk


def describe_http_classifier_retry():
    def it_retries_on_a_transient_5xx_then_succeeds(monkeypatch):
        # 503 once, then 200. The classifier must retry and return the
        # parsed schema instance, not raise.
        from fetcher.commands.classify import http as http_module
        monkeypatch.setattr(http_module, "_BACKOFF_BASE_S", 0)
        calls = {"n": 0}

        def mock_post(self, url, *, json, headers, timeout):
            calls["n"] += 1
            req = httpx.Request("POST", url)
            if calls["n"] == 1:
                return httpx.Response(503, text="overloaded", request=req)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"is_about_ml": true}'}}]},
                request=req,
            )

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        clf = http_classifier("qwen3:8b")
        result = clf.call("any prompt", _Out)

        assert calls["n"] == 2
        assert result.is_about_ml is True

    def it_does_not_retry_on_a_non_retryable_4xx(monkeypatch):
        # 401 is a credentials bug, not a blip; surface immediately.
        from fetcher.commands.classify import http as http_module
        monkeypatch.setattr(http_module, "_BACKOFF_BASE_S", 0)
        calls = {"n": 0}

        def mock_post(self, url, *, json, headers, timeout):
            calls["n"] += 1
            return httpx.Response(401, text="bad token",
                                  request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        clf = http_classifier("qwen3:8b")

        with pytest.raises(ClassifyError):
            clf.call("any prompt", _Out)
        assert calls["n"] == 1

    def it_gives_up_after_three_attempts(monkeypatch):
        from fetcher.commands.classify import http as http_module
        monkeypatch.setattr(http_module, "_BACKOFF_BASE_S", 0)
        calls = {"n": 0}

        def mock_post(self, url, *, json, headers, timeout):
            calls["n"] += 1
            return httpx.Response(503, text="still overloaded",
                                  request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx.Client, "post", mock_post)

        clf = http_classifier("qwen3:8b")
        with pytest.raises(ClassifyError):
            clf.call("any prompt", _Out)
        assert calls["n"] == 3
