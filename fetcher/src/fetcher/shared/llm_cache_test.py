"""Unit tests for the LLM cachetta wrapper.

Mirror the on-disk caching pattern in ``download_test.py``: build the
cache against a tmp dir, run a decorated function twice, assert the
second call did not invoke the underlying callable.
"""

from __future__ import annotations

import pytest

from fetcher.shared.llm_cache import cache_key, cache_path, make_llm_cache


def describe_cache_key():
    def it_is_stable_for_the_same_inputs():
        a = cache_key("phi4:14b", "abstract X", '{"k":1}')
        b = cache_key("phi4:14b", "abstract X", '{"k":1}')
        assert a == b

    def it_differs_when_model_changes():
        a = cache_key("phi4:14b", "abstract X", '{"k":1}')
        b = cache_key("qwen3:8b", "abstract X", '{"k":1}')
        assert a != b

    def it_differs_when_prompt_changes():
        a = cache_key("phi4:14b", "abstract X", '{"k":1}')
        b = cache_key("phi4:14b", "abstract Y", '{"k":1}')
        assert a != b

    def it_differs_when_schema_changes():
        a = cache_key("phi4:14b", "abstract X", '{"k":1}')
        b = cache_key("phi4:14b", "abstract X", '{"k":2}')
        assert a != b

    def it_is_immune_to_input_concatenation_collisions():
        # ("ab", "c") and ("a", "bc") would collide under naive a+b
        # concat; the NUL separator inside cache_key prevents that.
        a = cache_key("ab", "c", "{}")
        b = cache_key("a", "bc", "{}")
        assert a != b


def describe_cache_path():
    def it_routes_under_an_llm_subdir(tmp_path):
        p = cache_path(tmp_path, "phi4:14b", "abstract", "{}")
        assert p.parent == tmp_path / "llm"
        assert p.suffix == ".pkl"


def describe_make_llm_cache():
    def it_serves_a_repeat_call_from_disk(tmp_path):
        cache = make_llm_cache(tmp_path)
        calls = {"n": 0}

        @cache
        def fake_send(model: str, prompt: str, schema_json: str) -> str:
            calls["n"] += 1
            return '{"output": true}'

        a = fake_send("phi4:14b", "hello", '{"k":1}')
        b = fake_send("phi4:14b", "hello", '{"k":1}')

        assert a == b == '{"output": true}'
        assert calls["n"] == 1  # second call read from disk, no invocation

    def it_keys_separately_per_prompt(tmp_path):
        cache = make_llm_cache(tmp_path)
        calls = {"n": 0}

        @cache
        def fake_send(model, prompt, schema_json) -> str:
            calls["n"] += 1
            return f'{{"prompt": "{prompt}"}}'

        fake_send("phi4:14b", "one", "{}")
        fake_send("phi4:14b", "two", "{}")
        fake_send("phi4:14b", "one", "{}")  # repeat -- cached

        assert calls["n"] == 2

    def it_does_not_cache_when_the_function_raises(tmp_path):
        cache = make_llm_cache(tmp_path)
        attempts = {"n": 0}

        @cache
        def fake_send(model, prompt, schema_json) -> str:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("transient")
            return '{"output": true}'

        with pytest.raises(RuntimeError):
            fake_send("phi4:14b", "hello", "{}")
        # Second attempt: cache had nothing to serve, function runs.
        result = fake_send("phi4:14b", "hello", "{}")
        assert result == '{"output": true}'
        assert attempts["n"] == 2

    def it_does_not_cache_an_empty_response(tmp_path):
        # condition rejects "" / whitespace-only -- a model returning a
        # blank content string is a bug, not a stable answer.
        cache = make_llm_cache(tmp_path)
        attempts = {"n": 0}

        @cache
        def fake_send(model, prompt, schema_json) -> str:
            attempts["n"] += 1
            return "" if attempts["n"] == 1 else '{"output": true}'

        first = fake_send("phi4:14b", "hello", "{}")
        second = fake_send("phi4:14b", "hello", "{}")

        assert first == ""
        assert second == '{"output": true}'
        assert attempts["n"] == 2  # first response was not cached
