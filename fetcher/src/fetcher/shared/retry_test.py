"""Unit tests for shared.retry.with_retry."""

from __future__ import annotations

import pytest

from fetcher.shared.retry import with_retry


class _Retryable(Exception):
    pass


class _Fatal(Exception):
    pass


def _is_retryable(exc: Exception) -> bool:
    return isinstance(exc, _Retryable)


def describe_with_retry():
    def it_returns_on_the_first_success():
        assert with_retry(
            lambda: "ok",
            is_retryable=_is_retryable,
            sleep=lambda _s: None,
        ) == "ok"

    def it_succeeds_after_two_retryable_failures():
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise _Retryable("transient")
            return "ok"

        slept = []
        assert with_retry(
            flaky, is_retryable=_is_retryable, sleep=slept.append,
        ) == "ok"
        assert len(attempts) == 3
        assert slept == [1, 2]  # base*2**0, base*2**1 -- before retries 2 and 3

    def it_gives_up_after_the_attempt_limit():
        attempts = []

        def always_fail():
            attempts.append(1)
            raise _Retryable("still failing")

        with pytest.raises(_Retryable):
            with_retry(
                always_fail,
                is_retryable=_is_retryable,
                attempts=3,
                sleep=lambda _s: None,
            )
        assert len(attempts) == 3

    def it_reraises_a_non_retryable_immediately():
        attempts = []

        def always_fatal():
            attempts.append(1)
            raise _Fatal("nope")

        with pytest.raises(_Fatal):
            with_retry(
                always_fatal, is_retryable=_is_retryable, sleep=lambda _s: None,
            )
        assert len(attempts) == 1

    def it_honors_a_custom_base():
        attempts = []
        slept = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise _Retryable("transient")
            return "ok"

        with_retry(
            flaky,
            is_retryable=_is_retryable,
            base=0.5,
            sleep=slept.append,
        )
        assert slept == [0.5, 1.0]
