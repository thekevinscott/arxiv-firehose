"""Unit tests for the transport's retry classifier.

``http_get`` itself is exercised through its integration callers with
``unittest.mock.patch.object`` -- no live network in tests.
"""

import httpx

from fetcher.shared.http.is_retryable import is_retryable


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://arxiv.org/x")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError(f"HTTP {code}", request=request, response=response)


def describe_is_retryable():
    def it_retries_a_503():
        assert is_retryable(_status_error(503)) is True

    def it_does_not_retry_a_404():
        assert is_retryable(_status_error(404)) is False

    def it_retries_a_transport_error():
        exc = httpx.ConnectError("connection refused")
        assert is_retryable(exc) is True

    def it_does_not_retry_an_unrelated_exception():
        assert is_retryable(ValueError("nope")) is False
