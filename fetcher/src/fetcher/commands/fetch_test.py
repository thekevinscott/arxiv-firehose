"""Unit tests for fetch's download-error summarising.

Tar extraction now lives in convert.py (the LaTeX fallback owns it) -- its
tests moved to convert_test.py.
"""

import httpx

from fetcher.commands.fetch import _http_error_summary


def _status_error(code: int) -> httpx.HTTPStatusError:
    """Build an httpx.HTTPStatusError the way raise_for_status() would.

    httpx hardcodes a multi-line MDN documentation link into the message.
    """
    request = httpx.Request("GET", "https://arxiv.org/pdf/2602.04555v2")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError(
        f"Client error '{code}' for url '{request.url}'\n"
        "For more information check: "
        f"https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/{code}",
        request=request,
        response=response,
    )


def describe_http_error_summary():
    def it_reduces_a_status_error_to_the_status_code():
        assert _http_error_summary(_status_error(404)) == "HTTP 404"

    def it_drops_the_mdn_documentation_link():
        assert "developer.mozilla" not in _http_error_summary(_status_error(404))

    def it_summarises_a_non_http_error_by_type_and_message():
        assert _http_error_summary(ValueError("response was not a PDF")) == (
            "ValueError: response was not a PDF"
        )
