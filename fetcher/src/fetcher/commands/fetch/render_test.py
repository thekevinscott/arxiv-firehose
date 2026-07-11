"""Unit tests for the render stage's download-error summarising.

Tar extraction now lives in convert.py (the LaTeX fallback owns it) -- its
tests moved to convert_test.py.
"""

import json
import logging
from unittest.mock import patch

import httpx

from fetcher.commands.fetch import render
from fetcher.commands.fetch.render import _http_error_summary
from fetcher.shared.config import Config


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


def _paper_dir(root, arxiv_id: str) -> None:
    d = root / arxiv_id
    d.mkdir()
    (d / "metadata.json").write_text(
        json.dumps(
            {
                "arxiv_id": arxiv_id,
                "html_url": f"https://arxiv.org/abs/{arxiv_id}v1",
                "source_url": f"https://arxiv.org/src/{arxiv_id}v1",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}v1",
            }
        )
    )


def describe_run():
    def it_stops_the_run_at_the_first_429(tmp_path):
        _paper_dir(tmp_path, "2401.00001")
        _paper_dir(tmp_path, "2401.00002")
        calls: list[str] = []

        def fake(url):
            calls.append(url)
            raise _status_error(429)

        with (
            patch.object(render.download, "fetch_html", side_effect=fake),
            patch.object(render.download, "fetch_paper", side_effect=fake),
        ):
            render.run(tmp_path, Config(), logging.getLogger("test"))

        assert len(calls) == 1

    def it_writes_no_false_absent_marker_on_a_429(tmp_path):
        _paper_dir(tmp_path, "2401.00001")

        def fake(url):
            raise _status_error(429)

        with (
            patch.object(render.download, "fetch_html", side_effect=fake),
            patch.object(render.download, "fetch_paper", side_effect=fake),
        ):
            render.run(tmp_path, Config(), logging.getLogger("test"))

        assert not (tmp_path / "2401.00001" / ".no_markdown").exists()


def describe_http_error_summary():
    def it_reduces_a_status_error_to_the_status_code():
        assert _http_error_summary(_status_error(404)) == "HTTP 404"

    def it_drops_the_mdn_documentation_link():
        assert "developer.mozilla" not in _http_error_summary(_status_error(404))

    def it_summarises_a_non_http_error_by_type_and_message():
        assert _http_error_summary(ValueError("response was not a PDF")) == (
            "ValueError: response was not a PDF"
        )
