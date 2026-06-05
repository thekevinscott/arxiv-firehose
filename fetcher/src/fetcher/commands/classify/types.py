"""classify: the injection seam and shared types.

The LLM backend is hidden behind a tiny callable contract: ``Classifier``
forwards a rendered prompt plus a Pydantic schema and returns a validated
instance of that schema. The real wiring (HTTP, retries, pooling) lives in
``http.py``; tests inject a fake here directly. Mirrors the
``Transport`` / ``Converter`` pattern (see AGENTS.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel


class ClassifyError(Exception):
    """The LLM response could not be parsed into the requested schema."""


# A backend is a callable ``(payload_json, timeout_s) -> response_json``. The
# default uses httpx; tests inject a fake. Decoupling the HTTP send from the
# classifier lets unit tests assert on the exact payload going over the wire
# without monkeypatching.
HttpBackend = Callable[[dict, float], dict]


@dataclass(frozen=True)
class Classifier:
    """Inject seam: ``call(prompt, schema) -> schema-instance``.

    The default implementation POSTs to an OpenAI-compatible chat-completions
    endpoint; tests inject a fake. fetcher's SDK builds the default
    ``Classifier`` from config at call time.
    """
    call: Callable[[str, type[BaseModel]], BaseModel]
