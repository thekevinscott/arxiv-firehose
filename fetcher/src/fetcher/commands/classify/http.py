"""http_classifier: a Classifier backed by an OpenAI-compatible LLM.

Thin adapter over ``shared.llm.LLM``: binds a model name to a
Classifier, translates the Pydantic schema into the cache-key/payload
JSON, and converts ``LLMError`` (generic) into ``ClassifyError``
(classify's surface).

Everything substantive -- HTTP, retries, pooling, cachetta-backed
response cache, api_key resolution -- lives in ``shared.llm``. This
module exists so classify can:

- bind a single model name to a Classifier (callers don't thread
  ``model`` through every ``call``);
- compute the Pydantic-schema -> ``schema_json`` value (used both as
  part of the cache key by ``LLM`` and as the
  ``response_format.json_schema`` payload, sort_keys-stable across
  Pydantic releases);
- raise ``ClassifyError`` so callers don't depend on shared.llm
  internals or Pydantic's ValidationError.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from ...shared.llm import LLM, LLMError
from .types import Classifier, ClassifyError


def http_classifier(model: str, llm: LLM) -> Classifier:
    """Wrap *llm* into a Classifier bound to *model*.

    The Classifier exposes ``call(prompt, schema) -> schema-instance``.
    Internally that POSTs to *llm*'s OpenAI-compatible endpoint, with
    transparent retries and disk caching handled inside ``LLM``.

    *llm* carries every connection-time concern (base_url, api_key,
    timeout, cache, backend). Callers build one ``LLM`` per process and
    pass it here for each (model, category) Classifier they need; the
    underlying httpx.Client and cache are shared across them.
    """
    def call(prompt: str, schema: type[BaseModel]) -> BaseModel:
        # sort_keys keeps the cache key stable across runs even if
        # Pydantic shuffles dict order between releases.
        schema_json = json.dumps(schema.model_json_schema(), sort_keys=True)
        try:
            content = llm.send_chat_completion(model, prompt, schema_json)
        except LLMError as exc:
            raise ClassifyError(str(exc)) from exc
        try:
            return schema.model_validate_json(content)
        except (ValidationError, ValueError) as exc:
            raise ClassifyError(f"invalid LLM response: {exc}") from exc

    return Classifier(call=call)
