"""classify: label each paper's abstract with binary topic flags.

A compiled coaxer prompt artifact + an OpenAI-compatible chat-completions
endpoint per flag, written to ``classification.json`` per paper. The LLM
backend is an injection seam (``Classifier``); the default speaks the
de-facto cross-vendor ``/v1/chat/completions`` shape, so swapping Ollama,
vLLM, llama.cpp, OpenAI or LiteLLM is a config edit, not a code edit.

Public surface re-exported from sibling modules:
- types: Classifier, ClassifyError, HttpBackend
- http:  http_classifier
- coaxed: load_coaxed
- run:   run
"""

from .coaxed import load_coaxed
from .http import API_KEY_ENV, http_classifier
from .run import run
from .types import Classifier, ClassifyError, HttpBackend

__all__ = [
    "API_KEY_ENV",
    "Classifier",
    "ClassifyError",
    "HttpBackend",
    "http_classifier",
    "load_coaxed",
    "run",
]
