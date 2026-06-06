"""classify: label each paper's abstract with binary topic flags.

A compiled coaxer prompt artifact + an OpenAI-compatible chat-completions
endpoint per flag, written to ``classifications/<cat>.json`` per paper.
The LLM backend is an injection seam (``Classifier``); the default
speaks the de-facto cross-vendor ``/v1/chat/completions`` shape via
``shared.llm.LLM``, so swapping Ollama, vLLM, llama.cpp, OpenAI or
LiteLLM is a config edit, not a code edit.

Public surface re-exported from sibling modules:
- types:  Classifier, ClassifyError
- http:   http_classifier
- coaxed: load_coaxed
- run:    run
"""

from .coaxed import load_coaxed
from .http import http_classifier
from .run import run
from .types import Classifier, ClassifyError

__all__ = [
    "Classifier",
    "ClassifyError",
    "http_classifier",
    "load_coaxed",
    "run",
]
