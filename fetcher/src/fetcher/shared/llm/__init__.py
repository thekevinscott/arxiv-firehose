"""Shared OpenAI-compatible chat-completion client.

Public surface:
- ``LLM``, ``LLMError`` -- the client class and error.
- ``build_default_backend``, ``HttpBackend`` -- the byte-fetching seam.
- ``llm`` -- the process-wide singleton; consumers should import this
  rather than constructing their own.
"""

from ..build_default_backend import HttpBackend, build_default_backend
from .llm import LLM, LLMError

llm = LLM()

__all__ = [
    "HttpBackend",
    "LLM",
    "LLMError",
    "build_default_backend",
    "llm",
]
