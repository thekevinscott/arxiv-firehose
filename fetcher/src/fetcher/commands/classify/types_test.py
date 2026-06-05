"""Unit tests for the Classifier injection seam (no LLM, no HTTP)."""

from __future__ import annotations

from pydantic import BaseModel

from fetcher.commands.classify import Classifier


def describe_Classifier():
    def it_is_a_simple_callable_seam():
        # Classifier is a wrapper around a callable -- the dataclass holds
        # the function and forwards arguments untouched. Real wiring is
        # exercised through http_classifier and the integration suite.
        def fake_call(prompt, schema):
            return schema(is_about_ml=("ml" in prompt.lower()))

        class Out(BaseModel):
            is_about_ml: bool

        clf = Classifier(call=fake_call)
        result = clf.call("ML paper abstract", Out)
        assert isinstance(result, Out)
        assert result.is_about_ml is True
