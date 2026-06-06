"""Unit tests for the deterministic cache-key hash."""

from __future__ import annotations

from pathlib import Path

from fetcher.shared.hash import hash


def describe_hash():
    def it_is_stable_for_the_same_inputs():
        assert hash("a", "b") == hash("a", "b")

    def it_differs_for_different_positional_args():
        assert hash("a") != hash("b")

    def it_differs_for_different_kwargs():
        assert hash(x=1) != hash(x=2)

    def it_is_kwarg_order_independent():
        assert hash(x=1, y=2) == hash(y=2, x=1)

    def it_distinguishes_positional_from_kwargs():
        assert hash("a") != hash(x="a")

    def it_coerces_non_json_native_via_str():
        # Path is not JSON-native; default=str must serialize it.
        a = hash(Path("/tmp/x"))
        b = hash(Path("/tmp/x"))
        assert a == b

    def it_returns_a_16_hex_string():
        h = hash("anything")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)
