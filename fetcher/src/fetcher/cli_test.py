"""Smoke tests for the CLI wrapper.

cli.py is a thin typer layer over ``api.py``; behavior lives in the SDK
and is covered by SDK tests. This file exercises just enough of each
command to prove the wiring (arg parsing, delegation, output line).
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from fetcher.cli import app

runner = CliRunner()


def describe_cli_embed():
    def it_runs_embed_dry_run_and_prints_counts(tmp_path: Path):
        # Empty data dir: nothing to embed, dry-run writes nothing.
        result = runner.invoke(
            app, ["embed", "--data-dir", str(tmp_path), "--dry-run"]
        )
        assert result.exit_code == 0, result.output
        assert "embedded=0" in result.output
        assert "total=0" in result.output
