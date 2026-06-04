"""TOML configuration: load, validate, and bootstrap a default."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# Arxiv data: organized paper folders, kept beside the package so the default
# is stable no matter which directory the command is invoked from. config.py
# lives at fetcher/src/fetcher/config.py -> parents[2] is the fetcher/
# package root, so the default data dir is fetcher/data. This holds only for
# an editable/in-repo install; a packaged (wheel) install must pass --data-dir.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
# Cache: cachetta's separate download store.
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "arxiv-firehose"

DEFAULT_CONFIG_TOML = """\
[categories]
# arxiv subject classifiers to track
include = ["cs.LG", "cs.CL", "cs.AI"]

[fetch]
# "arxiv" (HTTPS, rate-limited, free) is the only source in this prototype.
source = "arxiv"
# Hard cap on concurrent downloads. arxiv source must stay at 1.
concurrency = 1
# When a paper has no arxiv HTML, fall back to converting its LaTeX source.
latex_fallback = true
# Last resort: when a paper has no HTML and no usable LaTeX, convert its PDF.
pdf_fallback = true

[ingest]
# Skip papers older than this many days on first sync. 0 = take all of RSS.
backfill_days = 0

[classify]
# Compiled coaxer prompt artifacts -- one per binary flag. Empty list
# disables classify cleanly (the daily cron stays green while labels are
# still being authored).
prompts_dirs = []
# OpenAI-compatible /v1/chat/completions endpoint and the model tag it
# serves. The default points at a local Ollama; swap base_url + model to
# point at vLLM, llama.cpp, OpenAI, or any other compatible gateway.
model = "phi4:14b"
base_url = "http://localhost:11434/v1"
api_key = ""
timeout_s = 60.0
"""


class CategoriesConfig(BaseModel):
    include: list[str] = Field(default_factory=lambda: ["cs.LG", "cs.CL", "cs.AI"])


class FetchConfig(BaseModel):
    source: Literal["arxiv"] = "arxiv"
    concurrency: int = 1
    latex_fallback: bool = True
    pdf_fallback: bool = True


class IngestConfig(BaseModel):
    backfill_days: int = 0


class ClassifyConfig(BaseModel):
    prompts_dirs: list[str] = Field(default_factory=list)
    model: str = "phi4:14b"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    timeout_s: float = 60.0


class Config(BaseModel):
    categories: CategoriesConfig = Field(default_factory=CategoriesConfig)
    fetch: FetchConfig = Field(default_factory=FetchConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    classify: ClassifyConfig = Field(default_factory=ClassifyConfig)

    def model_post_init(self, __context: object) -> None:
        if self.fetch.source == "arxiv" and self.fetch.concurrency != 1:
            # arxiv's rate policy makes concurrency >1 a banning offense.
            self.fetch.concurrency = 1


def config_path(data_dir: Path) -> Path:
    return data_dir / "config.toml"


def load_config(data_dir: Path, config_file: Path | None = None) -> Config:
    """Load config from *config_file* (or {data_dir}/config.toml).

    If no config file exists, write the default and use it.
    """
    path = config_file or config_path(data_dir)
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONFIG_TOML)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return Config.model_validate(raw)
