"""TOML configuration: load, validate, and bootstrap a default."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from cachetta import Cachetta
from datetime import timedelta

# Arxiv data: organized paper folders, kept beside the package so the default
# is stable no matter which directory the command is invoked from. config.py
# lives at fetcher/src/fetcher/config.py -> parents[2] is the fetcher/
# package root, so the default data dir is fetcher/data. This holds only for
# an editable/in-repo install; a packaged (wheel) install must pass --data-dir.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "arxiv-firehose"
DEFAULT_CACHE_DURATION = timedelta(days=365)

# Default OpenAI-compatible LLM endpoint for classify. Points at a local
# Ollama. Not user-configurable -- the LLM client reads these constants
# directly. If you need to point at vLLM, llama.cpp, OpenAI, or another
# compatible gateway, edit these values.
DEFAULT_CLASSIFY_BASE_URL = "http://localhost:11434/v1"
DEFAULT_CLASSIFY_TIMEOUT_S = 60.0

DEFAULT_CONFIG_TOML = """\
[categories]
# arxiv subject classifiers to track. RSS feed per category; daily cron
# fetches each. Tuned for ML + adjacent (security, ethics, audio) + HCI.
include = [
    "cs.LG",   # Machine Learning
    "cs.AI",   # Artificial Intelligence
    "cs.CL",   # Computation and Language (NLP)
    "cs.CV",   # Computer Vision
    "cs.NE",   # Neural and Evolutionary Computing
    "cs.IR",   # Information Retrieval
    "cs.RO",   # Robotics
    "cs.MA",   # Multi-Agent Systems
    "stat.ML", # Machine Learning (statistics)
    "cs.HC",   # Human-Computer Interaction
    "cs.CY",   # Computers and Society (AI ethics, policy)
    "cs.CR",   # Cryptography and Security (adversarial ML)
    "eess.AS", # Audio and Speech Processing
    "cs.SD",   # Sound
]

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
# Model tag served by the OpenAI-compatible endpoint (see
# DEFAULT_CLASSIFY_BASE_URL in shared/config.py). Defaults to a local
# Ollama tag; change to whatever your gateway serves.
model = "phi4:14b"
"""


class CategoriesConfig(BaseModel):
    include: list[str] = Field(
        default_factory=lambda: [
            "cs.LG", "cs.AI", "cs.CL", "cs.CV", "cs.NE", "cs.IR", "cs.RO",
            "cs.MA", "stat.ML", "cs.HC", "cs.CY", "cs.CR", "eess.AS", "cs.SD",
        ]
    )


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

cache = Cachetta(
    path=DEFAULT_CACHE_DIR,
    duration=DEFAULT_CACHE_DURATION,
)
