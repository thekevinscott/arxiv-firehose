"""TOML configuration: load, validate, and bootstrap a default."""

from __future__ import annotations

import os
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
# Cache root. Honors ``ARXIV_FIREHOSE_CACHE_DIR`` for deployments that store
# the cache off the home volume (e.g. tower keeps it on /mnt/bertha so a
# disk swap doesn't lose the 100-year paper bytes).
DEFAULT_CACHE_DIR = Path(
    os.environ.get("ARXIV_FIREHOSE_CACHE_DIR")
    or Path.home() / ".cache" / "arxiv-firehose"
)
DEFAULT_CACHE_DURATION = timedelta(days=3650)

DEFAULT_CONFIG_TOML = """\
[categories]
# arxiv subject classifiers to track. The daily cron queries the export
# API for all of them in one day-slice per day of the lookback window.
# Tuned for ML + adjacent (security, ethics, audio) + HCI.
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
# Hard cap on concurrent requests. arxiv source must stay at 1.
concurrency = 1

[ingest]
# Lookback window: every sync re-reads this many days of export-API day
# slices (settled slices come from the ~forever cache, so only new or
# missed days cost a real request). A missed cron day self-heals as long
# as another run happens within the window.
backfill_days = 90
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


class IngestConfig(BaseModel):
    backfill_days: int = 90


class Config(BaseModel):
    categories: CategoriesConfig = Field(default_factory=CategoriesConfig)
    fetch: FetchConfig = Field(default_factory=FetchConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)

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
