"""arxiv ID parsing and on-disk path derivation.

Two ID formats exist:
  - modern (post-2007-04): ``YYMM.NNNNN``        e.g. ``2401.12345``
  - legacy (pre-2007-04):  ``archive/YYMMNNN``   e.g. ``cs/0501001``

Layout: one folder per paper, named by arxiv id, directly under the data dir.

    {data_dir}/
      config.toml
      last_sync.json
      logs/
      {arxiv_id}/
        metadata.json
        paper.md           the markdown rendering (or .no_markdown if none)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_MODERN_RE = re.compile(r"^(?P<num>\d{4}\.\d{4,5})$")
_LEGACY_RE = re.compile(r"^(?P<archive>[a-z-]+(?:\.[A-Z]{2})?)/(?P<num>\d{7})$")

# Reserved names at the data-dir root that are not paper folders.
RESERVED = {"logs", "config.toml", "last_sync.json"}


@dataclass(frozen=True)
class ArxivId:
    """A parsed arxiv identifier."""

    raw: str          # canonical id, e.g. '2401.12345' or 'cs/0501001'
    yymm: str         # 4-char year+month, e.g. '2401'
    is_legacy: bool

    @property
    def slug(self) -> str:
        """Filesystem-safe stem (legacy ids contain a '/')."""
        return self.raw.replace("/", "_")


def parse_id(value: str) -> ArxivId:
    """Parse an arxiv id string, stripping any trailing version suffix.

    Raises ValueError on anything that matches neither known format.
    """
    cleaned = re.sub(r"v\d+$", "", value.strip())

    m = _MODERN_RE.match(cleaned)
    if m:
        return ArxivId(raw=cleaned, yymm=cleaned[:4], is_legacy=False)

    m = _LEGACY_RE.match(cleaned)
    if m:
        return ArxivId(raw=cleaned, yymm=m.group("num")[:4], is_legacy=True)

    raise ValueError(f"unrecognized arxiv id: {value!r}")


def id_from_entry_id(entry_id: str) -> str:
    """Extract the bare id from an RSS/Atom entry id URL.

    e.g. 'oai:arXiv.org:2401.12345v2'          -> '2401.12345'
         'http://arxiv.org/abs/cs/0501001v1'   -> 'cs/0501001'
    """
    m = re.search(r"abs/([a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})", entry_id)
    if m:
        return m.group(1)
    tail = entry_id.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", tail)


def version_from_entry_id(entry_id: str) -> int:
    """Extract the version integer from an entry id; default 1 if absent."""
    m = re.search(r"v(\d+)$", entry_id.strip())
    return int(m.group(1)) if m else 1


def paper_dir(data_dir: Path, arxiv_id: str) -> Path:
    """The folder for a single paper, named by its (slugified) arxiv id."""
    return data_dir / parse_id(arxiv_id).slug


def markdown_path(data_dir: Path, arxiv_id: str) -> Path:
    """Path to a paper's markdown rendering inside its folder."""
    return paper_dir(data_dir, arxiv_id) / "paper.md"


def metadata_path(data_dir: Path, arxiv_id: str) -> Path:
    """Path to a paper's metadata.json inside its folder."""
    return paper_dir(data_dir, arxiv_id) / "metadata.json"


def iter_paper_dirs(data_dir: Path):
    """Yield every paper folder under *data_dir* (those with a metadata.json)."""
    if not data_dir.exists():
        return
    for child in sorted(data_dir.iterdir()):
        if child.is_dir() and child.name not in RESERVED:
            if (child / "metadata.json").exists():
                yield child
