"""sync-metadata: pull recent paper metadata from arxiv RSS feeds.

Writes one ``metadata.json`` per paper folder. No database -- the filesystem
is the state. RSS feeds are fetched through ``shared.download.fetch_feed``
(cachetta-cached one day), so repeated runs within a day never re-hit arxiv.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import httpx

from ...shared import download
from ...shared.atomic_write import atomic_write_json
from ...shared.config import Config
from ...shared.paths import id_from_entry_id, metadata_path, parse_id, version_from_entry_id


@dataclass
class PaperRecord:
    arxiv_id: str
    version: int
    title: str
    authors: list[str]
    abstract: str
    primary_category: str
    categories: set[str] = field(default_factory=set)
    # arxiv RSS <pubDate> is the announcement date, not the submission date.
    announced_at: str = ""
    updated_at: str = ""
    pdf_url: str = ""

    def to_metadata(self, synced_at: str) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "version": self.version,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "primary_category": self.primary_category,
            "categories": sorted(self.categories),
            "announced_at": self.announced_at,
            "updated_at": self.updated_at,
            "pdf_url": self.pdf_url,
            "html_url": f"https://arxiv.org/html/{self.arxiv_id}v{self.version}",
            "source_url": f"https://arxiv.org/e-print/{self.arxiv_id}",
            "synced_at": synced_at,
        }


def _clean_abstract(summary: str) -> str:
    """Strip the 'arXiv:... Announce Type:...' header arxiv RSS prepends."""
    text = summary.strip()
    if "Abstract:" in text:
        text = text.split("Abstract:", 1)[1]
    return text.strip()


def _announce_type(summary: str) -> str:
    """Read arxiv's 'Announce Type:' value from an RSS item description.

    One of new / cross / replace / replace-cross; '' when the header is
    absent. 'new' is a first announcement; the rest are cross-lists or
    revisions of existing (often years-old) papers.
    """
    marker = "Announce Type:"
    if marker not in summary:
        return ""
    rest = summary.split(marker, 1)[1].strip()
    return rest.split()[0].lower() if rest else ""


def _parse_entry(entry: feedparser.FeedParserDict) -> PaperRecord | None:
    # fetcher mirrors only papers first announced this week: drop cross-lists
    # and replacements, which would otherwise pull in old arxiv ids.
    if _announce_type(entry.get("summary", "")) != "new":
        return None

    raw_id = entry.get("id", "") or entry.get("link", "")
    try:
        arxiv_id = id_from_entry_id(raw_id)
        parse_id(arxiv_id)  # validate
    except ValueError:
        return None
    version = version_from_entry_id(raw_id)

    tags = [t.get("term", "") for t in entry.get("tags", []) if t.get("term")]
    primary = tags[0] if tags else "unknown"

    author_raw = entry.get("author", "")
    authors = [a.strip() for a in author_raw.split(",") if a.strip()]

    published = entry.get("published", "") or entry.get("updated", "")
    updated = entry.get("updated", "") or published

    return PaperRecord(
        arxiv_id=arxiv_id,
        version=version,
        title=" ".join(entry.get("title", "").split()),
        authors=authors,
        abstract=_clean_abstract(entry.get("summary", "")),
        primary_category=primary,
        categories=set(tags),
        announced_at=published,
        updated_at=updated,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}v{version}",
    )


def collect_records(
    config: Config,
    log: logging.Logger,
    dry_run: bool = False,
) -> dict[str, PaperRecord]:
    """Fetch every tracked feed, dedupe by arxiv_id, merge categories.

    Calls ``download.fetch_feed`` directly: that function is the
    cachetta-cached seam. Tests patch it (or the underlying ``_http_get``)
    with ``unittest.mock.patch.object``.
    """
    records: dict[str, PaperRecord] = {}
    cutoff = None
    if config.ingest.backfill_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=config.ingest.backfill_days)

    for category in config.categories.include:
        if dry_run:
            log.info("[dry-run] would fetch feed for %s", category)
            continue
        try:
            content = download.fetch_feed(category)  # cached 1 day
        except httpx.HTTPError as exc:
            log.error("feed fetch failed for %s: %s", category, exc)
            continue

        feed = feedparser.parse(content)
        log.info("feed %s: %d entries", category, len(feed.entries))
        for entry in feed.entries:
            rec = _parse_entry(entry)
            if rec is None:
                continue
            if cutoff is not None:
                try:
                    dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    if dt < cutoff:
                        continue
                except (TypeError, ValueError):
                    pass
            if rec.arxiv_id in records:
                records[rec.arxiv_id].categories |= rec.categories
            else:
                records[rec.arxiv_id] = rec
    return records


def run(
    data_dir: Path,
    config: Config,
    log: logging.Logger,
    limit: int | None = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Execute sync-metadata. Returns (added, updated) folder counts."""
    now = datetime.now(timezone.utc).isoformat()
    log.info("sync-metadata start: categories=%s", config.categories.include)

    records = collect_records(config, log, dry_run=dry_run)
    items = list(records.values())
    if limit is not None:
        items = items[:limit]

    if dry_run:
        log.info("[dry-run] would write %d metadata.json files", len(items))
        return (0, 0)

    added = updated = 0
    for rec in items:
        meta_file = metadata_path(data_dir, rec.arxiv_id)
        is_new = not meta_file.exists()
        atomic_write_json(meta_file, rec.to_metadata(now))
        if is_new:
            added += 1
            log.debug("added %s", rec.arxiv_id)
        else:
            updated += 1

    atomic_write_json(data_dir / "last_sync.json", {
        "finished_at": now,
        "categories": config.categories.include,
        "papers_added": added,
        "papers_updated": updated,
    })

    log.info("sync-metadata done: added=%d updated=%d", added, updated)
    return (added, updated)
