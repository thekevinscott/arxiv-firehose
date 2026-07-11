"""sync-metadata: pull paper metadata from the arxiv export API.

Writes one ``metadata.json`` per paper folder. No database -- the filesystem
is the state. The API is queried in per-day submittedDate slices through
``fetch.download.fetch_day`` (cachetta-cached: settled days ~forever, the
trailing few days one day), so a daily run re-reads its whole lookback
window at the cost of at most a handful of real requests. Any day a cron
run missed is simply fetched by the next run that covers it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import feedparser
import httpx

from . import download
from .download.api import MAX_RESULTS
from ...shared.atomic_write import atomic_write_json
from ...shared.config import Config
from ...shared.paths import id_from_entry_id, metadata_path, parse_id, version_from_entry_id

RFC_2822 = "%a, %d %b %Y %H:%M:%S +0000"


@dataclass
class PaperRecord:
    arxiv_id: str
    version: int
    title: str
    authors: list[str]
    abstract: str
    primary_category: str
    categories: set[str] = field(default_factory=set)
    # The export API's <published> timestamp (v1 submission time), rendered
    # in the RFC-2822 shape the RSS era wrote so the existing corpus and
    # the papers view's strptime stay uniform.
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


def _rfc2822(parsed) -> str:
    """Render feedparser's parsed struct_time in the corpus's RFC-2822 shape."""
    if isinstance(parsed, time.struct_time):
        dt = datetime(*parsed[:6], tzinfo=timezone.utc)
        return dt.strftime(RFC_2822)
    return ""


def _primary_category(entry: feedparser.FeedParserDict, tags: list[str]) -> str:
    """The entry's primary category: arxiv's namespaced element when
    feedparser surfaces it, else the first <category> term."""
    primary = entry.get("arxiv_primary_category")
    if isinstance(primary, dict) and primary.get("term"):
        return primary["term"]
    return tags[0] if tags else "unknown"


def _parse_entry(
    entry: feedparser.FeedParserDict, tracked: set[str]
) -> PaperRecord | None:
    raw_id = entry.get("id", "") or entry.get("link", "")
    try:
        arxiv_id = id_from_entry_id(raw_id)
        parse_id(arxiv_id)  # validate
    except ValueError:
        return None

    # The API id carries the latest version; anything past v1 is a revision
    # of an old paper resurfacing in the window. Mirror first versions only.
    version = version_from_entry_id(raw_id)
    if version != 1:
        return None

    tags = [t.get("term", "") for t in entry.get("tags", []) if t.get("term")]
    primary = _primary_category(entry, tags)
    # A cat: query also matches cross-lists; keep only papers that live in
    # a tracked category, matching the RSS era's new-announcements-only rule.
    if primary not in tracked:
        return None

    authors = [a.get("name", "").strip() for a in entry.get("authors", [])]
    published = _rfc2822(entry.get("published_parsed"))
    updated = _rfc2822(entry.get("updated_parsed")) or published

    return PaperRecord(
        arxiv_id=arxiv_id,
        version=version,
        title=" ".join(entry.get("title", "").split()),
        authors=[a for a in authors if a],
        abstract=entry.get("summary", "").strip(),
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
    """Fetch every day-slice in the lookback window, dedupe by arxiv_id.

    Walks from today back through ``ingest.backfill_days``, newest first
    (so ``limit`` favors fresh papers). ``download.fetch_day`` is the
    cachetta-cached seam; the integration fixture answers through the
    fake ``http_get`` underneath it.
    """
    categories = tuple(config.categories.include)
    tracked = set(categories)
    days = [date.today() - timedelta(days=n) for n in range(config.ingest.backfill_days + 1)]

    records: dict[str, PaperRecord] = {}
    for day in days:
        if dry_run:
            log.info("[dry-run] would fetch day slice %s", day.isoformat())
            continue
        try:
            content = download.fetch_day(categories, day)
        except httpx.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429:
                # Firing more requests into a 429 wall can extend the ban.
                # Older days stay uncached, so the next run resumes here.
                log.warning(
                    "arxiv rate limited (429) at %s; deferring older days to the next run",
                    day.isoformat(),
                )
                break
            log.error("day slice fetch failed for %s: %s", day.isoformat(), exc)
            continue

        feed = feedparser.parse(content)
        log.info("day %s: %d entries", day.isoformat(), len(feed.entries))
        if len(feed.entries) >= MAX_RESULTS:
            log.warning(
                "day %s returned a full page (%d); slice may be truncated",
                day.isoformat(), len(feed.entries),
            )
        for entry in feed.entries:
            rec = _parse_entry(entry, tracked)
            if rec is None:
                continue
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
    """Execute sync-metadata. Returns (added, existing) folder counts."""
    now = datetime.now(timezone.utc).isoformat()
    log.info("sync-metadata start: categories=%s", config.categories.include)

    records = collect_records(config, log, dry_run=dry_run)
    items = list(records.values())
    if limit is not None:
        items = items[:limit]

    if dry_run:
        log.info("[dry-run] would write %d metadata.json files", len(items))
        return (0, 0)

    added = existing = 0
    for rec in items:
        meta_file = metadata_path(data_dir, rec.arxiv_id)
        # The lookback window re-serves already-mirrored papers every run;
        # rewriting them would churn synced_at and hammer the data volume.
        if meta_file.exists():
            existing += 1
            continue
        atomic_write_json(meta_file, rec.to_metadata(now))
        added += 1
        log.debug("added %s", rec.arxiv_id)

    atomic_write_json(data_dir / "last_sync.json", {
        "finished_at": now,
        "categories": config.categories.include,
        "papers_added": added,
        "papers_existing": existing,
    })

    log.info("sync-metadata done: added=%d existing=%d", added, existing)
    return (added, existing)
