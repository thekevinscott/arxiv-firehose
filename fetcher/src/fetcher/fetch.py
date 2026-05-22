"""fetch: produce a markdown rendering of every known paper.

Every paper is fetched on every run -- there is no "already on disk, skip"
shortcut. Network reads go through the cachetta-backed downloaders (see
download.py), which serve bytes from the on-disk cache or the network
transparently. Two conversion paths yield the markdown:

  1. arxiv native HTML (the primary path) -> markdown.
  2. for a paper with no arxiv HTML, the LaTeX e-print source -> markdown.

Only the markdown lands in the data dir: ``{id}/paper.md`` (or a
``.no_markdown`` marker when neither path produces anything). The PDF and the
raw LaTeX are never written to disk -- intermediate bytes live only in the
cachetta cache.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from .config import Config
from .convert import REAL_CONVERTER, Converter, _is_substantial
from .download import Transport, make_downloader, make_html_fetcher
from .paths import iter_paper_dirs, markdown_path


def _http_error_summary(exc: Exception) -> str:
    """Condense a download exception to one short log-friendly line.

    httpx embeds a multi-line MDN documentation link in every
    ``HTTPStatusError`` message; reduce a status error to just ``HTTP <code>``
    and any other exception to ``<Type>: <message>``.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return f"{type(exc).__name__}: {exc}"


def _write_markdown(md: str, dest: Path) -> int:
    """Atomically write markdown *md* into the data folder. Returns char count."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_text(md, encoding="utf-8")
    tmp.rename(dest)
    return len(md)


def _markdown_from_html(
    fetch_html, url: str, converter: Converter, log: logging.Logger, arxiv_id: str
) -> str | None:
    """Fetch arxiv HTML and convert it. None if no usable HTML/markdown.

    A 404 (no arxiv HTML for this paper) is expected for ~3% of papers and is
    logged quietly -- the caller falls back to the LaTeX path.
    """
    try:
        html = fetch_html(url)
    except httpx.HTTPStatusError as exc:
        log.debug("html %s: %s", arxiv_id, _http_error_summary(exc))
        return None
    except httpx.HTTPError as exc:
        log.warning("html %s: %s", arxiv_id, _http_error_summary(exc))
        return None
    try:
        md = converter.html(html)
    except Exception as exc:  # noqa: BLE001 -- a converter blow-up must not abort
        log.warning("html %s: conversion failed: %s", arxiv_id, exc)
        return None
    if not _is_substantial(md):
        log.debug("html %s: render too thin, ignoring", arxiv_id)
        return None
    log.info("html %s: HTTP 200", arxiv_id)
    return md


def _markdown_from_latex(
    download, url: str, converter: Converter, log: logging.Logger, arxiv_id: str
) -> str | None:
    """Fetch the LaTeX e-print archive and convert it. None if it yields none."""
    try:
        body = download(url)
    except httpx.HTTPError as exc:
        log.debug("tex  %s: %s", arxiv_id, _http_error_summary(exc))
        return None
    try:
        md = converter.latex(body)
    except ValueError as exc:
        # No LaTeX in the archive (a PDF-only e-print) -- an expected outcome.
        log.debug("tex  %s: %s", arxiv_id, exc)
        return None
    except Exception as exc:  # noqa: BLE001 -- a pandoc/converter blow-up
        log.warning("tex  %s: conversion failed: %s", arxiv_id, exc)
        return None
    if not _is_substantial(md):
        log.debug("tex  %s: render too thin, ignoring", arxiv_id)
        return None
    log.info("tex  %s: HTTP 200", arxiv_id)
    return md


def run(
    data_dir: Path,
    cache_dir: Path,
    config: Config,
    log: logging.Logger,
    limit: int | None = None,
    dry_run: bool = False,
    transport: Transport | None = None,
    converter: Converter = REAL_CONVERTER,
) -> dict[str, int]:
    """Execute fetch. Returns a counts dict."""
    # iter_paper_dirs yields folders sorted by arxiv id; --limit therefore
    # takes a deterministic prefix.
    paper_dirs = list(iter_paper_dirs(data_dir))

    counts = {"html": 0, "latex": 0, "absent": 0, "failed": 0, "skipped": 0}
    processed = 0

    log.info("fetch start: %d papers known, cache=%s, latex_fallback=%s",
             len(paper_dirs), cache_dir, config.fetch.latex_fallback)
    fetch_html = make_html_fetcher(cache_dir, transport)
    download = make_downloader(cache_dir, transport)

    for pd in paper_dirs:
        if limit is not None and processed >= limit:
            break
        # A single unreadable metadata.json must not abort the whole run --
        # log it, count it, and move on to the next paper.
        try:
            meta = json.loads((pd / "metadata.json").read_text())
            arxiv_id = meta["arxiv_id"]
            html_url = meta["html_url"]
            source_url = meta["source_url"]
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            counts["skipped"] += 1
            log.error("skip %s: bad metadata.json", pd.name)
            continue
        processed += 1

        if dry_run:
            log.info("[dry-run] would fetch %s", arxiv_id)
            continue

        marker = pd / ".no_markdown"
        try:
            # Primary: arxiv native HTML. Fallback: the LaTeX e-print source.
            md = _markdown_from_html(fetch_html, html_url, converter, log, arxiv_id)
            source = "html"
            if md is None and config.fetch.latex_fallback:
                md = _markdown_from_latex(
                    download, source_url, converter, log, arxiv_id
                )
                source = "latex"

            if md is not None:
                size = _write_markdown(md, markdown_path(data_dir, arxiv_id))
                marker.unlink(missing_ok=True)
                counts[source] += 1
                log.info("md   %s: wrote paper.md (%d chars, via %s)",
                         arxiv_id, size, source)
            else:
                marker.write_text("no markdown representation available\n")
                counts["absent"] += 1
                log.info("md   %s: no markdown available", arxiv_id)
        except Exception as exc:  # noqa: BLE001
            counts["failed"] += 1
            log.error("md   %s: %s", arxiv_id, _http_error_summary(exc))

    log.info("fetch done: %s", counts)
    return counts
