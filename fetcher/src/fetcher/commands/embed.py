"""embed: populate ``embeddings.json`` at the data-dir root.

One row per paper. The invariant is convergence, not correctness of any
single run: every paper whose arxiv_id is not yet in
``embeddings.json`` gets embedded on the next call -- new arrivals
from today's ``sync-metadata`` and historical gaps go through the same
"missing → embed" path. There is no ``--force`` in the common flow;
running to convergence is what we want by default.

Storage: a single JSON array at ``data_dir/embeddings.json``, one object
per paper: ``{"arxiv_id": str, "embedding": [float, ...]}`` (1024 dims,
rounded to 6 decimals). dirsql scans this file into the ``embeddings``
table (see ``shared/dirsql_schema.py``), where sqlite-vec's
``vec_distance_cosine`` powers ``/search`` -- the same SQLite surface as
``/sql``, no separate engine. The whole file is rewritten each run
(atomic via ``.part + rename``); at ~34 K rows the 1024-dim vectors make
it ~300 MB, still a sub-second write on local disk.

Model: ``Qwen3-Embedding-0.6B`` served by tower's llama-server over its
OpenAI-compatible ``/v1/embeddings`` endpoint (GPU, local, free). The
endpoint and model id are overridable via ``ARXIV_FIREHOSE_EMBED_BASE_URL``
and ``ARXIV_FIREHOSE_EMBED_MODEL`` so a laptop run can point elsewhere.
The HTTP client is built lazily (``load_embedder``) so ``import fetcher``
(used by status, sync-metadata, tests) pays no startup cost.

Per-paper errors (bad metadata.json, empty abstract) are logged and
skipped. A single broken folder must not abort the run.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import httpx

from ..shared.atomic_write import atomic_write_text
from ..shared.paths import iter_paper_dirs
from ..shared.retry import with_retry

# tower's llama-server router (see systems/tower.md); host 8180 -> container
# 8080. A dummy value is fine for a laptop run that overrides the env var.
DEFAULT_BASE_URL = "http://tower.tail790bbc.ts.net:8180"
MODEL_NAME = os.environ.get("ARXIV_FIREHOSE_EMBED_MODEL", "Qwen3-Embedding-0.6B-f16")
EMBED_DIM = 1024
EMBEDDINGS_FILE = "embeddings.json"

# Per-request abstract count. Small enough to keep each request cheap and
# well under the server's batch limits; the whole pending set is chunked.
_BATCH = 32
_TIMEOUT = 120.0


def embeddings_path(data_dir: Path) -> Path:
    """The consolidated embeddings JSON at the data-dir root."""
    return data_dir / EMBEDDINGS_FILE


def _read_rows(data_dir: Path) -> list[dict]:
    """Every ``{"arxiv_id", "embedding"}`` row in the current file (or [])."""
    path = embeddings_path(data_dir)
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A corrupt file self-heals: treat as empty so this run re-embeds
        # everything and rewrites it atomically.
        return []
    return rows if isinstance(rows, list) else []


def _iter_pending(
    data_dir: Path,
    existing: set[str],
    log: logging.Logger,
) -> list[tuple[str, str]]:
    """Return ``(arxiv_id, abstract)`` for every paper missing an embedding.

    Papers with unreadable metadata.json or an empty abstract are logged
    and skipped -- they simply reappear next run once the underlying
    problem is fixed (or the paper is dropped from the sync feed).
    """
    pending: list[tuple[str, str]] = []
    for pd in iter_paper_dirs(data_dir):
        arxiv_id = pd.name
        if arxiv_id in existing:
            continue
        try:
            meta = json.loads((pd / "metadata.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("embed skip %s: bad metadata.json (%s)", arxiv_id, exc)
            continue
        # The paper folder is named by the slugified arxiv id; metadata's
        # own arxiv_id can carry the legacy 'archive/NNN' form. Prefer
        # the folder name as the row key -- it's what the embeddings table
        # joins against papers.arxiv_id on.
        abstract = (meta.get("abstract") or "").strip()
        if not abstract:
            log.warning("embed skip %s: empty abstract", arxiv_id)
            continue
        pending.append((arxiv_id, abstract))
    return pending


def _is_retryable(exc: Exception) -> bool:
    """Transport hiccups and transient 5xx/429 are worth a retry.

    The router loads models on demand, so the first request after an idle
    period (or one that evicts a chat model) can briefly 503 while the GGUF
    warms up -- exactly the case retrying rescues."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


class _HTTPEmbedder:
    """``.encode(list[str]) -> list[list[float]]`` over an OpenAI-compatible
    ``/v1/embeddings`` endpoint -- the same call shape ``embed.run`` and
    ``/search`` expect from the old model2vec ``StaticModel`` seam.

    Inputs are chunked into ``batch`` per request; each request retries the
    on-demand model load with exponential backoff. Response rows are
    reordered by their ``index`` so a batch's vectors line up with inputs."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = _TIMEOUT,
        batch: int = _BATCH,
    ) -> None:
        self._url = base_url.rstrip("/") + "/v1/embeddings"
        self._model = model
        self._timeout = timeout
        self._batch = batch

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            vectors.extend(self._embed_chunk(texts[start : start + self._batch]))
        return vectors

    def _embed_chunk(self, chunk: list[str]) -> list[list[float]]:
        def _call() -> dict:
            resp = httpx.post(
                self._url,
                json={"model": self._model, "input": chunk},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()

        payload = with_retry(
            _call, is_retryable=_is_retryable, attempts=5, base=2.0
        )
        rows = sorted(payload["data"], key=lambda d: d["index"])
        return [row["embedding"] for row in rows]


def load_embedder() -> _HTTPEmbedder:
    """Build the HTTP embedder from env (or the tower defaults).

    Shared by ``run`` and ``serve``'s ``/search`` so documents and queries
    ride the same model. Cheap to construct (no network at build time), so
    callers may cache it or not."""
    base_url = os.environ.get("ARXIV_FIREHOSE_EMBED_BASE_URL", DEFAULT_BASE_URL)
    return _HTTPEmbedder(base_url, MODEL_NAME)


def _write_embeddings(
    data_dir: Path, prior: list[dict], ids: list[str], vecs
) -> None:
    """Merge new rows with the prior file and rewrite atomically.

    Prior arxiv_ids were filtered out upstream via ``existing``, so the
    merge cannot produce duplicates. Vectors are stored as plain JSON
    arrays (rounded to 6 decimals) -- sqlite-vec reads JSON vectors
    directly, so no binary encoding is needed. Compact separators keep
    the ~11 K-row file small.
    """
    rows = list(prior)
    for aid, vec in zip(ids, vecs):
        rows.append(
            {"arxiv_id": aid, "embedding": [round(float(x), 6) for x in vec]}
        )
    atomic_write_text(
        embeddings_path(data_dir),
        json.dumps(rows, separators=(",", ":")),
    )


def run(
    data_dir: Path,
    log: logging.Logger,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    model=None,
) -> dict[str, int]:
    """Embed every paper missing from ``embeddings.json``.

    Returns ``{"embedded", "skipped", "total"}``:
    - embedded -- rows added this run
    - skipped  -- rows already in the file at the start of the run
    - total    -- row count in the resulting file

    Papers with unreadable metadata or an empty abstract are counted as
    neither embedded nor skipped -- they log a WARNING and re-surface on
    the next run.

    *model* is a test seam: any object with ``encode(list[str]) ->``
    a sequence of ``EMBED_DIM``-length vectors. ``None`` builds the real
    HTTP embedder against tower's llama-server.
    """
    prior = _read_rows(data_dir)
    existing = {r["arxiv_id"] for r in prior}
    pending = _iter_pending(data_dir, existing, log)
    if limit is not None:
        pending = pending[:limit]

    if not pending:
        log.info("embed: nothing to do (%d already embedded)", len(existing))
        return {"embedded": 0, "skipped": len(existing), "total": len(existing)}

    if dry_run:
        log.info("[dry-run] would embed %d abstracts", len(pending))
        return {"embedded": 0, "skipped": len(existing), "total": len(existing)}

    if model is None:
        log.info("embed: using %s", MODEL_NAME)
        model = load_embedder()
    log.info("embed: encoding %d abstracts", len(pending))
    ids = [p[0] for p in pending]
    texts = [p[1] for p in pending]
    vecs = model.encode(texts)

    _write_embeddings(data_dir, prior, ids, vecs)
    total = len(existing) + len(ids)
    log.info(
        "embed done: embedded=%d skipped=%d total=%d",
        len(ids),
        len(existing),
        total,
    )
    return {"embedded": len(ids), "skipped": len(existing), "total": total}
