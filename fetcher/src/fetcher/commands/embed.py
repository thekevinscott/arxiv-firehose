"""embed: populate ``embeddings.json`` at the data-dir root.

One row per paper. The invariant is convergence, not correctness of any
single run: every paper whose arxiv_id is not yet in
``embeddings.json`` gets embedded on the next call -- new arrivals
from today's ``sync-metadata`` and historical gaps go through the same
"missing → embed" path. There is no ``--force`` in the common flow;
running to convergence is what we want by default.

Storage: a single JSON array at ``data_dir/embeddings.json``, one object
per paper: ``{"arxiv_id": str, "embedding": [float, ...]}`` (256 dims,
rounded to 6 decimals). dirsql scans this file into the ``embeddings``
table (see ``shared/dirsql_schema.py``), where sqlite-vec's
``vec_distance_cosine`` powers ``/search`` -- the same SQLite surface as
``/sql``, no separate engine. Rewriting the whole file each run is fine
at ~11 K rows (~25 MB) and keeps the write atomic via ``.part + rename``.

Model: model2vec ``potion-base-8M`` -- static, CPU-only, 256-dim, fast.
Loaded lazily so ``import fetcher`` (used by status, sync-metadata,
tests) doesn't pay the model-load cost.

Per-paper errors (bad metadata.json, empty abstract) are logged and
skipped. A single broken folder must not abort the run -- same posture
as classify.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..shared.atomic_write import atomic_write_text
from ..shared.paths import iter_paper_dirs

MODEL_NAME = "minishlab/potion-base-8M"
EMBED_DIM = 256
EMBEDDINGS_FILE = "embeddings.json"


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


def _load_model():
    """Lazy import: ``import fetcher`` shouldn't drag in model2vec."""
    from model2vec import StaticModel

    return StaticModel.from_pretrained(MODEL_NAME)


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

    *model* is a test seam: any object with ``encode(list[str]) -> ndarray``
    (shape ``(N, EMBED_DIM)``). ``None`` loads the real potion-base-8M.
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
        log.info("embed: loading %s", MODEL_NAME)
        model = _load_model()
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
