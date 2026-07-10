"""embed: populate ``embeddings.parquet`` at the data-dir root.

One row per paper. The invariant is convergence, not correctness of any
single run: every paper whose arxiv_id is not yet in
``embeddings.parquet`` gets embedded on the next call -- new arrivals
from today's ``sync-metadata`` and historical gaps go through the same
"missing → embed" path. There is no ``--force`` in the common flow;
running to convergence is what we want by default.

Storage: a single parquet at ``data_dir/embeddings.parquet`` with
columns ``(arxiv_id VARCHAR, embedding FLOAT[256])``. Rewriting the
whole file each run is fine at 10 K rows (~10 MB) and lets us keep the
file atomic via ``.part + rename``.

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

import duckdb

from ..shared.paths import iter_paper_dirs

MODEL_NAME = "minishlab/potion-base-8M"
EMBED_DIM = 256
EMBEDDINGS_FILE = "embeddings.parquet"


def embeddings_path(data_dir: Path) -> Path:
    """The consolidated embeddings parquet at the data-dir root."""
    return data_dir / EMBEDDINGS_FILE


def _existing_ids(data_dir: Path) -> set[str]:
    parquet = embeddings_path(data_dir)
    if not parquet.exists():
        return set()
    con = duckdb.connect()
    try:
        rows = con.execute(
            "SELECT arxiv_id FROM read_parquet(?)", [str(parquet)]
        ).fetchall()
    finally:
        con.close()
    return {r[0] for r in rows}


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
        # the folder name as the row key -- it's what /search will look
        # up when joining against read_json_auto('data/*/metadata.json').
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


def _write_parquet(data_dir: Path, ids: list[str], vecs) -> None:
    """Merge new rows with any existing parquet and rewrite atomically.

    UNION ALL BY NAME: prior arxiv_ids were filtered out upstream via
    ``existing``, so the merge cannot produce duplicates. The cast to
    ``FLOAT[EMBED_DIM]`` is required because DuckDB's parquet reader
    surfaces fixed-size array columns as variable-length ``FLOAT[]``.
    """
    parquet = embeddings_path(data_dir)
    tmp = parquet.with_name(parquet.name + ".part")

    con = duckdb.connect()
    try:
        con.execute(
            f"CREATE TABLE new (arxiv_id VARCHAR, embedding FLOAT[{EMBED_DIM}])"
        )
        con.executemany(
            "INSERT INTO new VALUES (?, ?)",
            [(aid, [float(x) for x in vec]) for aid, vec in zip(ids, vecs)],
        )
        if parquet.exists():
            con.execute(
                f"CREATE TABLE prior AS "
                f"SELECT arxiv_id, "
                f"       embedding::FLOAT[{EMBED_DIM}] AS embedding "
                f"FROM read_parquet(?)",
                [str(parquet)],
            )
            con.execute(
                "COPY (SELECT * FROM prior "
                "      UNION ALL BY NAME "
                "      SELECT * FROM new) "
                "TO ? (FORMAT 'parquet')",
                [str(tmp)],
            )
        else:
            con.execute("COPY new TO ? (FORMAT 'parquet')", [str(tmp)])
    finally:
        con.close()

    tmp.rename(parquet)


def run(
    data_dir: Path,
    log: logging.Logger,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    model=None,
) -> dict[str, int]:
    """Embed every paper missing from ``embeddings.parquet``.

    Returns ``{"embedded", "skipped", "total"}``:
    - embedded -- rows added this run
    - skipped  -- rows already in the parquet at the start of the run
    - total    -- row count in the resulting parquet

    Papers with unreadable metadata or an empty abstract are counted as
    neither embedded nor skipped -- they log a WARNING and re-surface on
    the next run.

    *model* is a test seam: any object with ``encode(list[str]) -> ndarray``
    (shape ``(N, EMBED_DIM)``). ``None`` loads the real potion-base-8M.
    """
    existing = _existing_ids(data_dir)
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

    _write_parquet(data_dir, ids, vecs)
    total = len(existing) + len(ids)
    log.info(
        "embed done: embedded=%d skipped=%d total=%d",
        len(ids),
        len(existing),
        total,
    )
    return {"embedded": len(ids), "skipped": len(existing), "total": total}
