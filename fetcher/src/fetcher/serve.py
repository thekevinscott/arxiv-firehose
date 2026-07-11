"""HTTP API: thin wrapper over ``api.py`` for remote ops over tailnet.

The goal is to make ``status`` / ``fetch`` / ``classify`` callable from
anywhere on the tailnet without SSHing into the box. The web layer is
parallel to ``cli.py``: both are thin wrappers over the SDK in
``api.py``; the SDK functions hold all behavior.

Long jobs are fire-and-forget. ``fetch`` runs in minutes and ``classify``
in hours, so the HTTP path can't block until completion. ``POST /fetch``
and ``POST /classify`` spawn the CLI as a subprocess, register the job,
and return a ``Job`` immediately (HTTP 202). Clients tail
``GET /logs/{kind}`` (the shared cron log) or poll ``GET /jobs/{id}``
for pid + exit code.

Subprocess (not in-process ``api.fetch()``) is deliberate:

- crash isolation -- a broken classify never takes the API down
- one code path -- cron, manual CLI, and the HTTP API all run the same
  ``fetcher fetch`` / ``fetcher classify`` invocation
- no asyncio entanglement with cachetta's sync internals or httpx-sync
- log output lands in the same file the cron writes to, so ``tail -f``
  works regardless of how a run was triggered

Auth: none. The intended perimeter is whatever sits in front of the
host -- on tower that's home-router NAT (no port-forward for 8087)
plus tailnet routing. Bind ``0.0.0.0`` when the host is firewalled or
NATted; bind the tailscale IP when extra LAN isolation matters. Local
dev uses ``127.0.0.1``.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import api
from .commands import embed as embed_mod
from .shared.config import DEFAULT_DATA_DIR

JobKind = Literal["fetch", "classify", "embed", "pull", "render"]

# Default row cap when the client omits ``limit`` and doesn't write a
# custom SQL. The cap only matters for the built-in ORDER BY distance
# case; a user-supplied ``sql`` runs verbatim (with its own LIMIT if
# any). Tailnet-only, no auth -- trusted caller.
DEFAULT_SEARCH_LIMIT = 20

def _build_papers_view_sql(parquet_path: str, meta_glob: str) -> str:
    """CREATE VIEW SQL for the ``papers`` view exposed to /search.

    DuckDB rejects prepared parameters in CREATE VIEW, so the two paths
    are interpolated. They come from the server-side ``data_dir`` (a
    trusted config value), never from the request body -- no injection
    surface. Single quotes in each are escaped defensively.
    """
    parquet_lit = parquet_path.replace("'", "''")
    glob_lit = meta_glob.replace("'", "''")
    return f"""
        CREATE OR REPLACE VIEW papers AS
        SELECT
            m.arxiv_id,
            m.title,
            m.abstract,
            m.authors,
            m.primary_category,
            m.categories,
            m.announced_at,
            m.updated_at,
            m.html_url,
            array_cosine_distance(
                e.embedding::FLOAT[{embed_mod.EMBED_DIM}],
                (SELECT vec FROM _q)
            ) AS distance
        FROM read_parquet('{parquet_lit}') e
        JOIN read_json_auto(
            '{glob_lit}', union_by_name := true, filename := false
        ) m
          ON e.arxiv_id = m.arxiv_id
    """

# Per-job log file naming. We *also* write to the shared cron log so a
# single ``tail -f .../classify-cron.log`` shows every run regardless of
# how it was triggered. The shared file is the read path; per-job is just
# an idea we deliberately did not take.
CRON_LOG_NAME = "{kind}-cron.log"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8087


class Job(BaseModel):
    """A spawned subprocess. ``exit_code`` is ``None`` while running."""

    id: str
    kind: JobKind
    pid: int
    started_at: float
    exit_code: int | None
    log_path: str


class JobRegistry:
    """In-memory ring buffer of spawned jobs.

    The API server process owns the list. Restarting the API loses
    history -- acceptable, because the underlying log files on disk are
    durable and a fresh ``GET /logs/{kind}`` will still show every run.

    Eviction drops the oldest *finished* job once over capacity; running
    jobs are never evicted so a long classify can't disappear from
    ``GET /jobs`` mid-flight.
    """

    def __init__(self, capacity: int = 50) -> None:
        self._jobs: dict[str, tuple[Job, subprocess.Popen]] = {}
        self._capacity = capacity
        # FastAPI dispatches sync endpoints in a threadpool, so two
        # concurrent POSTs can race the check-then-spawn in
        # ``add_unless_running``. The lock makes that compound op atomic.
        self._lock = threading.Lock()

    def add(
        self, kind: JobKind, popen: subprocess.Popen, log_path: Path
    ) -> Job:
        with self._lock:
            return self._add_locked(kind, popen, log_path)

    def _add_locked(
        self, kind: JobKind, popen: subprocess.Popen, log_path: Path
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            pid=popen.pid,
            started_at=time.time(),
            exit_code=None,
            log_path=str(log_path),
        )
        self._jobs[job.id] = (job, popen)
        self._evict_locked()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            pair = self._jobs.get(job_id)
            if pair is None:
                return None
            job, popen = pair
            # poll() returns None while running, an int once exited;
            # cache it on the Job so a later GET still reports a finished
            # job's code even after the Popen is gone.
            if job.exit_code is None:
                job.exit_code = popen.poll()
            return job

    def list(self) -> list[Job]:
        with self._lock:
            ids = list(self._jobs)
        # Release the lock before calling get() to avoid reentrant
        # acquisition; get() takes the lock itself per id.
        return [j for j in (self.get(jid) for jid in ids) if j is not None]

    def running(self, kind: JobKind) -> Job | None:
        """Return any same-kind job whose subprocess is still running."""
        with self._lock:
            return self._running_locked(kind)

    def _running_locked(self, kind: JobKind) -> Job | None:
        for job, popen in self._jobs.values():
            if job.kind == kind and popen.poll() is None:
                return job
        return None

    def add_unless_running(
        self,
        kind: JobKind,
        spawner: Callable[[], subprocess.Popen],
        log_path: Path,
    ) -> tuple[Job, bool]:
        """Spawn via *spawner* unless a same-kind job is already running.

        Returns ``(job, is_new)``. When *is_new* is False, *job* is the
        already-running one and *spawner* was not called. The whole
        check-and-spawn happens under the lock so two concurrent
        callers can't both win the race.
        """
        with self._lock:
            existing = self._running_locked(kind)
            if existing is not None:
                return existing, False
            popen = spawner()
            return self._add_locked(kind, popen, log_path), True

    def _evict_locked(self) -> None:
        if len(self._jobs) <= self._capacity:
            return
        finished = sorted(
            (j.started_at, jid)
            for jid, (j, p) in self._jobs.items()
            if p.poll() is not None
        )
        for _, jid in finished:
            if len(self._jobs) <= self._capacity:
                break
            del self._jobs[jid]


class PullRequest(BaseModel):
    """Pull request body: the arxiv ids to mirror."""

    ids: list[str] = Field(min_length=1)


class SearchRequest(BaseModel):
    """Semantic search request body.

    - ``q``: the natural-language query. Embedded server-side into a
      256-dim vector and exposed to SQL as a pre-computed ``distance``
      column on the ``papers`` view.
    - ``sql``: optional DuckDB SQL executed verbatim against ``papers``.
      Omitted -> ``SELECT arxiv_id, title, distance FROM papers
      ORDER BY distance LIMIT :limit``. Arbitrary DuckDB is allowed
      (WHERE on category / date, JOINs, window functions, etc.);
      /search is tailnet-only so we don't sandbox the SQL.
    - ``limit``: cap for the default SQL only. Ignored when ``sql`` is
      set (put your own LIMIT in the SQL).
    """

    q: str
    sql: str | None = None
    limit: int = DEFAULT_SEARCH_LIMIT


# Process-lifetime embedding model cache. The static model is ~30 MB
# and loads in a few seconds; we amortize it across requests.
_MODEL = None
_MODEL_LOCK = threading.Lock()


def _get_model():
    """Return the module-level model2vec instance, loading it once."""
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                from model2vec import StaticModel

                _MODEL = StaticModel.from_pretrained(embed_mod.MODEL_NAME)
    return _MODEL


def _default_log_dir(data_dir: Path) -> Path:
    """Where cron writes its logs.

    On tower the convention is ``/mnt/bertha/data/arxiv-firehose/{kind}-cron.log``;
    ``data_dir`` is ``/mnt/bertha/data/arxiv-firehose/data``, so its parent
    is the right directory. Honor ``ARXIV_FIREHOSE_LOG_DIR`` for setups
    that put logs elsewhere.
    """
    env = os.environ.get("ARXIV_FIREHOSE_LOG_DIR")
    if env:
        return Path(env)
    return data_dir.parent


def _spawn_cli(
    kind: JobKind,
    data_dir: Path,
    log_path: Path,
    args: tuple[str, ...] = (),
    fetcher_bin: str = "fetcher",
) -> subprocess.Popen:
    """Launch ``fetcher {kind} [ARGS...] --data-dir DATA_DIR`` detached.

    *args* carries per-job positional arguments (pull's arxiv ids).
    ``start_new_session=True`` puts the child in its own process group so
    the API can restart without dragging classify down with it.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("ab")
    return subprocess.Popen(
        [fetcher_bin, kind, *args, "--data-dir", str(data_dir)],
        stdout=fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def _tail(log_path: Path, lines: int) -> list[str]:
    """Return the last *lines* lines of *log_path*; empty list if absent.

    Reads the whole file -- fine for cron logs in the MB range. If these
    grow to GBs, switch to a seek-from-end strategy.
    """
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return text.splitlines()[-lines:]


def make_app(
    data_dir: Path = DEFAULT_DATA_DIR,
    config_file: Path | None = None,
    *,
    spawn: object = _spawn_cli,
    log_dir: Path | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    *spawn* and *log_dir* are test seams: an integration test injects a
    fake spawn (records args, returns a fake popen) and a temp log dir.
    Production uses the defaults.
    """
    app = FastAPI(title="arxiv-firehose", version="0.1")
    registry = JobRegistry()
    logs = log_dir or _default_log_dir(data_dir)

    def _start(kind: JobKind, args: tuple[str, ...] = ()) -> Job:
        log_path = logs / CRON_LOG_NAME.format(kind=kind)
        job, is_new = registry.add_unless_running(
            kind,
            spawner=lambda: spawn(kind, data_dir, log_path, args),
            log_path=log_path,
        )
        if not is_new:
            # 409 Conflict: a same-kind run is already in flight. Carry
            # the existing job in the response so the caller can poll
            # /jobs/{id} or tail /logs/{kind} without a second request.
            # This makes the API the single dedup point: cron, manual
            # CLI, and ad-hoc curl all funnel through here and never
            # race each other on the same paper-x-category pairs.
            raise HTTPException(
                status_code=409,
                detail={
                    "error": f"{kind} already running",
                    "job": job.model_dump(),
                },
            )
        return job

    @app.get("/status")
    def get_status() -> dict[str, str]:
        return {"report": api.status(data_dir, config_file)}

    @app.post("/fetch", status_code=202)
    def post_fetch() -> Job:
        return _start("fetch")

    @app.post("/classify", status_code=202)
    def post_classify() -> Job:
        return _start("classify")

    @app.post("/embed", status_code=202)
    def post_embed() -> Job:
        """Trigger a standalone embed run.

        Useful for the first backfill or after adding papers out-of-band,
        when waiting for the next fetch cycle isn't worth it. Fetch also
        runs embed as its last stage, so this is a shortcut, not a
        prerequisite.
        """
        return _start("embed")

    @app.post("/pull", status_code=202)
    def post_pull(req: PullRequest) -> Job:
        """Mirror specific papers by arxiv id as a background job.

        The bespoke retrieval path (e.g. tracing a paper's citations):
        spawns ``fetcher pull ID... `` like /fetch spawns the daily
        ingest. One pull at a time -- a second POST while one runs gets
        a 409 carrying the in-flight job.
        """
        return _start("pull", tuple(req.ids))

    @app.post("/render", status_code=202)
    def post_render() -> Job:
        """Render markdown for every paper missing one, as a background job.

        Explicit-only: no cron triggers rendering. This is the heavy
        path (up to three paced downloads per paper), so it runs only
        when a human asks for paper bodies.
        """
        return _start("render")

    @app.get("/jobs")
    def list_jobs() -> list[Job]:
        return registry.list()

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> Job:
        job = registry.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job

    @app.get("/logs/{kind}")
    def get_log(kind: JobKind, lines: int = 50) -> dict[str, object]:
        log_path = logs / CRON_LOG_NAME.format(kind=kind)
        return {"path": str(log_path), "lines": _tail(log_path, lines)}

    @app.post("/search")
    def post_search(req: SearchRequest) -> dict[str, object]:
        """Semantic search over paper abstracts, with arbitrary SQL sub-select.

        The server embeds ``req.q`` once, builds a per-request DuckDB
        session with a ``papers`` view that carries a pre-computed
        ``distance`` column against that vector, then runs either the
        default SQL (``ORDER BY distance LIMIT :limit``) or the
        client-supplied ``req.sql`` verbatim. Rows come back as a list
        of dicts.
        """
        import duckdb  # local: keep `fetcher status` fast

        parquet = embed_mod.embeddings_path(data_dir)
        if not parquet.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "embeddings.parquet not built yet -- run `fetcher fetch` "
                    "or wait for the next cron cycle"
                ),
            )

        query_vec = [float(x) for x in _get_model().encode([req.q])[0]]
        meta_glob = str(data_dir / "*" / "metadata.json")

        con = duckdb.connect()
        try:
            # Query vector as a scalar the view can reference. Using a
            # temp table (not a Python-side format-string) keeps the
            # vector out of the SQL text and lets DuckDB constant-fold
            # the subquery once for all rows.
            con.execute(
                f"CREATE TEMP TABLE _q AS "
                f"SELECT ?::FLOAT[{embed_mod.EMBED_DIM}] AS vec",
                [query_vec],
            )
            con.execute(_build_papers_view_sql(str(parquet), meta_glob))

            if req.sql:
                sql = req.sql
            else:
                sql = (
                    "SELECT arxiv_id, title, distance "
                    "FROM papers ORDER BY distance LIMIT ?"
                )
            try:
                cur = con.execute(sql, [int(req.limit)] if not req.sql else [])
                # Avoid pyarrow: it's an optional duckdb extra.
                # cursor.description + fetchall() is the light-weight path --
                # fine for /search response sizes (10s to 100s of rows).
                # fetchall() is inside the try: value conversion can also
                # raise (e.g. TIMESTAMPTZ columns need pytz installed).
                columns = [d[0] for d in cur.description]
                rows = [dict(zip(columns, r)) for r in cur.fetchall()]
            except (duckdb.Error, ImportError) as exc:
                # The SQL is client-authored: surface the real DuckDB error
                # as a 400 instead of an opaque 500 -- the caller is
                # debugging their query, not our server.
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"sql": sql, "count": len(rows), "rows": rows}
        finally:
            con.close()

    return app


def serve(
    data_dir: Path = DEFAULT_DATA_DIR,
    config_file: Path | None = None,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    """Run the HTTP API. Blocks. Use a systemd unit for production."""
    import uvicorn  # local import: keeps `fetcher status` fast

    uvicorn.run(
        make_app(data_dir, config_file),
        host=host,
        port=port,
        log_level="info",
    )
