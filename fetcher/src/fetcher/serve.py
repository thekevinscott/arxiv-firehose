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

Auth: none. Bind to the tailscale IP (or 0.0.0.0 behind a firewall) and
let the tailnet ACL be the perimeter. For local dev, bind 127.0.0.1.
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import api
from .shared.config import DEFAULT_DATA_DIR

JobKind = Literal["fetch", "classify"]

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

    def add(
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
        self._evict()
        return job

    def get(self, job_id: str) -> Job | None:
        pair = self._jobs.get(job_id)
        if pair is None:
            return None
        job, popen = pair
        # poll() returns None while running, an int once exited; cache it
        # on the Job so a later GET still reports a finished job's code
        # even after the Popen has been reaped.
        if job.exit_code is None:
            job.exit_code = popen.poll()
        return job

    def list(self) -> list[Job]:
        return [j for j in (self.get(jid) for jid in list(self._jobs)) if j is not None]

    def _evict(self) -> None:
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
    fetcher_bin: str = "fetcher",
) -> subprocess.Popen:
    """Launch ``fetcher {kind} --data-dir DATA_DIR`` detached.

    ``start_new_session=True`` puts the child in its own process group so
    the API can restart without dragging classify down with it.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("ab")
    return subprocess.Popen(
        [fetcher_bin, kind, "--data-dir", str(data_dir)],
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

    def _start(kind: JobKind) -> Job:
        log_path = logs / CRON_LOG_NAME.format(kind=kind)
        popen = spawn(kind, data_dir, log_path)
        return registry.add(kind, popen, log_path)

    @app.get("/status")
    def get_status() -> dict[str, str]:
        return {"report": api.status(data_dir, config_file)}

    @app.post("/fetch", status_code=202)
    def post_fetch() -> Job:
        return _start("fetch")

    @app.post("/classify", status_code=202)
    def post_classify() -> Job:
        return _start("classify")

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
