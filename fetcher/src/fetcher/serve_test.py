"""Unit tests for the HTTP API.

The FastAPI app is exercised in-process through starlette's
``TestClient`` -- no port binding, no uvicorn. Subprocess spawning is
replaced with a recording fake so a test never starts a real
``fetcher classify`` run.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from fetcher import serve


class FakePopen:
    """Minimal stand-in for ``subprocess.Popen``.

    Records construction args and lets a test set ``returncode`` to
    drive the ``poll() is None`` branch.
    """

    def __init__(self, pid: int = 4242, returncode: int | None = None) -> None:
        self.pid = pid
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    d = tmp_path / "logs"
    d.mkdir()
    return d


@pytest.fixture
def spawns(log_dir: Path):
    """A recording spawn that opens the log file and returns a FakePopen."""
    calls: list[tuple[str, Path, Path]] = []

    def spawn(kind, data_dir, log_path):
        calls.append((kind, data_dir, log_path))
        # Touch the file so /logs has something to tail in tests that
        # want to see the read path work end-to-end.
        log_path.touch()
        return FakePopen()

    return spawn, calls


@pytest.fixture
def client(tmp_path: Path, spawns, log_dir: Path):
    spawn, _ = spawns
    app = serve.make_app(data_dir=tmp_path / "data", spawn=spawn, log_dir=log_dir)
    with TestClient(app) as c:
        yield c


def describe_status():
    def it_returns_a_report_string(client: TestClient, tmp_path: Path):
        # api.status reads the data dir; a non-existent data dir just yields
        # zeroes -- enough to assert the wiring without bootstrapping data.
        with patch("fetcher.serve.api.status", return_value="ok"):
            r = client.get("/status")
        assert r.status_code == 200
        assert r.json() == {"report": "ok"}


def describe_post_fetch():
    def it_spawns_and_returns_a_job(client: TestClient, spawns):
        _, calls = spawns
        r = client.post("/fetch")
        assert r.status_code == 202
        body = r.json()
        assert body["kind"] == "fetch"
        assert body["pid"] == 4242
        assert body["exit_code"] is None
        assert body["log_path"].endswith("fetch-cron.log")
        assert len(calls) == 1
        assert calls[0][0] == "fetch"


def describe_post_classify():
    def it_spawns_and_returns_a_job(client: TestClient, spawns):
        _, calls = spawns
        r = client.post("/classify")
        assert r.status_code == 202
        body = r.json()
        assert body["kind"] == "classify"
        assert body["log_path"].endswith("classify-cron.log")
        assert calls[0][0] == "classify"


def describe_get_jobs():
    def it_lists_started_jobs(client: TestClient):
        client.post("/fetch")
        client.post("/classify")
        r = client.get("/jobs")
        assert r.status_code == 200
        jobs = r.json()
        assert {j["kind"] for j in jobs} == {"fetch", "classify"}


def describe_get_job():
    def it_returns_a_known_job(client: TestClient):
        job_id = client.post("/fetch").json()["id"]
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        assert r.json()["id"] == job_id

    def it_404s_an_unknown_job(client: TestClient):
        r = client.get("/jobs/does-not-exist")
        assert r.status_code == 404


def describe_get_log():
    def it_tails_the_cron_log_file(client: TestClient, log_dir: Path):
        log = log_dir / "classify-cron.log"
        log.write_text("\n".join(f"line {i}" for i in range(100)) + "\n")
        r = client.get("/logs/classify", params={"lines": 5})
        assert r.status_code == 200
        body = r.json()
        assert body["path"].endswith("classify-cron.log")
        assert body["lines"] == [f"line {i}" for i in range(95, 100)]

    def it_returns_empty_when_log_missing(client: TestClient):
        r = client.get("/logs/fetch")
        assert r.status_code == 200
        assert r.json()["lines"] == []


def describe_JobRegistry():
    def it_records_pid_and_started_at():
        r = serve.JobRegistry()
        before = time.time()
        job = r.add("fetch", FakePopen(pid=99), Path("/tmp/x.log"))
        assert job.pid == 99
        assert job.started_at >= before

    def it_caches_exit_code_after_first_observed():
        r = serve.JobRegistry()
        popen = FakePopen()
        job = r.add("fetch", popen, Path("/tmp/x.log"))
        assert r.get(job.id).exit_code is None
        popen.returncode = 0
        assert r.get(job.id).exit_code == 0
        # Even if the underlying popen "forgets", we keep what we saw.
        popen.returncode = None
        assert r.get(job.id).exit_code == 0

    def it_evicts_oldest_finished_when_over_capacity():
        r = serve.JobRegistry(capacity=2)
        a = r.add("fetch", FakePopen(returncode=0), Path("/tmp/a.log"))
        b = r.add("fetch", FakePopen(returncode=0), Path("/tmp/b.log"))
        # Force a's started_at < b's so the eviction order is deterministic.
        r._jobs[a.id][0].started_at = 1.0
        r._jobs[b.id][0].started_at = 2.0
        r.add("fetch", FakePopen(), Path("/tmp/c.log"))
        ids = {j.id for j in r.list()}
        assert a.id not in ids
        assert b.id in ids

    def it_never_evicts_a_running_job():
        r = serve.JobRegistry(capacity=1)
        running = r.add("fetch", FakePopen(returncode=None), Path("/tmp/a.log"))
        running.started_at = 1.0
        finished = r.add("fetch", FakePopen(returncode=0), Path("/tmp/b.log"))
        finished.started_at = 2.0
        ids = {j.id for j in r.list()}
        assert running.id in ids
