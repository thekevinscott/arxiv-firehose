"""Unit tests for the HTTP API.

The FastAPI app is exercised in-process through starlette's
``TestClient`` -- no port binding, no uvicorn. Subprocess spawning is
replaced with a recording fake so a test never starts a real
``fetcher classify`` run.
"""

from __future__ import annotations

import json
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
    """A recording spawn that opens the log file and returns a FakePopen.

    Tests can pre-seed ``next_popens`` to override the default
    (a fresh running FakePopen) for individual spawns -- useful when a
    test needs the spawned process to look already-finished.
    """
    calls: list[tuple[str, Path, Path, tuple[str, ...]]] = []
    next_popens: list[FakePopen] = []

    def spawn(kind, data_dir, log_path, args=()):
        calls.append((kind, data_dir, log_path, tuple(args)))
        # Touch the file so /logs has something to tail in tests that
        # want to see the read path work end-to-end.
        log_path.touch()
        return next_popens.pop(0) if next_popens else FakePopen()

    return spawn, calls, next_popens


@pytest.fixture
def client(tmp_path: Path, spawns, log_dir: Path):
    spawn, _, _ = spawns
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
        _, calls, _ = spawns
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
        _, calls, _ = spawns
        r = client.post("/classify")
        assert r.status_code == 202
        body = r.json()
        assert body["kind"] == "classify"
        assert body["log_path"].endswith("classify-cron.log")
        assert calls[0][0] == "classify"


def describe_post_render():
    def it_spawns_and_returns_a_job(client: TestClient, spawns):
        # Rendering paper bodies is explicit-only (not a fetch stage);
        # this endpoint is the HTTP trigger for it.
        _, calls, _ = spawns
        r = client.post("/render")
        assert r.status_code == 202
        body = r.json()
        assert body["kind"] == "render"
        assert body["log_path"].endswith("render-cron.log")
        assert calls[0][0] == "render"


def describe_post_pull():
    def it_spawns_a_pull_job_carrying_the_requested_ids(
        client: TestClient, spawns
    ):
        _, calls, _ = spawns
        r = client.post("/pull", json={"ids": ["2401.00001", "2012.09999"]})
        assert r.status_code == 202
        body = r.json()
        assert body["kind"] == "pull"
        assert body["log_path"].endswith("pull-cron.log")
        kind, _, _, args = calls[0]
        assert kind == "pull"
        assert args == ("2401.00001", "2012.09999")

    def it_rejects_an_empty_id_list(client: TestClient, spawns):
        _, calls, _ = spawns
        r = client.post("/pull", json={"ids": []})
        assert r.status_code == 422
        assert calls == []


def describe_duplicate_concurrent_jobs():
    def it_returns_409_with_existing_job_when_same_kind_in_flight(
        client: TestClient, spawns
    ):
        _, calls, _ = spawns
        first = client.post("/classify").json()
        # Second POST while the first is still "running" (FakePopen
        # default returncode is None) should be rejected, not spawn again.
        r = client.post("/classify")
        assert r.status_code == 409
        body = r.json()
        assert body["detail"]["error"] == "classify already running"
        assert body["detail"]["job"]["id"] == first["id"]
        # Crucially: only one spawn happened.
        assert len(calls) == 1

    def it_allows_concurrent_fetch_and_classify(
        client: TestClient, spawns
    ):
        _, calls, _ = spawns
        assert client.post("/fetch").status_code == 202
        assert client.post("/classify").status_code == 202
        assert len(calls) == 2

    def it_allows_a_new_run_after_the_previous_finished(
        client: TestClient, spawns
    ):
        _, calls, next_popens = spawns
        # First spawn returns a popen we can flip to "finished" later.
        finished = FakePopen()
        next_popens.append(finished)
        first = client.post("/classify").json()
        finished.returncode = 0
        # Now a new POST should succeed -- the prior job exited.
        second = client.post("/classify").json()
        assert second["id"] != first["id"]
        assert len(calls) == 2


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


@pytest.fixture
def sql_data_dir(tmp_path: Path) -> Path:
    """A data dir seeded with three papers for /sql tests."""
    d = tmp_path / "data"
    d.mkdir()
    # announced_at is stored in the corpus's RFC-2822 shape; the dirsql
    # on_file callback normalizes it to UTC ISO at index time.
    papers = [
        ("2401.00001", "Diffusion", "cs.LG",
         "A paper about diffusion models.", "Mon, 01 Jan 2024 00:00:00 +0000"),
        ("2401.00002", "Compiler", "cs.PL",
         "A paper about compiler optimizations.",
         "Wed, 15 May 2024 00:00:00 +0000"),
        ("2401.00003", "Protein", "q-bio.BM",
         "A paper about protein folding.", "Sun, 01 Sep 2024 00:00:00 +0000"),
    ]
    for aid, title, primary, abstract, announced in papers:
        pd = d / aid
        pd.mkdir()
        (pd / "metadata.json").write_text(json.dumps({
            "arxiv_id": aid,
            "title": title,
            "abstract": abstract,
            "primary_category": primary,
            "categories": [primary],
            "authors": ["A"],
            "announced_at": announced,
            "updated_at": announced,
            "html_url": f"https://arxiv.org/html/{aid}v1",
        }))
    return d


@pytest.fixture
def sql_client(sql_data_dir: Path, spawns, log_dir: Path):
    """Client wired to the seeded data dir for /sql tests."""
    spawn, _, _ = spawns
    app = serve.make_app(
        data_dir=sql_data_dir, spawn=spawn, log_dir=log_dir
    )
    with TestClient(app) as c:
        yield c


def describe_post_sql():
    def it_counts_papers_via_the_dirsql_schema(sql_client: TestClient):
        r = sql_client.post(
            "/sql", json={"sql": "SELECT COUNT(*) AS n FROM papers"}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 1
        assert body["rows"][0]["n"] == 3

    def it_reads_the_metadata_eav_table(sql_client: TestClient):
        # title lives in the EAV table (abstract/arxiv_id are excluded).
        r = sql_client.post("/sql", json={
            "sql": "SELECT value FROM metadata "
                   "WHERE key = 'title' ORDER BY value",
        })
        assert r.status_code == 200
        titles = [row["value"] for row in r.json()["rows"]]
        assert titles == ["Compiler", "Diffusion", "Protein"]

    def it_rejects_a_write_with_400(sql_client: TestClient):
        # dirsql's authorizer refuses non-read statements; the endpoint
        # turns that into a 400 rather than a 500.
        r = sql_client.post("/sql", json={"sql": "DELETE FROM papers"})
        assert r.status_code == 400

    def it_returns_400_for_bad_sql(sql_client: TestClient):
        r = sql_client.post(
            "/sql", json={"sql": "SELECT nonexistent FROM papers"}
        )
        assert r.status_code == 400

    def it_filters_by_the_normalized_announced_at_iso(
        sql_client: TestClient
    ):
        # announced_at is normalized to UTC ISO at index time, so a plain
        # lexical string compare is a correct date filter (the RFC-2822
        # form sorted lexically and silently leaked out-of-window rows).
        # Only the Sep paper is on/after the June cutoff.
        r = sql_client.post("/sql", json={
            "sql": (
                "SELECT arxiv_id, announced_at FROM papers "
                "WHERE announced_at >= '2024-06-01' "
                "ORDER BY announced_at"
            ),
        })
        assert r.status_code == 200
        body = r.json()
        assert [row["arxiv_id"] for row in body["rows"]] == ["2401.00003"]


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
