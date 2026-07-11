"""Unit tests for the HTTP API.

The FastAPI app is exercised in-process through starlette's
``TestClient`` -- no port binding, no uvicorn. Subprocess spawning is
replaced with a recording fake so a test never starts a real
``fetcher classify`` run.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fetcher import serve
from fetcher.commands import embed as embed_mod


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


def describe_post_embed():
    def it_spawns_and_returns_a_job(client: TestClient, spawns):
        _, calls, _ = spawns
        r = client.post("/embed")
        assert r.status_code == 202
        body = r.json()
        assert body["kind"] == "embed"
        assert body["log_path"].endswith("embed-cron.log")
        assert calls[0][0] == "embed"


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


class _SearchFakeModel:
    """Fake ``StaticModel`` for /search tests: one axis per keyword.

    Encodes text into a fixed vector by counting occurrences of a small
    keyword set on distinct axes. That lets a test assert semantic
    ordering (a "diffusion" query is nearest a "diffusion" abstract)
    without loading a real embedding model.
    """

    KEYWORDS = ("diffusion", "compiler", "protein", "quantum")

    def encode(self, texts: list[str]) -> np.ndarray:
        arr = np.zeros((len(texts), embed_mod.EMBED_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            lower = t.lower()
            for j, kw in enumerate(self.KEYWORDS):
                if kw in lower:
                    arr[i, j] = 1.0
            # Zero vectors would make cosine distance NaN; nudge unrelated
            # texts onto a dedicated axis so distance stays defined.
            if not arr[i, : len(self.KEYWORDS)].any():
                arr[i, len(self.KEYWORDS)] = 1.0
        return arr


@pytest.fixture
def search_data_dir(tmp_path: Path) -> Path:
    """A data dir seeded with three papers + their embeddings.parquet.

    Uses the deterministic keyword-axis fake so /search tests can assert
    exact ordering. Real model2vec is never loaded in these tests.
    """
    d = tmp_path / "data"
    d.mkdir()
    papers = [
        ("2401.00001", "Diffusion", "cs.LG",
         "A paper about diffusion models."),
        ("2401.00002", "Compiler", "cs.PL",
         "A paper about compiler optimizations."),
        ("2401.00003", "Protein", "q-bio.BM",
         "A paper about protein folding."),
    ]
    for aid, title, primary, abstract in papers:
        pd = d / aid
        pd.mkdir()
        (pd / "metadata.json").write_text(json.dumps({
            "arxiv_id": aid,
            "title": title,
            "abstract": abstract,
            "primary_category": primary,
            "categories": [primary],
            "authors": ["A"],
            "announced_at": "2024-01-01",
            "updated_at": "2024-01-01",
            "html_url": f"https://arxiv.org/html/{aid}v1",
        }))
    embed_mod.run(
        d, logging.getLogger("test.embed"), model=_SearchFakeModel()
    )
    return d


@pytest.fixture
def search_client(search_data_dir: Path, spawns, log_dir: Path):
    """FastAPI test client wired to the seeded data dir + fake model.

    The module-level ``serve._MODEL`` is patched inside a with-block so
    a test's fake never leaks into subsequent tests.
    """
    spawn, _, _ = spawns
    app = serve.make_app(
        data_dir=search_data_dir, spawn=spawn, log_dir=log_dir
    )
    with patch.object(serve, "_MODEL", _SearchFakeModel()):
        with TestClient(app) as c:
            yield c


def describe_post_search():
    def it_returns_503_when_embeddings_parquet_is_missing(
        client: TestClient
    ):
        # The default ``client`` fixture points at a nonexistent data dir;
        # /search must refuse with 503 rather than crash.
        r = client.post("/search", json={"q": "anything"})
        assert r.status_code == 503
        assert "embeddings.parquet" in r.json()["detail"]

    def it_ranks_papers_by_semantic_distance_with_default_sql(
        search_client: TestClient
    ):
        r = search_client.post(
            "/search", json={"q": "diffusion models", "limit": 3}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 3
        # The diffusion paper is nearest; the others follow.
        assert body["rows"][0]["arxiv_id"] == "2401.00001"
        assert body["rows"][0]["distance"] == pytest.approx(0.0, abs=1e-6)
        # Row shape: arxiv_id, title, distance.
        assert set(body["rows"][0]) == {"arxiv_id", "title", "distance"}

    def it_honors_limit_on_default_sql(search_client: TestClient):
        r = search_client.post(
            "/search", json={"q": "diffusion", "limit": 1}
        )
        assert r.status_code == 200
        assert r.json()["count"] == 1

    def it_runs_custom_sql_with_where_and_orderby(
        search_client: TestClient
    ):
        # Filter by primary_category via the papers view. Proves the
        # metadata JOIN carries through and the client SQL runs verbatim.
        r = search_client.post("/search", json={
            "q": "biology",
            "sql": (
                "SELECT arxiv_id, primary_category, distance "
                "FROM papers "
                "WHERE primary_category LIKE 'q-bio%' "
                "ORDER BY distance"
            ),
        })
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 1
        assert body["rows"][0]["arxiv_id"] == "2401.00003"
        assert body["rows"][0]["primary_category"] == "q-bio.BM"

    def it_supports_aggregate_sql(search_client: TestClient):
        r = search_client.post("/search", json={
            "q": "anything",
            "sql": "SELECT COUNT(*) AS n FROM papers",
        })
        assert r.status_code == 200
        assert r.json()["rows"][0]["n"] == 3

    def it_echoes_the_sql_that_ran(search_client: TestClient):
        # Handy for debugging clients; the response body carries the
        # exact string DuckDB executed, whether default or custom.
        r = search_client.post("/search", json={"q": "x", "limit": 5})
        assert r.status_code == 200
        assert "ORDER BY distance" in r.json()["sql"]

    def it_returns_400_with_the_duckdb_message_for_bad_sql(
        search_client: TestClient
    ):
        # Client SQL is arbitrary; a typo must come back as a 400 with
        # DuckDB's own message, not an opaque 500.
        r = search_client.post("/search", json={
            "q": "x",
            "sql": "SELECT nonexistent_column FROM papers",
        })
        assert r.status_code == 400
        assert "nonexistent_column" in r.json()["detail"]


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
