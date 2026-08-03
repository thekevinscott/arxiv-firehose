# fetcher

A CLI tool that maintains a local mirror of arxiv paper *metadata* in chosen
categories. For each paper it stores the metadata (title, authors, abstract,
categories, dates) as a JSON file on disk — one folder per paper, named by
arxiv id. It also derives an embedding of each abstract for semantic search.
No paper bodies (PDF/HTML/LaTeX) are downloaded. No database.

This is plumbing. It knows nothing about summarization or filtering — a
downstream tool reads the folders (or the HTTP API) for that.

## Install

Requires Python 3.13+. Built with [`uv`](https://docs.astral.sh/uv/).

```sh
uv sync
uv run fetcher --help
```

## Two directories

fetcher keeps two separate trees:

| Tree       | Default                   | Contents |
|------------|---------------------------|----------|
| Data dir   | `./arxiv-firehose/data`   | Organized paper folders — the deliverable. |
| Cache dir  | `~/.cache/arxiv-firehose` | [cachetta](https://github.com/thekevinscott/cachetta) request cache. |

Override the data dir with `--data-dir`; override the cache location with the
`ARXIV_FIREHOSE_CACHE_DIR` env var. They are deliberately separate: the cache
is disposable (delete it anytime, the next fetch just re-requests), the data
dir is what you keep.

### Data layout

```
{data_dir}/
  config.toml
  last_sync.json
  logs/
  embeddings.json              one abstract embedding per paper (for /search)
  {arxiv_id}/                  e.g. 2401.12345/  (legacy ids: cs_0501001/)
    metadata.json
```

### Cache layout

```
{cache_dir}/
  slices/{hash}                cached export-API day slice — settled days never expire
```

## Caching strategy

Everything that touches arxiv goes through the
[cachetta](https://github.com/thekevinscott/cachetta) cache, so arxiv is hit as
rarely as possible. Export-API day slices for settled days are immutable, so
those cache entries effectively never expire; a re-run of the same lookback
window only makes real requests for days that are new or previously missed.

## Commands

```sh
fetcher fetch              # daily ingest: sync metadata, then embed abstracts
fetcher pull <ids...>      # mirror specific papers' metadata by id
fetcher embed              # standalone embedding backfill (also a fetch stage)
fetcher status             # print counts
fetcher sql "<query>"      # read-only SQL over the dirsql schema
fetcher serve              # HTTP API for the above (tailnet-only)
```

Flags: `--data-dir`, `--config`, `--verbose/-v`, `--limit N`, `--dry-run`.

`sync_metadata` is SDK-only (`api.sync_metadata`); for granular debugging call
it from a REPL.

## How fetching works

`fetch` runs two stages in order:

1. **sync** queries the arxiv export API over the lookback window
   (`[ingest] backfill_days`) and writes a `metadata.json` for every new paper
   in each tracked category. Settled day slices come from the ~forever cache,
   so only new or missed days cost a real request.
2. **embed** reads each paper's `metadata.json.abstract`, computes a
   model2vec embedding (`potion-base-8M`, 256-dim, CPU-only), and writes it to
   `embeddings.json`. It runs to convergence: papers already embedded are
   skipped, so a missed run self-heals on the next fetch.

Both stages are idempotent and resumable: a re-run rewrites the same bytes,
and a missed cron day recovers on the next run as long as it falls within the
lookback window.

```sh
fetcher fetch
fetcher fetch --data-dir /tmp/mirror --limit 5
```

## Configuration

A TOML file at `{data_dir}/config.toml`, created with defaults on first run:

```toml
[categories]
include = ["cs.LG", "cs.CL", "cs.AI"]

[fetch]
source = "arxiv"        # only "arxiv" is implemented
concurrency = 1         # arxiv source must stay at 1

[ingest]
backfill_days = 90      # re-read this many days of export-API day slices each run
```

## Querying

The data dir is queryable in place through [dirsql](https://github.com/thekevinscott/dirsql),
which materializes the JSON files into an in-process SQLite schema:

| Table        | Rows |
|--------------|------|
| `papers`     | one per paper: `arxiv_id`, `announced_at` (normalized UTC ISO), `primary_category`. |
| `metadata`   | EAV: `(id, paper_id, key, value)`, one per metadata field except the abstract. |
| `embeddings` | one per paper: `paper_id`, `embedding` (JSON-array TEXT for sqlite-vec). |

```sh
fetcher sql "SELECT primary_category, COUNT(*) n FROM papers GROUP BY 1 ORDER BY 2 DESC"
```

Writes are rejected by dirsql's authorizer, so `sql` is effectively read-only.

## HTTP API

`fetcher serve` runs a small FastAPI app exposing the commands over HTTP, so a
remote client (or another agent) can trigger and inspect runs without SSHing
into the box. Bind to a tailscale IP and let the tailnet ACL be the perimeter —
there is no auth.

```sh
fetcher serve --host 100.x.y.z --port 8087    # tower (tailscale IP)
fetcher serve                                  # 127.0.0.1:8087 (local dev)
```

| Endpoint                     | Behavior |
|------------------------------|----------|
| `GET  /status`               | Same report as `fetcher status`. |
| `POST /fetch`                | Spawns `fetcher fetch`, returns a `Job` (HTTP 202). 409 if a fetch is already running. |
| `POST /embed`                | Spawns `fetcher embed`, returns a `Job` (HTTP 202). |
| `POST /pull`                 | Spawns `fetcher pull` for the posted `ids`. |
| `POST /sql`                  | Read-only SQL over the dirsql schema (`{"sql": "..."}`). |
| `POST /search`              | Embedding search + arbitrary SQL over the `search` relation. |
| `GET  /jobs`                 | Jobs spawned by this API process (ring buffer). |
| `GET  /jobs/{id}`            | One job: pid, started_at, exit_code, log_path. |
| `GET  /logs/{fetch,embed,pull}` | Tail the shared cron log (`?lines=50` default). |
| `GET  /docs`                 | OpenAPI / Swagger UI. |

Long jobs are fire-and-forget: a `POST` returns immediately with a job id; the
child runs detached (`start_new_session`) and survives an API restart. A
duplicate POST while a same-kind run is in flight returns `409 Conflict`
carrying the existing Job.

Deployment is a systemd unit; see `fetcher/deploy/`.

## Daily cron

arxiv announces new papers roughly once per weekday (~20:00 US Eastern). One run
per day keeps the mirror complete; the `backfill_days` window means a missed day
is recovered automatically.

## Prototype scope

- Only the `arxiv` source is implemented.
- No incremental re-fetch of updated versions (v2, v3); the first version seen
  is the one kept.
- arxiv's rate limit is real: network requests are spaced 3s apart. Cache hits
  are exempt. Do not raise `concurrency`.
