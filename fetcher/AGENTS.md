# fetcher — Agent Guide

Workflow and process for agents working in this repo. Architecture and behavior
live in `README.md`; this file is about *how to work*, not *what the tool does*.

Many conventions here are pulled from
[thekevinscott/dirsql](https://github.com/thekevinscott/dirsql) — when in doubt,
that repo's `AGENTS.md` is the reference.

## Ops: prefer the HTTP API over SSH

The fetcher exposes an HTTP API (`fetcher serve`) so `status` / `fetch` /
`classify` can be triggered from anywhere on the tailnet without SSHing
into the deployment box. On tower it runs as `fetcher-api.service`
(user systemd) bound to `0.0.0.0:8087`. Tower is behind home-router
NAT with no port-forward for 8087, so the API is reachable from the
tailnet (100.x) and the home LAN (192.168.x) but not from the public
internet.

**Default for any "what's tower doing?" or "kick a run" task: curl, not
SSH.** SSH commands need to be whitelisted per-call by the user; HTTP
calls do not. Reach for SSH only when you need something the API can't
do: editing config, reading the systemd journal, repairing data,
deploying new code.

```sh
BASE=http://tower.tail790bbc.ts.net:8087

curl -s $BASE/status                          # paper / classified counts
curl -s $BASE/logs/classify?lines=30          # tail the classify cron log
curl -s $BASE/logs/fetch?lines=30             # tail the fetch cron log
curl -s $BASE/jobs                            # list spawned jobs (this API process)

curl -sX POST $BASE/fetch                     # kick a fetch run; returns Job
curl -sX POST $BASE/classify                  # kick a classify run; returns Job
curl -sX POST $BASE/embed                     # embed any un-embedded abstracts; returns Job
curl -s $BASE/jobs/{id}                       # poll one job for exit_code

curl -s --json @/tmp/query.json $BASE/search  # semantic search (see below)
```

POST endpoints return immediately with a `Job` (id, pid, started_at,
exit_code=null). The work continues in the background as a subprocess
that survives an API restart. Watch progress with `/logs/{kind}` (tails
the shared cron log) or `/jobs/{id}` (per-process pid + exit code).

A POST while a same-kind job is already running returns `409 Conflict`
carrying the existing `Job` in `detail.job` -- safe to spam-trigger; the
API is the dedup point so you cannot accidentally double up on a long
classify run.

OpenAPI docs at `$BASE/docs` if a curl call is going sideways.

## Semantic search (`POST /search`)

The common flow: the user drops in with a question ("papers about X, last
week only, cs.AI only") and the agent translates it into a `/search` call
and runs it. Everything needed for that is below.

### Request shape

JSON body with three fields:

- `q` (required) — natural-language query. Embedded server-side
  (model2vec potion-base-8M) and compared against every abstract.
- `sql` (optional) — a DuckDB SELECT against the `papers` view. When
  omitted, the default is the top-`limit` nearest neighbors.
- `limit` (optional, default 10) — only applies to the default query;
  custom SQL controls its own LIMIT.

Response: `{"sql": ..., "count": N, "rows": [...]}`. Bad SQL returns
`400` with the DuckDB error message — iterate on the SQL and re-POST.

### The `papers` view

One row per paper, with `distance` (cosine distance to `q`, lower =
more similar) precomputed:

| column           | type      | notes                                        |
|------------------|-----------|----------------------------------------------|
| arxiv_id         | VARCHAR   | e.g. `2401.12345`                            |
| title            | VARCHAR   |                                              |
| abstract         | VARCHAR   |                                              |
| authors          | VARCHAR[] | list of names                                |
| primary_category | VARCHAR   | e.g. `cs.CL`                                 |
| categories       | VARCHAR[] | all categories                               |
| announced_at     | VARCHAR   | RFC-2822: `Fri, 22 May 2026 00:00:00 -0400`  |
| updated_at       | VARCHAR   | same format                                  |
| html_url         | VARCHAR   |                                              |
| distance         | DOUBLE    | cosine distance to `q`                       |

`announced_at` is a *string*; parse it with
`try_strptime(announced_at, '%a, %d %b %Y %H:%M:%S %z')` before
comparing to timestamps. (Do not use a `GMT`-suffix format — the offset
is numeric — and always `try_strptime`, never `strptime`, so unparseable
rows become NULL instead of erroring.)

### Recipes

Write the body to `/tmp/query.json`, then
`curl -s --json @/tmp/query.json $BASE/search`.

**Semantic question only** (default SQL, top 10):

```json
{"q": "agents that learn user preferences from feedback"}
```

**Restrict by date** (last 7 days):

```json
{"q": "agents that learn user preferences from feedback",
 "sql": "SELECT arxiv_id, title, announced_at, distance FROM papers WHERE try_strptime(announced_at, '%a, %d %b %Y %H:%M:%S %z') > now() - INTERVAL 7 DAY ORDER BY distance LIMIT 10"}
```

**Restrict by category** (primary, or any category):

```json
{"q": "retrieval augmented generation",
 "sql": "SELECT arxiv_id, title, primary_category, distance FROM papers WHERE primary_category = 'cs.CL' ORDER BY distance LIMIT 10"}
```

```sql
-- any-category variant: match against the full list
WHERE list_contains(categories, 'cs.HC')
```

**Keyword + semantic** (belt and suspenders):

```sql
WHERE abstract ILIKE '%preference%' ORDER BY distance LIMIT 10
```

**Distance threshold** instead of top-N (useful for "is there anything
about X at all?" — empty result means no):

```sql
WHERE distance < 0.55 ORDER BY distance LIMIT 50
```

Rough calibration: < 0.5 strongly related, 0.5–0.7 loosely related,
> 0.8 noise.

**Aggregates** work too — any SELECT is fine:

```sql
-- where does this topic live? category histogram of the 100 NN
SELECT primary_category, count(*) AS n
FROM (SELECT * FROM papers ORDER BY distance LIMIT 100)
GROUP BY 1 ORDER BY n DESC
```

**Author search**:

```sql
WHERE list_contains(authors, 'Exact Name')          -- exact
WHERE len(list_filter(authors, a -> a ILIKE '%scott%')) > 0  -- fuzzy
```

### Executing without permission prompts

`curl --json` with an inline body triggers a permission prompt every
time. Two prompt-free patterns:

1. Write the JSON body to `/tmp/<something>.json` (Write tool), then
   `curl -s --json @/tmp/<something>.json $BASE/search`.
2. Or write a small urllib script to `/tmp/*.py` and run it with
   `uv run python /tmp/whatever.py` from the `fetcher/` directory —
   `uv run` is allowlisted. This also gives you `json.dumps(..., indent=2)`
   for readable output.

If embeddings look stale (a paper you expect is missing), `POST /embed`
first — it fills any gap and is safe to spam (409 dedup).

## Scratch files

Write scratch/temporary files to `/tmp`, not into the repo. Use unique
filenames to avoid collisions with other sessions. Temporary scripts (Python or
shell helpers) also go in `/tmp` and run from there.

## Shell commands

**Do not chain commands** with `;`, `&&`, or `||`. Chaining forces a single
bulk permission decision for the whole pipeline; each command must be evaluated
on its own. Run each as a separate call.

Exceptions: a pipe (`|`) is fine when it is genuinely one operation
(`cmd | jq`). Heredocs are fine. `cd path && cmd` is **not** — pass an absolute
path or `cd` as its own call.

## Testing

Never monkey patch.

### Red/Green development

Follow **red/green** (test-first) methodology:

1. **Write the test first** — it must capture the desired behavior.
2. **Run it and confirm it fails (RED).** Do not proceed until the test is
   reliably red. A test that passes before the implementation exists proves
   nothing.
3. **Make the minimal change to pass (GREEN).** Only then write the
   implementation.
4. **Refactor** if needed, keeping tests green.

### TDD order: outside-in

Write tests before implementation, starting from the outermost layer:

1. **Integration test first** — proves the feature works from the consumer's
   (SDK caller's) perspective.
2. **Unit tests** — written as you implement each module.

A feature is not done until its integration test passes.

### Test layers

- **Unit tests** — colocated with source: `foo.py` → `foo_test.py` in the same
  directory. Test pure functions and small classes in isolation. Dependencies
  are passed as arguments, never patched onto modules.
- **Integration tests** — `tests/integration/`. Exercise the **Python SDK's
  public API** (`fetcher.sync_metadata`, `.fetch`, `.status`, `.run`),
  with third-party I/O (here: all `httpx` network calls) replaced by
  **fixture-backed fakes injected through the public API**. Never test the CLI
  here — the CLI is a thin wrapper over the SDK and has nothing worth testing
  in-process.

### What to mock, what not to

**Monkeypatching is a code smell.** If a test replaces a module-level attribute
on the system under test (`monkeypatch.setattr(download, "httpx", ...)`), the
test is at the wrong layer or the production code is missing a seam.

Concrete rules:

1. **Integration tests hit the SDK's public API**, not the CLI.
2. **Third-party dependencies are injected, not patched.** The network
   transport is a parameter (`transport=`) threaded from the SDK down to the
   downloader. Integration tests pass a fake transport that serves bytes from
   `tests/integration/__fixtures__/`. Production code keeps the real,
   rate-limited httpx transport as the default.
3. **Unit tests** receive their dependencies as arguments or constructor
   params. If a unit can only be tested by patching its imports, refactor for
   dependency injection.

If you reach for `monkeypatch.setattr(some_production_module, ...)`, stop: this
is almost always an integration test that should go through the public API, or
a unit whose target needs an injected parameter.

### pytest-describe

Tests use [`pytest-describe`](https://pypi.org/project/pytest-describe/) BDD
blocks: a `describe_<thing>()` function containing `it_<behaves>()` cases.
Group by the unit under test; name cases for the behavior they assert.

```python
def describe_parse_id():
    def it_parses_modern_ids():
        assert parse_id("2401.12345").yymm == "2401"

    def it_rejects_garbage():
        with pytest.raises(ValueError):
            parse_id("nonsense")
```

### Running tests

`just` is run through uv (the PyPI `just` package is broken; use `rust-just`):

```sh
uvx --from rust-just just              # list recipes
uvx --from rust-just just test-unit    # colocated unit tests, fast
uvx --from rust-just just test-integration
uvx --from rust-just just test         # both
```

If `just` is installed natively, plain `just test` also works.

## SDK / CLI split

Every command is a function in the Python SDK (`fetcher/api.py`, re-exported
from `fetcher/__init__.py`). The CLI (`fetcher/cli.py`) is a thin
typer wrapper: it parses flags and calls the SDK. New behavior goes in the SDK
and gets an integration test there; the CLI only ever grows argument plumbing.

## Package layout

```
src/fetcher/
  api.py            # SDK surface (orchestrator)
  cli.py            # typer wrapper
  commands/         # one subpackage / module per CLI command
    fetch/          # daily ingest: sync metadata + render markdown
      __init__.py   #   the composite (sync → render) lives here
      sync.py       #   stage 1: arxiv RSS -> metadata.json per paper
      render.py     #   stage 2: HTML/LaTeX/PDF -> paper.md per paper
    classify/       # daily classify: abstract -> topic flags via LLM
    status.py       # read-only counts (single-file command)
    train_categories.py  # developer: compile labels/ -> prompts/ (cached)
  shared/           # cross-command utilities (config, paths, logsetup,
                    # download, convert, dirsql_schema)
```

CLI surface mirrors the commands/ folder: `fetcher fetch`, `fetcher
classify`, `fetcher status`, `fetcher train-categories`. The fetch stages
(sync, render) are SDK-only (`api.sync_metadata`, `api.render_markdown`);
they live as separate modules inside `commands/fetch/` so each has its
own integration test file and can be called granularly.

A command is a flat module while it fits in one file (`commands/status.py`,
`commands/train_categories.py`). The moment it needs internal modules
(multiple stages, a backend, a store, a prompt loader) it gets promoted
to a subpackage (`commands/fetch/run.py` + siblings,
`commands/classify/run.py` + siblings). A module belongs in `shared/`
only if it's imported by more than one command or by `api.py` -- not a
dumping ground.
