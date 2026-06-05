# fetcher

A CLI tool that maintains a local mirror of arxiv papers in chosen categories.
For each paper it stores the metadata, the PDF, and the LaTeX source as plain
files on disk — one folder per paper, named by arxiv id. No database.

This is plumbing. It knows nothing about LLMs, filtering, or summarization — a
downstream tool reads the folders for that.

## Install

Requires Python 3.12+. Built with [`uv`](https://docs.astral.sh/uv/).

```sh
uv sync
uv run fetcher --help
```

## Two directories

fetcher keeps two separate trees:

| Tree       | Default                   | Contents |
|------------|---------------------------|----------|
| Data dir   | `./arxiv-firehose/data`   | Organized paper folders — the deliverable. |
| Cache dir  | `~/.cache/arxiv-firehose` | [cachetta](https://github.com/thekevinscott/cachetta) download cache. |

Override with `--data-dir` and `--cache-dir`. They are deliberately separate:
the cache is disposable (delete it anytime, the next fetch just re-downloads),
the data dir is what you keep.

### Data layout

```
{data_dir}/
  config.toml
  last_sync.json
  logs/
  {arxiv_id}/                  e.g. 2401.12345/  (legacy ids: cs_0501001/)
    metadata.json
    {arxiv_id}.pdf
    source/                    extracted LaTeX .tex/.bib/figures
    .no_latex                  marker: arxiv has no LaTeX source for this paper
```

### Cache layout

```
{cache_dir}/
  rss/{category}.pkl           cached RSS feed   — expires after 1 day
  pdf/{arxiv_id}vN.pkl         cached PDF bytes  — never expires
  eprint/{arxiv_id}.pkl        cached e-print archive bytes — never expires
```

## Caching strategy

Everything that touches arxiv goes through the [cachetta](https://github.com/thekevinscott/cachetta)
cache, so arxiv is hit as rarely as possible:

| Request          | Cache duration | Why |
|------------------|----------------|-----|
| RSS feed         | 1 day          | arxiv regenerates each feed once a day, after the daily announcement. |
| PDF / LaTeX      | never expires  | A paper's content is immutable for a fixed version. |

Consequences, all verifiable by re-running:

- **First run** downloads a full week of papers — rate-limited to one request
  per 3 s, so a few hundred papers takes on the order of an hour or two.
- **Re-running the same day** is near-instant: the feed is still fresh in the
  cache and every paper is already on disk — zero arxiv requests.
- **Running the next day** re-fetches the feeds (the 1-day cache has expired)
  and downloads only the papers that are genuinely new — slightly longer than
  an instant no-op, far shorter than the first run.

Prune the data dir however you like; the cache lets a re-fetch skip the network
entirely. Prune the cache too if it grows large — re-fetching just costs arxiv
bandwidth, nothing else.

## Commands

```sh
fetcher fetch              # daily ingest: sync RSS metadata, render markdown
fetcher classify           # daily labeling: abstract -> topic flags (LLM)
fetcher status             # print counts
fetcher train-categories   # developer: compile labels/ -> prompts/
```

Flags: `--data-dir`, `--cache-dir`, `--config`, `--verbose/-v`, `--limit N`,
`--dry-run`.

The fetch stages are also callable from the SDK (`api.sync_metadata`,
`api.render_markdown`) when granular control is wanted; they are not
exposed on the CLI.

`train-categories` is a developer command, not a cron one: it walks
`labels/`, treats every subdir with a `_schema.json` as a category, and
compiles each into `prompts/<category>/`. The output field name is the
category name with hyphens swapped for underscores
(`is-about-control` -> `is_about_control`) -- same key the runtime
classify writes per paper. Each compile is content-cached at
`~/.cache/arxiv-firehose/classify/{hash}/`; unchanged labels copy from
cache. With `--optimizer gepa` it drives a DSPy round-trip against
`--model`/`--base-url`.

```sh
fetcher train-categories                                  # labels/ -> prompts/
fetcher train-categories --optimizer gepa --model phi4:14b
```

## How fetching works

`fetch` runs two stages in order:

1. **sync** writes a `metadata.json` for every paper in the RSS feed of each
   tracked category (RSS carries ~1 week of submissions).
2. **render** walks every paper folder and produces a markdown rendering
   for *each* one, every run — there is no "already on disk, skip" shortcut.
   It re-fetches and rewrites the data-dir files unconditionally. Three
   conversion paths are tried in order: arxiv native HTML → LaTeX e-print
   → PDF.
3. Every download goes through the cachetta-backed downloader, which serves
   the bytes from the on-disk cache or the network **transparently**. arxiv
   content is immutable for a fixed paper version, so cache entries never
   expire. From `fetch`'s perspective every paper is a fresh fetch; the
   cache only decides whether a request touches the network. A file deleted
   or corrupted in the data dir is rewritten from the cache with **no
   network call and no rate-limit delay**.

Both stages are idempotent and resumable: a re-run rewrites the same bytes,
and after a crash the next run simply fetches everything again — cheaply,
since the cache absorbs it.

```sh
fetcher fetch
fetcher fetch --data-dir /tmp/mirror --cache-dir /tmp/cache --limit 5
```

## Configuration

A TOML file at `{data_dir}/config.toml`, created with defaults on first run:

```toml
[categories]
include = ["cs.LG", "cs.CL", "cs.AI"]

[fetch]
source = "arxiv"        # only "arxiv" is implemented
concurrency = 1         # arxiv source must stay at 1
latex = true            # also pull the LaTeX source tarball

[ingest]
backfill_days = 0       # skip papers older than N days; 0 = all of RSS
```

## Daily cron

arxiv announces new papers roughly once per weekday (~20:00 US Eastern). One run
per day keeps the mirror complete; the RSS window is ~1 week so a missed day is
recovered automatically.

```cron
0 3 * * *  cd ~/work && fetcher fetch >> ~/.cache/arxiv-firehose/cron.log 2>&1
```

Hourly polling is safe — it is idempotent and a no-op when nothing is new — but
arxiv publishes once a day, so most hourly runs find nothing. Use hourly only
if you want to pick up the daily drop within the hour.

## Disk planning

At ~500 papers/day across the ML cluster, ~3 MB average PDF, expect roughly
**1.5 GB/day, ~550 GB/year** in the data dir. LaTeX source adds ~10-20%.

Note: the cachetta cache holds a **second copy** of every downloaded PDF and
e-print archive, so the cache dir grows at a similar rate. Put it on a volume
you are willing to wipe, or delete it periodically — it only exists to spare
arxiv repeat traffic, and losing it costs nothing but bandwidth.

## Prototype scope

- Only the `arxiv` source is implemented.
- No incremental re-fetch of updated versions (v2, v3); the first version seen
  is the one kept.
- arxiv's rate limit is real: network requests are spaced 3s apart. Cache hits
  are exempt. Do not raise `concurrency`.
