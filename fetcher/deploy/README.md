# Deployment

Artifacts for running fetcher on tower (or any Linux box with systemd).

## Deploying new code

Code lives at `~/apps/arxiv-firehose` (a git checkout) on the deploy user's
account; the systemd units below run out of `~/apps/arxiv-firehose/fetcher`.
There is no build step -- `uv run` resolves the venv on each invocation -- so
a deploy is just: update the checkout, then bounce the resident API.

```sh
cd ~/apps/arxiv-firehose
git pull

# The API (fetcher serve) is the only long-running process; restart it to
# pick up new code.
systemctl --user restart fetcher-api.service
systemctl --user status fetcher-api.service
```

The `fetcher-fetch.timer` needs no restart -- it runs a fresh
`uv run fetcher fetch` each firing and picks up new code on its own. Re-copy
the unit files only if the `.service`/`.timer` themselves changed (then
`daemon-reload`).

Sanity-check that the running API actually matches the checkout. The
`/logs/{kind}` enum is a cheap version tell: current code exposes only
`fetch`, `embed`, `pull`, so a stale build is obvious.

```sh
# Current build -> 422 listing exactly {fetch, embed, pull}.
# A stale build answers 200 (or lists classify/render) -> it predates the
# metadata-only refactor and is still running retired stages. Redeploy.
curl -s http://localhost:8087/logs/classify
```

## Retiring a stage's schedule

When a pipeline stage is removed from the code (e.g. `classify` was dropped in
the metadata-only refactor), its scheduled trigger does **not** disappear on
its own. A leftover timer or cron line keeps invoking the old entrypoint --
now against a missing dependency -- and errors every morning. Symptom: a
`{stage}-cron.log` that is still growing, or `/logs/{stage}` still returning
rows, for a stage the current code no longer has.

The trigger lives in one of two places (fetcher stages started on cron and
were later migrated to user timers, so a retired one may hide in either).
Check both:

```sh
# 1. user systemd timers
systemctl --user list-timers --all --no-pager
# if a `fetcher-<stage>.timer` is listed:
systemctl --user disable --now fetcher-<stage>.timer
rm ~/.config/systemd/user/fetcher-<stage>.service \
   ~/.config/systemd/user/fetcher-<stage>.timer
systemctl --user daemon-reload

# 2. crontab
crontab -l                 # look for a `... fetcher <stage> ...` line
crontab -e                 # delete it
```

Verify the next morning that `<stage>-cron.log` has gone quiet and
`/logs/<stage>` stops gaining rows.

## fetcher-api.service

The HTTP API (`fetcher serve`) as a **user** systemd unit -- no sudo
required to install, restart, or read logs.

Install (run as the deploy user, e.g. `tower`):

```sh
# One-time: let user units survive logout / start at boot.
# `loginctl` accepts this without sudo for one's own user on modern systemd.
loginctl enable-linger "$USER"

mkdir -p ~/.config/systemd/user
cp ~/apps/arxiv-firehose/fetcher/deploy/fetcher-api.service \
    ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now fetcher-api.service
systemctl --user status fetcher-api.service
```

Verify:

```sh
curl -s http://localhost:8087/status
curl -s http://tower.tail790bbc.ts.net:8087/status    # any tailnet client
```

Restart / inspect later (still no sudo):

```sh
systemctl --user restart fetcher-api.service
journalctl --user -u fetcher-api.service -n 50
```

Edit the unit if paths differ (the defaults assume `~/apps/arxiv-firehose/...`
for code and `/mnt/bertha/...` for data and logs; `%h` resolves to the
deploy user's home).

## fetcher-fetch.service + fetcher-fetch.timer (optional)

Optional replacement for the cron entry. `fetch` now pulls only metadata
and embeds abstracts (no PDF/markdown conversion), so it is light; the
timer is a convenience, not a fix. It adds three things cron does not give
you:

- **A cgroup memory cap (`MemoryMax=2G`).** Defense-in-depth: any runaway
  leak trips the cap and systemd SIGKILLs the unit instead of the kernel
  OOM-killer picking an arbitrary victim on tower.
- **Structured journal integration.** `journalctl --user -u fetcher-fetch`
  gets a clean per-invocation history. The append-mode log file at
  `/mnt/bertha/.../fetcher-cron.log` is still written for the HTTP API's
  `/logs/fetch` tail endpoint.
- **A jittered start (0-5 min).** Nothing else on tower cares, but
  removes coincident wakeups if we add more timers later.

Install (as the deploy user):

```sh
mkdir -p ~/.config/systemd/user
cp ~/apps/arxiv-firehose/fetcher/deploy/fetcher-fetch.service \
    ~/apps/arxiv-firehose/fetcher/deploy/fetcher-fetch.timer \
    ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now fetcher-fetch.timer
```

**Then remove the crontab entry for `fetcher fetch`** -- otherwise both
fire and the API's 409-guard rejects the second one:

```sh
crontab -e   # delete the "0 5 * * * ... fetcher fetch ..." line
```

Verify the timer is armed:

```sh
systemctl --user list-timers fetcher-fetch.timer
# NEXT ELAPSES LEFT  LAST PASSED UNIT                 ACTIVATES
# Fri 2026-07-05 05:00:00 EDT ...  fetcher-fetch.timer fetcher-fetch.service
```

Manual trigger (skips the timer, runs the service now):

```sh
systemctl --user start fetcher-fetch.service
systemctl --user status fetcher-fetch.service
journalctl --user -u fetcher-fetch.service -f
```

Verify the memory cap after a run:

```sh
systemctl --user show fetcher-fetch.service \
    -p MemoryMax -p MemoryPeak -p Result
# MemoryMax=2147483648
# MemoryPeak=...         <-- metadata-only fetch stays well under 2G
# Result=success
```
