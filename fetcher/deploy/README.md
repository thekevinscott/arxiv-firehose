# Deployment

Artifacts for running fetcher on tower (or any Linux box with systemd).

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

**Not required for the OOM fix.** The subprocess-isolated PDF converter
in `shared/convert.py` is what actually stopped the 2026-07-03/04 tower
OOMs. Tower stays on its existing cron entry unless you opt in.

If you *do* want to replace cron with a systemd timer, it adds three
things cron does not give you:

- **A cgroup memory cap (`MemoryMax=2G`).** After the 2026-07-03/04 OOM
  where a single pathological paper's PDF (2607.02140) pushed the fetch
  process past 30 GB anon-rss, systemd will now SIGKILL the whole unit
  if it exceeds 2 GB -- the kernel OOM-killer no longer gets to pick
  an arbitrary victim on tower.
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
# MemoryPeak=...         <-- should be well under 2G with the subprocess-
#                            isolated PDF converter (shared/convert.py)
# Result=success
```

## Rollout on tower (2026-07-04 OOM fix)

The whole rollout is the pull; cron keeps firing at 05:00 with the fixed
code. Installing the systemd timer (above) is optional and orthogonal.

```sh
ssh tower@tower.tail790bbc.ts.net -p 22884

cd ~/apps/arxiv-firehose/fetcher
git checkout main                          # tower had been on a feature branch
git pull                                   # picks up shared/convert.py fix
uv sync                                    # no new deps, but idempotent

# Sanity-check on the known-pathological paper without waiting for
# tomorrow's 05:00 timer. This runs the full fetch (~3.5 h); use a
# scratch data dir so the production data isn't touched:
mkdir -p /tmp/fetch-smoketest
/home/tower/.local/bin/uv run fetcher fetch --data-dir /tmp/fetch-smoketest \
    2>&1 | tee /tmp/fetch-smoketest.log
# Look for "pdf 2607.02140: conversion failed: pdf conversion grandchild
# died (exit=1)" -- that's the subprocess isolation catching the memory
# bomb. Peak RSS of the fetcher process itself should stay < 500 MB.

# Watch the next scheduled run:
journalctl --user -u fetcher-fetch.service -f
systemctl --user show fetcher-fetch.service -p MemoryPeak
```

The cron jobs (classify, and anything else) continue to run independently
of the API and the fetch timer. The API only adds an on-demand path; it
does not replace scheduled runs.
