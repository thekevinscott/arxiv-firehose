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

The cron jobs continue to run independently of the API. The API only
adds an on-demand path; it does not replace scheduled runs.
