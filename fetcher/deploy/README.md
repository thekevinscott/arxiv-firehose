# Deployment

Artifacts for running fetcher on tower (or any Linux box with systemd).

## fetcher-api.service

The HTTP API (`fetcher serve`) as a systemd unit.

Install:

```sh
# As tower, with the repo already at /home/tower/apps/arxiv-firehose and
# `uv sync` already done.
sudo cp /home/tower/apps/arxiv-firehose/fetcher/deploy/fetcher-api.service \
    /etc/systemd/system/fetcher-api.service
sudo systemctl daemon-reload
sudo systemctl enable --now fetcher-api.service
systemctl status fetcher-api.service
```

Verify:

```sh
curl -s http://localhost:8087/status
curl -s http://tower.tail790bbc.ts.net:8087/status   # from any tailnet client
```

Edit the unit if paths differ (the defaults assume `/home/tower/apps/...`
for code and `/mnt/bertha/...` for data and logs).

The cron jobs continue to run independently of the API. The API only
adds an on-demand path; it does not replace scheduled runs.
