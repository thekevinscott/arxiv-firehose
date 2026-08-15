# arxiv-reader

Per-paper reading UI on top of arxiv-firehose. Modeled on
transcript-to-prose: a stdlib Python HTTP server (`http.server`, no web
framework), one vanilla-JS page per view, no build step (PDF.js is
vendored under `static/vendor/`).

- `/` — catalogue: recent papers from the firehose corpus on tower
  (title search; paste an arxiv id to jump straight to it).
- `/paper/<arxiv-id>` — reader. Left pane shows the PDF (rendered with
  PDF.js) with a toggle to arxiv's HTML version. Right column:
  - **Paper** — metadata + abstract (arxiv export API, works for any id).
  - **Chat** — an ongoing conversation with Claude (claude-agent-sdk,
    session-resumed per turn). Selecting text in the HTML view or drawing
    a box on the PDF (Select region) auto-attaches that context as a chip
    on the next message; screenshots go up as images.
  - **Citations** — arxiv ids extracted from the paper's HTML, linking to
    their own reader pages in new tabs.

Paper artifacts (metadata, PDF, HTML) are proxied through the server and
cached in `~/.cache/arxiv-reader/`. Proxying the HTML is what makes it
same-origin, which is what makes text selection readable. The corpus on
tower is only consulted for the catalogue and citation titles; when it is
busy (ingest holds the SQLite lock) those degrade gracefully.

## Run

```
uv run main.py
```

Binds the tailnet interface, port 8789 (`READER_HOST` / `READER_PORT`
to override): http://duncan.tail790bbc.ts.net:8789
