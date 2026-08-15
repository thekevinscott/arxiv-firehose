"""Stdlib HTTP server. No web framework.

Routes:
  GET  /                    -> static/index.html   (catalogue)
  GET  /paper/<id>          -> static/reader.html  (per-paper reader)
  GET  /<file>              -> static/<file>
  GET  /api/papers?q=&limit= -> catalogue rows from tower
  GET  /api/paper/<id>      -> metadata (arxiv export API, cached)
  GET  /api/pdf/<id>        -> proxied+cached PDF bytes
  GET  /api/html/<id>       -> proxied+cached arxiv HTML (same-origin)
  GET  /api/citations/<id>  -> arxiv ids cited by the paper (+known titles)
  GET  /api/persona         -> the reader persona (markdown, may be empty)
  POST /api/chat            -> NDJSON stream of one Claude chat turn
"""

import asyncio
import json
import os
import subprocess
import sys
import traceback
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import arxiv, chat, persona, tower

STATIC = Path(__file__).resolve().parent.parent / "static"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".json": "application/json",
}

MAX_BODY = 30_000_000  # chat turns can carry base64 screenshots


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "arxiv-reader"

    # ---- helpers ---------------------------------------------------------

    def _send(self, status, body=b"", content_type="text/plain; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_json(self, status, obj):
        self._send(status, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def _serve_static(self, path):
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (STATIC / rel).resolve()
        if not str(target).startswith(str(STATIC)) or not target.is_file():
            self._send(404, b"not found")
            return
        self._send(
            200,
            target.read_bytes(),
            CONTENT_TYPES.get(target.suffix, "application/octet-stream"),
        )

    def _paper_id(self, path, prefix):
        arxiv_id = path[len(prefix):]
        if not arxiv.valid_id(arxiv_id):
            self._send_json(400, {"error": f"not an arxiv id: {arxiv_id}"})
            return None
        return arxiv_id

    # ---- routes ----------------------------------------------------------

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        try:
            if path.startswith("/paper/"):
                self._serve_static("reader.html")
            elif path == "/api/papers":
                qs = parse_qs(url.query)
                try:
                    rows = tower.catalogue(
                        q=(qs.get("q") or [None])[0],
                        limit=(qs.get("limit") or ["100"])[0],
                    )
                    self._send_json(200, {"papers": rows, "count": tower.count()})
                except Exception as exc:
                    # tower being busy (ingest holds the db lock) is routine;
                    # the catalogue degrades instead of erroring the page.
                    self._send_json(200, {"papers": [], "count": None,
                                          "error": f"corpus unavailable: {exc}"})
            elif path.startswith("/api/paper/"):
                arxiv_id = self._paper_id(path, "/api/paper/")
                if arxiv_id:
                    self._send_json(200, arxiv.metadata(arxiv_id))
            elif path.startswith("/api/pdf/"):
                arxiv_id = self._paper_id(path, "/api/pdf/")
                if arxiv_id:
                    self._send(200, arxiv.pdf(arxiv_id), "application/pdf")
            elif path.startswith("/api/html/"):
                arxiv_id = self._paper_id(path, "/api/html/")
                if arxiv_id:
                    self._send(200, arxiv.html(arxiv_id).encode(),
                               "text/html; charset=utf-8")
            elif path == "/api/persona":
                self._send_json(200, {"persona": persona.text()})
            elif path.startswith("/api/citations/"):
                arxiv_id = self._paper_id(path, "/api/citations/")
                if arxiv_id:
                    ids = arxiv.citations(arxiv_id)
                    try:
                        titles = tower.titles_for(ids)
                    except Exception:
                        titles = {}
                    self._send_json(200, {"citations": [
                        {"id": i, "title": titles.get(i)} for i in ids
                    ]})
            else:
                self._serve_static(path)
        except LookupError as exc:
            self._send_json(404, {"error": str(exc)})
        except urllib.error.HTTPError as exc:
            self._send_json(502, {"error": f"arxiv returned HTTP {exc.code}"})
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/chat":
            self._send(404, b"not found")
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                self._send_json(400, {"error": "missing or oversized body"})
                return
            payload = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "bad json"})
            return

        meta = None
        arxiv_id = payload.get("arxiv_id")
        if arxiv_id and arxiv.valid_id(arxiv_id):
            try:
                meta = arxiv.metadata(arxiv_id)
            except Exception:
                pass

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        client_gone = False

        def emit(event):
            nonlocal client_gone
            if client_gone:
                return
            line = (json.dumps(event) + "\n").encode()
            try:
                self.wfile.write(b"%x\r\n%s\r\n" % (len(line), line))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                client_gone = True

        try:
            asyncio.run(chat.run(payload, meta, emit))
        except Exception as exc:
            traceback.print_exc()
            emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        if client_gone:
            self.close_connection = True
            return
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


def tailnet_ip():
    """Bind to the tailnet interface only, so this isn't exposed to the LAN."""
    try:
        out = subprocess.run(
            ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip().splitlines()[0].strip() or None
    except Exception:
        return None


def main():
    host = os.environ.get("READER_HOST") or tailnet_ip() or "127.0.0.1"
    port = int(os.environ.get("READER_PORT", "8789"))
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    print(f"arxiv-reader listening on http://{host}:{port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
