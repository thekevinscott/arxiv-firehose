"""Fetch + disk-cache arxiv artifacts: metadata, PDF bytes, HTML, citations.

Everything comes straight from arxiv.org (export API for metadata), so the
reader works for any paper, whether or not the firehose corpus mirrors it.
The HTML is proxied so it can live in a same-origin iframe -- that is what
makes text selection readable from the parent page.
"""

import gzip
import json
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

CACHE = Path.home() / ".cache" / "arxiv-reader"
UA = "arxiv-reader/0.1 (personal research tool)"

# New-style (2401.00001, optional version) or old-style (cs/0501001) ids.
ID_RE = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


def valid_id(arxiv_id):
    return bool(ID_RE.match(arxiv_id))


def _cache_path(arxiv_id, suffix):
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{arxiv_id.replace('/', '_')}{suffix}"


def _get(url, timeout=60):
    """GET with UA + gzip handling. Returns (final_url, bytes)."""
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        return resp.url, body


def metadata(arxiv_id):
    """Paper metadata from the arxiv export API, cached forever on disk."""
    path = _cache_path(arxiv_id, ".meta.json")
    if path.exists():
        return json.loads(path.read_text())

    _, body = _get(f"https://export.arxiv.org/api/query?id_list={arxiv_id}")
    feed = ET.fromstring(body)
    entry = feed.find(f"{ATOM}entry")
    if entry is None:
        raise LookupError(f"{arxiv_id}: no entry in export API response")
    id_url = entry.findtext(f"{ATOM}id") or ""
    if "/abs/" not in id_url:
        # The API returns a stub entry (title "Error") for unknown ids.
        raise LookupError(f"{arxiv_id}: not found on arxiv")

    primary = entry.find(f"{ARXIV}primary_category")
    meta = {
        "arxiv_id": arxiv_id,
        "title": " ".join((entry.findtext(f"{ATOM}title") or "").split()),
        "abstract": " ".join((entry.findtext(f"{ATOM}summary") or "").split()),
        "authors": [
            a.findtext(f"{ATOM}name")
            for a in entry.findall(f"{ATOM}author")
            if a.findtext(f"{ATOM}name")
        ],
        "primary_category": primary.get("term") if primary is not None else None,
        "categories": [
            c.get("term") for c in entry.findall(f"{ATOM}category") if c.get("term")
        ],
        "published": entry.findtext(f"{ATOM}published"),
        "updated": entry.findtext(f"{ATOM}updated"),
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
    }
    path.write_text(json.dumps(meta))
    return meta


def pdf(arxiv_id):
    """PDF bytes, cached on disk."""
    path = _cache_path(arxiv_id, ".pdf")
    if path.exists():
        return path.read_bytes()
    _, body = _get(f"https://arxiv.org/pdf/{arxiv_id}", timeout=120)
    path.write_bytes(body)
    return body


def html(arxiv_id):
    """Arxiv's HTML rendering with a <base> injected so relative assets
    (figures, CSS) resolve back to arxiv.org. Raises LookupError when the
    paper has no HTML version."""
    path = _cache_path(arxiv_id, ".html")
    if path.exists():
        return path.read_text()

    try:
        final_url, body = _get(f"https://arxiv.org/html/{arxiv_id}", timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise LookupError(f"{arxiv_id}: no HTML version") from exc
        raise
    text = body.decode("utf-8", errors="replace")
    base = final_url if final_url.endswith("/") else final_url + "/"
    tag = f'<base href="{base}">'
    if re.search(r"<head[^>]*>", text, re.IGNORECASE):
        text = re.sub(r"(<head[^>]*>)", r"\1" + tag, text, count=1, flags=re.IGNORECASE)
    else:
        text = tag + text
    path.write_text(text)
    return text


_DEADWEIGHT_RE = re.compile(
    r"<head\b.*?</head>|<script\b.*?</script>|<style\b.*?</style>|<!--.*?-->",
    re.DOTALL | re.IGNORECASE,
)


def html_for_llm(arxiv_id, cap=400_000):
    """The paper's HTML source slimmed for a model prompt: head/scripts/
    styles and comments dropped (no paper content, pure token weight),
    whitespace collapsed, capped so a huge paper can't blow the context."""
    body = _DEADWEIGHT_RE.sub("", html(arxiv_id))
    body = re.sub(r"[ \t]+", " ", body)
    if len(body) > cap:
        body = body[:cap] + "\n\n[source truncated here]"
    return body


_CITE_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})"
    r"|arXiv[.:]\s*(\d{4}\.\d{4,5})",
    re.IGNORECASE,
)


def citations(arxiv_id):
    """Arxiv ids referenced by the paper's HTML, in order of appearance.

    Extraction is textual (hrefs + "arXiv:NNNN.NNNNN" mentions), which is
    exactly what the bibliography of arxiv-rendered HTML contains.
    """
    body = html(arxiv_id)  # raises LookupError when there is no HTML
    bare = arxiv_id.split("v")[0]
    seen, out = set(), []
    for m in _CITE_RE.finditer(body):
        cited = (m.group(1) or m.group(2)).split("v")[0]
        if cited != bare and cited not in seen:
            seen.add(cited)
            out.append(cited)
    return out
