"""Read-only queries against the firehose corpus on tower (POST /sql).

Used only for the catalogue and for labeling citations with titles the
corpus already knows. The reader page itself never depends on tower --
paper metadata comes straight from arxiv, so any id works.
"""

import json
import urllib.request

TOWER_SQL = "http://tower.tail790bbc.ts.net:8087/sql"


def query(sql, timeout=30):
    req = urllib.request.Request(
        TOWER_SQL,
        data=json.dumps({"sql": sql}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["rows"]


def _quote(s):
    return "'" + s.replace("'", "''") + "'"


def catalogue(q=None, limit=100):
    """Most recent papers, optionally title-filtered."""
    limit = max(1, min(int(limit), 500))
    where = f"AND m.value LIKE {_quote('%' + q + '%')}" if q else ""
    rows = query(
        f"""
        SELECT p.arxiv_id, p.announced_at, p.primary_category, m.value AS title
        FROM papers p
        JOIN metadata m ON m.paper_id = p.arxiv_id AND m.key = 'title'
        WHERE 1=1 {where}
        ORDER BY p.announced_at DESC
        LIMIT {limit}
        """
    )
    return rows


def count():
    return query("SELECT COUNT(*) AS n FROM papers")[0]["n"]


def titles_for(ids):
    """{arxiv_id: title} for whichever of `ids` the corpus knows."""
    if not ids:
        return {}
    id_list = ", ".join(_quote(i) for i in ids)
    rows = query(
        f"SELECT paper_id, value FROM metadata "
        f"WHERE key = 'title' AND paper_id IN ({id_list})"
    )
    return {r["paper_id"]: r["value"] for r in rows}
