"use strict";

const $ = (id) => document.getElementById(id);

const ID_RE = /^(\d{4}\.\d{4,5}|[a-z-]+(\.[A-Z]{2})?\/\d{7})(v\d+)?$/;

function fmtDate(iso) {
  return iso ? iso.slice(0, 10) : "";
}

function render(data) {
  const list = $("papers");
  list.innerHTML = "";
  if (data.error) {
    $("status").textContent = data.error;
  } else {
    $("status").textContent = data.papers.length ? "" : "no matches";
  }
  if (data.count != null) {
    $("count").textContent = data.count.toLocaleString() + " papers in corpus";
  }
  for (const p of data.papers) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = "/paper/" + p.arxiv_id;
    a.textContent = p.title || p.arxiv_id;
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = ` ${p.arxiv_id} · ${p.primary_category || ""} · ${fmtDate(p.announced_at)}`;
    li.appendChild(a);
    li.appendChild(meta);
    list.appendChild(li);
  }
}

let inFlight = null;
async function load(q) {
  if (inFlight) inFlight.abort();
  inFlight = new AbortController();
  $("status").textContent = "loading…";
  try {
    const url = "/api/papers?limit=100" + (q ? "&q=" + encodeURIComponent(q) : "");
    const resp = await fetch(url, { signal: inFlight.signal });
    render(await resp.json());
  } catch (e) {
    if (e.name !== "AbortError") $("status").textContent = "error: " + e;
  }
}

$("search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (ID_RE.test(q)) {
    location.href = "/paper/" + q;
    return;
  }
  load(q);
});

let timer = null;
$("q").addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(() => load($("q").value.trim()), 300);
});

load("");
