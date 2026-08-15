"use strict";

import * as pdfjs from "/vendor/pdf.min.mjs";
import { marked } from "/vendor/marked.esm.js";
import DOMPurify from "/vendor/purify.es.mjs";
pdfjs.GlobalWorkerOptions.workerSrc = "/vendor/pdf.worker.min.mjs";

// Sanitized: the reply can echo content that originated in an arbitrary
// paper's HTML, so it never reaches innerHTML unscrubbed.
function renderMarkdown(el, text) {
  el.innerHTML = DOMPurify.sanitize(marked.parse(text));
}

const $ = (id) => document.getElementById(id);
const arxivId = decodeURIComponent(location.pathname.replace(/^\/paper\//, ""));

let mode = "pdf"; // "pdf" | "html"
let htmlLoaded = false;
let htmlAvailable = null; // null = unknown yet
let sessionId = null;
let chips = []; // {type:"text",text} | {type:"image",data,media_type,dataUrl}
const transcript = []; // {role, text, chips?} -- mirrored into the URL fragment

// ---- paper info ----------------------------------------------------------

async function loadInfo() {
  try {
    const resp = await fetch("/api/paper/" + arxivId);
    const meta = await resp.json();
    if (!resp.ok) throw new Error(meta.error || resp.status);
    document.title = meta.title + " · arxiv reader";
  } catch {
    document.title = arxivId + " · arxiv reader";
  }
}

// ---- PDF pane ------------------------------------------------------------

async function renderPdf() {
  const status = $("viewer-status");
  status.textContent = "loading PDF…";
  try {
    const doc = await pdfjs.getDocument("/api/pdf/" + arxivId).promise;
    status.textContent = `rendering ${doc.numPages} pages…`;
    const container = $("pdf-pages");
    const width = container.clientWidth - 2;
    for (let n = 1; n <= doc.numPages; n++) {
      const page = await doc.getPage(n);
      const base = page.getViewport({ scale: 1 });
      const scale = width / base.width;
      const dpr = window.devicePixelRatio || 1;
      const viewport = page.getViewport({ scale: scale * dpr });
      const canvas = document.createElement("canvas");
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      canvas.style.width = `${viewport.width / dpr}px`;
      canvas.style.height = `${viewport.height / dpr}px`;
      canvas.className = "pdf-page";
      container.appendChild(canvas);
      await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
    }
    status.textContent = "";
  } catch (e) {
    status.textContent = "PDF failed to load: " + (e.message || e);
  }
}

// ---- HTML pane + text selection -----------------------------------------

function loadHtml() {
  const frame = $("html-frame");
  if (htmlLoaded) return;
  htmlLoaded = true;
  frame.addEventListener("load", () => {
    try {
      const doc = frame.contentDocument;
      // Same-origin (proxied), so selections are readable from here.
      doc.addEventListener("mouseup", () => {
        const text = String(frame.contentWindow.getSelection() || "").trim();
        if (text.length > 3) addChip({ type: "text", text });
      });
      htmlAvailable = true;
    } catch {
      /* ignore */
    }
  });
  frame.src = "/api/html/" + arxivId;
}

async function checkHtml() {
  // HEAD-ish probe so the toggle can say up front when there is no HTML.
  try {
    const resp = await fetch("/api/html/" + arxivId, { method: "GET" });
    htmlAvailable = resp.ok;
  } catch {
    htmlAvailable = false;
  }
  if (!htmlAvailable) {
    $("mode-html").disabled = true;
    $("mode-html").title = "This paper has no arxiv HTML version";
  }
}

function setMode(next) {
  mode = next;
  $("pdf-pages").hidden = mode !== "pdf";
  $("html-frame").hidden = mode !== "html";
  $("mode-pdf").classList.toggle("active", mode === "pdf");
  $("mode-html").classList.toggle("active", mode === "html");
  $("select-region").hidden = mode !== "pdf";
  if (mode === "html") loadHtml();
}

$("mode-pdf").addEventListener("click", () => setMode("pdf"));
$("mode-html").addEventListener("click", () => setMode("html"));

// ---- PDF region selection -> screenshot chip ----------------------------

let selecting = false;

$("select-region").addEventListener("click", () => {
  selecting = !selecting;
  $("select-region").classList.toggle("active", selecting);
  $("pdf-pages").classList.toggle("selecting", selecting);
});

(function wireRegionSelect() {
  const container = $("pdf-pages");
  let startX, startY, rect = null;

  function box(x1, y1, x2, y2) {
    return {
      left: Math.min(x1, x2), top: Math.min(y1, y2),
      width: Math.abs(x1 - x2), height: Math.abs(y1 - y2),
    };
  }

  container.addEventListener("mousedown", (e) => {
    if (!selecting) return;
    e.preventDefault();
    startX = e.pageX;
    startY = e.pageY;
    rect = document.createElement("div");
    rect.className = "select-rect";
    document.body.appendChild(rect);
  });

  window.addEventListener("mousemove", (e) => {
    if (!rect) return;
    const b = box(startX, startY, e.pageX, e.pageY);
    Object.assign(rect.style, {
      left: b.left + "px", top: b.top + "px",
      width: b.width + "px", height: b.height + "px",
    });
  });

  window.addEventListener("mouseup", (e) => {
    if (!rect) return;
    const b = box(startX, startY, e.pageX, e.pageY);
    rect.remove();
    rect = null;
    selecting = false;
    $("select-region").classList.remove("active");
    container.classList.remove("selecting");
    if (b.width < 8 || b.height < 8) return;
    capture(b);
  });

  function capture(b) {
    // Find the page canvas with the largest overlap, then crop from its
    // backing store (which is devicePixelRatio times the CSS size).
    let best = null;
    for (const canvas of container.querySelectorAll("canvas.pdf-page")) {
      const r = canvas.getBoundingClientRect();
      const abs = {
        left: r.left + window.scrollX, top: r.top + window.scrollY,
        right: r.right + window.scrollX, bottom: r.bottom + window.scrollY,
      };
      const overlap =
        Math.max(0, Math.min(b.left + b.width, abs.right) - Math.max(b.left, abs.left)) *
        Math.max(0, Math.min(b.top + b.height, abs.bottom) - Math.max(b.top, abs.top));
      if (overlap > 0 && (!best || overlap > best.overlap)) {
        best = { canvas, abs, overlap };
      }
    }
    if (!best) return;
    const { canvas, abs } = best;
    const scale = canvas.width / (abs.right - abs.left);
    const sx = Math.max(0, (b.left - abs.left) * scale);
    const sy = Math.max(0, (b.top - abs.top) * scale);
    const sw = Math.min(canvas.width - sx, b.width * scale);
    const sh = Math.min(canvas.height - sy, b.height * scale);
    if (sw < 4 || sh < 4) return;
    const crop = document.createElement("canvas");
    crop.width = sw;
    crop.height = sh;
    crop.getContext("2d").drawImage(canvas, sx, sy, sw, sh, 0, 0, sw, sh);
    const dataUrl = crop.toDataURL("image/png");
    addChip({
      type: "image",
      media_type: "image/png",
      data: dataUrl.split(",", 2)[1],
      dataUrl,
    });
  }
})();

// ---- context chips -------------------------------------------------------

function addChip(chip) {
  chips.push(chip);
  renderChips();
  $("chat-input").focus();
}

function renderChips() {
  const holder = $("context-chips");
  holder.innerHTML = "";
  chips.forEach((chip, i) => {
    const el = document.createElement("div");
    el.className = "chip";
    if (chip.type === "image") {
      const img = document.createElement("img");
      img.src = chip.dataUrl;
      el.appendChild(img);
    } else {
      const span = document.createElement("span");
      span.textContent = chip.text.length > 160 ? chip.text.slice(0, 160) + "…" : chip.text;
      span.title = chip.text;
      el.appendChild(span);
    }
    const x = document.createElement("button");
    x.textContent = "×";
    x.className = "chip-x";
    x.addEventListener("click", () => {
      chips.splice(i, 1);
      renderChips();
    });
    el.appendChild(x);
    holder.appendChild(el);
  });
}

// ---- chat ----------------------------------------------------------------

function addMessage(role) {
  const el = document.createElement("div");
  el.className = "msg " + role;
  $("chat-log").appendChild(el);
  return el;
}

function scrollLog() {
  const log = $("chat-log");
  log.scrollTop = log.scrollHeight;
}

function renderTurn(turn) {
  const el = addMessage(turn.role);
  for (const chip of turn.chips || []) {
    if (chip.type === "image") {
      if (chip.dataUrl) {
        const img = document.createElement("img");
        img.src = chip.dataUrl;
        img.className = "msg-img";
        el.appendChild(img);
      } else {
        // Restored from the URL, where image bytes don't fit; the real
        // image still lives in the server-side session.
        const ph = document.createElement("div");
        ph.className = "meta";
        ph.textContent = "[screenshot]";
        el.appendChild(ph);
      }
    } else {
      const q = document.createElement("blockquote");
      q.textContent = chip.text;
      el.appendChild(q);
    }
  }
  if (turn.role === "user") {
    const p = document.createElement("p");
    p.textContent = turn.text;
    el.appendChild(p);
  } else {
    renderMarkdown(el, turn.text);
  }
  return el;
}

// ---- conversation in the URL --------------------------------------------
// Deflate + base64url in the fragment: never sent to the server, a few KB
// for a typical conversation, restorable by anyone who has this server.
// Image chips are stored as placeholders (base64 PNG doesn't compress and
// would blow the URL); resume fidelity is unaffected -- images live in the
// server-side SDK session.

async function deflate(text) {
  const stream = new Blob([text]).stream()
    .pipeThrough(new CompressionStream("deflate-raw"));
  const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

async function inflate(b64url) {
  const s = atob(b64url.replaceAll("-", "+").replaceAll("_", "/"));
  const bytes = Uint8Array.from(s, (c) => c.charCodeAt(0));
  const stream = new Blob([bytes]).stream()
    .pipeThrough(new DecompressionStream("deflate-raw"));
  return new Response(stream).text();
}

async function saveState() {
  try {
    const slim = transcript.map((t) => ({
      ...t,
      chips: (t.chips || []).map((c) =>
        c.type === "image" ? { type: "image" } : { type: "text", text: c.text }),
    }));
    const state = { v: 1, session_id: sessionId, transcript: slim };
    const hash = "#c=" + (await deflate(JSON.stringify(state)));
    history.replaceState(null, "", location.pathname + hash);
  } catch { /* best-effort */ }
}

async function loadState() {
  if (!location.hash.startsWith("#c=")) return;
  try {
    const state = JSON.parse(await inflate(location.hash.slice(3)));
    sessionId = state.session_id || null;
    for (const turn of state.transcript || []) {
      transcript.push(turn);
      renderTurn(turn);
    }
    scrollLog();
  } catch { /* stale or foreign fragment; start fresh */ }
}

async function sendChat(message) {
  const sent = chips;
  chips = [];
  renderChips();

  const userTurn = { role: "user", text: message, chips: sent };
  transcript.push(userTurn);
  renderTurn(userTurn);

  const botEl = addMessage("assistant");
  let botText = "";
  scrollLog();
  $("chat-status").textContent = "thinking…";
  $("chat-send").disabled = true;

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        arxiv_id: arxivId,
        session_id: sessionId,
        message,
        context: sent.map(({ type, text, data, media_type }) =>
          type === "image" ? { type, data, media_type } : { type, text }),
      }),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        const event = JSON.parse(line);
        if (event.type === "session") sessionId = event.session_id;
        else if (event.type === "delta") {
          botText += event.text;
          renderMarkdown(botEl, botText);
          scrollLog();
        } else if (event.type === "error") {
          botText += "\n\n`[error] " + event.message + "`";
          renderMarkdown(botEl, botText);
        }
      }
    }
  } catch (e) {
    botText += "\n\n`[error] " + e + "`";
    renderMarkdown(botEl, botText);
  }
  transcript.push({ role: "assistant", text: botText });
  saveState();
  $("chat-status").textContent = "";
  $("chat-send").disabled = false;
  scrollLog();
}

$("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const message = $("chat-input").value.trim();
  if (!message) return;
  $("chat-input").value = "";
  sendChat(message);
});

$("chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("chat-form").requestSubmit();
  }
});

// ---- citations -----------------------------------------------------------

async function loadCitations() {
  const status = $("cite-status");
  status.textContent = "extracting…";
  try {
    const resp = await fetch("/api/citations/" + arxivId);
    const data = await resp.json();
    if (!resp.ok) {
      status.textContent = data.error && data.error.includes("no HTML")
        ? "needs the HTML version (none for this paper)"
        : "unavailable";
      return;
    }
    status.textContent = data.citations.length ? "" : "none found";
    const list = $("citations");
    for (const c of data.citations) {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = "/paper/" + c.id;
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = c.title || c.id;
      li.appendChild(a);
      if (c.title) {
        const meta = document.createElement("span");
        meta.className = "meta";
        meta.textContent = " " + c.id;
        li.appendChild(meta);
      }
      list.appendChild(li);
    }
  } catch (e) {
    status.textContent = "error: " + e;
  }
}

// ---- persona -------------------------------------------------------------

async function loadAccordion(url, field, id) {
  try {
    const data = await (await fetch(url)).json();
    if (!data[field]) return;
    renderMarkdown($(id + "-body"), data[field]);
    $(id).hidden = false;
  } catch (e) { /* pane stays hidden */ }
}

function loadPersona() {
  loadAccordion("/api/persona", "persona", "persona");
  loadAccordion("/api/system-prompt", "system_prompt", "system-prompt");
}

// ---- panes: resize + collapse --------------------------------------------

const store = {
  get(k) { try { return localStorage.getItem("reader." + k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem("reader." + k, v); } catch (e) {} },
};

function drag(bar, onMove, onDone) {
  bar.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    bar.setPointerCapture(e.pointerId);
    bar.classList.add("dragging");
    const move = (ev) => onMove(ev);
    const up = () => {
      bar.classList.remove("dragging");
      bar.removeEventListener("pointermove", move);
      onDone();
    };
    bar.addEventListener("pointermove", move);
    bar.addEventListener("pointerup", up, { once: true });
  });
}

function initSplitX() {
  const mainEl = document.querySelector(".reader-main");
  const size = (w) => { mainEl.style.gridTemplateColumns = `1fr 6px ${w}px`; };
  const saved = parseInt(store.get("sideW"), 10);
  let width = saved || 420;
  if (saved) size(saved);
  drag($("split-x"),
    (ev) => {
      width = Math.min(Math.max(window.innerWidth - ev.clientX, 260),
                       Math.round(window.innerWidth * 0.7));
      size(width);
    },
    () => store.set("sideW", width));
}

function initSplitY() {
  const pane = $("citations-pane");
  const size = (h) => {
    pane.style.height = h + "px";
    pane.style.maxHeight = "none";
    pane.style.flex = "none";
  };
  const saved = parseInt(store.get("citeH"), 10);
  let height = saved || 0;
  if (saved) size(saved);
  drag($("split-y"),
    (ev) => {
      const side = $("side").getBoundingClientRect();
      height = Math.min(Math.max(side.bottom - ev.clientY - 6, 60),
                        Math.round(side.height * 0.8));
      size(height);
    },
    () => store.set("citeH", height));
}

function initCollapse(paneId, key) {
  const pane = $(paneId);
  pane.classList.toggle("collapsed", store.get(key) === "1");
  pane.querySelector(".pane-head h2").addEventListener("click", () => {
    const collapsed = pane.classList.toggle("collapsed");
    store.set(key, collapsed ? "1" : "0");
  });
}

initSplitX();
initSplitY();
initCollapse("chat-pane", "chatCollapsed");
initCollapse("citations-pane", "citeCollapsed");

// ---- boot ----------------------------------------------------------------

setMode("pdf");
loadInfo();
renderPdf();
checkHtml();
loadCitations();
loadPersona();
loadState();
