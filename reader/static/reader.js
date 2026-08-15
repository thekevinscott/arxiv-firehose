"use strict";

import * as pdfjs from "/vendor/pdf.min.mjs";
pdfjs.GlobalWorkerOptions.workerSrc = "/vendor/pdf.worker.min.mjs";

const $ = (id) => document.getElementById(id);
const arxivId = decodeURIComponent(location.pathname.replace(/^\/paper\//, ""));

let mode = "pdf"; // "pdf" | "html"
let htmlLoaded = false;
let htmlAvailable = null; // null = unknown yet
let sessionId = null;
let chips = []; // {type:"text",text} | {type:"image",data,media_type,dataUrl}

// ---- paper info ----------------------------------------------------------

async function loadInfo() {
  try {
    const resp = await fetch("/api/paper/" + arxivId);
    const meta = await resp.json();
    if (!resp.ok) throw new Error(meta.error || resp.status);
    document.title = meta.title + " · arxiv reader";
    $("paper-title").textContent = meta.title;
    $("paper-title").title = meta.title;
    $("info-authors").textContent = (meta.authors || []).join(", ");
    $("info-cats").textContent =
      `${arxivId} · ${(meta.categories || []).join(" ")} · ${(meta.published || "").slice(0, 10)}`;
    $("info-abstract").textContent = meta.abstract;
    $("abs-link").href = meta.abs_url;
    return meta;
  } catch (e) {
    $("paper-title").textContent = arxivId;
    $("info-abstract").textContent = "metadata unavailable: " + e.message;
    return null;
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
    $("toggle-view").disabled = true;
    $("toggle-view").title = "This paper has no arxiv HTML version";
  }
}

function setMode(next) {
  mode = next;
  $("pdf-pages").hidden = mode !== "pdf";
  $("html-frame").hidden = mode !== "html";
  $("toggle-view").textContent = mode === "pdf" ? "HTML" : "PDF";
  $("select-region").hidden = mode !== "pdf";
  if (mode === "html") loadHtml();
}

$("toggle-view").addEventListener("click", () => {
  setMode(mode === "pdf" ? "html" : "pdf");
});

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

async function sendChat(message) {
  const sent = chips;
  chips = [];
  renderChips();

  const userEl = addMessage("user");
  for (const chip of sent) {
    if (chip.type === "image") {
      const img = document.createElement("img");
      img.src = chip.dataUrl;
      img.className = "msg-img";
      userEl.appendChild(img);
    } else {
      const q = document.createElement("blockquote");
      q.textContent = chip.text;
      userEl.appendChild(q);
    }
  }
  const p = document.createElement("p");
  p.textContent = message;
  userEl.appendChild(p);

  const botEl = addMessage("assistant");
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
          botEl.textContent += event.text;
          scrollLog();
        } else if (event.type === "error") {
          botEl.textContent += "\n[error] " + event.message;
        }
      }
    }
  } catch (e) {
    botEl.textContent += "\n[error] " + e;
  }
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

// ---- boot ----------------------------------------------------------------

setMode("pdf");
loadInfo();
renderPdf();
checkHtml();
loadCitations();
