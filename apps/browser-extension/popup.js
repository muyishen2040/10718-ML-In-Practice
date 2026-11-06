"use strict";

const $ = id => document.getElementById(id);
const els = { search: $("search"), mode: $("mode"), hint: $("modeHint"), scenarioRow: $("scenarioRow"), scenario: $("scenario"),
  claim: $("claim"), check: $("check"), useSel: $("useSelection"), result: $("result"), backend: $("backend") };

const HINTS = {
  mock: "Built-in samples for trying the interface. Not a real check.",
  benchmark: "Known benchmark claims only, served by the local backend. Not a check of arbitrary text.",
  live: "Checking new claims is not available yet; you will see an explanation instead of a result.",
};
const MODE_BADGE = { mock: "Mock sample", benchmark: "Benchmark replay (saved)", live: "Live" };
// The backend says where a benchmark result came from; label it truthfully.
const PROVENANCE_BADGE = {
  fixture_replay: "Benchmark replay (synthetic sample)",
  baseline_distilbert_bm25_top3: "Baseline model, benchmark claim (not a live check)",
};
const VERDICT_CLASS = { "Supported": "Supported", "Refuted": "Refuted",
  "Not Enough Evidence": "NEI", "Conflicting Evidence": "Conflicting" };
const VERDICT_ICON = { "Supported": "✓", "Refuted": "✕", "Not Enough Evidence": "?", "Conflicting Evidence": "↔" };
const MAX_SOURCES = 3;

function domainOf(href) { try { return new URL(href).hostname.replace(/^www\./, ""); } catch { return href; } }

let benchmarkClaims = []; // [{claim_id, claim}]

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text; // never innerHTML: claim/source text is untrusted
  return n;
}

function safeHttpUrl(u) {
  try { const p = new URL(u); return (p.protocol === "https:" || p.protocol === "http:") ? p.href : null; }
  catch { return null; }
}

function show(node) { els.result.replaceChildren(node); }

function renderLoading() {
  const c = el("div", "card");
  c.append(el("span", "spinner"), document.createTextNode("Checking…"));
  show(c);
}

function renderMessage(kind, title, body, badge) {
  const c = el("div", "card msg-" + kind);
  if (badge) c.append(el("div", "badge", badge));
  c.append(el("div", "title", title), el("p", "note", body));
  show(c);
}

function renderResponse(r) {
  const badge = PROVENANCE_BADGE[r.provenance] || MODE_BADGE[r.mode] || "";
  if (r.status === "insufficient_sources") {
    return renderMessage("warn", "Not enough sources to check this",
      (r.explanation || "") + (r.uncertainty_note ? " " + r.uncertainty_note : ""), badge);
  }
  if (r.status !== "completed" || !r.verdict) {
    return renderMessage("error", "Something went wrong", r.explanation || "The check could not be completed.", badge);
  }
  const wrap = el("div");
  const card = el("div", "card verdict-card " + (VERDICT_CLASS[r.verdict] || ""));
  card.append(el("div", "badge", badge));
  // The claim box already shows the claim in benchmark mode; repeat it only if it differs.
  if (r.claim && r.claim !== els.claim.value.trim()) card.append(el("p", "claimline", "Claim checked: " + r.claim));
  const v = el("div", "verdict");
  v.append(el("span", "icon", VERDICT_ICON[r.verdict] || "•"), document.createTextNode(r.verdict));
  card.append(v, el("div", null, r.explanation || ""));
  if (r.uncertainty_note) card.append(el("p", "note", r.uncertainty_note));
  card.append(el("p", "note", "Preliminary screening only. Read the sources and decide for yourself."));
  wrap.append(card);

  const sources = (r.evidence || []).slice().sort((a, b) => a.rank - b.rank).slice(0, MAX_SOURCES);
  if (sources.length) {
    wrap.append(el("h2", null, "Sources, most relevant first"));
    const list = el("ol");
    sources.forEach((s, i) => {
      const href = safeHttpUrl(s.url);
      const li = el("li");
      const d = el("details", "src");
      const sum = el("summary");
      const head = el("div", "src-head");
      head.append(el("span", "src-rank", String(i + 1)), el("span", "src-domain", href ? domainOf(href) : "Source link unavailable"),
        el("span", "src-chev", "›"));
      sum.append(head, el("div", "preview", s.text));
      const body = el("div", "src-body");
      body.append(el("blockquote", null, s.text));
      if (href) {
        const a = el("a", null, href);
        a.href = href; a.target = "_blank"; a.rel = "noopener noreferrer";
        body.append(a);
      }
      d.append(sum, body);
      li.append(d);
      list.append(li);
    });
    wrap.append(list);
  }
  show(wrap);
}

async function postVerify(body) {
  const base = els.backend.value.replace(/\/+$/, "");
  let resp;
  try {
    resp = await fetch(base + "/verify", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) });
  } catch {
    return { mode: body.mode, status: "error", verdict: null, evidence: [],
      explanation: "Could not reach the backend at " + base + ". Is it running?" };
  }
  try { return await resp.json(); }
  catch { return { mode: body.mode, status: "error", verdict: null, evidence: [], explanation: "The backend sent an unreadable reply." }; }
}

async function runCheck() {
  const mode = els.mode.value;
  const claim = els.claim.value.trim();
  if (!claim) return renderMessage("warn", "Enter a claim first", "Select text on a page or type one above.");
  els.check.disabled = true;
  renderLoading();
  try {
    let r;
    if (mode === "mock") r = await mockVerify(claim, els.scenario.value);
    else if (mode === "live") r = await postVerify({ mode: "live", claim, page_url: null });
    else {
      const picked = benchmarkClaims.find(c => c.claim_id === els.scenario.value);
      r = picked ? await postVerify({ mode: "benchmark", claim_id: picked.claim_id })
        : { mode, status: "error", verdict: null, evidence: [], explanation: "Choose a demo claim." };
    }
    renderResponse(r);
    els.result.scrollIntoView({ block: "start", behavior: "smooth" }); // popup is short: bring the verdict into view
  } finally { els.check.disabled = false; }
}

async function loadBenchmarkClaims() {
  els.scenario.replaceChildren();
  let reachable = true;
  try {
    const base = els.backend.value.replace(/\/+$/, "");
    const q = encodeURIComponent(els.search.value.trim());
    const data = await (await fetch(base + "/benchmark/claims?limit=50&q=" + q)).json();
    benchmarkClaims = data.claims || [];
  } catch { benchmarkClaims = []; reachable = false; }
  if (!reachable) {
    renderMessage("error", "Backend not reachable", "Start it with: python3 apps/backend/server.py", MODE_BADGE.benchmark);
    return;
  }
  if (!benchmarkClaims.length) {
    els.claim.value = "";
    renderMessage("warn", "No demo claims match", "Try a different search.");
    return;
  }
  benchmarkClaims.forEach(c => els.scenario.append(new Option(c.claim_id + " · " + c.claim.slice(0, 60), c.claim_id)));
  els.claim.value = benchmarkClaims[0].claim;
  els.result.replaceChildren();
}

function applyMode() {
  const mode = els.mode.value;
  els.hint.textContent = HINTS[mode];
  els.scenarioRow.hidden = mode === "live";
  els.search.hidden = mode !== "benchmark";
  els.claim.readOnly = mode === "benchmark"; // benchmark replays a fixed demo claim only
  els.claim.placeholder = mode === "benchmark" ? "Pick a demo claim above"
    : "Select text on a page, or type or paste a claim";
  $("scenarioLabel").textContent = mode === "mock" ? "Sample outcome" : "Demo claim";
  els.result.replaceChildren();
  if (mode === "mock") {
    els.scenario.replaceChildren(...Object.entries(MOCK_SCENARIOS).map(([k, v]) => new Option(v.label, k)));
  } else if (mode === "benchmark") {
    loadBenchmarkClaims();
  }
}

const NO_SELECTION_HELP = "Nothing could be read from the page. Some sites (for example Google Docs) " +
  "don't expose selected text to extensions. Copy the text, then use Paste from clipboard.";

function benchmarkLocked() {
  if (els.mode.value !== "benchmark") return false;
  renderMessage("warn", "Demo claims only", "Benchmark demo replays saved results for the listed demo claims. Switch to Mock or Live to use your own text.");
  return true;
}

async function useSelection(quiet) {
  if (els.mode.value === "benchmark") { if (!quiet) benchmarkLocked(); return; }
  if (typeof chrome === "undefined" || !chrome.tabs || !chrome.scripting) return;
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const results = await chrome.scripting.executeScript({ target: { tabId: tab.id, allFrames: true },
      func: () => String(window.getSelection()) });
    const text = results.map(r => (r.result || "").trim()).sort((a, b) => b.length - a.length)[0] || "";
    if (text) els.claim.value = text.slice(0, 2000);
    else if (!quiet) renderMessage("warn", "No text selected", NO_SELECTION_HELP);
  } catch {
    if (!quiet) renderMessage("warn", "Can't read this page", "Browser pages and the web store block this. Paste the claim instead.");
  }
}

async function pasteClipboard() {
  if (benchmarkLocked()) return;
  try {
    const text = (await navigator.clipboard.readText()).trim();
    if (text) { els.claim.value = text.slice(0, 2000); els.result.replaceChildren(); }
    else renderMessage("warn", "Clipboard is empty", "Copy the claim text first.");
  } catch {
    renderMessage("warn", "Clipboard not available", "Click in the Claim box and paste with Cmd/Ctrl+V.");
  }
}

document.querySelectorAll(".seg button").forEach(b => b.addEventListener("click", () => {
  if (els.mode.value === b.dataset.mode) return;
  els.mode.value = b.dataset.mode;
  document.querySelectorAll(".seg button").forEach(x => x.setAttribute("aria-checked", String(x === b)));
  applyMode();
}));
let searchTimer;
els.search.addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadBenchmarkClaims, 250); });
els.scenario.addEventListener("change", () => {
  if (els.mode.value !== "benchmark") return;
  const c = benchmarkClaims.find(x => x.claim_id === els.scenario.value);
  if (c) els.claim.value = c.claim;
});
els.check.addEventListener("click", runCheck);
els.useSel.addEventListener("click", () => useSelection(false));
$("pasteClip").addEventListener("click", pasteClipboard);
applyMode();
useSelection(true); // quietly prefill if text is already selected
