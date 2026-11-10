"use strict";

const $ = id => document.getElementById(id);
const els = { search: $("search"), mode: $("mode"), hint: $("modeHint"), scenarioRow: $("scenarioRow"), scenario: $("scenario"),
  claim: $("claim"), check: $("check"), useSel: $("useSelection"), result: $("result"), backend: $("backend"),
  model: $("model"), modelRow: $("modelRow"), modelHint: $("modelHint") };

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
let models = [], preferredModel = "distilbert_bm25", revision = 0, catalogRequest = 0, modelRequest = 0;

async function saveSettings() {
  const settings = { mode: els.mode.value, backend: els.backend.value, model: preferredModel };
  try {
    if (typeof chrome !== "undefined" && chrome.storage?.local) await chrome.storage.local.set({ claimCheckSettings: settings });
    else localStorage.setItem("claimCheckSettings", JSON.stringify(settings));
  } catch { /* Preferences are optional; verification still works. */ }
}

function clearResult() {
  revision++;
  els.result.replaceChildren();
  els.check.disabled = els.mode.value === "benchmark" && !models.some(m => m.id === els.model.value && m.available);
}

function describeModel() {
  const picked = models.find(m => m.id === els.model.value);
  els.modelHint.textContent = picked ? picked.description : "No model available.";
}

async function loadModels() {
  const current = ++modelRequest, base = els.backend.value.replace(/\/+$/, "");
  models = [];
  els.model.disabled = true;
  els.model.replaceChildren(new Option("Loading models…", ""));
  els.check.disabled = true;
  try {
    const response = await fetch(base + "/models");
    if (!response.ok) throw new Error("No model catalog");
    const data = await response.json();
    if (current !== modelRequest || els.mode.value !== 'benchmark') return;
    if (!Array.isArray(data.models) || !data.models.length) throw new Error("Empty model catalog");
    models = data.models;
    els.model.replaceChildren(...models.map(m => {
      const option = new Option(m.label + (m.available ? "" : " (unavailable)"), m.id);
      option.disabled = !m.available;
      option.title = m.unavailable_reason || m.description;
      return option;
    }));
    const chosen = models.find(m => m.id === preferredModel && m.available)
      || models.find(m => m.id === data.default_model_id && m.available);
    if (!chosen) throw new Error("No model loaded");
    els.model.value = chosen.id;
    preferredModel = chosen.id;
    els.model.disabled = false;
    els.check.disabled = false;
    describeModel();
    saveSettings();
  } catch {
    if (current !== modelRequest || els.mode.value !== 'benchmark') return;
    models = [];
    els.model.replaceChildren(new Option("Backend unavailable", ""));
    els.modelHint.textContent = "Start or update the backend, then reopen Benchmark.";
  }
}

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
  const badge = r.mode === "benchmark" && r.model_label && r.provenance !== "fixture_replay"
    ? r.model_label + " · benchmark claim" : PROVENANCE_BADGE[r.provenance] || MODE_BADGE[r.mode] || "";
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
  const current = ++revision;
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
      r = picked ? await postVerify({ mode: "benchmark", claim_id: picked.claim_id, model_id: els.model.value })
        : { mode, status: "error", verdict: null, evidence: [], explanation: "Choose a demo claim." };
    }
    if (current !== revision) return;
    renderResponse(r);
    els.result.scrollIntoView({ block: "start", behavior: "smooth" }); // popup is short: bring the verdict into view
  } finally { if (current === revision) els.check.disabled = false; }
}

async function loadBenchmarkClaims() {
  if (els.mode.value !== "benchmark") return;
  const current = ++catalogRequest;
  clearResult();
  els.claim.value = "";
  els.check.disabled = true;
  els.scenario.replaceChildren();
  let reachable = true;
  try {
    const base = els.backend.value.replace(/\/+$/, "");
    const q = encodeURIComponent(els.search.value.trim());
    const data = await (await fetch(base + "/benchmark/claims?limit=50&q=" + q)).json();
    if (current !== catalogRequest || els.mode.value !== 'benchmark') return;
    benchmarkClaims = data.claims || [];
  } catch {
    if (current !== catalogRequest || els.mode.value !== 'benchmark') return;
    benchmarkClaims = []; reachable = false;
  }
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
  clearResult();
}

function applyMode() {
  modelRequest++;
  catalogRequest++;
  const mode = els.mode.value;
  els.hint.textContent = HINTS[mode];
  els.scenarioRow.hidden = mode === "live";
  els.search.hidden = mode !== "benchmark";
  els.claim.readOnly = mode === "benchmark"; // benchmark replays a fixed demo claim only
  els.claim.placeholder = mode === "benchmark" ? "Pick a demo claim above"
    : "Select text on a page, or type or paste a claim";
  $("scenarioLabel").textContent = mode === "mock" ? "Sample outcome" : "Demo claim";
  clearResult();
  els.modelRow.hidden = mode !== "benchmark";
  document.querySelectorAll(".seg button").forEach(b => b.setAttribute("aria-checked", String(b.dataset.mode === mode)));
  if (mode === "mock") {
    els.scenario.replaceChildren(...Object.entries(MOCK_SCENARIOS).map(([k, v]) => new Option(v.label, k)));
  } else if (mode === "benchmark") {
    loadModels();
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
  saveSettings();
}));
let searchTimer;
els.search.addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadBenchmarkClaims, 250); });
els.scenario.addEventListener("change", () => {
  clearResult();
  if (els.mode.value !== "benchmark") return;
  const c = benchmarkClaims.find(x => x.claim_id === els.scenario.value);
  if (c) els.claim.value = c.claim;
});
els.model.addEventListener("change", () => {
  preferredModel = els.model.value;
  clearResult();
  describeModel();
  saveSettings();
});
els.backend.addEventListener("change", () => { applyMode(); saveSettings(); });
els.claim.addEventListener("input", clearResult);
els.check.addEventListener("click", runCheck);
els.useSel.addEventListener("click", () => useSelection(false));
$("pasteClip").addEventListener("click", pasteClipboard);
async function initialize() {
  try {
    const saved = typeof chrome !== "undefined" && chrome.storage?.local
      ? (await chrome.storage.local.get("claimCheckSettings")).claimCheckSettings
      : JSON.parse(localStorage.getItem("claimCheckSettings") || "null");
    if (saved) {
      if (HINTS[saved.mode]) els.mode.value = saved.mode;
      if (typeof saved.backend === "string") els.backend.value = saved.backend;
      if (typeof saved.model === "string") preferredModel = saved.model;
    }
  } catch { /* Start with defaults if preferences cannot be read. */ }
  applyMode();
  useSelection(true);
}
initialize();
