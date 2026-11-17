"use strict";
/* Claim Check user study page. Static: shows saved model outputs from study_claims.js (no model runs here).
   Flow: consent -> background -> instructions -> practice -> two blocks of 6 -> short survey -> save.
   Which block gets the tool, and which claim set, comes from ?g=1..4 (counterbalancing groups). */

const CONFIG = {
  // Web-app URL of the Google Apps Script receiver (docs/USER_STUDY_PILOT.md). Empty = results are only
  // offered as a file download at the end. TOKEN must equal the receiver's STUDY_TOKEN; it is visible in
  // the page source, so it only keeps casual junk out. It is not a secret.
  SUBMIT_URL: "",
  TOKEN: "",
  STUDY_VERSION: "pilot-0.1",
};

const ANSWERS = [
  ["Supported", "Supported", "The claim is true or backed up."],
  ["Refuted", "Refuted", "The claim is false or contradicted."],
  ["Not Enough Evidence", "Not enough evidence", "You cannot tell from what is shown."],
  ["Conflicting Evidence", "Conflicting evidence", "Some evidence supports it and some contradicts it."],
];
const VERDICT_CLASS = { "Supported": "Supported", "Refuted": "Refuted", "Not Enough Evidence": "NEI", "Conflicting Evidence": "Conflicting" };
const VERDICT_ICON = { "Supported": "✓", "Refuted": "✕", "Not Enough Evidence": "?", "Conflicting Evidence": "↔" };

const PRACTICE = {
  claim_id: "practice", claim: "The town of Brookfield banned bicycles on every street in 2019.",
  verdict: "Refuted", uncertainty_note: "The model leaned toward this label, but not strongly. This is a simple baseline model. On the benchmark it agreed with the reference label about half the time, so treat it as a rough lead only.",
  explanation: "A baseline classifier read the claim and the three passages shown and chose this label. It does not give reasons.",
  evidence: [
    { text: "Brookfield's council voted in 2019 to ban bicycles from the main shopping street during market hours only.", url: "https://example.com/brookfield-council" },
    { text: "Cyclists remain free to use all other streets in Brookfield, according to the council's transport office.", url: "https://example.com/brookfield-transport" },
    { text: "A local blog complained about the new cycling rules in the town centre.", url: "https://example.com/brookfield-blog" },
  ],
};

const app = document.getElementById("app");
const state = { pid: "", group: 1, trials: [], idx: -1, background: {}, post: {}, startedAt: new Date().toISOString(), shownAt: 0, srcOpens: [] };

/* ---------- helpers ---------- */
function el(tag, cls, text) { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; }
function render(...nodes) { app.replaceChildren(...nodes); window.scrollTo(0, 0); }
function seededRandom(seed) { // small deterministic PRNG so a participant's order is reproducible
  let h = 1779033703 ^ seed.length;
  for (let i = 0; i < seed.length; i++) { h = Math.imul(h ^ seed.charCodeAt(i), 3432918353); h = (h << 13) | (h >>> 19); }
  return () => { h = Math.imul(h ^ (h >>> 16), 2246822507); h = Math.imul(h ^ (h >>> 13), 3266489909); h ^= h >>> 16; return (h >>> 0) / 4294967296; };
}
function shuffled(arr, rnd) { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
function button(label, cls, onClick) { const b = el("button", cls, label); b.type = "button"; b.addEventListener("click", onClick); return b; }
function progress(done, total) { const p = el("div", "progress"); const s = el("span"); s.style.width = Math.round(100 * done / total) + "%"; p.append(s); return p; }
function domainOf(u) { try { return new URL(u).hostname.replace(/^www\./, ""); } catch { return "source"; } }

function radioGroup(name, legend, options, onChange) {
  const fs = el("fieldset"); fs.append(el("legend", null, legend));
  options.forEach(([value, title, desc]) => {
    const lab = el("label", "opt"); const input = document.createElement("input");
    input.type = "radio"; input.name = name; input.value = value; input.addEventListener("change", onChange);
    const box = el("div"); box.append(el("b", null, title)); if (desc) box.append(el("span", null, desc));
    lab.append(input, box); fs.append(lab);
  });
  return fs;
}
function scale(name, legend, lo, hi, onChange) {
  const fs = el("fieldset"); fs.append(el("legend", null, legend));
  const row = el("div", "scale");
  for (let v = 1; v <= 5; v++) {
    const lab = el("label"); const input = document.createElement("input");
    input.type = "radio"; input.name = name; input.value = String(v); input.addEventListener("change", onChange);
    lab.append(input, document.createTextNode(String(v))); row.append(lab);
  }
  const ends = el("div", "scale-ends"); ends.append(el("span", null, lo), el("span", null, hi));
  fs.append(row, ends); return fs;
}
function picked(form, name) { const c = form.querySelector(`input[name="${name}"]:checked`); return c ? c.value : null; }

/* ---------- the tool panel (mirrors the browser extension popup) ---------- */
function toolPanel(item, logOpens) {
  const wrap = el("div");
  const card = el("div", "card verdict-card " + (VERDICT_CLASS[item.verdict] || ""));
  card.append(el("div", "tag", "Saved model output (not a live check)"));
  const v = el("div", "verdict"); v.append(el("span", "icon", VERDICT_ICON[item.verdict] || "•"), document.createTextNode(item.verdict));
  card.append(v, el("div", null, item.explanation));
  if (item.uncertainty_note) card.append(el("p", "muted", item.uncertainty_note));
  card.append(el("p", "muted", "Preliminary screening only. Read the sources and decide for yourself."));
  wrap.append(card, el("h2", null, "Sources, most relevant first"));
  item.evidence.slice(0, 3).forEach((s, i) => {
    const d = el("details", "src"); const sum = el("summary"); const head = el("div", "src-head");
    head.append(el("span", "src-rank", String(i + 1)), el("span", "src-domain", domainOf(s.url)));
    sum.append(head, el("div", "preview", s.text));
    const body = el("div", "src-body"); body.append(el("blockquote", null, s.text), el("div", "url", s.url)); // shown as text, not a link: no searching
    d.append(sum, body);
    d.addEventListener("toggle", () => { if (d.open && logOpens) logOpens(i + 1); });
    wrap.append(d);
  });
  return wrap;
}

/* ---------- screens ---------- */
function screenConsent() {
  const params = new URLSearchParams(location.search);
  const g = parseInt(params.get("g"), 10);
  const fromLink = [1, 2, 3, 4].includes(g);
  state.group = fromLink ? g : 1 + Math.floor(Math.random() * 4);
  state.groupSource = fromLink ? "link" : "random";
  state.pid = (params.get("p") || "P" + Math.random().toString(36).slice(2, 8)).slice(0, 40);
  const nodes = [el("h1", null, "Claim Check study"),
    el("p", null, "You will judge short factual claims that appeared online. For some claims you will also see the output of a prototype checking tool. It takes about 20 minutes."),
    el("div", "card", "Draft consent text: confirm wording and approval requirements with the course staff before recruiting. "
      + "Taking part is voluntary and you may stop at any time. We record your answers, confidence ratings and timings. "
      + "We do not ask for your name, email or other identifying details. Results are used only for a course project and reported in aggregate.")];
  let code = null;
  if (!fromLink) { // opened without a personal link: ask for the code the study team sent, like P01-2 (participant P01, group 2)
    const fs = el("fieldset"); fs.append(el("legend", null, "Your study code (from the study team)"));
    code = document.createElement("input"); code.type = "text"; code.placeholder = "for example P01-2"; code.autocomplete = "off"; fs.append(code);
    nodes.push(fs);
  }
  const a = el("div", "actions");
  a.append(button("I agree, start", "primary", () => {
    if (code && code.value.trim()) {
      const m = code.value.trim().match(/^([A-Za-z0-9_]{1,20})-([1-4])$/);
      if (!m) { code.setCustomValidity("Use the form P01-2"); code.reportValidity(); return; }
      state.pid = m[1]; state.group = Number(m[2]); state.groupSource = "code";
    }
    screenBackground();
  }));
  nodes.push(a);
  render(...nodes);
}

function screenBackground() {
  const form = el("form");
  const check = () => go.disabled = !picked(form, "factcheck_freq");
  form.append(el("h1", null, "A quick question"),
    radioGroup("factcheck_freq", "How often do you check whether a claim you see online is true?",
      [["never", "Almost never"], ["rarely", "Rarely"], ["sometimes", "Sometimes"], ["often", "Often"]], check));
  const go = button("Continue", "primary", () => { state.background = { factcheck_freq: picked(form, "factcheck_freq") }; screenInstructions(); });
  go.disabled = true; form.append(go); render(form);
}

function screenInstructions() {
  render(el("h1", null, "How this works"),
    el("p", null, "You will see 12 claims, one at a time, in two blocks of six. For each one, choose the option that best fits and say how sure you are."),
    el("div", "card", "Important: please do not search the web or look anything up. Judge from what is shown on this page and what you already know. There are no trick questions, and it is fine to say you cannot tell."),
    el("p", null, "In one block you will see only the claim. In the other block you will also see the output of a checking tool (a verdict and three source passages). The tool can be wrong. Use it as you see fit."),
    el("p", "muted", "First, a practice claim that is made up, so you can see how the tool looks."),
    (() => { const a = el("div", "actions"); a.append(button("Start practice", "primary", () => runTrial(PRACTICE, { practice: true, assisted: true, number: 0 }, screenReady))); return a; })());
}

function screenReady() {
  state.trials = buildTrials();
  state.idx = -1;
  render(el("h1", null, "Practice done"), el("p", null, "Now the real claims. Remember: no searching."),
    (() => { const a = el("div", "actions"); a.append(button("Begin", "primary", nextTrial)); return a; })());
}

/* assignment: group 1/2 see set A with the tool, group 3/4 see set B with it; groups 1/3 do the tool block first */
function buildTrials() {
  const claims = STUDY_DATA.claims;
  const assistedSet = state.group <= 2 ? "A" : "B";
  const toolFirst = state.group % 2 === 1;
  const rnd = seededRandom(state.pid + "|" + state.group);
  const block = (set, assisted) => shuffled(claims.filter(c => c.set === set), rnd).map(c => ({ item: c, assisted }));
  const withTool = block(assistedSet, true), without = block(assistedSet === "A" ? "B" : "A", false);
  const order = toolFirst ? [withTool, without] : [without, withTool];
  const trials = []; order.forEach((b, bi) => b.forEach((t, i) => trials.push({ ...t, block: bi + 1, positionInBlock: i + 1 })));
  trials.forEach((t, i) => t.number = i + 1);
  return trials;
}

function nextTrial() {
  state.idx += 1;
  if (state.idx >= state.trials.length) return screenSurvey();
  const t = state.trials[state.idx];
  if (t.positionInBlock === 1 && state.idx > 0) return screenBlockBreak(t, () => showTrial(t));
  showTrial(t);
}
function screenBlockBreak(t, then) {
  render(el("h1", null, "Next block"),
    el("p", null, t.assisted ? "In this block you will also see the checking tool's output." : "In this block you will see only the claim, without the tool."),
    el("p", "muted", "Still no searching, please."),
    (() => { const a = el("div", "actions"); a.append(button("Continue", "primary", then)); return a; })());
}
function showTrial(t) { runTrial(t.item, t, nextTrial, t); }

/* One claim. `record` is the trial object that receives the answer (null for practice). */
function runTrial(item, meta, done, record) {
  state.srcOpens = [];
  const form = el("form");
  const total = 12;
  form.append(progress(meta.practice ? 0 : meta.number - 1, total),
    el("div", "muted", meta.practice ? "Practice (made-up claim)" : `Claim ${meta.number} of ${total}`),
    (() => { const c = el("div", "card"); c.append(el("div", "claim", item.claim)); return c; })());
  if (meta.assisted) form.append(toolPanel(item, n => state.srcOpens.push({ source: n, t_ms: Math.round(performance.now() - state.shownAt) })));
  const check = () => next.disabled = !(picked(form, "answer") && picked(form, "confidence"));
  form.append(radioGroup("answer", "What is your judgement of this claim?", ANSWERS, check),
    scale("confidence", "How sure are you?", "Not at all sure", "Very sure", check));
  const next = button(meta.number === 12 ? "Finish claims" : "Next", "primary", () => {
    if (record) {
      record.answer = picked(form, "answer"); record.confidence = Number(picked(form, "confidence"));
      record.time_ms = Math.round(performance.now() - state.shownAt); record.sources_opened = state.srcOpens.slice();
      record.answered_at = new Date().toISOString();
    }
    done();
  });
  next.disabled = true;
  const a = el("div", "actions"); a.append(next); form.append(a);
  render(form);
  state.shownAt = performance.now();
}

function screenSurvey() {
  const form = el("form");
  const names = ["helpful", "overrelied", "would_use"];
  const check = () => go.disabled = !names.every(n => picked(form, n));
  form.append(el("h1", null, "Last few questions"),
    el("p", "muted", "These are about the block where you saw the checking tool."),
    scale("helpful", "The tool's output helped me decide.", "Strongly disagree", "Strongly agree", check),
    scale("overrelied", "I may have relied on the tool's verdict more than I should have.", "Strongly disagree", "Strongly agree", check),
    scale("would_use", "I would use something like this before sharing a claim.", "Strongly disagree", "Strongly agree", check));
  const fs = el("fieldset"); fs.append(el("legend", null, "Anything confusing or missing? (optional)"));
  const ta = document.createElement("textarea"); ta.rows = 4; fs.append(ta); form.append(fs);
  const go = button("Submit", "primary", () => {
    state.post = { helpful: +picked(form, "helpful"), overrelied: +picked(form, "overrelied"), would_use: +picked(form, "would_use"), comments: ta.value.trim().slice(0, 2000) };
    finish();
  });
  go.disabled = true; const a = el("div", "actions"); a.append(go); form.append(a); render(form);
}

function resultPayload() {
  return {
    study_version: CONFIG.STUDY_VERSION, data_source: STUDY_DATA.provenance, participant_id: state.pid, group: state.group,
    group_source: state.groupSource, started_at: state.startedAt, finished_at: new Date().toISOString(),
    background: state.background, post: state.post,
    trials: state.trials.map(t => ({ number: t.number, block: t.block, claim_id: t.item.claim_id, set: t.item.set, condition: t.assisted ? "assisted" : "unassisted",
      model_verdict_shown: t.assisted ? t.item.verdict : null, answer: t.answer, confidence: t.confidence, time_ms: t.time_ms,
      sources_opened: t.sources_opened || [] })),
  };
}

/* POST to the receiver. Try a normal request so the reply can be read; if the browser blocks reading it,
   send once more without reading (the receiver ignores a repeat of the same submission). */
async function submit(payload) {
  const body = JSON.stringify({ ...payload, token: CONFIG.TOKEN });
  const opts = { method: "POST", headers: { "Content-Type": "text/plain" }, body };
  try {
    const reply = await (await fetch(CONFIG.SUBMIT_URL, opts)).json();
    return reply.ok ? "confirmed" : "rejected";
  } catch {
    try { await fetch(CONFIG.SUBMIT_URL, { ...opts, mode: "no-cors" }); return "unconfirmed"; } catch { return "failed"; }
  }
}

async function finish() {
  const payload = resultPayload();
  const json = JSON.stringify(payload, null, 1);
  const status = el("p", "muted", CONFIG.SUBMIT_URL ? "Sending your answers…" : "Your answers were not sent anywhere. Please download the file below and send it to the study team.");
  render(el("h1", null, "Thank you"), el("p", null, "That is the end of the study."), status);
  if (CONFIG.SUBMIT_URL) {
    const result = await submit(payload);
    const text = { confirmed: "Your answers were saved. You can also download a copy below.",
      unconfirmed: "Your answers were sent, but we could not confirm they arrived. Please also download the file below and send it to the study team.",
      rejected: "The server did not accept your answers. Please download the file below and send it to the study team.",
      failed: "Sending failed. Please download the file below and send it to the study team." }[result];
    status.textContent = text; status.className = result === "confirmed" ? "muted" : "err";
  }
  const a = el("div", "actions");
  a.append(button("Download my answers", "secondary", () => {
    const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = `claimcheck_${state.pid}.json`; link.click(); URL.revokeObjectURL(url);
  }));
  app.append(a);
  window.__studyResult = payload; // for testing
}

if (typeof STUDY_DATA === "undefined") render(el("p", "err", "Study data missing: study_claims.js did not load."));
else screenConsent();
