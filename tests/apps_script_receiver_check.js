// Runs apps/study/apps_script/Code.gs against a small in-memory fake of Google's services.
// Used by tests/test_study_results.py (skipped if node is missing). Exits non-zero on failure.
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const code = fs.readFileSync(path.join(__dirname, "..", "apps", "study", "apps_script", "Code.gs"), "utf8");

function makeSheet() {
  const rows = [];
  return { rows,
    getLastRow: () => rows.length, appendRow: r => rows.push(r.slice()),
    getDataRange: () => ({ getValues: () => rows.map(r => r.slice()) }),
    getRange: (row, col, n) => ({ setValues: vals => { vals.forEach((v, i) => { rows[row - 1 + i] = v.slice(); }); } }) };
}
function makeEnv(token) {
  const sheets = {};
  return { sheets, ctx: vm.createContext({
    JSON, Date, String, Math,
    SpreadsheetApp: { getActiveSpreadsheet: () => ({ getSheetByName: n => sheets[n] || null, insertSheet: n => (sheets[n] = makeSheet()) }) },
    PropertiesService: { getScriptProperties: () => ({ getProperty: () => token }) },
    LockService: { getScriptLock: () => ({ waitLock() {}, releaseLock() {} }) },
    ContentService: { MimeType: { JSON: "json" }, createTextOutput: t => ({ text: t, setMimeType() { return this; } }) },
  }) };
}
function participant(overrides = {}) {
  const answers = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence"];
  const trials = Array.from({ length: 12 }, (_, i) => ({ number: i + 1, block: i < 6 ? 1 : 2, claim_id: `averitec:dev:${i}`, set: i % 2 ? "A" : "B",
    condition: i < 6 ? "assisted" : "unassisted", model_verdict_shown: i < 6 ? "Refuted" : null, answer: answers[i % 4], confidence: 1 + (i % 5),
    time_ms: 4000 + i, sources_opened: i < 3 ? [{ source: 1, t_ms: 10 }] : [] }));
  return { token: "secret", study_version: "pilot-0.1", participant_id: "P01", group: 1, group_source: "code", started_at: "s", finished_at: "f",
    background: { factcheck_freq: "often" }, post: { helpful: 4, overrelied: 2, would_use: 3, comments: "ok" }, trials, ...overrides };
}
function post(env, payload) { return JSON.parse(vm.runInContext("doPost", env.ctx)({ postData: { contents: typeof payload === "string" ? payload : JSON.stringify(payload) } }).text); }

vm.runInContext(code, makeEnv("secret").ctx); // parses

// 1. a valid submission writes 12 trial rows, one participant row and one raw row without the token
const env = makeEnv("secret");
vm.runInContext(code, env.ctx);
let r = post(env, participant());
assert.deepStrictEqual(r, { ok: true, stored_trials: 12 });
assert.strictEqual(env.sheets.trials.rows.length, 13);               // header + 12
assert.strictEqual(env.sheets.participants.rows.length, 2);
assert.strictEqual(env.sheets.raw.rows.length, 2);
assert.ok(!env.sheets.raw.rows[1][3].includes("secret"), "token must not be stored");
assert.strictEqual(JSON.parse(env.sheets.raw.rows[1][3]).trials.length, 12);
assert.strictEqual(env.sheets.trials.rows[1][8], "Refuted");        // model_verdict_shown
assert.strictEqual(env.sheets.trials.rows[7][8], "");               // blank when unassisted
assert.strictEqual(env.sheets.trials.rows[1][12], 1);               // sources_opened count

// 2. the same submission twice is stored once
r = post(env, participant());
assert.deepStrictEqual(r, { ok: true, duplicate: true });
assert.strictEqual(env.sheets.trials.rows.length, 13);

// 3. a different person is stored
r = post(env, participant({ participant_id: "P02", finished_at: "g" }));
assert.strictEqual(r.ok, true);
assert.strictEqual(env.sheets.participants.rows.length, 3);

// 4. rejects bad token, bad answer, bad id, bad confidence, junk, oversize
const bad = [
  [participant({ token: "nope" }), "bad token"],
  [participant({ participant_id: "x y" }), "bad participant_id"],
  [participant({ group: 9 }), "bad group"],
  [Object.assign(participant(), { trials: participant().trials.map((t, i) => (i === 2 ? { ...t, answer: "Maybe" } : t)) }), "bad answer in trial 3"],
  [Object.assign(participant(), { trials: participant().trials.map((t, i) => (i === 0 ? { ...t, confidence: 7 } : t)) }), "bad confidence in trial 1"],
  ["not json at all", "not json"],
];
for (const [payload, error] of bad) {
  const e2 = makeEnv("secret"); vm.runInContext(code, e2.ctx);
  assert.deepStrictEqual(post(e2, payload), { ok: false, error });
  assert.ok(!e2.sheets.trials, "nothing may be written for: " + error);
}
const e3 = makeEnv("secret"); vm.runInContext(code, e3.ctx);
assert.strictEqual(post(e3, "x".repeat(100001)).error, "too large");

// 5. no token configured on the server: refuse everything
const e4 = makeEnv(null); vm.runInContext(code, e4.ctx);
assert.strictEqual(post(e4, participant()).error, "server has no STUDY_TOKEN set");

// 6. doGet reports counts
assert.deepStrictEqual(JSON.parse(vm.runInContext("doGet", env.ctx)().text), { ok: true, participants: 2, trials: 24 });
console.log("receiver ok");
