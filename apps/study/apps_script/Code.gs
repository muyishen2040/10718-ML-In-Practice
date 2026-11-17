/**
 * Receiver for the Claim Check study page. Paste into a Google Apps Script project that is
 * bound to a Google Sheet (Extensions > Apps Script), set the script property STUDY_TOKEN,
 * and deploy as a web app. Steps: docs/USER_STUDY_PILOT.md, "Receiving data in a Google Sheet".
 *
 * Each submission writes three things:
 *   trials        one row per judgement (easy to filter and chart)
 *   participants  one row per person: group, survey answers, comment
 *   raw           one row per person with the full JSON, which scripts/analyze_study_results.py reads
 * Names, emails and IP addresses are never collected or stored.
 */

var SHEET_TRIALS = "trials";
var SHEET_PARTICIPANTS = "participants";
var SHEET_RAW = "raw";
var MAX_BODY_CHARS = 100000;
var ANSWERS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence"];

var TRIAL_HEADERS = ["server_time", "participant_id", "group", "number", "block", "claim_id", "set", "condition",
  "model_verdict_shown", "answer", "confidence", "time_ms", "sources_opened", "study_version"];
var PARTICIPANT_HEADERS = ["server_time", "participant_id", "group", "group_source", "started_at", "finished_at",
  "factcheck_freq", "helpful", "overrelied", "would_use", "comments", "study_version"];
var RAW_HEADERS = ["server_time", "participant_id", "finished_at", "json"];

/** Returns an error string, or "" when the submission is acceptable. Pure, so it can be tested. */
function validatePayload_(payload, expectedToken) {
  if (!expectedToken) return "server has no STUDY_TOKEN set";
  if (!payload || payload.token !== expectedToken) return "bad token";
  if (typeof payload.participant_id !== "string" || !/^[A-Za-z0-9_]{1,40}$/.test(payload.participant_id)) return "bad participant_id";
  if ([1, 2, 3, 4].indexOf(payload.group) < 0) return "bad group";
  if (!payload.trials || payload.trials.length < 1 || payload.trials.length > 20) return "bad trials";
  for (var i = 0; i < payload.trials.length; i++) {
    var t = payload.trials[i];
    if (ANSWERS.indexOf(t.answer) < 0) return "bad answer in trial " + (i + 1);
    if (typeof t.confidence !== "number" || t.confidence < 1 || t.confidence > 5) return "bad confidence in trial " + (i + 1);
    if (typeof t.claim_id !== "string") return "bad claim_id in trial " + (i + 1);
  }
  return "";
}

function trialRows_(payload, serverTime) {
  return payload.trials.map(function (t) {
    return [serverTime, payload.participant_id, payload.group, t.number, t.block, t.claim_id, t.set, t.condition,
      t.model_verdict_shown || "", t.answer, t.confidence, t.time_ms, (t.sources_opened || []).length, payload.study_version || ""];
  });
}

function participantRow_(payload, serverTime) {
  var bg = payload.background || {};
  var post = payload.post || {};
  return [serverTime, payload.participant_id, payload.group, payload.group_source || "", payload.started_at || "",
    payload.finished_at || "", bg.factcheck_freq || "", post.helpful || "", post.overrelied || "", post.would_use || "",
    String(post.comments || "").slice(0, 2000), payload.study_version || ""];
}

/** The stored copy never contains the token. */
function rawRow_(payload, serverTime) {
  var copy = JSON.parse(JSON.stringify(payload));
  delete copy.token;
  return [serverTime, payload.participant_id, payload.finished_at || "", JSON.stringify(copy)];
}

function sheet_(name, headers) {
  var book = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = book.getSheetByName(name) || book.insertSheet(name);
  if (sheet.getLastRow() === 0) sheet.appendRow(headers);
  return sheet;
}

function alreadyStored_(sheet, participantId, finishedAt) {
  var rows = sheet.getDataRange().getValues();
  for (var i = 1; i < rows.length; i++) {
    if (rows[i][1] === participantId && String(rows[i][2]) === String(finishedAt || "")) return true;
  }
  return false;
}

function reply_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function doPost(e) {
  var body = (e && e.postData && e.postData.contents) || "";
  if (body.length > MAX_BODY_CHARS) return reply_({ ok: false, error: "too large" });
  var payload;
  try { payload = JSON.parse(body); } catch (err) { return reply_({ ok: false, error: "not json" }); }
  var token = PropertiesService.getScriptProperties().getProperty("STUDY_TOKEN");
  var problem = validatePayload_(payload, token);
  if (problem) return reply_({ ok: false, error: problem });

  var lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    var raw = sheet_(SHEET_RAW, RAW_HEADERS);
    if (alreadyStored_(raw, payload.participant_id, payload.finished_at)) return reply_({ ok: true, duplicate: true });
    var now = new Date().toISOString();
    var trials = sheet_(SHEET_TRIALS, TRIAL_HEADERS);
    var rows = trialRows_(payload, now);
    trials.getRange(trials.getLastRow() + 1, 1, rows.length, TRIAL_HEADERS.length).setValues(rows);
    sheet_(SHEET_PARTICIPANTS, PARTICIPANT_HEADERS).appendRow(participantRow_(payload, now));
    raw.appendRow(rawRow_(payload, now));
    return reply_({ ok: true, stored_trials: rows.length });
  } finally {
    lock.releaseLock();
  }
}

/** Opening the web app URL in a browser shows counts, so you can confirm the deployment works. */
function doGet() {
  var book = SpreadsheetApp.getActiveSpreadsheet();
  var count = function (name) { var s = book.getSheetByName(name); return s ? Math.max(0, s.getLastRow() - 1) : 0; };
  return reply_({ ok: true, participants: count(SHEET_PARTICIPANTS), trials: count(SHEET_TRIALS) });
}
