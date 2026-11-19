// Built-in SYNTHETIC samples so the UI can be exercised without a backend.
// These are not real checks and not model output. Same shape as the contract.
const MOCK_SCENARIOS = {
  supported: { label: "Supported (2 sources)", status: "completed", verdict: "Supported",
    explanation: "The sample passages agree with the claim.", uncertainty_note: "Sample data only.",
    evidence: [
      { passage_id: "mock-1", rank: 1, url: "https://example.com/a", text: "Sample passage that agrees with the claim." },
      { passage_id: "mock-2", rank: 2, url: "https://example.com/b", text: "A second sample passage with the same conclusion." }] },
  refuted: { label: "Refuted (3 sources)", status: "completed", verdict: "Refuted",
    explanation: "The sample passages contradict the claim.", uncertainty_note: "Sample data only.",
    evidence: [1, 2, 3].map(i => ({ passage_id: "mock-r" + i, rank: i, url: "https://example.com/r" + i,
      text: "Sample passage " + i + " that contradicts the claim." })) },
  nei: { label: "Not Enough Evidence", status: "completed", verdict: "Not Enough Evidence",
    explanation: "The sample passages do not settle the claim.", uncertainty_note: "Additional sources would be needed.",
    evidence: [{ passage_id: "mock-n1", rank: 1, url: "https://example.com/n", text: "Related but inconclusive sample passage." }] },
  conflicting: { label: "Conflicting Evidence", status: "completed", verdict: "Conflicting Evidence",
    explanation: "One sample passage supports the claim and another contradicts it.", uncertainty_note: "The sources disagree; read both.",
    evidence: [
      { passage_id: "mock-c1", rank: 1, url: "https://example.com/c1", text: "Sample passage in favour." },
      { passage_id: "mock-c2", rank: 2, url: "https://example.com/c2", text: "Sample passage against." }] },
  insufficient: { label: "No sources found", status: "insufficient_sources", verdict: null,
    explanation: "No usable source passages were found for this claim.", uncertainty_note: null, evidence: [] },
  error: { label: "Service error", status: "error", verdict: null,
    explanation: "The service could not complete the check.", uncertainty_note: null, evidence: [] },
};

async function mockVerify(claim, scenarioKey) {
  await new Promise(r => setTimeout(r, 700)); // visible loading state
  return Object.assign({ mode: "mock", claim: claim }, MOCK_SCENARIOS[scenarioKey]);
}
