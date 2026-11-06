# Interface update: mock browser extension and fixture backend

- Owner / workstream: Teammate A, interface/extension
- Written date / run date: 2026-11-06 / 2026-11-06
- Completion: completed for the mock stage; real adapter not yet integrated
- Git branch and commit: `team/teammate-a-interface`, based on `origin/main`
  `0a3a858`; the work is the tip of that branch. Branch not yet pushed.
- Hypothesis and the single change being tested: none. This is a product
  milestone, not an ML experiment.
- Comparator and why it is a fair comparison: not applicable.

## Inputs and prediction-time availability

- Dataset/split/evaluation release: not used. `configs/team_evaluation.json`
  status was `pending`. No AVeriTeC data was read.
- Evidence available at prediction time: synthetic fixture passages only.
- Privileged annotations: none present.

## Reproduction and artifacts

- Startup steps: see [apps/SETUP.md](../../apps/SETUP.md).
- Backend contract version: INTEGRATION_CONTRACT as of `0a3a858`, plus an
  optional additive `provenance` field (needs team agreement).
- Modes: mock (in extension), benchmark replay (fixtures, labeled), live
  (returns `insufficient_sources`).
- Demo location: none recorded yet; no screen recording made.

## Results (tested flows, no ML scores)

- `python3 -m unittest discover -s apps/backend/tests`: 6 tests pass. They cover
  valid labels, the three-item limit and ordering, NEI vs insufficient vs error,
  live mode attaching no evidence, and malformed requests.
- Popup states exercised in a browser tab at 420 px width (served over HTTP,
  not as an installed extension): empty claim, mock (supported, refuted, NEI,
  conflicting, no sources, error), benchmark replay (refuted, no sources,
  simulated error), live unavailable. Source links render as http(s) only.
- Later checked by hand in Chrome as an unpacked extension: install, selected
  text on a news page, mock mode, benchmark replay with the backend running,
  and benchmark mode locking the claim box to the demo claim. Google Docs does
  not expose selected text; use Paste from clipboard there.
- Not tested: dark mode appearance, Paste from clipboard button, any real adapter.

## Interpretation and next action

- Establishes: the UI handles the contract's response shapes and states.
- Unverified: usability, and the unpacked-extension flow in Chrome.
- Dependency: a verifier adapter from B (local) or C (hosted) with saved
  benchmark predictions for real AVeriTeC claim IDs; a live retrieval path from C.
- Next: load unpacked in Chrome and record a short demo; agree the
  `provenance` field with B and C.
