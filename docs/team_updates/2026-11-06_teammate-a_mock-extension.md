# Interface update: mock browser extension and fixture backend

- Owner / workstream: Teammate A, interface/extension
- Written date / run date: 2026-11-06 / 2026-11-06
- Completion: mock stage completed; baseline adapter added for benchmark dev claims
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
- Later added `apps/backend/baseline_adapter.py`: runs the saved BM25-top-3
  DistilBERT baseline (Drive: `outputs/initial_ml_solution_rerun_bs16/transformer/bm25_top3/final_model`)
  on the 500 official dev claims, using the saved top-3 BM25 passages from
  `bm25/dev_bm25_top20_v2/dev_rankings.jsonl`. Inputs are built as in
  `scripts/train_transformer_verifier.py::compose_text`; gold labels are not read;
  claims without passages return `insufficient_sources`. Started with
  `--baseline-dir` and `--dev-rankings` (see SETUP.md). 13 backend tests pass,
  including one real-model test; one claim (`averitec:dev:00000`) was checked end
  to end in the popup (Refuted, three passages). The baseline's own dev result is
  macro-F1 0.347, accuracy 0.52 on the 500 dev claims, so single outputs are not
  evidence of quality; only a handful of claims were viewed here.
- Popup redesigned (segmented mode control, verdict banner, collapsible sources);
  checked in a 400 px browser tab in light and dark mode.
- Not tested: Paste from clipboard button, redesigned popup inside installed Chrome
  for all states, any live retrieval.

## Interpretation and next action

- Establishes: the UI handles the contract's response shapes and states.
- Unverified: usability, and the unpacked-extension flow in Chrome.
- Dependency: a verifier adapter from B (local) or C (hosted) with saved
  benchmark predictions for real AVeriTeC claim IDs; a live retrieval path from C.
- Next: load unpacked in Chrome and record a short demo; agree the
  `provenance` field with B and C.
