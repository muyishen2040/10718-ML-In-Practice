# Experiment update: descriptive name

Use a new filename such as `YYYY-MM-DD_teammate-a_extension-demo.md` or
`YYYY-MM-DD_teammate-c_api-pilot.md`.
Write enough context that another agent can understand and reproduce the work
without a conversation transcript. Replace every placeholder before review.

- Owner / workstream:
- Written date / run date:
- Completion: planned, running, interrupted, or completed:
- Git branch and commit, or portable-notebook source hash:
- Hypothesis and the single change being tested:
- Comparator and why it is a fair comparison:

## Inputs and prediction-time availability

- Dataset version and split-manifest path/hash:
- Shared evaluation protocol ID, release revision and fold ID:
- Training, validation and evaluation claim counts and dates:
- Shared input paths relative to `DATA_ROOT`, and hashes:
- Evidence available at prediction time:
- Training-only labels or privileged annotations, and how they are excluded
  from prediction features:

## Reproduction and artifacts

- Exact notebook/script and parameters, seeds, dependencies, CPU/GPU:
- Run output path and stable Drive folder link with teammate access:
- Metrics JSON, predictions JSONL, configuration, selection record and checkpoint:
- Runtime, peak memory if measured, and resume instructions:
- If interrupted: completed work, missing work and the last usable checkpoint:
- For API runs: provider/model version, prompt hash, token use, cost, latency,
  failed/retried calls and whether external search was enabled:
- For interface work: backend contract version, mock/benchmark/live mode,
  startup steps and demo location:

## Results

Give macro-F1, accuracy, per-class precision/recall/F1, FR/FA counts with their
denominators, and relevant calibration/paired intervals. For retrieval give the
relevance definition, judged population, Hit@3 and explicitly scoped MRR.
Keep holdout, official dev, controls and post-selection diagnostics separate.
For interface-only updates, report tested user flows, loading/error/uncertainty
states, evidence links and integration checks instead of inventing ML scores.

## Interpretation and next action

- What the results establish:
- What remains a hypothesis or unverified:
- Failure cases or subsets where the comparator is better:
- Whether the result justifies expanding the experiment:
- Dependency or concrete request for another workstream:
