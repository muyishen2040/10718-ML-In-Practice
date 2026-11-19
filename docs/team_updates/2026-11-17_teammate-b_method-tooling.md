# Experiment update: Teammate B method tooling (pre-release preparation)

- Owner / workstream: Teammate B, local ML method ([brief](../tasks/TEAMMATE_B_METHOD.md)).
- Written date / run date: 2026-11-17. Only offline unit tests were run; no Drive data was run.
- Completion: **planned**. The code for brief steps 1–5 and the backend adapter is implemented
  and tested on synthetic fixtures. No experiment on AVeriTeC data has run yet.
- Git branch and commit: written in a local copy that has no `.git` directory. Before teammates
  check it out, commit it to `team/teammate-b-method` from `origin/main` and confirm that the
  branch exists on the remote. Record that commit here.
- Hypothesis and single change: none tested yet. This update adds the tools for the brief's
  bounded sequence: the mechanics check, released-fold fitting, the decisive-evidence pilot,
  the selector and frozen-verifier input intervention.
- Comparator: not applicable until runs exist. The planned comparators are BM25 top-three
  evidence fed to the same frozen raw reader (selector step) and fold-refit DistilBERT and
  `raw_sample_d01` configurations (step 2).

## Inputs and prediction-time availability

- Dataset version and split manifest: the release in `configs/team_evaluation.json` is still
  **pending**, so there is no manifest path or hash. Every new command refuses a pending release.
  The exception is `--provisional-all-train` on `sheet`/`audit`. It is marked provisional in its
  output, and rows outside the later fit IDs are dropped at audit/train time.
- Protocol ID / fold: none yet.
- Shared inputs that the planned runs will read (relative to `DATA_ROOT`): `processed/averitec/{train,dev}.jsonl`,
  `outputs/initial_ml_solution_answer_aligned_top100_v1/bm25/train_bm25_top100_v2/train_rankings.jsonl`,
  `outputs/evidence_aware_reader_v1/prepared/` and `outputs/evidence_reader_followup_v2/pair/raw_sample_d01/`.
- Evidence at prediction time: claim text, claim date and saved candidate passages. The selector
  features are BM25 rank/score, lexical/number overlap, source-type/date flags, and cached NLI
  scores used only when the cached text equals the candidate text.
- Privileged training annotations: annotated answers of training-fit claims are written only to
  `claim_context_annotators_only.jsonl` to guide human labelling. `url_matches_annotated_source`
  is a sheet column for annotators, never a selector feature or an automatic negative. Claim
  verdicts are not copied to passages.

## What was implemented

| Brief step | Code | Command |
| --- | --- | --- |
| 1 Mechanics check | `src/verification/reader_mechanics.py` | `run_evidence_reader.py mechanics --run RUN --output OUT [--data-root --prepared --rescore-limit N]` |
| 2 Released folds | `src/evaluation/team_release.py`, `pair_training.py`, `train_transformer_verifier.py` | `run_evidence_reader.py train ... --release-config configs/team_evaluation.json --fold-id F [--fold-epoch-policy fixed\|best]` |
| 3–4 Decisive pilot | `src/retrieval/decisive_evidence.py` | `run_evidence_selector.py sheet` / `audit` |
| 5 Selector + same verifier | `src/retrieval/evidence_selector.py`, `src/verification/local_reader.py` | `run_evidence_selector.py train` / `apply`, then `run_evidence_reader.py infer` |
| Backend adapter | `LocalReaderVerifier.verify` in `local_reader.py` | Called from A's backend; no separate HTTP service |

Design choices that teammates should review:

- **Proposed manifest schema.** The schema is in the `team_release.py` docstring:
  `protocol_id`, `source_claims_sha256`, and `folds[]` with `fold_id`,
  `fit_claim_ids`, `validation_claim_ids` and optional date ranges. The loader
  checks the pointer status, manifest SHA-256, released fold IDs, duplicate,
  unknown and overlapping IDs, dev/test IDs, and fit-before-validation dates.
  This schema still has to be agreed with the release owner and C.
- **Fold runs never refit on all of train and never score dev.** Their result is
  the fold's validation score. For the pair reader, `--fold-epoch-policy fixed`
  (the default) reports the last epoch with no selection on validation. It
  still records the per-epoch validation curve for brief step 6. `best` and
  multiple learning rates are flagged as optimistic in `model_selection.json`.
  The DistilBERT script keeps its best-epoch-per-trial behaviour, with the same
  caveat recorded.
- **Unchanged historical runs.** Without the release flags, the run config is
  byte-identical, so existing run directories still resume.
- **Decisive labels.** The labels are `direct_support`,
  `direct_contradiction`, `related_insufficient`, `ambiguous` and `unrelated`.
  Decisive labels need an exact quote from the passage. Only `reviewed` rows
  train the selector by default; `silver` needs `--allow-silver`. Ambiguous
  rows are excluded rather than treated as negatives.
- **Pilot sheet.** Each sampled claim gets 10 candidates: the BM25 top three
  (what users see), three answer-aligned annotation aids, and same-topic
  candidates spread over the remaining ranks. Claims are sampled with label
  balance so rare verdicts appear in a 20-claim pilot.
- **Selector metric scope.** Hit@3 and MRR are computed over *reviewed
  candidates only*, with claim-grouped cross-validation on fit claims. They are
  compared with BM25 order on the same candidates. This is not a full-pool
  retrieval metric and not URL-proxy MRR.
- **Adapter output.** The adapter returns at most three ranked evidence items
  and a plain-language uncertainty note with no percentages. Empty sources
  return `insufficient_sources` and a reader failure returns `error`, with
  `verdict: null` and never NEI. The contract version is labelled
  `proposed-v1`.

## Reproduction and artifacts

- Tests: `python -m pytest -q` → 120 passed, on CPU with Python 3.13, torch CPU and
  transformers 4.57.1. The new file is `tests/test_teammate_b_method.py`; its two torch tests
  are skipped without torch. With a tiny offline BERT they cover release-fold training (fixed
  epoch, validation curve, no refit, resume), the mechanics check with re-scoring (max
  probability difference < 1e-5), frozen inference on reordered passages, and adapter
  responses.
- Run outputs, Drive links, metrics, predictions and checkpoints: **none yet**.
- Suggested personal output root: `/content/drive/MyDrive/10718_runs/teammate_b/`.

Next commands, in order (Colab, after mounting Drive and checking out the published branch):

```bash
# Step 1 now (mechanics only; not an independent validation result)
python scripts/run_evidence_reader.py mechanics \
  --run "$DATA_ROOT/outputs/evidence_reader_followup_v2/pair/raw_sample_d01" \
  --data-root "$DATA_ROOT" --prepared "$DATA_ROOT/outputs/evidence_aware_reader_v1/prepared" \
  --rescore-limit 32 --device cuda --output "$OUT/mechanics_raw_sample_d01"

# Step 3 pilot sheet now (provisional until fit IDs are released)
python scripts/run_evidence_selector.py sheet --provisional-all-train \
  --claims "$DATA_ROOT/processed/averitec/train.jsonl" \
  --rankings "$DATA_ROOT/outputs/initial_ml_solution_answer_aligned_top100_v1/bm25/train_bm25_top100_v2/train_rankings.jsonl" \
  --claims-count 20 --per-claim 10 --output "$OUT/decisive_pilot_v1"
```

Confirm in `$OUT/mechanics_raw_sample_d01/mechanics_report.md` that the prepared directory and
`raw` mode match the run before relying on evidence-ID checks.

## Results

There are no ML results. The checks above are software tests on synthetic fixtures. They do
not support any claim about AVeriTeC macro-F1, FR/FA or retrieval quality.

## Interpretation and next action

- What this establishes: the tooling runs end to end offline and enforces the release, leakage
  and quote rules described above.
- Still unverified: behaviour on the real Drive artifacts. In particular, check that
  `raw_sample_d01` saved `dev/predictions.jsonl` and `final_model/*.safetensors` where
  `mechanics` expects them. It is also unverified whether the top-100 rankings carry `score`;
  if not, the selector uses the missing-score flag.
- Expansion: label the 20-claim pilot first. Expand to about 100 claims only if the labels are
  credible (agreement on a double-labelled subset, quotes found, few `ambiguous`).
- Requests:
  - Release owner / C: agree on or amend the manifest schema in `team_release.py`.
  - A: confirm that `LocalReaderVerifier.verify(claim, passages)` matches the backend boundary,
    and supply the shared mock fixtures.
  - C: optional assisted labelling of the pilot sheet. Proposed labels stay `silver` until
    reviewed.
