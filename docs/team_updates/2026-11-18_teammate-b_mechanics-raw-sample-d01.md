# Experiment update: mechanics check of the `raw_sample_d01` pair reader

- Owner / workstream: Teammate B, local ML method (brief step 1).
- Written date / run date: 2026-11-18 (Colab, Tesla T4 15 GB).
- Completion: **completed** (mechanics check only).
- Git branch and commit: `team/teammate-b-method` at `0ceac8e`.
- Hypothesis and single change: no model change. The question was whether the saved
  `raw_sample_d01` checkpoint, its predictions and its recorded configuration are internally
  consistent and reproduce the recorded reference score.
- Comparator: the recorded reference in [TEAM_PROGRESS.md](../TEAM_PROGRESS.md) and the
  [B brief](../tasks/TEAMMATE_B_METHOD.md): macro-F1 0.3604, 49 false reassurances, 68 false alarms.

## Inputs and prediction-time availability

- Split manifest / protocol / fold: none. The shared release is still pending. This run uses
  official dev and the historical grouped train holdout, and it is **not** an independent
  validation result.
- Inputs (relative to `DATA_ROOT = 10718_averitec/v2_source_documents`):
  - Saved run: `outputs/evidence_reader_followup_v2/pair/raw_sample_d01/`
  - Prepared evidence: `outputs/evidence_aware_reader_v1/prepared/`, mode `raw`. The run is bound
    to this `preparation.json` by hash.
  - Claims: `processed/averitec/dev.jsonl` (500 official dev claims).
- Weights: `final_model/model.safetensors`, SHA-256
  `ef316e8d5900951114ab24ad3de620c3287abec3735a8731408436886803148d`.
- Evidence at prediction time: the exact original BM25 top three per claim (`raw` mode). One
  dev claim has no retrieved passage; the reader sees its "No evidence available" placeholder.
- Gold labels: joined only after inference for scoring.

## Reproduction and artifacts

```python
!python scripts/run_evidence_reader.py mechanics --run "{RUN}" --data-root "{DATA_ROOT}" \
    --prepared "{PREPARED}" --rescore-limit 32 --device cuda --output "{MECH}"
```

- Output: `/content/drive/MyDrive/10718_runs/teammate_b/mechanics_raw_sample_d01_v1/`
  (`mechanics_report.json`, `mechanics_report.md`). Stable Drive link with teammate access:
  https://drive.google.com/drive/folders/1FJ_-Tu_Fgwa14e_0p72HPg3qVm-xx8_W?usp=drive_link.
- Runtime: a few minutes on a T4 (not timed). Peak memory: not measured.
- Dependencies: `requirements/reader-colab.txt` on Colab's torch `2.14.1+cu130`, transformers `4.57.1`.
  pip reported `huggingface-hub` conflicts with Colab's preinstalled gradio/diffusers. Those
  packages are unused here.
- Outputs of the first attempt were overwritten by this re-run at the same path. That attempt
  differed only by a false FAIL from a check bug, which `0ceac8e` fixes.

## Recorded configuration (read from the run, not re-chosen)

| Item | Value |
| --- | --- |
| Encoder | `cross-encoder/nli-deberta-v3-small` @ `fa2804872c3b4bd748f38c0185cc85775361e735` |
| Verdict label order | Supported, Refuted, Not Enough Evidence, Conflicting Evidence |
| NLI label order | contradiction, entailment, neutral (model and preparation agree) |
| Text lengths | max length 384, claim budget 96; tokenizer `DebertaV2Tokenizer` |
| Training | lr 3e-5 (single trial), 3 epochs, batch 2 × accumulation 8, seed 718, fp16 autocast |
| Loss | class-weighted `sample_mean` [0.904, 0.440, 2.719, 3.932]; NLI distillation weight 0.1; word embeddings frozen |
| Selection | grouped train holdout, 2,764 fit / 303 holdout claims (disjoint). Best epoch 3.0 of 3; holdout macro-F1 0.3910 |
| Refit | all 3,067 train claims for 3 epochs (576 steps), last checkpoint at epoch 3.0 |
| Saved weights | 117 tensors, all float32 |

## Results

The population is all 500 official dev claims, which have been inspected repeatedly and are
exploratory. Rates use all 500 claims as the denominator.

| Metric | Value |
| --- | --- |
| Macro-F1 | 0.3604 (recomputed from predictions equals the saved value exactly) |
| Accuracy | 0.564 |
| False reassurances (non-Supported predicted Supported) | 49 / 500 |
| False alarms (Supported predicted otherwise) | 68 / 500 |
| Predicted label counts | Refuted 293, Supported 103, NEI 70, Conflicting 34 |

Consistency checks: all 18 pass. The checks cover:

- unique IDs and exact coverage of the 500 dev claims, with correct gold-label joins;
- normalised probabilities, with each predicted label equal to the argmax;
- evidence IDs equal to the prepared top three for every claim.

Re-running frozen inference on the first 32 dev claims agreed with all 32 saved verdicts. The
largest probability difference was 0.00035, which is consistent with half-precision GPU
arithmetic. Per-class precision/recall/F1, the confusion matrix and calibration were not part of
this check. They are in the run's own `dev/metrics.json`.

## Interpretation and next action

- What this establishes:
  - The saved checkpoint, predictions and recorded configuration are internally consistent.
  - They reproduce the reference 0.3604 / 49 / 68.
  - The label orders and evidence joins are correct.
- What it does not establish:
  - Any new validation result. The model saw all train claims, including any future
    temporal-fold validation claims, so it **cannot** serve as a fold baseline (brief step 2).
  - A selection result: only one learning rate was tried, and official dev is exploratory.
- Observations to carry forward (hypotheses, not findings):
  - The best epoch equalled the last epoch, so the holdout curve may not have plateaued. Check
    this on the released-fold validation curve before training longer (step 6).
  - Predictions lean heavily toward Refuted (293 of 500). Compare this with the dev gold
    distribution when analysing failures.
  - The holdout score (0.391 on 303 claims) exceeds the dev score (0.360).
- Next action: label the 20-claim decisive-evidence pilot
  (`10718_runs/teammate_b/decisive_pilot_v1/decisive_sheet.csv`). Once the release is ready,
  refit `raw_sample_d01`'s configuration and DistilBERT on each released fold with `--fold-epoch-policy fixed`.
- Requests: release owner, publish the split (or comment on the proposed manifest schema in
  `src/evaluation/team_release.py`).
