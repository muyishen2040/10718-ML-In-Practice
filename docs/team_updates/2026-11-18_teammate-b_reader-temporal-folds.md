# Experiment update: NLI pair reader on the released temporal folds

- Owner / workstream: Teammate B, local ML method (brief step 2).
- Written date / run date: 2026-11-18 (Colab, Tesla T4).
- Completion: **completed** for both released windows.
- Git branch and commit: `team/teammate-b-method` at `cdae1c9` (contains the release branch
  `feat/part1-temporal-evaluation`).
- Hypothesis and single change: on the same later claims and the same evidence, does the
  NLI-initialised pair reader make better Supported-versus-other trust decisions than DistilBERT?
  The model family changes. The release protocol, partitions, evidence, threshold grid and selection
  rule stay the same.
- Comparator: the release's temporal DistilBERT results (Part 1 revision table;
  [2026-11-17_part1-temporal-completed.md](2026-11-17_part1-temporal-completed.md)). The comparison is fair on
  these points:
  - same fold partitions;
  - same validation-only choice of the Supported cutoff τ, frozen before evaluation, with no refit;
  - identical BM25 top-three passages for every evaluation claim (checked below).

  It is unequal on two points:
  - DistilBERT searched four learning-rate/epoch settings per window, the reader only one;
  - the reader starts from an NLI-trained encoder (external pretraining, uncontrolled).

## Inputs and prediction-time availability

- Split manifest: `processed/averitec/team_splits/trust_temporal_20261117_v1/split_manifest.json`,
  SHA-256 `af2d5436167ed39f658255c099b41acab16cdd1ecc89e25e42a95bfd0af20d5e`.
  The loader verified this hash, plus the source train hash and partition checks, before training.
- Protocol `trust_temporal_20261117_v1`, folds `spring_2020` and `summer_2020`:

  | Fold | Fit | Validation | Evaluation |
  | --- | --- | --- | --- |
  | spring_2020 | 1,411 through 2019-11-08 | 355, 2019-11-09 to 2020-02-29 | 575, 2020-03-01 to 2020-06-03 |
  | summer_2020 | 1,870 through 2020-03-15 | 468, 2020-03-16 to 2020-06-03 | 611, 2020-06-04 to 2020-08-25 |

- Shared inputs (relative to `DATA_ROOT`):
  - `processed/averitec/train.jsonl`;
  - prepared evidence `outputs/evidence_aware_reader_v1/prepared/`, mode `raw`.
- Evidence at prediction time: the original BM25 top three passages per claim. For all 575 spring and
  611 summer evaluation claims, the prepared `raw` top-three passage IDs are identical to those in
  `outputs/bm25/train_bm25_top20_v2/train_rankings.jsonl`, the rankings the release's DistilBERT used.
- Labels: class weights and training use only fit labels. Validation labels choose the cutoff.
  Evaluation labels are joined only for scoring. The NLI distillation targets are cached scores from
  the frozen NLI model and use no verdicts. Official dev is not used.

## Reproduction and artifacts

```python
!python scripts/run_evidence_reader.py train --prepared "{PREPARED}" --data-root "{DATA_ROOT}" --mode raw \
    --learning-rates 3e-5 --epochs 3 --batch-size 2 --accumulation 8 --distillation-weight 0.1 \
    --release-config configs/team_evaluation.json --fold-id spring_2020 --device cuda \
    --output "{OUT}/reader_fold_spring_2020_v1"
# then --fold-id summer_2020 with --output "{OUT}/reader_fold_summer_2020_v1"
!python scripts/score_trust_predictions.py --data-root "{DATA_ROOT}" --manifest "{MANIFEST}" --fold FOLD \
    --predictions "{RUN}/evaluation_predictions.jsonl" --threshold TAU --output "{RUN}/team_score.json"
```

- Configuration: the `raw_sample_d01` settings.
  - Encoder `cross-encoder/nli-deberta-v3-small` at the revision recorded in `preparation.json`.
  - Max length 384, claim budget 96.
  - Learning rate 3e-5 (single trial), 3 epochs, last epoch kept.
  - Batch 2 × accumulation 8, seed 718, fp16.
  - Class-weighted `sample_mean` loss, NLI distillation 0.1, word embeddings frozen.
- Output folders (`OUT = MyDrive/10718_runs/teammate_b`), with teammate access:
  - `reader_fold_spring_2020_v1/`:
    https://drive.google.com/drive/folders/1d1EcsysFEvqw-I0itd8Dh7ooVG56UIOe?usp=drive_link
  - `reader_fold_summer_2020_v1/`:
    https://drive.google.com/drive/folders/1CZr_qMhU02Rc11pl7YlO58bzVo2PwtCy?usp=drive_link
- Files in each folder:
  - `reader_run.json`: run configuration and release binding.
  - `trials/lr_3e-05/summary.json`: validation curve and threshold curve.
  - `trials/lr_3e-05/validation_predictions.jsonl` and `trials/lr_3e-05/model/`: weights used for evaluation.
  - `selection_before_evaluation.json`: the frozen choice.
  - `evaluation_predictions.jsonl`, `evaluation_complete.json`, `team_score.json`: results from the team scorer.
- Runtime: training took about 3.7 minutes (spring) and 4.7 minutes (summer) on the T4, plus
  inference. There were no interruptions. Peak memory was not measured. Re-running the same command
  reuses the completed trial and evaluation.

## Results

Evaluation partitions only. The choices were frozen on validation beforehand.

- **FR** (false reassurance): a non-Supported claim predicted Supported, out of other claims.
- **FA** (false alarm): a Supported claim predicted otherwise, out of Supported claims.
- Spring has 97 Supported and 478 other claims; summer has 172 Supported and 439 other.

| Window | Model | Selected τ | Balanced accuracy, argmax → τ | FR, argmax → τ | FA, argmax → τ |
| --- | --- | --- | --- | --- | --- |
| Spring | DistilBERT (release) | LR 2e-5, 2 epochs; 0.20 | 0.5827 → 0.6874 | 54 → 151 | 70 → 30 |
| Spring | NLI pair reader | LR 3e-5, 3 epochs; 0.25 | 0.6722 → **0.6984** | 67 → **106** | 50 → 37 |
| Summer | DistilBERT (release) | LR 2e-5, 3 epochs; 0.15 | 0.6465 → **0.6628** | 73 → **194** | 93 → 40 |
| Summer | NLI pair reader | LR 3e-5, 3 epochs; 0.15 | 0.6282 → 0.6391 | 84 → 225 | 95 → 36 |

Validation (used for the cutoff choice only):

| Fold | Chosen policy: balanced accuracy, FR, FA | Argmax balanced accuracy by epoch 1/2/3 | Macro-F1 by epoch 1/2/3 |
| --- | --- | --- | --- |
| Spring (82 / 273) | τ 0.25: 0.6681, 88/273, 28/82 | 0.577 / 0.618 / 0.643 | 0.290 / 0.323 / 0.348 |
| Summer (83 / 385) | τ 0.15: 0.7108, 167/385, 12/83 | 0.679 / 0.689 / 0.656 | 0.341 / 0.336 / 0.341 |

Secondary four-class diagnostics at the selected cutoff (in `team_score.json`):

- Spring: macro-F1 0.354, accuracy 0.617. NEI F1 0.09, Conflicting F1 0.11. ECE 0.265.
- Summer: macro-F1 0.282, accuracy 0.529. It **never predicted NEI or Conflicting**. ECE 0.194.

No paired interval was computed against DistilBERT. That would need a paired bootstrap on the release
run's `evaluation_predictions.jsonl`.

## Interpretation and next action

- What this establishes:
  - The pair reader runs end to end under the shared temporal protocol, with selection frozen before
    evaluation.
  - On identical evidence the comparison is mixed:
    - Spring: the reader is 1.1 points higher and makes 45 fewer false reassurances at its cutoff,
      with 7 more false alarms.
    - Summer: it is 2.4 points lower and makes 31 more false reassurances, with 4 fewer false alarms.
  - At argmax, the reader is clearly higher in spring (+9.0 points) and lower in summer (−1.8).
- What it does not establish:
  - That either model is better. The differences are small relative to 97 and 172 Supported claims,
    there is no paired interval, and the two windows share training claims.
  - Anything about official dev or test.
- Failure cases:
  - The selected cutoffs again trade false alarms for false reassurance. In summer, false reassurance
    rises from 84 to 225 of 439 (19% → 51%), so balanced accuracy alone should not set the cutoff,
    as the report already notes.
  - In summer, validation balanced accuracy peaked at epoch 2, and the model predicted no NEI or
    Conflicting claims.
- Expansion: not justified yet. Cheap next checks:
  1. a paired bootstrap against DistilBERT's saved evaluation predictions;
  2. a second learning rate (1e-5), to match DistilBERT's search more fairly.

  The larger bet remains the decisive-evidence selector (brief steps 3–5), with this run as the
  BM25-evidence baseline.
- Requests:
  - Release owner: the path of the temporal DistilBERT run's `evaluation_predictions.jsonl`, for the
    paired comparison.
  - Reviewers: whether to add these rows to `TEAM_PROGRESS.md`.
