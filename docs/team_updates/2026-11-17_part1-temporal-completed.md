# Part 1 revision: temporal evaluation and trust-decision metrics

- Owner / workstream: Mu-Yi, evaluation and report revisions.
- Completed November 17, 2026, after resuming the earlier interrupted run.
- Code: `feat/part1-temporal-evaluation`, training release commit `f453171`.
- Source snapshot SHA-256:
  `06a8d8edaabebe2de940d56855664248eba4c2838f13af05aea7bf84436c675b`.
- Scope: recalculate the historical table with a task-aligned binary metric,
  compare supervised models across two chronological windows, and test
  validation-selected decision thresholds. No new retrieval run was required.

## Inputs, partitions and availability

Shared inputs are in `v2_source_documents` under the
[shared Drive root](https://drive.google.com/drive/folders/19tuNxxB1y9arSmLnrQcxRuPvDQBf6YeK).
Claims: `processed/averitec/train.jsonl`. Evidence:
`outputs/bm25/train_bm25_top20_v2/train_rankings.jsonl`, truncated to the first
three passages per claim. Claim labels and annotated answers are excluded from
prediction text. Models, vocabulary and class weights are fitted on fit IDs only.

Protocol: `trust_temporal_20261117_v1`. Shared manifest:
`processed/averitec/team_splits/trust_temporal_20261117_v1/split_manifest.json`.
SHA-256: `af2d5436167ed39f658255c099b41acab16cdd1ecc89e25e42a95bfd0af20d5e`.

| Window | Fit claims / last date | Validation claims / dates | Future evaluation claims / dates |
| --- | --- | --- | --- |
| Spring | 1,411 / Nov 8, 2019 | 355 / Nov 9, 2019–Feb 29, 2020 | 575 / Mar 1–Jun 3, 2020 |
| Summer | 1,870 / Mar 15, 2020 | 468 / Mar 16–Jun 3, 2020 | 611 / Jun 4–Aug 25, 2020 |

The manifest excludes 105 missing/unparseable dates and quarantines nine
pre-2000 source date values without guessing corrections. Earlier duplicate
claim text is purged across partition boundaries: one fit record in spring and
four earlier records in summer. Date ties stay together. Duplicates within a
partition remain claim records. Official dev/test are not reassigned.

## Metric, explored settings and selection

Primary metric: Supported-versus-other **binary balanced accuracy**, the mean
of Supported acceptance recall and other-claim withholding recall. “Other”
combines Refuted, Not Enough Evidence and Conflicting Evidence. Both constant
decisions score 0.5. Conditional FR divides by other claims; conditional FA
divides by Supported claims. Four-class macro-F1 remains a secondary diagnostic.

- Logistic regression: word TF-IDF unigrams/bigrams, sublinear TF, balanced
  class weights, max 2,000 optimization iterations; `C={0.1,1,10}`.
- DistilBERT: `distilbert-base-uncased`, revision
  `12040accade4e8a0f71eabdb258fecc2e7e948be`; learning rates `{2e-5,5e-5}`
  × epochs `{2,3}`; class-balanced loss, batch 8, gradient accumulation 2,
  weight decay 0.01 and warmup ratio 0.06. Token budgets: 96 claim tokens,
  110 per passage, 512 total. Seed 718; fp16 on a T4.
- Each trial compares four-way argmax with 17 Supported-score thresholds,
  0.10–0.90 in steps of 0.05. A threshold accepts a claim as Supported when
  its score meets the cutoff; otherwise the highest-scoring other label is used.
- Within a family/window, highest validation balanced accuracy wins. Ties prefer
  lower conditional FR, lower conditional FA, then argmax on an exact tie.
  Selection is saved before future prediction. No refit on validation is used.

All six logistic-regression fits and eight transformer fits completed. The
argmax comparator below uses the same selected checkpoint as its threshold row;
it is not a separately tuned argmax model.

| Window | Model | Selected trial | Supported threshold | Validation BA |
| --- | --- | --- | ---: | ---: |
| Spring | Logistic regression | C=10 | 0.25 | 0.6638 |
| Spring | DistilBERT | LR=2e-5, 2 epochs | 0.20 | 0.6485 |
| Summer | Logistic regression | C=1 | 0.20 | 0.6414 |
| Summer | DistilBERT | LR=2e-5, 3 epochs | 0.15 | 0.6940 |

## Future-window results

Each spring row uses 97 Supported / 478 other claims. Each summer row uses
172 Supported / 439 other claims. FR and FA below are counts.

| Window | Model | Policy | Balanced accuracy | Binary accuracy | FR | FA |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Spring | Logistic regression | Argmax | 0.5841 | 0.8209 | 28 | 75 |
| Spring | Logistic regression | Selected threshold | 0.6229 | 0.7009 | 124 | 48 |
| Spring | DistilBERT | Argmax | 0.5827 | 0.7843 | 54 | 70 |
| Spring | DistilBERT | Selected threshold | 0.6874 | 0.6852 | 151 | 30 |
| Summer | Logistic regression | Argmax | 0.6212 | 0.7250 | 62 | 106 |
| Summer | Logistic regression | Selected threshold | 0.6245 | 0.5188 | 271 | 23 |
| Summer | DistilBERT | Argmax | 0.6465 | 0.7283 | 73 | 93 |
| Summer | DistilBERT | Selected threshold | 0.6628 | 0.6170 | 194 | 40 |

Thresholding increases balanced accuracy for each selected checkpoint, while
reducing false alarms and increasing false reassurance. The summer logistic
model's BA rises by only 0.0032, but FR rises from 62/439 (14.1%) to 271/439
(61.7%). Equal weighting of the two class recalls is not a false-reassurance
safety constraint. These policies should not be described as safer or deployed
automatically. The windows use overlapping historical training data and should
be reported separately. No statistical-significance or seed-robustness claim is
made from this single-seed comparison.

## Historical metric revision

All 11 original report rows were recalculated on the same 500 official dev
claims, with 122 Supported and 378 other. No dev threshold tuning was done.
Selected examples are below; controls and the full table are in the artifacts.

| Existing model / evidence | Binary balanced accuracy | FR / 378 | FA / 122 |
| --- | ---: | ---: | ---: |
| Qwen3-4B / BM25 top three | 0.7356 | 110 | 29 |
| TF-IDF + logistic regression / BM25 top three | 0.5996 | 61 | 78 |
| DistilBERT / BM25 top three | 0.5824 | 74 | 78 |
| BM25-trained DistilBERT / BGE dev passages | 0.6015 | 72 | 74 |
| BM25-trained DistilBERT / fused dev passages | 0.6111 | 74 | 71 |

Qwen leads these historical retrieved-evidence rows on the revised metric,
because it accepts more genuinely Supported claims. It also makes more false
reassurances. This is retrospective rescoring, not independent model selection.
The BGE/fusion rows retain the original BM25-trained classifier.

## Artifacts, checks and reproduction

Output directory:
`/content/drive/MyDrive/10718_runs/muyi/report_part1_20261117_v1`.
This is a personal output location, not automatically shared through the input
folder; teammates need access to read full artifacts. The shared manifest and
this Git summary are available through the existing handoff.

- `historical_metrics.{json,csv}`: all 11 original report rows under the new metric.
- `temporal_results.{json,csv}`: four model/window runs, each with two policies.
- `temporal/WINDOW/FAMILY/evaluation_complete.json`: full per-class metrics,
  confusion matrices, calibration and both decision policies.
- The same directory has evaluation predictions, the validation threshold
  curve, `selection_before_evaluation.json`, and all trial configurations,
  validation predictions and saved models.
- `verification_report.json`: independent coverage, gold-label, selection and
  score checks. All four evaluations matched scikit-learn balanced accuracy.
- `experiment_config.json`, `protocol_release.json`, `split_manifest.json` and
  `source_snapshot.zip`: bound inputs, dependency versions and exact source.
- `part1_revision_results.zip`: small verified result/config/prediction bundle;
  excludes model weights. Downloaded to ignored `local/part1_revision_results/`.
  Readable detailed summary: ignored `local/PART1_REVISION_RESULTS.md`.

The resumed training workflow took approximately 11 minutes, including model
downloads and Drive writes. Peak memory and experiment-only compute cost were
not measured. The experiment T4 was terminated after downloading and verifying
results; the unrelated `mk64-multitrack` runtime was left untouched.
The code test directory passed 110 tests.

Reproduction uses `scripts/run_part1_revision.py` stages `prepare`, `historical`,
`train`, `summarize`; see [EVALUATION.md](../EVALUATION.md). Completed trials are
reused from Drive. For exact resumption, restore `source_snapshot.zip` and the
versions in `experiment_config.json`; changing bytes/settings requires a new
run directory. The saved epoch checkpoints for unfinished trials are local to a
runtime, but all completed trial weights and results persist in Drive.

## Implication for the report and next work

Part 1 can now report the revised binary metric, an explicit threshold policy,
the hyperparameter/threshold search and two chronological evaluations. URL-proxy
Hit@3 remains a secondary retrieval diagnostic; it does not sort verdict models.
The primary metric gives equal class weight. Its relation to a natural-prevalence
human task score must be stated explicitly, and a randomized field evaluation
must test whether better model decisions improve participant decisions.

The immediate remaining decision is how strongly the product should penalize
incorrect endorsements. A stricter safety constraint or different error costs
must be chosen before another evaluation, rather than selecting a cutoff from
these future-window scores. No large new retrieval/model architecture experiment
is required to document the completed Part 1 revisions.
