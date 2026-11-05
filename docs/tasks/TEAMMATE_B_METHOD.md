# Teammate B: improve the local ML method

## Project and your outcome

Improve the evidence selection and local verdict model for a misinformation
screening assistant. Given a claim and at most three selected source passages,
predict Supported, Refuted, Not Enough Evidence or Conflicting Evidence. Our
goal is a reproducible gain on later claims with fewer false reassurances, while
keeping the user able to inspect the sources.

- Repository: <https://github.com/muyishen2040/10718-ML-In-Practice>
- Access, submitted report and artifact map: [TEAM_HANDOFF.md](../TEAM_HANDOFF.md).
- Experiments and current limitations: [TEAM_PROGRESS.md](../TEAM_PROGRESS.md).
- Shared split release: [EVALUATION.md](../EVALUATION.md).
- Shared service boundary: [INTEGRATION_CONTRACT.md](../INTEGRATION_CONTRACT.md).

The time-based training split release is tracked in
[team_evaluation.json](../../configs/team_evaluation.json).
Do not create a different split for the team's new comparison. Existing notebooks
do not automatically consume this pointer. Inspect/reproduce prior mechanics
while it is pending; use the ready manifest before new fitting or model selection.

## Current evidence and starting point

The submitted BM25 + DistilBERT result is macro-F1 0.3469 on 500 official dev
claims, with 74 false reassurances and 78 false alarms. A later corrected
NLI-initialized raw-passage reader, `raw_sample_d01`, achieved 0.3604 with 49 and
68 respectively. This is a useful research reference, not the recorded train-only
winner of that later study. Claim-only DistilBERT scored 0.3970; annotated answers
scored 0.5284 but are unavailable as new-claim prediction inputs.

QA packing looked strongest on the 303-claim train holdout, yet its same-checkpoint
dev score fell from 0.4314 to 0.3364, then to 0.3250 after full-train refitting.
That supports checking temporal robustness and refit stability. It does not
prove retrieval is the sole bottleneck or that extracted answers are decisive.

## Ownership and reusable artifacts

Suggested branch: `team/teammate-b-method`, from the remotely verified shared
base in the handoff. Main areas: `src/retrieval/`, local-reader files in
`src/verification/`, relevant training scripts, notebooks and method tests.
Agree on changes to common split/scoring interfaces before implementation.

Start with these files rather than choosing a model from the full script list:

- [Reader CLI](../../scripts/run_evidence_reader.py): preparation, training,
  frozen-model inference and result export.
- [Reader architecture and loss](../../src/verification/pair_reader.py) and
  [training/selection loop](../../src/verification/pair_training.py).
- [Evidence preparation](../../src/retrieval/evidence_pipeline.py) and
  [validated claim/evidence joins](../../src/data/evidence.py).
- [Latest reference notebook](../../notebooks/evidence_reader_followup_colab.ipynb):
  historical configurations and stage order; its old holdout is not the pending
  temporal release. Inspect it before intentionally reproducing a run.

After the shared Drive shortcut is mounted, use
`DATA_ROOT=/content/drive/MyDrive/10718_averitec/v2_source_documents`:

| Input relative to DATA_ROOT | Purpose |
| --- | --- |
| `processed/averitec/train.jsonl` | 3,067 usable train claims, labels and annotations |
| `outputs/bm25/train_bm25_top20_v2/train_rankings.jsonl` | Existing train evidence |
| `outputs/initial_ml_solution_answer_aligned_top100_v1/bm25/train_bm25_top100_v2/train_rankings.jsonl` | Larger cached training candidate pool |
| `outputs/evidence_aware_reader_v1/prepared/` | Cached raw/clean/NLI-selected passages and NLI scores |
| `outputs/evidence_reader_followup_v2/pair/raw_sample_d01/` | Corrected raw-reader configuration, selection and checkpoints |

See the handoff for matching dev paths. New run outputs belong in a personal,
unique Drive directory. The full 22.85-GB dev corpus and three large train archives
usually need not be scanned again when training from saved rankings.

## First bounded experiment sequence

1. Reproduce saved-checkpoint inference and inspect label order, tokenization,
   evidence IDs, precision, best epoch and refit schedule. Preserve this as a
   mechanics check, not a new independent validation result.
2. Once the shared release is ready, fit the original DistilBERT and
   corrected raw reader on each released fit subset and score its held-out IDs.
   An old model trained on all 3,067 claims already saw the new validation claims
   and cannot be reused as an independent fold baseline.
3. Start one decisive-evidence pilot on released training-fit claims. Review about
   20 claims, distinguishing direct support, direct contradiction, related but
   insufficient, and ambiguous cases. Record exact source quotes and missing
   qualifications. Do not copy claim verdicts to every passage or treat every
   unmatched URL as a reliable negative.
4. If the labels are credible, expand to roughly 100 fit claims and ten selected
   candidates each, including same-topic hard negatives. Coordinate optional
   assisted labeling with C. Training gold answers can guide supervision creation;
   they must not appear as validation/new-claim prediction features. Automatically
   proposed labels remain silver unless reviewed.
5. Test a small selector on saved BM25 top-100 candidates. Feed three full passages
   to the same verifier to isolate selection, then consider matched retraining.
   Preserve context, dates and qualifications. Do not default to QA packing after
   its weak transfer results.
6. For a promising configuration, compare a few controlled seeds and extend
   training only if the released validation curve warrants it. Keep cached NLI
   distillation scores aligned with the actual source text. No improvement is
   guaranteed by ten epochs or a larger model alone.

## Evaluation and completion

Use the published fold IDs, source hashes, scorer and stopping/selection
rules. Record every trial. Report macro-F1, accuracy, per-class scores, FR/FA
counts and denominators, runtime and relevant uncertainty. Keep the historical
500-claim official-dev table separate; test remains untouched.

Deliver a reproducible script/notebook, input and model-selection records,
rankings and predictions, model checkpoints in Drive, and failure analysis.
Preserve progress bars, input/config binding and resume behavior. Expose the
chosen local verifier through A's common backend boundary instead of building
a separate frontend. Coordinate provider-file changes with C.

Add an update using [the template](../team_updates/TEMPLATE.md), with a PR,
stable Drive links, exact relative artifact paths, split protocol/hash, completed
and missing work, limitations and the next hypothesis. Negative results are
valid deliverables; do not expand an unsuccessful pilot unchanged.
