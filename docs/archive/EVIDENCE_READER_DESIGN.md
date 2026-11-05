> Archived reference, November 4, 2026. Start with [AGENTS.md](../../AGENTS.md) and [the current handoff](../TEAM_HANDOFF.md). Historical status and evaluation rules below do not override the current [evaluation release](../EVALUATION.md). Run any old commands from the repository root.

# Evidence-aware reader

Branch: `feat/evidence-aware-reader`

Start with [the portable Colab notebook](../../notebooks/evidence_aware_reader_colab.ipynb).
It embeds the implementation, so an unpushed branch does not cause missing-script
errors in Colab. Use a fresh GPU runtime. Original data, rankings, reports and
models are preserved; the default new run is `outputs/evidence_aware_reader_v1/`
under the existing Drive data root.

## Method

1. Reuse BM25 candidate lists at the same depth for train and dev: top 100 when
   available for both splits, otherwise top 20 for both. Never rebuild the corpus.
2. Remove search-engine result pages and exact/near-duplicate passage text. Drop
   a later publication only when explicit publication metadata establishes its
   date. Unknown dates are retained and counted. An archive capture time or a
   future year mentioned in prose does not establish publication time.
3. Score up to 24 cleaned candidates per claim with a frozen NLI model. The
   passage is the premise and the claim is the hypothesis. Source domain/type
   is included explicitly. Preserve all scored candidates in the append-only
   journals, including scores for raw-control passages rejected by cleaning.
4. Form three matched variants: original BM25 top three (`raw`), cleaned BM25
   top three (`clean`), and cleaned stance/coverage/rank selection with source
   diversity (`clean_nli`). All variants retain the same claims, including
   claims with no remaining evidence.
5. Initialize a reader from the same NLI model. Encode each of the three
   claim–passage pairs separately, using at most 384 tokens per pair. Combine
   pair representations, NLI probabilities and source-quality features with
   learned attention, max pooling, and a four-class prediction head. Aggregation
   is invariant to passage order.
6. Fine-tune the transformer layers and aggregation head, keeping vocabulary
   embeddings frozen to limit optimizer memory and overfitting on 3,067 claims.
   Train using balanced claim-level cross-entropy. A small KL distillation term
   retains the cached pretrained NLI behavior; it does not turn AVeriTeC verdicts
   into passage labels. Gold answers and qrels never enter this model's inputs.

Default backbone: [cross-encoder/nli-deberta-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small).
The model card documents training on SNLI/MultiNLI and the logit order
`contradiction, entailment, neutral`. The resolved immutable revision is saved
before scoring and reused for initialization. NLI probabilities are neither
source-trust probabilities nor calibrated four-way verdict probabilities.

## Comparisons

The notebook includes a quick logistic classifier over frozen NLI/quality
features, three frozen original-DistilBERT input controls, matched raw/clean
DistilBERT training, and the new pair reader on all three evidence variants.

All new training uses a fixed, approximately stratified 10% holdout grouped by
normalized claim text. Duplicate claim text cannot cross that internal boundary.
Selection uses holdout macro-F1, then false reassurance and false alarm counts.
Learning-rate trials use at most three epochs; refitting uses the selected epoch.
The notebook records the chosen pipeline before scoring the new trained models
on official dev. The original baseline used a different internal holdout, so
the newly matched controls are reported separately.

Every primary verdict row uses all 500 official dev claims. URL-proxy retrieval
metrics keep the original qrels and denominator, including claims whose matched
source was filtered. A secondary verdict subset excludes normalized train/dev
text overlaps. Per-class results, calibration, safety errors and paired intervals
are exported to `evidence_reader_results.json`.

The new reader processes more tokens than the original concatenation baseline.
The product constraint remains three displayed passages, but reading cost is
higher and should be reported. Full candidate scoring and ablations can take
hours on a T4; the notebook includes a small GPU smoke test and cost controls.
Keeping every trial/refit checkpoint can consume tens of GB of Drive space.

## Code layout

| Module | Responsibility |
| --- | --- |
| `src/data/evidence.py` | Strict claim/ranking joins shared with legacy baselines |
| `src/retrieval/evidence_quality.py` | Pure source rules, deduplication, provenance features and selection |
| `src/retrieval/evidence_pipeline.py` | Resumable frozen-NLI scoring and matched evidence preparation |
| `src/verification/pair_reader.py` | Trainable reader, tokenization and collation |
| `src/verification/pair_training.py` | Train-only selection, refit and cached dev inference |
| `src/verification/evidence_experiments.py` | Fast aggregation, frozen controls and comparison export |
| `src/utils/checkpoints.py` | Atomic artifacts and recoverable JSONL/Trainer checkpoints |
| `scripts/run_evidence_reader.py` | Thin command-line entrypoint |

Legacy baseline commands remain available. The TF-IDF/transformer baselines now
share validated evidence joins; the transformer baseline shares complete
checkpoint discovery and supports grouped validation.

## Resume and verification

Keep source snapshot, input files, run names and settings fixed when resuming.
Configuration and input hashes reject stale reuse. NLI scoring saves each claim;
reader fitting saves complete optimizer checkpoints each epoch; dev inference
saves each batch. A torn final journal line is repaired, while earlier corruption
and duplicate IDs are rejected. Output files are finalized atomically.

The tests include real offline BERT and DeBERTa forward/backward passes, empty
evidence handling, passage-order invariance and save/load round trips. A tiny
offline training test covers prepare, selection-only mode, refit and resuming
finished runs. These test mechanics, not AVeriTeC performance; the full GPU
experiment still needs to run in Colab.

## Limits

The cleaning rules were motivated by already-inspected dev examples. These dev
results are exploratory, not an untouched test. Unknown publication dates,
updated pages, source credibility, quote authentication and multi-hop evidence
remain unresolved. The manually reviewed dev sample is not used for training.
Removing unhelpful pages cannot recover decisive evidence absent from the saved
candidate lists. A frozen final test and a user study remain necessary.
