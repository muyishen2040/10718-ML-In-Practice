# Shared experiment progress

Updated November 5, 2026. This is the portable project record for teammates and
agents. Run JSONs and configurations are authoritative; rounded tables below
summarize them. The submitted report and later research are separate snapshots.

## Source and verification record

- [Implement and Evaluate Baselines report](https://docs.google.com/document/d/1Mg5oxNHiU_7iHtqfiym8EL19Eb0xX3ViJKu4Ium9ufE/edit?usp=sharing), the earlier assignment supplied by the coordinator.
- [Initial ML Solution report](https://docs.google.com/document/d/1foHMotcyZf8koHD3wpsCdVbcg45mAVq15SVNWQkEYeA/edit), read directly on November 4.
- [Shared Drive root](https://drive.google.com/drive/folders/19tuNxxB1y9arSmLnrQcxRuPvDQBf6YeK), with active artifacts under `v2_source_documents`.
- [Latest reader results JSON](https://drive.google.com/file/d/1ppMT_l3C4tvalDe_zaYzWvdbApnXWgjn/view), read directly on November 4.
- [Recorded train-only selection](https://drive.google.com/file/d/1wD7YoNfCZMuWg0YLuA9vQ_AL3PI--5yO/view), read directly on November 4.
- [Refit diagnostic summary](https://drive.google.com/file/d/10H2IHH387gVaU5uuQ78VIkrn-t9k6p01/view), read directly on November 4.
- Earlier exploratory results below were supplied from completed Colab runs.
  Their parent folders are visible in Drive; individual old run files were not
  all re-read during this handoff. Verify exact values against JSONs for a new
  report or statistical claim.

`DATA_ROOT` means your mounted
`/content/drive/MyDrive/10718_averitec/v2_source_documents`.
FR and FA below are counts out of all 500 official dev claims. Holdout results
are identified explicitly and must not be compared as if they used dev.
FR (false reassurance) means a non-Supported claim predicted Supported; FA
(false alarm) means a Supported claim predicted another label. NLI means
natural language inference, used here to score support/contradiction between a
passage and claim; QA means question answering over a supplied passage.

## Submitted report: retrieval

These retrievers rank passages inside each development claim's candidate pool.
The display budget is three passages. Relevance is a source-URL proxy and the
denominator is the same 386 judged dev claims.

| Retrieval method | URL-proxy Hit@3 | Hits / 386 | MRR over saved top 20 |
| --- | ---: | ---: | ---: |
| BM25 | 0.1969 | 76 | 0.1835 |
| BGE-small | 0.1813 | 70 | 0.1760 |
| BM25+BGE reciprocal-rank fusion | 0.1839 | 71 | 0.1847 |

Dense retrieval and fusion did not improve source Hit@3. This does not imply
that their passages cannot help a verifier: a proxy URL match and a useful
verdict input are different measures.

## Submitted report: verdicts

All rows evaluate the same 500 official dev claims. The BGE/fusion DistilBERT
rows hold the BM25-trained verifier fixed and change only dev evidence; training
on corresponding full-train dense/fusion rankings was not completed.

| Model | Evidence | Macro-F1 | Accuracy | FR | FA |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-4B | BM25 top three | 0.3346 | 0.496 | 110 | 29 |
| Qwen3-4B | Claim only | 0.2538 | 0.372 | 32 | 97 |
| Qwen3-4B | Annotated answers | 0.4228 | 0.554 | 26 | 41 |
| TF-IDF + logistic regression | BM25 top three | 0.2896 | 0.626 | 61 | 78 |
| TF-IDF + logistic regression | Claim only | 0.3940 | 0.616 | 57 | 80 |
| TF-IDF + logistic regression | Annotated answers | 0.4587 | 0.650 | 50 | 69 |
| DistilBERT | BM25 top three | 0.3469 | 0.520 | 74 | 78 |
| BM25-trained DistilBERT | BGE-small dev top three | 0.3575 | 0.518 | 72 | 74 |
| BM25-trained DistilBERT | Fusion dev top three | 0.3808 | 0.562 | 74 | 71 |
| DistilBERT | Claim only | 0.3970 | 0.576 | 68 | 68 |
| DistilBERT | Annotated answers | 0.5284 | 0.636 | 58 | 44 |

Original DistilBERT: `distilbert-base-uncased`, learning rates `{2e-5, 5e-5}`
and epochs `{2, 3}`, four trials per input condition. Selection used macro-F1
on a 307-claim stratified training holdout, with safety errors breaking ties.
All three reported conditions selected `5e-5`, three epochs, followed by refit
on 3,067 train claims. The selected BM25 model's holdout macro-F1 was 0.4020.
Claim and evidence budgets were 96 and 110 tokens per passage, with 512 total.

Interpretation: the trained BM25 verifier modestly improved macro-F1 over the
two BM25 verifier baselines, but its safety and accuracy gains were mixed. The
claim-only control remained stronger than every retrieved-evidence variant in
the report. Annotated answers show that informative inputs can help, but their
wording and availability differ from raw source passages; 0.5284 is not a
guaranteed ceiling attainable by reranking the current candidate pool.

## Other completed directions: concise context

These were exploratory and are not all included in the submitted report.

| Direction | Observed result | What it established |
| --- | --- | --- |
| Larger BM25 candidate lists | URL-proxy Hit@20 = 0.4223; Hit@100 = 0.6477 | A matching source often exists deeper in the saved pool. These are candidate-coverage bounds, not achieved top-three results or decisive-passage judgments. |
| Simple learned embedding projection | Dev URL-proxy Hit@3 = 0.1943 vs BM25 0.1969 | This inexpensive adaptation did not improve the main retrieval proxy. |
| Answer-aligned cross-encoder plus source diversity | Dev URL-proxy Hit@3 = 0.2280; matched verifier macro-F1 about 0.3455 | More source matches did not produce a corresponding verdict gain. Automatic alignment was silver supervision. |
| Claim-generated questions for reranking | Dev URL-proxy Hit@3 = 0.2332; matched DistilBERT macro-F1 0.3472 with full passages, 0.3597 with excerpts | Question guidance improved the proxy, but did not produce a large end-to-end gain. |
| Source cleaning, generic NLI scores and learned passage aggregation | Initial raw NLI pair reader macro-F1 0.3251; original DistilBERT 0.3469 | The initial pair readers predicted only Supported/Refuted. Cleaning or generic NLI did not solve minority-class learning. |

Do not interpret the shared candidate pools as open-web retrieval or these
experiments as evidence of a single proven bottleneck. Snopes/DisinfoMM remains
secondary work; no completed external evaluation is established by this report
and the latest verified study.

## Latest study: corrected weighting and answer-focused inputs

Artifact root: `DATA_ROOT/outputs/evidence_reader_followup_v2/`.
All variants use the same NLI-initialized DeBERTa pair-reader architecture and
same 303-claim grouped training holdout. They use one fixed learning rate from
the earlier raw reader's training holdout and at most three epochs. The selected
epoch is refit from initialization on all train claims before primary dev scoring.

The legacy weighted cross-entropy divided by the sum of target weights within
each microbatch. With batch size two, single-class batches can cancel those
weights. The correction averages weighted per-example losses by sample count.
Distillation retains cached pretrained NLI behavior; it does not use verdicts
as passage labels. QA variants disable it because the original teacher scores
were computed on different passage text.

| Run ID | Inputs and training change | Holdout macro-F1 /303 | Refit dev macro-F1 /500 | Accuracy | FR | FA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `raw_legacy_d01` | Raw BM25 three, legacy loss, distillation 0.1 | 0.3424 | 0.3234 | 0.654 | 68 | 56 |
| `raw_sample_d01` | Raw BM25 three, corrected loss, distillation 0.1 | 0.3910 | 0.3604 | 0.564 | 49 | 68 |
| `raw_sample_d0` | Raw BM25 three, corrected loss, no distillation | 0.3665 | 0.3460 | 0.608 | 84 | 47 |
| `qa_same3_d0` | Same three sources, generated questions and verbatim answer contexts | 0.4314 | 0.3250 | 0.596 | 75 | 72 |
| `qa_selected3_d0` | QA selection from at most six cached raw/NLI-selected candidates | 0.4230 | 0.3369 | 0.576 | 69 | 68 |

The recorded train-only choice was **`qa_same3_d0`**, despite
`raw_sample_d01` later producing the strongest dev score in this batch. Preserve
that distinction. A better dev ablation is a research lead, not a retroactive
train-only model-selection result.

Corrected weighting improved dev macro-F1 by 0.0370 relative to the legacy
control; the class-stratified paired 95% interval was `[0.0042, 0.0759]`. This
interval excludes retraining randomness, correlated claims and repeated dev
inspection. It is evidence for the correction on these claims, not a complete
generalization guarantee. The QA and selection contrasts did not establish a
clear dev gain; their paired intervals included zero.

## Diagnostic: where the QA holdout gain disappeared

The saved selected checkpoint was evaluated on both its original holdout and
official dev. It was then compared with the already-completed full-train refit.
This requires no new model training.

| Model | Holdout macro-F1 | Same checkpoint dev macro-F1 | Refit dev macro-F1 |
| --- | ---: | ---: | ---: |
| QA, same three passages | 0.4314 | 0.3364 | 0.3250 |
| Corrected raw + distillation | 0.3910 | 0.3655 | 0.3604 |

The QA drop is 9.50 points before refitting, plus 1.14 points after refitting.
The raw model drops 2.55 points before refitting, plus 0.51 afterward. QA conflict
predictions on the same dev set fall from 46 before refit to one afterward;
the raw model's conflict predictions remain similar, 33 versus 34.

Both train and dev questions came from the earlier claim-only generator, with
one fallback among 3,067 train claims and none among 500 dev claims. Candidate
answerability was 14,355 of 31,476 question-passage pairs in train and 2,228 of
5,124 in dev, approximately 45.6% and 43.5%. Thus there is no obvious old/new
generator-origin mismatch, but these aggregates do not establish equal question
quality, equal model revisions, or decisive extracted answers.

## Current conclusions and open hypotheses

**Established:** weighting affects rare-class learning; the QA holdout advantage
did not transfer; refitting additionally changed rare-class behavior; source
URL overlap is an incomplete relevance proxy. A later chronological dev set is
not interchangeable with a mixed-period train holdout.

**Hypotheses to test:** extracted spans may omit negation, dates, quantities or
qualifications; the model may rely on topic/wording correlations; chronological
validation may rank methods differently; full-train refitting may change the
optimization schedule and decision boundaries. Current results do not isolate
one of these as the sole cause.

Next proposed work: freeze chronological training validation, audit decisive
passage labels using training data, and stabilize the full-context reader before
another expensive end-to-end run. No large improvement is claimed for this plan.

## Team organization for the next phase

The proposed assignments are now A: browser extension/interface and backend
integration; B: local ML and evidence selection; C: hosted API-model experiments.
The coordinator owns validation, shared scoring and the report. This replaces
the earlier split into three research-only evaluation/evidence/reader roles.

The hosted-model track first uses the same three supplied passages as the local
baselines. A web-assisted system is a separate evidence setting. The interface
can start with a labeled benchmark demo and mock responses; arbitrary browser
claims need a new retrieval path, since saved rankings belong to known claim
pools. No extension, deployed service or new hosted-model score is established
by this organizational update. Details and dependencies are in
[the work plan](TEAM_WORKPLAN.md).

## Updating this record

Repository cleanup on November 4 moved 11 older notebooks into
`notebooks/archive/` and long historical specifications into `docs/archive/`.
Current catalogs link to the new locations. A second cleanup removed the root
compatibility files, moved dependency profiles under `requirements/`, and
preserved private notes/reports in ignored `local/`. Ten archived setup cells
were updated only for external install paths; portable snapshots stayed frozen.
Historical result artifacts and experimental helper
code were retained. This organization change establishes no new ML results.

The coordinator now owns construction/publication of chronological splits and
the common evaluation path. `configs/team_evaluation.json` records the release
state, initially pending. Teammates use the published path/hash rather than
defining separate splits. Three standalone briefs are in `docs/tasks/`; the
shared protocol is described in [EVALUATION.md](EVALUATION.md).

Each owner writes a new file under `docs/team_updates/`, using the template.
Include negative results, interrupted runs and unresolved questions. The
coordinator reviews those updates, adds accepted results here, and updates the
handoff's artifact map and publication status. Do not replace the submitted
report's historical numbers with later exploratory results.
