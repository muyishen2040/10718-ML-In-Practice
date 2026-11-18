# Script map

Start with [your task brief](../docs/README.md) and the shared
[evaluation release](../docs/EVALUATION.md), not the alphabetical script list.
Existing script paths remain stable because tests and other runners import
their helpers. Experimental CLI entry points are not necessarily unused code.

## Current reader and baseline building blocks

| Entry point | Purpose | Current guidance |
| --- | --- | --- |
| [run_evidence_reader.py](run_evidence_reader.py) | Prepare cached evidence, train/read, and compare the NLI pair reader | Main local-method starting point; published split-manifest support still needs wiring |
| [run_answer_followup.py](run_answer_followup.py) | Reuse/generate questions and extract QA contexts | Reproduction/reference for completed weighting/QA study; QA is not the new default |
| [train_transformer_verifier.py](train_transformer_verifier.py) | DistilBERT and other transformer baselines | Keep as a comparator and shared input helper |
| [train_tfidf_logreg.py](train_tfidf_logreg.py) | Cheap supervised baseline and evidence joins | Useful CPU comparator |
| [run_llm_baseline.py](run_llm_baseline.py) | Resumable hosted OpenAI benchmark | Use the shared evaluation population and saved rankings |
| [run_hosted_pilot.py](run_hosted_pilot.py) | Bounded synthetic smoke test or designated training-fit pilot | Offline preflight by default; `--live` sends API requests |
| [run_local_llm_baseline.py](run_local_llm_baseline.py) | Local Qwen baseline | Earlier report comparator; reuse saved results when possible |
| [evaluate_end_to_end.py](evaluate_end_to_end.py), [generate_baseline_report.py](generate_baseline_report.py) | Earlier scoring/report entry points | Do not confuse these with the planned shared evaluator that has not been released |

### Resume a hosted benchmark

Use Python 3.11 or newer and install `requirements/hosted.txt`. Run the
synthetic preflight with `python scripts/run_hosted_pilot.py`. Use
`--check-key` to verify model access and `--live` to generate responses.
The launcher prompts for the project key with input hidden when
`OPENAI_API_KEY` is absent. Run files are saved under `local/pilot/runs`.

The pilot selects `gpt-5.4-mini-2026-03-17`, reasoning effort `none`,
temperature 0, standard service tier, and 512 completion tokens. It enforces
ten API attempts across restarts and a local cost preflight threshold of
$0.10 using recorded standard token prices. The
[November 16 smoke-test update](../docs/team_updates/2026-11-16_teammate-c_synthetic-smoke.md)
contains the observed results.

Run the same command again with the same input files, arguments, and
`--run-name`. The claims file defines the evaluation population.
Use a new run name for a different prompt, model, or set of inputs. Legacy
outputs without a matching binding file require a new run name or an explicit
`--overwrite`.

The runner saves successful predictions and provider responses to
`*_raw.jsonl` and records each failed provider attempt in
`*_failed_attempts.jsonl`. A completed invocation writes the current
`*_predictions.jsonl`, `*_errors.jsonl`, and metrics with completion coverage.
A missing rankings row counts as an input error. An empty evidence list
produces a completed Not Enough Evidence verdict locally.

The reusable `HostedVerifier(model=...)` in
`src/verification/hosted_verifier.py` accepts `verify(claim, passages)`.
It returns `claim`, `status`, `verdict`, `explanation`,
`uncertainty_note`, and `evidence` for A's backend to wrap in its HTTP response.
The adapter validates source IDs and returns excerpts of the supplied text.

## Saved-data preparation and retrieval

The shared dataset/rankings already exist. These are recovery/reproduction
commands, not mandatory onboarding steps:

- Claims/archive parsing: `prepare_averitec.py`,
  `inspect_averitec_knowledge_store.py`, `normalize_averitec_knowledge_store.py`,
  `build_averitec_passage_corpus.py`.
- Downloads/large file utility: `download_averitec_knowledge_store.py`,
  `concat_jsonl.py`. Prefer existing Drive files; a combined train corpus is
  unnecessary for the archive-streaming BM25 workflow.
- BM25: `run_bm25.py`, `run_averitec_archive_bm25.py`.
- Dense/fusion: `run_dense_retrieval.py`, `run_hybrid_retrieval.py`. The full dense
  pass is expensive and did not improve the primary retrieval proxy in the report.

## Historical experimental methods and diagnostics

These support archived workflows and regression tests. Their presence does not
make them recommended next experiments:

- Generic reranking and source diversity: `run_cross_encoder_reranker.py`,
  `run_source_diversity.py`, `run_question_guided_retrieval.py`.
- Weak URL/projection/alignment training: `train_weak_url_reranker.py`,
  `train_projection_reranker.py`, `train_cross_encoder_weak.py`,
  `prepare_answer_aligned_labels.py`, `supervised_rerank_common.py`.
- Earlier answer-aware selection: `run_answer_reader.py`, `train_answer_selector.py`.
- Ablations and audits: `prepare_verifier_variant_rankings.py`,
  `run_evidence_ablation.py`, `run_limitation_diagnostics.py`.
- Deferred external dataset: `prepare_snopes.py`, `build_snopes_manifest.py`,
  `sample_snopes_external_subset.py`, `evaluate_external_tfidf.py`. Infrastructure
  is not evidence of a completed external evaluation.

## Notebook builders and dependencies

`build_evidence_reader_notebook.py` and `build_answer_followup_notebook.py` build
the current reader reproduction notebooks. `build_limitation_notebook.py` now
targets the notebook archive. Existing archived source snapshots are retained
without regeneration; rebuilding creates a new source snapshot and must not be
described as the original executed artifact.

| Dependency profile | Use |
| --- | --- |
| `requirements/base.txt` (or root `requirements.txt`) | Core data/metrics, BM25, TF-IDF and local tests |
| `requirements/reader-colab.txt` | Current neural reader; pinned text-only Transformers stack |
| `requirements/local-llm-colab.txt` | Earlier quantized local Qwen workflow |
| `requirements/colab.txt` | Older broad retrieval/hosted-baseline environment; includes sentence-transformers |

Do not install every profile into one Colab session. Use a fresh session and the
chosen workflow's setup. Provider/model compatibility is checked when C builds
the API runner. Secrets, raw API responses and checkpoints do not belong in Git.
See [the dependency guide](../requirements/README.md) for installation commands.
