# Script map

Start with [your task brief](../docs/README.md) and the coordinator's
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
| [run_llm_baseline.py](run_llm_baseline.py) | Hosted OpenAI baseline skeleton | C must address resume/cache binding and incomplete-claim handling before a full run |
| [run_local_llm_baseline.py](run_local_llm_baseline.py) | Local Qwen baseline | Earlier report comparator; reuse saved results when possible |
| [evaluate_end_to_end.py](evaluate_end_to_end.py), [generate_baseline_report.py](generate_baseline_report.py) | Earlier scoring/report entry points | Do not confuse these with a new coordinator evaluator that has not been released |

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
