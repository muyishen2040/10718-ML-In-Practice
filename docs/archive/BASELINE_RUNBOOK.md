# Historical baseline runbook

Preserved from the repository README on November 4, 2026, before the team
onboarding cleanup. These are earlier command recipes, not the current progress
record. Existing shared data and completed rankings should normally be reused.
Run commands from the repository root. Start new work from
[the current README](../../README.md) and [team handoff](../TEAM_HANDOFF.md).

---

# Misinformation Screening Assistant

An evidence-first claim-verification project for 10-718 ML in Practice. See
[IMPLEMENTATION_PLAN.md](INITIAL_IMPLEMENTATION_PLAN.md) for the agreed architecture
and experiment sequence. See [COLAB.md](BASELINE_COLAB.md) for the large-corpus and
LLM-baseline workflow.

Teammates and coding agents should start with [the team handoff](../../docs/TEAM_HANDOFF.md),
[current results](../../docs/TEAM_PROGRESS.md), and [the team work plan](../../docs/TEAM_WORKPLAN.md).
These explain Drive shortcuts, reusable artifacts, Git publication status and
how to return a self-contained experiment update.

The new [evidence-aware reader](EVIDENCE_READER_DESIGN.md) adds source cleaning,
cached NLI passage scoring, and a trainable three-passage aggregator. Its
[self-contained Colab notebook](../../notebooks/evidence_aware_reader_colab.ipynb)
reuses saved rankings and supports interrupted runs.

## Current status

The repository implements runnable checkpoint baselines: deterministic BM25,
TF-IDF + multinomial logistic regression, and a zero-shot, evidence-only LLM
runner. It also includes data-quality audits, retrieval/classification/error
metrics, run manifests, and a Colab runbook. The full AVeriTeC knowledge store
is intentionally not included in Git.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare AVeriTeC claim data

The following command downloads the official train and development claim JSON
files, validates their labels, and writes canonical JSONL records plus an audit
report. It intentionally does not download the knowledge store.

```powershell
python scripts/prepare_averitec.py --download-claims
```

The official AVeriTeC knowledge store is large. For BM25, prepare or place a
static passage corpus at:

```text
data/processed/averitec/<split>_evidence_corpus.jsonl
```

Each line must contain at least `passage_id` and `text`; `url` and `metadata`
are optional. The corpus must be built from the official candidate documents,
not from gold answers.

## Run BM25

After preparation and after writing relevance judgments to
`data/processed/averitec/<split>_qrels.jsonl`, run:

```powershell
python scripts/run_bm25.py --split dev
```

The runner deterministically retrieves exactly three passages, saves rankings,
and reports evidence Hit@3, passage Recall@3, Precision@3, and MRR when qrels
are provided. The corpus/qrels builder labels URL-based evidence coverage
explicitly; it does not claim exact passage-level human relevance.

The large archive normalizer and passage builder stream records to disk. The
official AVeriTeC archive is recognized as one candidate pool per claim, and
BM25 indexes one such pool at a time—preventing cross-claim retrieval and
avoiding a global in-memory index. Details, including the large train-shard
requirement, are in [COLAB.md](BASELINE_COLAB.md).

## Run the simple verifier baselines

After preparing the AVeriTeC claims, run the claim-only diagnostic baseline:

```powershell
python scripts/train_tfidf_logreg.py --evidence-mode claim_only
```

To test the verifier in isolation using human-annotated AVeriTeC answers:

```powershell
python scripts/train_tfidf_logreg.py --evidence-mode gold
```

`gold` is a gold-evidence verification experiment only. Its score must not be
reported as an end-to-end system result; the future BM25-evidence run will use
the same classifier interface with retrieved passages instead.

Use `--evidence-mode gold_all` only as an uncapped oracle upper bound. It is
not comparable to the user-facing top-three-evidence setting.

## Run the LLM baseline

For Colab GPU, the recommended runner is the 4-bit local Qwen3 baseline in
[COLAB.md](BASELINE_COLAB.md). It caches one completed claim at a time so an interrupted
GPU session can resume. It supports the primary BM25-top-3 run plus clearly
labeled claim-only and capped gold-evidence diagnostics. The hosted OpenAI
runner remains available as an alternative; it requires `requirements/colab.txt`
and `OPENAI_API_KEY` outside the repository.

## Preserve each run for review

The BM25, TF-IDF/logistic-regression, and LLM runners accept `--run-name`.
Use a new descriptive name for each frozen experiment, such as
`dev_bm25_top20_v1`; this creates a separate child directory under
`--output-dir` rather than replacing another experiment's files. Each baseline
run saves predictions, metrics, configuration, and a manifest. Evidence-aware
verifiers additionally save `*_evidence_coverage.jsonl` and
`*_retrieval_conditioned_metrics.json`, which separate verdict performance
when an annotated source was covered from performance when it was missed.

## Prepare the external Snopes records

This creates a clean, three-class DisinfoMM/Snopes claim set and preserves its
cited-source URLs. It does **not** fetch web pages or use the Snopes explanation
as evidence.

```powershell
python scripts/prepare_snopes.py --download
```

Then build the deduplicated source manifest before attempting any fetching:

```powershell
python scripts/build_snopes_manifest.py
```

For the future evidence-aware external test, freeze a manageable stratified
subset before any fetching:

```powershell
python scripts/sample_snopes_external_subset.py --size 300 --seed 718
```

The following source-corpus stage will fetch only permitted cited URLs, log
provenance/failures, filter verdict-revealing material, and construct passages.

Before that corpus exists, run the valid claim-only external-transfer diagnostic:

```powershell
python scripts/evaluate_external_tfidf.py
```

It trains only on AVeriTeC's three label-compatible classes and evaluates on
Snopes. It is not a substitute for the later evidence-aware external pipeline.

## Tests

The next GPU study is [evidence_reader_followup_colab.ipynb](../../notebooks/evidence_reader_followup_colab.ipynb).
Upload it directly to Colab. It includes its source and reuses the completed
`evidence_aware_reader_v1/prepared` artifacts in Drive. It tests corrected
rare-class weighting, NLI distillation, and answer-focused reading of up to six
cached candidate passages. Progress and resume are built in; no retrieval index
or full-corpus embeddings are rebuilt. New outputs use `evidence_reader_followup_v2`.

```powershell
python -m pytest
```

## Data policy

Do not commit datasets, model outputs, credentials, or API responses. Store
dataset source URLs, hashes, versions, and run configurations in audit files.
