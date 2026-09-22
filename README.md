# Misinformation Screening Assistant

An evidence-first claim-verification project for 10-718 ML in Practice. See
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the agreed architecture
and experiment sequence.

## Current status

The repository currently implements the foundation for the AVeriTeC data
pipeline and a BM25 retrieval runner. Classifier and LLM baselines are the next
phases.

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
data/processed/averitec/evidence_corpus.jsonl
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
and reports Recall@3 and MRR when qrels are provided.

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

## Tests

```powershell
python -m pytest
```

## Data policy

Do not commit datasets, model outputs, credentials, or API responses. Store
dataset source URLs, hashes, versions, and run configurations in audit files.
