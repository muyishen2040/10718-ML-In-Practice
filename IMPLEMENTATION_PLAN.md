# Implementation Plan — Misinformation Screening Assistant

## 1. Goal and scope

Build an evidence-first misinformation screening assistant. Given a claim, the system retrieves up to three evidence passages and predicts an evidential status. It always presents evidence and uncertainty as decision support; the user makes the final judgment.

The project has two deliberately separate evaluation tracks:

1. **Primary in-domain track — AVeriTeC.** Develop and evaluate retrieval and verification using the official data split and static evidence collection.
2. **External-generalization track — DisinfoMM / Snopes.** Train only on AVeriTeC, then evaluate the completed pipeline against a separately built, noisier corpus derived from cited source URLs. This is not additional training data.

The checkpoint scope is the AVeriTeC baselines. Snopes infrastructure is added after those baselines are working.

## 2. Evaluation contract

### AVeriTeC

- Respect the official train/dev/test split.
- Use train for fitting, dev for all choices (hyperparameters, prompt revisions, preprocessing), and test once for final reporting.
- Retrieve exactly **three** passages for every end-to-end prediction.
- Retrieval metrics: Recall@3 and MRR against annotated AVeriTeC evidence.
- Verdict metrics: Macro-F1 (primary), accuracy, per-class precision/recall/F1, and a confusion matrix.
- Four verdict labels: `Supported`, `Refuted`, `Not Enough Evidence`, `Conflicting Evidence`.

### Snopes external evaluation

- Filter DisinfoMM to English records from `Website == snopes` with valid claims and labels.
- Map labels before model evaluation:
  - `True`, `Mostly True` → `Supported`
  - `False`, `Mostly False` → `Refuted`
  - `Incomplete` → `Not Enough Evidence`
  - Unknown/missing/ambiguous values → excluded and counted in the audit report.
- This is a three-class external task. Build external-compatible versions of verifiers using only AVeriTeC examples from those three classes; do not invent a Conflicting label.
- Report Macro-F1, accuracy, per-class F1, confusion matrix, coverage, and calibration where probabilities are available.
- Do **not** initially report external Recall@3/MRR: DisinfoMM source URLs are candidates, not passage-level gold evidence.

## 3. Baselines for the checkpoint

| Baseline | Evidence input | Output | Primary measurement |
|---|---|---|---|
| BM25 retrieval | static AVeriTeC corpus | ranked top 3 passages | Recall@3, MRR |
| Claim-only logistic regression | claim | four-way verdict | Macro-F1 |
| BM25 + TF-IDF logistic regression | claim + same top 3 BM25 passages | four-way verdict | Macro-F1 |
| Zero-shot LLM | claim + same top 3 BM25 passages | four-way verdict | Macro-F1 |

The LLM run must use a frozen evidence-only prompt, structured four-label output, deterministic settings where supported, and saved raw responses. Record exact provider, model/version, date, parameters, prompt, retry behavior, and cost.

Dense retrieval, a ModernBERT-style verifier, and Laya are later models that must be compared with these baselines under the same frozen protocol.

## 4. Repository layout

```text
misinfo_screening_assistant/
├── AGENT.md
├── IMPLEMENTATION_PLAN.md
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── retrieval.yaml
│   ├── verifier.yaml
│   ├── llm_baseline.yaml
│   └── snopes.yaml
├── data/                         # ignored except small metadata/manifests
│   ├── raw/averitec/
│   ├── processed/averitec/
│   └── external/snopes/
│       ├── raw/
│       ├── manifests/
│       ├── documents/
│       ├── passages/
│       └── processed/
├── src/
│   ├── data/
│   │   ├── schema.py
│   │   ├── averitec.py
│   │   ├── snopes.py
│   │   └── preprocessing.py
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── dense.py              # later
│   │   └── metrics.py
│   ├── verification/
│   │   ├── tfidf_logreg.py
│   │   ├── llm_zero_shot.py
│   │   ├── encoder_classifier.py # later
│   │   ├── laya_verifier.py      # later
│   │   └── calibration.py
│   ├── evaluation/
│   │   ├── classification.py
│   │   ├── retrieval.py
│   │   └── end_to_end.py
│   ├── external/
│   │   ├── fetch_sources.py
│   │   ├── extract_text.py
│   │   ├── build_snopes_corpus.py
│   │   └── audit_snopes_corpus.py
│   └── utils/
├── scripts/
│   ├── prepare_averitec.py
│   ├── run_bm25.py
│   ├── train_tfidf_logreg.py
│   ├── run_llm_baseline.py
│   ├── evaluate_averitec.py
│   ├── prepare_snopes.py
│   ├── build_snopes_corpus.py
│   └── evaluate_external.py
├── outputs/                      # ignored: predictions, metrics, figures
└── tests/
```

## 5. Shared data and interface contracts

All dataset loaders produce a common record shape:

```python
Record = {
    "claim_id": str,
    "claim": str,
    "label": str | None,
    "source_urls": list[str],
    "evidence": list[EvidenceItem],
    "metadata": dict,
}

EvidenceItem = {
    "passage_id": str,
    "text": str,
    "url": str | None,
    "domain": str | None,
    "score": float | None,
    "metadata": dict,
}
```

Required model interfaces:

```python
retrieve(claim: str, k: int = 3) -> list[EvidenceItem]

predict(claim: str, evidence: list[str]) -> {
    "label": str,
    "probabilities": dict[str, float] | None,
}
```

Every evaluation run saves a configuration copy, dataset/version identifier, seed, predictions, retrieved passage IDs/scores, and metrics to `outputs/`.

## 6. Snopes corpus rules

- Use only cited source URLs as candidate evidence, never the Snopes verdict, rating, explanation, or verdict-revealing title.
- Respect robots.txt, site terms, rate limits, authentication/paywalls, and failed requests.
- Cache fetched documents and create a manifest recording URL, domain, retrieval timestamp, HTTP status, content hash, title, extracted-text length, and failure/exclusion reason.
- Deduplicate documents and preserve provenance for each passage.
- Maintain a blocklist/review process for fact-check or otherwise verdict-revealing pages that appear among cited sources.
- Treat the result as a frozen corpus snapshot for reproducibility.
- Do not crawl the complete URL manifest by default. First create a fixed,
  stratified claim subset for evidence-aware external evaluation and document its
  seed, class counts, source-URL count, and per-claim source cap if used.
- If manually annotating retrieval, pool candidates from multiple retrievers (for example BM25 top 20 plus dense top 20). Report metrics as **pooled** Recall@3/MRR.

## 7. Implementation phases and completion criteria

### Phase 0 — Foundation

Create the repository skeleton, environment specification, git ignore rules, configuration files, shared schemas, logging helpers, and minimal tests.

**Done when:** a clean environment installs dependencies and test discovery runs.

### Phase 1 — AVeriTeC preparation

Implement downloading/loading, label normalization, immutable split handling, evidence parsing, corpus creation, and an inspection/audit report.

**Done when:** every record is validated, split counts and label distributions are reported, and no test labels are read by training code.

### Phase 2 — BM25 baseline

Index the static evidence corpus, retrieve top three deterministically, save rankings, and compute Recall@3/MRR.

**Done when:** a single command produces rankings and retrieval metrics for each requested split.

### Phase 3 — Simple ML baselines

Implement claim-only and BM25-evidence TF-IDF + multinomial logistic-regression verifiers. Tune only on dev and generate a final test prediction artifact.

**Done when:** both baselines produce the same standardized metrics and confusion-matrix outputs.

### Phase 4 — Zero-shot LLM baseline

Implement the frozen prompt runner on identical BM25 evidence, resilient structured-output parsing, caching, and run metadata.

**Done when:** each prediction is reproducible from stored evidence/prompt/model metadata and can be evaluated with the shared classification code.

### Phase 5 — Checkpoint report package

Create a compact generated results table, error-analysis examples, and README instructions linking commands to artifacts.

**Done when:** the team can write Parts 2–5 of the checkpoint PDF directly from saved outputs.

### Phase 6 — Snopes preparation and corpus build

Implement DisinfoMM filtering/normalization/auditing first, then the compliant source-fetching and text/passage extraction pipeline.

**Done when:** the external corpus is accompanied by a complete provenance/failure manifest and contains no prohibited Snopes verdict material.

### Phase 7 — External generalization evaluation

Run frozen AVeriTeC-trained three-class models over the external corpus. Keep this evaluation separate from in-domain results.

**Done when:** external metrics, exclusions, overlap checks, and data limitations are explicitly reported.

### Phase 8 — Later project models

Add dense retrieval, ModernBERT, Laya, calibration, and optionally the manually annotated Snopes retrieval subset. Compare them only with the established baselines under the same protocol.

## 8. Decisions intentionally deferred

- Exact AVeriTeC download/version and parsing details, to be resolved by inspecting the official release.
- LLM provider/model, pending available credentials, budget, and instructor expectations.
- Dense encoder, cross-encoder reranker, ModernBERT variant, and Laya training configuration.
- Size of a manually annotated Snopes subset; it is not required for the checkpoint.

## 9. Immediate next action

Implement only Phases 0–2 first: create the skeleton, inspect/prepare AVeriTeC, and deliver a reproducible BM25 retrieval baseline before adding any classifier, LLM call, or web scraping.
