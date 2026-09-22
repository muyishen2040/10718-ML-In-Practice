# AGENT.md — Misinformation Screening Assistant

## Project Overview

Build an evidence-grounded **Misinformation Screening Assistant** for everyday internet users.

The user provides a questionable factual claim. The system should:
1. retrieve the most relevant evidence,
2. show the top 3 evidence items,
3. predict an evidence status,
4. show uncertainty/confidence when useful,
5. let the **user make the final judgment**.

Core principle:

> **Evidence first. User decides.**

The system is decision support, not an automatic authority on truth.

---

## Primary User Flow

```text
Claim
  ↓
Evidence retrieval
  ↓
Top 3 evidence passages
  ↓
Evidence-aware verifier
  ↓
Supported / Refuted /
Not Enough Evidence / Conflicting
  ↓
User reviews evidence and decides
```

The system should help users make **better-informed judgments** with less effort.

---

# 1. Datasets

## 1.1 Primary Dataset — AVeriTeC

Use **AVeriTeC** as the main dataset for training, development, retrieval evaluation, and in-domain testing.

Official resources:

- Dataset page: https://fever.ai/dataset/averitec.html
- FEVER / AVeriTeC 2025 shared task and revised evidence collection:
  https://fever.ai/2025/task.html

Important properties:

- 4,568 real-world claims
- claims collected from 50 fact-checking organizations
- real web evidence
- human-annotated fact-checking questions and answers
- source URLs
- justifications
- four verdict labels:
  - Supported
  - Refuted
  - Not Enough Evidence
  - Conflicting Evidence / Cherry-picking

Typical split:

- train: 3,068
- dev: 500
- test: 1,000

Use the official train/dev/test split.

### Important AVeriTeC use

AVeriTeC should support two separate experiments:

#### A. Gold-evidence verification

```text
claim + human-annotated evidence
        ↓
verifier
        ↓
4-way verdict
```

This isolates the verifier from retrieval errors.

#### B. End-to-end verification

```text
claim
  ↓
our retrieval model
  ↓
top 3 evidence
  ↓
verifier
  ↓
4-way verdict
```

This measures the actual system.

### Evidence collection

Prefer the revised AVeriTeC / FEVER 2025 evidence collection when possible because it addresses known temporal-leakage issues in earlier document collections.

For the MVP, use the provided static evidence/document collection rather than live web search.

---

## 1.2 Secondary Dataset — Snopes-derived external evaluation

Use a Snopes-derived dataset primarily for **external/generalization evaluation**, not initial training.

Recommended practical dataset:

### DisinfoMM — Snopes subset

Dataset:
https://huggingface.co/datasets/Syokan/DisinfoMM

Useful fields may include:

- claim
- original Snopes verdict
- harmonized evaluation
- explanation
- Snopes fact-check URL
- source/article URLs
- dates
- keywords/tags

Use only the English Snopes-derived examples for the initial external test.

### Label mapping for external evaluation

Do not force Snopes into AVeriTeC's exact four-class ontology.

Suggested mapping:

```text
True + Mostly True
    → Supported

False + Mostly False
    → Refuted

Incomplete
    → Not Enough Evidence
```

Use a **3-class external evaluation**:

- Supported
- Refuted
- Not Enough Evidence

Do not invent a Conflicting label if the source dataset does not support it.

### Optional richer Snopes dataset

UKP Snopes Corpus:
https://tudatalib.ulb.tu-darmstadt.de/items/0d997a9b-72e9-4e72-b5c7-899265c5d897

This dataset is attractive for retrieval because it contains evidence snippets and source documents, but access may require a research request.

If access is available later, use it as an additional retrieval/generalization benchmark.

---

# 2. Modeling Plan

Keep the modeling plan layered and interpretable.

## 2.1 Non-ML baseline

### BM25 retrieval only

```text
Claim
  ↓
BM25
  ↓
Top 3 evidence
  ↓
User reads evidence
```

No automatic verdict.

Purpose:
- represent a search-based baseline,
- establish retrieval performance before neural models.

Primary retrieval metrics:
- Recall@3
- MRR

---

## 2.2 Simple ML baseline

```text
Claim
  ↓
BM25
  ↓
Top 3 evidence
  ↓
TF-IDF(claim + evidence)
  ↓
Multinomial Logistic Regression
  ↓
4-way verdict
```

This is the main simple ML baseline.

Evaluate with:
- Macro-F1
- accuracy
- per-class precision / recall / F1
- confusion matrix

Also run a diagnostic version:

```text
Claim only
  ↓
TF-IDF
  ↓
Logistic Regression
```

Compare claim-only vs claim+evidence to measure how much evidence actually helps.

---

## 2.3 Main neural verifier

Use a standard evidence-aware encoder/classifier.

Recommended starting point:
- ModernBERT-base or a similar BERT/DeBERTa-style encoder.

Architecture:

```text
claim + top 3 evidence
        ↓
text encoder
        ↓
fixed 4-class classification head
        ↓
Supported / Refuted /
Not Enough Evidence / Conflicting
```

This is the conventional specialist model.

---

## 2.4 Laya as an additional method

Laya is an open-source, Jev-compatible typed decision model.

Resources:

- Laya GitHub:
  https://github.com/NandhaKishorM/laya

- Laya Hugging Face:
  https://huggingface.co/convaiinnovations/laya

Important behavior:

Laya is not a normal fixed-output classifier.

Instead of:

```text
encoder → fixed 4-class head
```

Laya takes:

```text
state + question + candidate answers
        ↓
ModernBERT-based encoder
        ↓
generic decision head
        ↓
score each candidate answer
```

For this project:

```text
State:
- claim
- evidence 1
- evidence 2
- evidence 3

Question:
"Based only on the supplied evidence, what is the evidential status of the claim?"

Choices:
- Supported
- Refuted
- Not Enough Evidence
- Conflicting Evidence
```

Laya returns:
- selected choice
- probability for each choice
- confidence-related outputs

### Important: do NOT judge Laya from zero-shot performance

The base checkpoint may perform poorly zero-shot on a new typed-decision task.

Use **task-specific fine-tuning** before comparing it fairly with trained classifiers.

The Laya repository contains a fine-tuning notebook for typed decisions:

```text
notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
```

Adapt that pipeline to AVeriTeC.

### Fair comparison

Compare:

```text
ModernBERT + fixed 4-class head
vs
Fine-tuned Laya
```

using:
- the same AVeriTeC training data,
- the same gold/retrieved evidence,
- the same dev/test splits.

Research question:

> Does a general typed-decision model provide useful accuracy, calibration, or flexibility advantages over a conventional task-specific classifier for evidence-grounded misinformation verification?

---

# 3. Retrieval Plan

Build retrieval separately from classification.

## Phase 1 — BM25

Retrieve top 3 evidence passages.

Metrics:
- Recall@3
- MRR

## Phase 2 — Semantic retrieval

Use a sentence embedding / dense retrieval model.

A reasonable starting point:
- BGE-base or similar retrieval encoder.

Architecture:

```text
Claim
  ↓
embedding model
  ↓
rank candidate evidence passages
  ↓
Top 3
```

Optional stretch goal:

```text
dense retrieval top 20
  ↓
cross-encoder reranker
  ↓
top 3
```

Do not make reranking mandatory for the MVP.

---

# 4. Build Order

Follow this order to reduce risk.

## Stage A — Data pipeline

Implement:
- dataset loading
- label normalization
- evidence parsing
- train/dev/test split handling
- reproducible preprocessing

Do not touch the test labels during model development.

---

## Stage B — Gold-evidence classification

First ignore retrieval.

Run:

```text
AVeriTeC claim + gold evidence
        ↓
verifier
```

Compare:
1. TF-IDF + Logistic Regression
2. ModernBERT-style classifier
3. Fine-tuned Laya

This tells us whether the verifier can reason over good evidence.

---

## Stage C — Retrieval

Implement:
1. BM25
2. semantic retrieval

Evaluate:
- Recall@3
- MRR

Inspect failure cases manually.

---

## Stage D — End-to-end system

Connect:

```text
claim
  ↓
retrieval
  ↓
top 3 evidence
  ↓
verifier
  ↓
verdict
```

Main combinations:

```text
BM25 + Logistic Regression
BM25 + ModernBERT
Semantic retrieval + ModernBERT
Semantic retrieval + Fine-tuned Laya
```

Do not explode the experiment matrix unless needed.

---

## Stage E — External Snopes evaluation

Train on AVeriTeC only.

Then test/adapt on Snopes-derived examples.

Goal:

> Measure how much performance drops when the fact-checking source, label style, and claim distribution change.

This is an explicit generalization / deployment-risk experiment.

---

# 5. Evaluation

## 5.1 Retrieval

Primary:
- Recall@3
- MRR

Recall@3 is especially important because the user-facing system shows only 3 evidence items.

---

## 5.2 Classification

Primary:
- Macro-F1

Also report:
- accuracy
- per-class precision
- per-class recall
- per-class F1
- confusion matrix

Macro-F1 is important because AVeriTeC is class-imbalanced.

---

## 5.3 Calibration

For probabilistic models, especially Laya, evaluate:
- Expected Calibration Error (ECE)
- Brier score
- reliability diagram if time permits

Do not assume Laya probabilities are automatically calibrated on misinformation.

If needed, apply post-hoc calibration such as:
- temperature scaling on dev data

Calibration is important because this is a user-facing decision-support system.

---

## 5.4 End-to-end evaluation

Measure:
- whether useful evidence is retrieved,
- whether the resulting verdict is correct.

Keep retrieval and classification metrics separate so failure sources remain diagnosable.

---

## 5.5 Human evaluation

If time allows, recruit about 20–30 accessible adult users.

Compare:

```text
Condition A:
claim only / normal verification process

Condition B:
top 3 evidence
+ evidence status
+ uncertainty
```

Measure:
- claim-assessment accuracy
- decision time
- user confidence
- confidently incorrect judgments
- perceived usefulness

Main real-world outcome:

> Does the assistant improve user judgment without adding too much time or causing over-trust?

---

# 6. Failure Modes / Risks

The project should explicitly connect to ML-in-Practice failure modes.

## 6.1 Validation & leakage

Risk:
- pretrained models may have seen benchmark claims or fact-check pages,
- evaluation may overestimate real-world performance.

Mitigation:
- use official splits,
- use revised AVeriTeC evidence collection,
- test on newer/unseen claims when feasible,
- compare claim-only vs evidence-conditioned models,
- use Snopes as external evaluation.

---

## 6.2 Human & workflow

Risk:
- users may over-trust the predicted verdict and ignore evidence.

Mitigation:
- always show the supporting evidence,
- include uncertainty,
- keep the user as final decision-maker,
- evaluate over-trust in user study.

---

## 6.3 Metric mismatch

Risk:
- better Recall@3 or Macro-F1 may not improve real decisions.

Mitigation:
- evaluate human accuracy and decision time,
- treat model metrics as intermediate outcomes, not final success.

---

## 6.4 Retrieval failure

Risk:
- the verifier cannot make a good decision if relevant evidence is missing.

Mitigation:
- evaluate retrieval independently,
- inspect retrieval failures,
- allow Not Enough Evidence.

---

## 6.5 Distribution shift

Risk:
- AVeriTeC performance may not transfer to claims from other fact-checking sources.

Mitigation:
- external evaluation on Snopes-derived data.

---

# 7. Ethics and Product Principles

Always preserve these constraints:

1. Do not present the model as an authority on truth.
2. Show evidence alongside the predicted status.
3. Keep an explicit "Not Enough Evidence" outcome.
4. Do not automatically censor or remove content.
5. Do not hide uncertainty.
6. Avoid sensitive personal attributes as predictive shortcuts.
7. User retains the final decision.
8. Clearly communicate limitations and possible errors.

---

# 8. Expected Repository Structure

Prefer a clean modular project layout such as:

```text
misinformation-screening/
├── AGENT.md
├── README.md
├── requirements.txt
├── configs/
│   ├── retrieval.yaml
│   ├── verifier.yaml
│   └── laya.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_bm25_baseline.ipynb
│   ├── 03_gold_evidence_classifier.ipynb
│   ├── 04_laya_finetuning.ipynb
│   └── 05_end_to_end_eval.ipynb
├── src/
│   ├── data/
│   │   ├── averitec.py
│   │   ├── snopes.py
│   │   └── preprocessing.py
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   └── metrics.py
│   ├── verification/
│   │   ├── tfidf_logreg.py
│   │   ├── encoder_classifier.py
│   │   ├── laya_verifier.py
│   │   └── calibration.py
│   ├── evaluation/
│   │   ├── classification.py
│   │   ├── retrieval.py
│   │   └── end_to_end.py
│   └── utils/
├── scripts/
│   ├── prepare_averitec.py
│   ├── train_classifier.py
│   ├── train_laya.py
│   ├── evaluate_retrieval.py
│   └── evaluate_external.py
└── tests/
```

Do not create unnecessary abstractions early.

---

# 9. Engineering Guidelines for Codex

When implementing:

- Prefer simple, readable code over framework-heavy abstractions.
- Keep experiments reproducible with explicit seeds.
- Keep data splits immutable.
- Never use test labels for model selection.
- Log model configuration and dataset version.
- Save predictions to disk for error analysis.
- Keep retrieval and classification evaluation separate.
- Avoid data leakage between train/dev/test.
- Make all model interfaces compatible with a shared record structure:

```python
{
    "claim": str,
    "evidence": list[str],
    "label": str | None,
    "metadata": dict,
}
```

Preferred verifier interface:

```python
predict(claim: str, evidence: list[str]) -> {
    "label": str,
    "probabilities": dict[str, float] | None
}
```

Preferred retriever interface:

```python
retrieve(claim: str, k: int = 3) -> list[EvidenceItem]
```

---

# 10. Compute Constraints

Assume development may happen on:
- Windows laptop
- Google Colab
- Kaggle T4 x2 when needed

Design experiments so that:
- inference can run on CPU when practical,
- main neural fine-tuning fits on a T4-class GPU,
- Laya fine-tuning may use Kaggle T4 x2 if necessary,
- no experiment should require large multi-node infrastructure.

Prefer:
- mixed precision,
- gradient accumulation,
- gradient checkpointing when needed,
- small batch sizes,
- cached embeddings.

---

# 11. Project Milestones

## Milestone 1
- load AVeriTeC
- inspect label distribution
- inspect evidence structure
- create clean train/dev/test preprocessing

## Milestone 2
- BM25 retrieval baseline
- Recall@3 + MRR

## Milestone 3
- TF-IDF + Logistic Regression verifier
- gold-evidence evaluation

## Milestone 4
- ModernBERT-style verifier
- gold-evidence evaluation

## Milestone 5
- fine-tune Laya on AVeriTeC
- compare with standard classifier
- evaluate calibration

## Milestone 6
- semantic retrieval
- end-to-end pipeline

## Milestone 7
- external Snopes evaluation

## Milestone 8
- user-facing prototype
- optional human study

---

# 12. Minimum Viable Project

If time becomes limited, prioritize:

1. AVeriTeC only
2. BM25 retrieval
3. TF-IDF + Logistic Regression baseline
4. ModernBERT verifier
5. Laya verifier
6. Recall@3 + Macro-F1
7. simple end-to-end demo

Snopes, reranking, calibration, and user study can be reduced or treated as stretch goals if necessary.

---

# 13. Main Research Questions

The project should answer:

1. **Can semantic retrieval find useful evidence more reliably than lexical retrieval?**
2. **How much does evidence improve verdict prediction compared with claim-only classification?**
3. **How does a standard task-specific classifier compare with a fine-tuned Laya decision model?**
4. **Are the model probabilities well calibrated enough to support user decisions?**
5. **How well does a model trained on AVeriTeC generalize to Snopes-derived claims?**
6. **Does the full system actually help users make better judgments?**

---

# 14. Final Project Story

Use this framing when making implementation decisions:

> We build an evidence-grounded misinformation screening assistant using AVeriTeC, compare lexical and semantic evidence retrieval, compare conventional classification with the Laya typed-decision model, test generalization on Snopes-derived fact checks, and evaluate whether the resulting system helps users make more accurate judgments.

Every experiment should support that story.

Do not add unrelated modeling complexity unless it helps answer one of the research questions above.
