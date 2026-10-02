# Checkpoint Handoff — Misinformation Screening Assistant

**Status date:** October 2, 2026
**Purpose:** Give a collaborating agent the current, evidence-based project state for finishing the baseline checkpoint report and presentation. This document supersedes the *checkpoint-status* portions of `AGENT.md`; it does not replace the longer-term project plan there.

## 1. Project in one paragraph

We are building an evidence-grounded misinformation-screening assistant. A user supplies a factual claim; the system retrieves up to three passages from a candidate evidence pool and predicts one of four AVeriTeC verdicts: **Supported**, **Refuted**, **Not Enough Evidence**, or **Conflicting Evidence**. The user remains the decision maker. The intended outcome is lower susceptibility to misinformation, especially fewer cases where unsupported or misleading claims receive false reassurance as “Supported.”

The project has two separable tasks:

1. **Retrieval:** rank candidate evidence passages for a claim. The UI constraint is the top three passages.
2. **Verdict classification:** predict the four-way verdict using the claim, optionally with retrieved evidence.

## 2. Fixed evaluation protocol

### Data split

- **Training:** official AVeriTeC train split; 3,068 raw records, of which 3,067 have nonempty claims and are usable by the TF-IDF/logistic-regression baselines.
- **Evaluation:** official AVeriTeC development split; all **500** claims.
- **Held-out test:** untouched. Do not use it for model selection or checkpoint reporting.

This mirrors deployment better than a random split: a system is fitted on prior fact-checked claims and evaluated on distinct later/unseen claims.

### Retrieval labels and scope

The knowledge store supplies candidate source documents, but its records are not exact human-labeled passage judgments. We created qrels by matching a candidate document’s normalized URL against AVeriTeC’s annotated evidence-source URL, then treating passages from matching documents as relevant. Therefore:

- call the main metric **URL-proxy evidence hit@3**, not exact human passage Recall@3;
- report **MRR** over the same URL-proxy qrels;
- rank 20 passages for auditing, but calculate the decision-constrained result using the top **3**;
- report results only over claims with qrels: **386 / 500** dev claims. The other 114 claims are unjudged for retrieval, not automatically retrieval failures;
- one dev claim (`averitec:dev:00417`) has no candidate pool. It is retained in classification, with an explicit no-passage fallback.

URL matching is a practical first-pass proxy. It does **not** prove that every matched chunk is decisive evidence, and it should not be represented as a fully human-annotated passage-level retrieval benchmark.

### Verdict metrics

All verdict models are evaluated on the same 500 dev labels.

- Primary classification metric: **macro-F1**, which gives the rare “Not Enough Evidence” and “Conflicting Evidence” labels equal weight.
- Secondary: accuracy and per-class precision/recall/F1.
- Safety-oriented outcomes:
  - **False reassurance:** gold label is Refuted, Not Enough Evidence, or Conflicting Evidence, but the model predicts Supported.
  - **False alarm:** gold label is Supported, but the model predicts any non-Supported label.
- Logistic regression additionally records multiclass Brier score and ECE. LLM self-reported confidence is not calibrated and is not used for calibration metrics.

## 3. Data-pipeline history and current primary retrieval corpus

### Superseded v1 corpus

The first corpus split every extracted sentence into a passage. It created 138,301,514 passages, which made BM25 inefficient and reduced useful context per item. Its result remains auditable but is not the primary checkpoint result.

### Primary v2 source-document corpus

The current corpus first joins extracted sentence records by source URL/document and then chunks documents into 160-word windows with 40-word overlap. Each claim is still searched only within its AVeriTeC candidate pool.

- Documents: **146,307**
- Passages: **16,050,740**
- Candidate-pool scoped: yes
- Empty documents rejected: 0
- Candidate documents unmatched during construction: 0
- Dev qrel-covered claims: 386
- Annotated source URLs not resolved by normalized matching: 437
- Corpus build time in Colab: about 23 minutes

The v2 segmentation contains 88.4% fewer passages than v1 while retaining source-document context. Passage-recall denominators differ across segmentations, so do **not** make a strong v1-versus-v2 comparison using passage recall alone.

## 4. Completed primary baseline results

### Retrieval baseline: BM25 v2

**What it is:** BM25 keyword retrieval over the per-claim AVeriTeC source-document corpus. It is both a realistic manual-search proxy and the required non-ML retrieval baseline. Ties are deterministically broken by descending score and then passage ID.

| Result | Value |
| --- | ---: |
| Evaluated dev claims | 500 |
| Qrel-judged claims | 386 |
| Ranking depth / displayed depth | 20 / 3 |
| URL-proxy evidence hit@3 | **0.1969** |
| MRR | **0.1835** |
| URL-proxy precision@3 | 0.1002 |
| URL-proxy passage recall@3 | 0.0388 |
| Runtime | about 57 minutes |

For context, v1 obtained hit@3 0.1658 and MRR 0.1438. The improved v2 result is the one to present: hit@3 increased by 3.11 percentage points and MRR by 3.97 points, under the URL-proxy caveat.

### Zero-shot LLM baseline: Qwen3-4B + BM25 v2 top 3

**What it is:** A local 4-bit zero-shot `Qwen/Qwen3-4B` verifier receives a claim and exactly the three BM25 v2 passages. It must choose one of the four labels using only those passages and return cited passage IDs. This is the main deployable zero-shot baseline.

| Metric | Value |
| --- | ---: |
| Claims evaluated | 500 |
| Macro-F1 | **0.3346** |
| Accuracy | **0.4960** |
| False reassurance | **0.220** (110 claims) |
| False alarm | **0.058** (29 claims) |
| Supported F1 | 0.5723 |
| Refuted F1 | 0.5955 |
| Not Enough Evidence F1 | 0.1250 |
| Conflicting Evidence F1 | 0.0455 |

Configuration: model revision `1cfa9a7208912126459214e8b04321603b3df60c`; 4-bit Transformers inference on Colab GPU; seed 718; deterministic generation (`do_sample=false`); `max_new_tokens=256`; Qwen thinking disabled; 1,200 maximum characters per passage. Ten malformed/non-candidate citation selections were logged but retained as classification predictions; one zero-passage fallback was used. The final run recovered the one initially failed prediction, so all 500 claims are included.

### Diagnostic classification baselines

These are useful to locate the retrieval bottleneck, but neither is deployable as an end-to-end product result.

| System | Evidence supplied | Macro-F1 | Accuracy | False reassurance | False alarm | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-4B | Claim only | 0.2538 | 0.372 | 0.064 | 0.194 | No-evidence diagnostic; can exploit claim wording or priors and must not be described as verification. |
| Qwen3-4B | Up to 3 human-annotated gold answers | 0.4228 | 0.554 | 0.052 | 0.082 | Non-deployable oracle. Its gap from BM25+Qwen indicates headroom if retrieval/evidence use improves. |
| TF-IDF + balanced multinomial logistic regression | Claim only | 0.3940 | 0.616 | 0.114 | 0.160 | Simple supervised diagnostic; high score also shows class/wording shortcuts and class imbalance. |
| TF-IDF + balanced multinomial logistic regression | Up to 3 gold evidence answers | 0.4587 | 0.650 | 0.100 | 0.138 | Non-deployable oracle verifier experiment. Gold evidence raises macro-F1 by 0.0647 over the claim-only logistic model. |

The logistic models use word TF-IDF unigrams/bigrams, sublinear term frequency, `min_df=1`, class-balanced multinomial logistic regression, seed 718, and `max_iter=2000`; they train on all 3,067 usable official train claims and evaluate on all 500 dev claims. Claim-only LR also reports Brier 0.6028 / ECE 0.1996; gold-evidence LR reports Brier 0.5384 / ECE 0.1972.

### Retrieval-conditioned Qwen diagnostic

On the 386 claims with URL-proxy qrels, BM25 hit a proxy-relevant source in the top 3 for 76 claims and missed it for 310.

| Group | Claims | Macro-F1 | Accuracy | False reassurance |
| --- | ---: | ---: | ---: | ---: |
| URL-proxy source covered | 76 | 0.2491 | 0.4605 | 0.3289 |
| URL-proxy source missed | 310 | 0.3787 | 0.5419 | 0.1903 |

This is **not causal evidence that retrieval hurts**. The proxy is incomplete, a matching URL need not contain a decisive chunk, group label distributions differ, and the LLM may not use supplied evidence faithfully. It is a diagnostic that motivates better retrieval metrics and manual error review.

## 5. Prompt required for reproducibility

The evidence-grounded Qwen baseline used this frozen system prompt:

```text
You are an evidence-grounded claim-verification baseline.
Use only the supplied passages. Do not use outside knowledge, the claim's likely
truth, or information that is absent from the passages. Return one label:
- Supported: evidence establishes the claim.
- Refuted: evidence establishes that the claim is false or materially wrong.
- Not Enough Evidence: supplied evidence does not establish either conclusion.
- Conflicting Evidence: supplied evidence contains material support and refutation.

Select the passage IDs that most directly support your decision. Your reported
confidence is a self-assessment, not a probability. Return the required JSON
object only, with exactly these keys:
{"label": "one of the four labels", "selected_passage_ids": ["passage ID"],
 "reported_confidence": 0.0, "brief_rationale": "one brief explanation"}.
```

Claim-only uses a parallel prompt that explicitly says no evidence was provided and forbids claims of retrieval or verification.

## 6. What is complete versus pending

### Complete and ready for the checkpoint

- Official dev data and source-document v2 corpus.
- BM25 non-ML retrieval baseline with top-3 constrained results.
- Qwen3-4B zero-shot evidence-grounded baseline over exactly those BM25 passages.
- Claim-only and gold-evidence diagnostics for both Qwen and TF-IDF logistic regression.
- False-reassurance/false-alarm reporting and retrieval-conditioned diagnostic.
- Reproducible Colab commands, outputs, configs, prompts, and a generated baseline-summary CSV/Markdown file.

### Not complete; do not present as completed

- **TF-IDF + logistic regression using train BM25 evidence and dev BM25 evidence.** This needs train BM25 rankings generated from official training knowledge-store archives.
- Dense semantic retrieval and an evidence-conditioned transformer (future main system).
- Snopes/DisinfoMM processing and external generalization experiment.
- Human user study.

The full train BM25 process is resumable but expensive: its last observed status was 11/1,000 claims in archive `train_0_999`, around 10.8 seconds/claim, or roughly three hours per archive shard. It is not cost-effective for the Monday baseline checkpoint, since a zero-shot LLM baseline is already appropriate and the assignment makes simple ML optional. Preserve its checkpoint rather than restarting it. If run later, either finish the full train pipeline or define a clearly labeled, fixed stratified development subset; never silently train on dev retrieval outputs.

## 7. Key conclusions to preserve in the writeup and presentation

1. **The non-ML baseline is real, not a strawman.** BM25 top-3 approximates keyword web search by a user and uses the same evidence budget the system would show.
2. **The LLM baseline is legitimate but safety-limited.** BM25+Qwen improves substantially from v1 to v2, yet 22% false reassurance is too high for automatic factual endorsement.
3. **The oracle gap is useful.** Gold evidence improves Qwen macro-F1 from 0.3346 to 0.4228 and accuracy from 0.496 to 0.554. This indicates retrieval/evidence selection remains a valuable target.
4. **Claim-only numbers are not success.** They expose label prevalence and lexical shortcuts. A verifier needs evidence, not a high claim-only score.
5. **The hard labels matter.** Both Qwen and logistic regression do poorly on Conflicting Evidence, while Not Enough Evidence also remains weak for Qwen. Macro-F1 and asymmetric errors are necessary alongside accuracy.
6. **A deployment bar should be stricter than macro-F1 alone.** A worthwhile system must improve macro-F1 and, especially, reduce false reassurance without creating excessive false alarms or burdening users with more than three pieces of evidence.

## 8. Where outputs live in Colab Drive

```text
/content/drive/MyDrive/10718_averitec/
├── outputs/                                      # v1 audit results
│   ├── bm25/dev_bm25_top20_v1/
│   └── local_llm/
│       ├── dev_qwen3_4b_claim_only_v1/
│       ├── dev_qwen3_4b_gold_top3_v1/
│       └── dev_qwen3_4b_top3_v1/
├── outputs/reports/final_baseline_summary/
│   ├── baseline_summary.md
│   └── baseline_summary.csv
└── v2_source_documents/
    ├── processed/averitec/                       # claims, v2 corpus, qrels
    └── outputs/
        ├── bm25/dev_bm25_top20_v2/
        │   ├── dev_metrics.json
        │   └── dev_rankings.jsonl
        ├── local_llm/dev_qwen3_4b_bm25_top3_v2/
        │   ├── classify_top3_config.json
        │   ├── classify_top3_metrics.json
        │   ├── classify_top3_evidence_coverage.jsonl
        │   └── classify_top3_retrieval_conditioned_metrics.json
        └── tfidf_logreg/
            ├── claim_only_train_to_dev_v2/claim_only_train_to_dev_metrics.json
            └── gold_train_to_dev_v2/gold_train_to_dev_metrics.json
```

## 9. Recommended immediate next steps

1. Turn `BASELINE_CHECKPOINT_REPORT_DRAFT.md` into the short PDF; keep the report concise and retain its caveats.
2. Build 7–8 minute update slides: problem/user flow; split and top-3 decision constraint; BM25 v2; Qwen BM25 v2; oracle/claim-only diagnostic chart; safety/errors; next steps.
3. Before submitting, rerun the report-generation command only if output paths change; do not rerun costly models unnecessarily.
4. After the checkpoint, prioritize a dense retriever and an evidence-conditioned classifier. Add a small manual passage-relevance audit to validate the URL proxy.
5. Treat DisinfoMM/Snopes as a separate external classification experiment, not retrieval evaluation, unless passage-level relevance annotations are created.
