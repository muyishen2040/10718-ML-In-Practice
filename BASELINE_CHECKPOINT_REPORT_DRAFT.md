# Implement and Evaluate Baselines — Draft Report

**Course:** CMU 10-718, Machine Learning in Practice
**Project:** Evidence-Grounded Misinformation Screening Assistant
**Checkpoint:** Baseline implementation and evaluation
**Data/evaluation date:** October 2, 2026

> **Drafting note.** This is a content-complete first version for conversion to a short PDF. Before submission, replace the author/team placeholders, confirm the due date in the course site, and trim Part 1 if the final PDF needs to be shorter. All reported primary retrieval and end-to-end numbers below use the same official AVeriTeC development split and the current source-document (“v2”) corpus.

## 1. Revised proposal responses

The following are the proposal questions that received feedback. Each entry shows the original response, the revision, and the substantive change.

### Question 2 — Goal

**Original.**

> We are trying to increase users’ accuracy in assessing questionable factual claims, measured by the percentage of claims they correctly assess as Supported, Refuted, Insufficient Evidence, or Conflicting Evidence. Secondary outcomes are reducing verification time and reducing confidently incorrect judgments.

**Revision.**

> We are trying to reduce users’ susceptibility to misinformation, measured by how often they correctly identify unsupported or misleading factual claims after using our system. Our primary concern is false reassurance: cases where a Refuted, Insufficient Evidence, or Conflicting Evidence claim is incorrectly accepted as Supported. However, excessive false alarms are also harmful because they could make users distrust legitimate information, so we will also measure how often supported claims are incorrectly flagged as questionable. Secondary outcomes are reducing verification time and reducing confidently incorrect judgments.

**What changed.** We broadened the goal from raw label accuracy to the user-centered outcome—resistance to misinformation—and made the asymmetric costs explicit: false reassurance is especially harmful, while false alarms can erode trust in legitimate information.

### Question 4b — Why show three evidence items?

**Original.**

> Users have limited attention and are unlikely to inspect many sources for one claim. Showing three evidence items keeps verification quick while still allowing comparison across multiple pieces of evidence. This capacity constraint directly motivates Recall@3 as our primary retrieval metric.

**Revision.**

> Users have limited attention, so we will show at most three evidence items per claim. In AVeriTeC, each claim has a candidate evidence pool and human-annotated gold evidence identifying which passages directly support or refute the claim. Recall@3 measures whether the top three retrieved items contain the gold relevant evidence. In deployment, we will retrieve from a curated set of trusted organizations and evaluate relevance with human annotation.

**What changed.** We tied the top-three constraint directly to the interface and clarified what retrieval relevance means. In the implemented checkpoint, the public knowledge-store format requires a URL-based relevance proxy rather than exact passage labels; we state that limitation below rather than claiming exact passage-level Recall@3.

### Question 5 — Data

**Original.**

> Our primary dataset is AVeriTeC, which contains 4,568 real-world claims collected from 50 fact-checking organizations. Each claim includes an annotated verdict—Supported, Refuted, Not Enough Evidence, or Conflicting Evidence/Cherry-picking—along with evidence identified by human fact-checkers. AVeriTeC also provides a document collection containing candidate evidence, so we can evaluate both parts of our system.
>
> For evidence retrieval, we will test whether the system places the annotated relevant evidence within the top 3 results using Recall@3 and MRR. For verdict classification, we will compare the predicted verdict with the human fact-check label using Macro-F1, accuracy, and per-class precision/recall. We can also evaluate the complete pipeline by requiring both relevant evidence retrieval and the correct verdict.
>
> The dataset is publicly available now. We will use its provided train/development/test splits. FEVER is our fallback or auxiliary dataset because it contains a much larger collection of claims with evidence. If end-to-end retrieval is initially too difficult, we can first train and evaluate the classifier using AVeriTeC’s annotated evidence, then add retrieval.

**Revision.**

> Our system has two ML tasks: (1) evidence retrieval, which retrieves relevant evidence for a claim, and (2) verdict classification, which uses the claim together with the retrieved evidence to predict Supported, Refuted, Not Enough Evidence, or Conflicting Evidence.
>
> Our primary dataset is AVeriTeC, with 4,568 real-world claims from 50 fact-checking organizations. It provides both claim-level verdict labels and a candidate evidence collection with human-annotated relevant evidence, so it supports evaluation of both tasks. We will evaluate retrieval using Recall@3/MRR and classification using Macro-F1 and per-class precision/recall.
>
> We will also use the Snopes subset of DisinfoMM as a secondary dataset for external classification/generalization testing. DisinfoMM provides Snopes claims, verdict labels, and cited source URLs. However, unlike AVeriTeC, it does not provide a gold passage-level retrieval candidate set with relevance annotations. Therefore, we can use it to test whether our classifier generalizes to a different fact-checking source, but we cannot cleanly evaluate Recall@3 without manually annotating relevant evidence passages. The original Snopes evidence corpus also has access limitations, which is why DisinfoMM is the more practical secondary source.
>
> AVeriTeC is publicly available now and will remain our main benchmark. If retrieval becomes difficult, we can first evaluate classification using AVeriTeC’s gold evidence and then add retrieval.

**What changed.** We made the retrieval/classification distinction explicit and added a feasible secondary dataset with the proper scope: DisinfoMM/Snopes can assess external classification generalization, not clean passage-retrieval quality without additional annotation.

### Question 7 — How would we know it worked?

**Original.**

> Offline, we will evaluate evidence retrieval using Recall@3 and MRR, and verdict classification using Macro-F1 and accuracy. We will also measure end-to-end performance: whether the system retrieves useful evidence and reaches the correct verdict. To identify where gains come from, we will compare claim-only classification, BM25-retrieved evidence, semantically retrieved evidence, and gold annotated evidence.
>
> If deployed, at 3 months we would look for improved user accuracy without a substantial increase in decision time. At 6 months, we would test whether the effect generalizes to newer claims and additional users. At 9 months, we would audit performance and overreliance across topics, sources, and verdict classes. At 12 months, we would evaluate whether improvements in accuracy and calibration persist with repeated use.
>
> Prospectively, we will test with approximately 20–30 students using unfamiliar claims. Participants will receive different claims under tool-assisted and unassisted conditions, counterbalanced across participants. We will compare assessment accuracy, decision time, confidence calibration, perceived usefulness, and overreliance when the system gives an incorrect verdict.

**Revision.**

> The system works if users become less likely to trust or share refuted or unsupported claims, while still correctly accepting supported claims. Our primary real-world evaluation would measure whether users make better trust decisions after seeing the system’s evidence and verdict. We would compare assisted vs. unassisted users on claim-assessment accuracy, willingness to trust/share the claim, decision time, confidence, and overreliance on incorrect system outputs.
>
> We would also analyze asymmetric errors. A false reassurance error—showing a misleading or unsupported claim as Supported—is especially harmful because it may increase trust in misinformation. A false alarm—flagging a supported claim as questionable—is also harmful because it may reduce trust in legitimate information. We will report both error types separately rather than relying only on overall accuracy.
>
> Offline ML metrics are proxies for this user outcome. Retrieval will be evaluated with Recall@3/MRR, while verdict prediction will use Macro-F1, per-class precision/recall, and calibration. We will also measure end-to-end performance: whether the retrieved evidence leads to the correct verdict. The interface will show the predicted verdict, its confidence, and the top three evidence items rather than only a ranked list of class labels.
>
> Ideally, with more resources, we would run a larger longitudinal study with diverse users. At 3 months we would measure trust decisions and false reassurance; at 6 months generalization to newer claims; at 9 months topic/source disparities and overreliance; and at 12 months whether improved judgment persists with repeated use.

**What changed.** We made the ultimate evaluation behavioral rather than only predictive, connected offline measures to that outcome as proxies, and added separate safety reporting for false reassurance and false alarms.

### Question 8 — Baselines

**Original.**

> The non-ML baseline is keyword/BM25 search. Given a claim, the system retrieves the top 3 keyword-matching evidence items but does not predict a verdict. The user reads the evidence and decides whether the claim is supported, refuted, conflicting, or lacks sufficient evidence.
>
> Our ML system adds semantic retrieval, which can find relevant evidence even when the wording differs, and an evidence-aware classifier that provides a preliminary evidence status to support the user’s final decision.

**Revision.**

> Our non-ML baseline is keyword/BM25 search followed by human judgment. Given a claim, BM25 retrieves the top 3 evidence passages from AVeriTeC’s candidate pool, and the user reads them and decides whether the claim is Supported, Refuted, Not Enough Evidence, or Conflicting Evidence. This approximates the current practice of manually searching for relevant information and deciding from the sources found.
>
> We will evaluate retrieval against AVeriTeC’s human-annotated gold evidence, not simply whether the returned passages look reasonable. The user’s final judgment will be compared with the dataset’s gold claim label. Therefore, if BM25 misses important evidence and the user chooses “Not Enough Evidence” for a claim that is actually Refuted, that is still counted as an error.
>
> Our ML system adds semantic retrieval and evidence-aware classification to improve both evidence discovery and the final claim assessment.
>
> Our simple ML baseline will use BM25 to retrieve the top 3 evidence items, followed by TF-IDF + multinomial logistic regression on the claim and retrieved evidence to predict Supported, Refuted, Insufficient Evidence, or Conflicting Evidence. Our main system will replace BM25 with dense semantic retrieval and use an evidence-conditioned transformer classifier. We aim to improve end-to-end Macro-F1 by at least 5 percentage points over the simple ML baseline and improve users’ claim-assessment accuracy without substantially increasing decision time.
>
> We will also test a claim-only version to measure how much evidence helps. As a stronger modern baseline, we will give a general-purpose LLM the same retrieved evidence and prompt it to output one of the four verdict labels.

**What changed.** We made BM25 a realistic human-search baseline, distinguished retrieval correctness from verdict correctness, and added a zero-shot LLM baseline receiving the same evidence budget. The checkpoint implements BM25 and the zero-shot LLM; the BM25+logistic end-to-end experiment remains a planned next baseline because producing train-set BM25 rankings is computationally expensive.

## 2. Problem and approach

People encounter factual claims online but often lack the time to locate and compare reliable evidence. Our assistant accepts one claim, retrieves at most three evidence passages, and gives a preliminary four-way evidence status: Supported, Refuted, Not Enough Evidence, or Conflicting Evidence. It is a decision-support tool, not an autonomous fact-checker: the user reads the retrieved passages and makes the final trust or sharing decision.

The top-three display is a product constraint. It keeps the interface realistic for limited user attention while forcing retrieval and classification to operate under the same evidence budget. The proposed later system will replace lexical retrieval with dense semantic retrieval and use an evidence-conditioned classifier; this checkpoint establishes the models that later work must beat.

## 3. Evaluation setup

### Dataset and split

Our primary dataset is AVeriTeC, containing 4,568 real-world fact-checking claims from 50 organizations. We use the official split:

- **Train:** 3,068 records; 3,067 nonempty claims are usable for supervised TF-IDF/logistic-regression runs.
- **Development:** 500 claims, used for every baseline table in this report.
- **Test:** untouched for the checkpoint.

The official development split represents evaluating a system on claims it did not train on. It is held fixed across baselines, preventing apparent improvements that actually arise from changing the evaluated claim set.

### Retrieval evaluation under a top-three constraint

For each dev claim, BM25 ranks candidate evidence passages within that claim’s AVeriTeC candidate pool. We retain 20 results for auditing but compute the user-facing metric at **k = 3**, because the UI will show no more than three passages. Ties are deterministically broken by descending BM25 score and then passage ID.

The knowledge-store release does not directly give exact human passage IDs for every chunk we construct. We therefore normalize source URLs and label a chunk relevant when its source URL matches an AVeriTeC annotated evidence-source URL. We call this **URL-proxy evidence hit@3**: whether at least one of the displayed three passages is from an annotated source. We also report MRR using the same qrels.

Of 500 dev claims, 386 have one or more matched URL-proxy judgments. The remaining 114 are **unjudged**, not assumed to be retrieval failures. This makes the retrieval metric a transparent first-pass proxy rather than a claim of exact human passage-level Recall@3.

### Classification and safety metrics

We evaluate every classifier against all 500 AVeriTeC dev verdict labels. The main classification metric is **macro-F1**, which weights each of the four labels equally despite the predominance of Refuted examples. We also report accuracy and per-class precision/recall/F1.

Because the project’s goal is user safety rather than only aggregate accuracy, we report:

- **False reassurance rate:** a Refuted, Not Enough Evidence, or Conflicting Evidence claim predicted as Supported.
- **False alarm rate:** a Supported claim predicted as any other label.

For logistic regression, we additionally calculate multiclass Brier score and expected calibration error (ECE). Qwen’s reported confidence is a self-assessment rather than a calibrated probability, so it is excluded from calibration metrics.

## 4. Implemented baselines

### A. Non-ML baseline: BM25 keyword search

BM25 is the non-ML baseline. Given a claim, it performs keyword-based search only within that claim’s candidate evidence pool and returns the three highest-scoring passages. This approximates the accessible current practice of searching for information and having a person judge the sources. It is not a strawman: keyword search is cheap, interpretable, deterministic, and often difficult to beat.

We use the current **source-document v2 corpus**: sentence-level extracted records are first joined into their source document and then chunked in 160-word windows with 40-word overlap. The corpus has 146,307 documents and 16,050,740 passages. The candidate pool remains claim-specific.

### B. Zero-shot LLM baseline: Qwen3-4B with BM25 top 3

The required LLM baseline uses `Qwen/Qwen3-4B` (exact Hugging Face revision `1cfa9a7208912126459214e8b04321603b3df60c`) in local 4-bit Transformers inference on a Colab GPU. For every dev claim, the LLM receives exactly the same three BM25 v2 passages that a user would see. It must choose one of the four labels and cite the supplied passage IDs. This makes it a stronger, contemporary baseline while holding the evidence budget fixed.

Generation is deterministic (`do_sample=false`), with seed 718, thinking disabled, and a maximum of 256 new tokens. Each passage is capped at 1,200 characters. The frozen system prompt is:

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

### C. Diagnostic baselines

We also ran diagnostic variants that help explain the retrieval/classification gap but should not be presented as deployable end-to-end systems:

- **Claim-only Qwen:** receives no evidence. It measures how much the LLM can infer from claim wording or label priors, not fact verification.
- **Gold-evidence Qwen:** receives up to three human-annotated AVeriTeC evidence answers. It is an oracle condition measuring the ceiling available to the same LLM if evidence were readily available.
- **Claim-only and gold-evidence TF-IDF + logistic regression:** supervised, class-balanced multinomial logistic regression using word TF-IDF unigrams/bigrams. They train on the 3,067 usable official train claims and evaluate on the 500 dev claims. The gold condition is an isolated verifier experiment, not an end-to-end retrieval result.

The originally proposed simple ML pipeline, TF-IDF/logistic regression over **BM25-retrieved** train and dev evidence, is not yet complete. Building fair train BM25 rankings requires processing the large official training knowledge-store archives. We do not substitute dev evidence for train evidence or report an incomplete result as if it were a valid end-to-end experiment.

## 5. First-pass results

### Retrieval: BM25 v2

| System | Dev claims | Qrel-judged claims | User-facing k | URL-proxy evidence hit@3 | MRR | URL-proxy precision@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25, source-document v2 | 500 | 386 | 3 | **0.1969** | **0.1835** | 0.1002 |

The current BM25 result was generated over 16,050,740 candidate passages in about 57 minutes. One dev claim had no candidate pool and was logged explicitly.

For an engineering comparison only, the earlier sentence-level v1 retrieval produced hit@3 = 0.1658 and MRR = 0.1438. The source-document corpus therefore improves the URL-proxy hit rate by 3.11 percentage points and MRR by 3.97 points, while reducing the number of passages by 88.4%. Since the chunking changes the number of relevant passages, we do not make a strong cross-version claim based on passage recall.

### Classification and end-to-end verdict results

| System | Evidence at prediction time | Macro-F1 | Accuracy | False reassurance | False alarm | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-4B | BM25 v2 top 3 | **0.3346** | **0.496** | **0.220** (110) | **0.058** (29) | Primary zero-shot end-to-end baseline |
| Qwen3-4B | Claim only | 0.2538 | 0.372 | 0.064 (32) | 0.194 (97) | No-evidence diagnostic |
| Qwen3-4B | Up to 3 gold evidence answers | 0.4228 | 0.554 | 0.052 (26) | 0.082 (41) | Oracle diagnostic |
| TF-IDF + logistic regression | Claim only | 0.3940 | 0.616 | 0.114 (57) | 0.160 (80) | Supervised no-evidence diagnostic |
| TF-IDF + logistic regression | Up to 3 gold evidence answers | 0.4587 | 0.650 | 0.100 (50) | 0.138 (69) | Oracle diagnostic |

The Qwen BM25 v2 classifier had per-class F1 of 0.5723 for Supported, 0.5955 for Refuted, 0.1250 for Not Enough Evidence, and 0.0455 for Conflicting Evidence. Thus, its nearly 0.50 accuracy should not obscure weak performance on the less frequent but important uncertainty/conflict labels.

The claim-only logistic baseline has a higher accuracy than Qwen+BM25, largely because Refuted is common in this dev split and supervised text features can exploit claim wording. It is not evidence that a no-evidence model is a good verifier. Its false alarm rate is also 16.0%. The gold-evidence oracle conditions instead show whether the verifiers could benefit from evidence: for Qwen, gold evidence increases macro-F1 by 0.0883 and accuracy by 0.058 compared with BM25 top 3; for logistic regression, it increases macro-F1 by 0.0647 and accuracy by 0.034 compared with claim only.

### Retrieval-conditioned diagnostic

For the Qwen+BM25 baseline, we also grouped the 386 qrel-judged claims according to whether BM25 retrieved a URL-proxy-relevant source within its top three.

| Group | Claims | Macro-F1 | Accuracy | False reassurance | False alarm |
| --- | ---: | ---: | ---: | ---: | ---: |
| URL-proxy evidence covered | 76 | 0.2491 | 0.4605 | 0.3289 | 0.0526 |
| URL-proxy evidence missed | 310 | 0.3787 | 0.5419 | 0.1903 | 0.0484 |

This is a diagnostic, **not causal attribution**. The URL proxy is incomplete; matching a cited source does not prove that the displayed chunk resolves the claim; the groups may have different label and difficulty distributions; and LLMs may not use their supplied evidence perfectly. We will not interpret the table as proof that retrieval is harmful. Instead, it motivates manual relevance checks and better retrieval/evidence-use evaluation.

## 6. What the results tell us

The baselines are meaningful and nontrivial. BM25 retrieval is a realistic current-practice comparator, while the Qwen3-4B verifier is a modern zero-shot baseline using the same three-passage budget as our eventual product. The source-document v2 engineering change produced a measurable retrieval improvement and made the full dev BM25 run more manageable.

At the same time, the deployable end-to-end baseline is not safe enough for autonomous endorsement: Qwen+BM25 falsely reassures users about 22.0% of dev claims. Its poor F1 on Not Enough Evidence and Conflicting Evidence also confirms that this is not merely a high-accuracy classification problem. The gap between BM25+Qwen and gold-evidence Qwen suggests real headroom for retrieval and evidence selection, but the URL-proxy qrels and diagnostic conditioning mean we cannot yet attribute that gap precisely to retrieval alone.

For a later model to be worth deploying, it should improve end-to-end macro-F1 over Qwen+BM25 while materially reducing false reassurance from 22%, without shifting the burden into excessive false alarms or requiring more than three displayed evidence items. We will judge it on the same dev split, evidence budget, and safety metrics. A user study would ultimately be needed to determine whether the system actually reduces willingness to trust or share unsupported claims without reducing appropriate trust in supported claims.

## 7. Next steps

1. Implement dense semantic retrieval and compare it with BM25 under the same candidate pool, top-three display limit, URL-proxy hit@3, and MRR.
2. Train an evidence-conditioned classifier using retrieved evidence; report the same macro-F1, per-class metrics, false reassurance, and false alarm rates.
3. Conduct a small manual audit of URL-proxy matches to estimate how well the retrieval proxy corresponds to decisive passage relevance.
4. Finish the fair train-BM25/logistic baseline only if computational budget permits. The current archive-streaming process is resumable but several hours per archive shard; it should not delay the checkpoint submission.
5. Add the Snopes subset of DisinfoMM as an external **classification** test, clearly separating it from AVeriTeC retrieval evaluation because it lacks compatible passage-level qrels.
6. In a future user study, compare assisted and unassisted participants on trust/share decisions, verdict accuracy, decision time, confidence, and overreliance on wrong system outputs.

## 8. Reproducibility and artifacts

The repository contains the scripts and Colab instructions needed to reproduce the reported runs. Major results are saved in Google Drive under:

```text
/content/drive/MyDrive/10718_averitec/
├── outputs/reports/final_baseline_summary/
│   ├── baseline_summary.md
│   └── baseline_summary.csv
└── v2_source_documents/outputs/
    ├── bm25/dev_bm25_top20_v2/
    │   ├── dev_metrics.json
    │   └── dev_rankings.jsonl
    ├── local_llm/dev_qwen3_4b_bm25_top3_v2/
    │   ├── classify_top3_config.json
    │   ├── classify_top3_metrics.json
    │   └── classify_top3_retrieval_conditioned_metrics.json
    └── tfidf_logreg/
        ├── claim_only_train_to_dev_v2/claim_only_train_to_dev_metrics.json
        └── gold_train_to_dev_v2/gold_train_to_dev_metrics.json
```

The final submission should include the public GitHub repository link and ensure instructor/TA access before the deadline.
