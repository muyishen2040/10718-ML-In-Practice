# Colab workflow

Use Colab for the large AVeriTeC knowledge-store archive and API-backed LLM
experiments. Keep source code in Git; keep datasets, Drive mounts, API keys,
and generated outputs out of Git.

## 1. Bootstrap

```python
!git clone https://github.com/muyishen2040/10718-ML-In-Practice.git
%cd 10718-ML-In-Practice
!pip install -r requirements-colab.txt
```

Either use Colab's local disk or mount Drive, then set a single data root:

```python
from pathlib import Path
DATA_ROOT = Path("/content/data")  # or Path("/content/drive/MyDrive/averitec_data")
```

## Resource gate

Normalization and passage chunking stream to disk. For the official AVeriTeC
layout, BM25 indexes one claim's own candidate pool at a time, so the full dev
run does not require a global corpus in RAM. First confirm every stage with a
small, explicitly non-reportable smoke test. The three train archives total
roughly 64 GB compressed, so the end-to-end train/dev verifier needs ample
disk (for example, Drive or a high-capacity runtime). Never describe a
capped-corpus score as the AVeriTeC baseline.

## 2. Prepare the compact claim data

```python
!python scripts/prepare_averitec.py --download-claims --data-root "{DATA_ROOT}"
```

This writes canonical train/dev records, hashes source files, audits the known
empty training claim, and checks exact normalized cross-split overlaps.

## 3. Inspect the official knowledge store before parsing it

For the initial BM25 result, use the development knowledge store rather than
downloading all training archives. The download command requires an explicit
confirmation because the archive is multi-GB:

```python
ARCHIVE = DATA_ROOT / "raw/averitec/dev_knowledge_store.zip"
!python scripts/download_averitec_knowledge_store.py --archive dev --output "{ARCHIVE}" --confirm-large-download
```

```python
ARCHIVE = DATA_ROOT / "raw/averitec/dev_knowledge_store.zip"
!python scripts/inspect_averitec_knowledge_store.py --archive "{ARCHIVE}" --output "{DATA_ROOT}/processed/averitec/dev_knowledge_store_layout.json"
```

Normalize recognized JSON, JSONL, TSV, or CSV URL/text records directly from
the archive. Review its audit and stop if the output does not recognize the
expected documents; do not substitute human gold answers as documents.

```python
!python scripts/normalize_averitec_knowledge_store.py \
  --archive "{ARCHIVE}" \
  --output-documents "{DATA_ROOT}/raw/averitec/dev_documents.jsonl"
```

For a smoke test only, add `--max-documents 10000` and write to a distinct
file such as `dev_documents_smoke.jsonl`. The resulting corpus is incomplete
and may be used only to validate the pipeline, never for the checkpoint table.

## 4. Build passages and URL-based relevance coverage

Once a normalized document file exists, build the per-claim candidate passages
and qrels. The official `output_dev/<claim-index>.json` format is recognized
directly; its candidate-pool boundary is retained:

```python
!python scripts/build_averitec_passage_corpus.py \
  --documents-jsonl "{DATA_ROOT}/raw/averitec/dev_documents.jsonl" \
  --split dev --data-root "{DATA_ROOT}" --max-words 160 --overlap-words 40
```

The generated qrels are an **annotated-source-URL proxy**: passages from a
human-annotated evidence URL are marked relevant. Call the resulting metric
`evidence_hit_at_3` / URL-based evidence coverage, not exact human
passage-level recall. The corpus audit reports unresolved annotated URLs.

## 5. Run retrieval and classifier baselines

```python
!python scripts/run_bm25.py --split dev --data-root "{DATA_ROOT}" --ranking-k 20 --metric-k 3 --resume \
  --output-dir outputs/bm25 --run-name dev_bm25_top20_v1
```

This writes 20 deterministic candidates per claim for the LLM reranking
baseline but evaluates the user-facing first three only. The TF-IDF verifier
and `classify_top3` LLM variant consume its first three saved passages.

The runner checkpoints every 10 completed claims in
`dev_rankings.partial.jsonl`. If Colab disconnects or you stop the cell, rerun
the exact command with `--resume`; completed claims are reused. The partial
file is removed automatically only after the final rankings and metrics have
been written.

For a fair BM25-evidence verifier, prepare the train and dev candidate-document
collections separately, using identical chunking/BM25 settings. The official
training store is split into three large archives; download, inspect, normalize,
and combine their resulting JSONL files in original claim-index order before
building the train corpus. Generate train rankings against the train collection
and dev rankings against the dev collection, then pass both saved rankings to
the classifier:

```python
for archive_name in ("train_0_999", "train_1000_1999", "train_2000_3067"):
    archive_path = DATA_ROOT / "raw/averitec" / f"{archive_name}.zip"
    !python scripts/download_averitec_knowledge_store.py --archive {archive_name} --output "{archive_path}" --confirm-large-download
    !python scripts/normalize_averitec_knowledge_store.py --archive "{archive_path}" --output-documents "{DATA_ROOT}/raw/averitec/{archive_name}.jsonl"
```

Combine the three normalized JSONL files into `train_documents.jsonl` in the
same order before calling `build_averitec_passage_corpus.py --split train`:

```python
!python scripts/concat_jsonl.py \
  --input "{DATA_ROOT}/raw/averitec/train_0_999.jsonl" \
  --input "{DATA_ROOT}/raw/averitec/train_1000_1999.jsonl" \
  --input "{DATA_ROOT}/raw/averitec/train_2000_3067.jsonl" \
  --output "{DATA_ROOT}/raw/averitec/train_documents.jsonl"
```

```python
!python scripts/build_averitec_passage_corpus.py \
  --documents-jsonl "{DATA_ROOT}/raw/averitec/train_documents.jsonl" \
  --split train --data-root "{DATA_ROOT}" --max-words 160 --overlap-words 40
```

```python
!python scripts/run_bm25.py --split train --data-root "{DATA_ROOT}" --ranking-k 20 --metric-k 3 \
  --output-dir outputs/bm25 --run-name train_bm25_top20_v1

!python scripts/train_tfidf_logreg.py --evidence-mode retrieved \
  --data-root "{DATA_ROOT}" \
  --train-rankings outputs/bm25/train_bm25_top20_v1/train_rankings.jsonl \
  --eval-rankings outputs/bm25/dev_bm25_top20_v1/dev_rankings.jsonl \
  --output-dir outputs/tfidf_logreg --run-name retrieved_train_to_dev_v1
```

The claim-only and gold modes are diagnostics. `gold` is capped at three
annotated answers; `gold_all` is an explicitly non-deployable oracle bound.

## 6. Run a zero-shot LLM baseline

### Recommended: local Qwen3 on the Colab GPU

This is the recommended checkpoint LLM baseline: it uses the open
`Qwen/Qwen3-4B` model, loaded in 4-bit mode on a Colab T4/L4 GPU. It uses the
same frozen evidence-only prompt and BM25 top three as the hosted runner, but
does not require an API key. Select a GPU in **Runtime > Change runtime type**,
then install the local-inference extras:

```python
!pip install -q -r requirements-local-llm-colab.txt
!nvidia-smi
```

Run a 20-claim pilot first. The raw output cache is appended after every claim,
so rerunning this exact command resumes successful calls after a disconnect:

```python
!python scripts/run_local_llm_baseline.py \
  --claims "{DATA_ROOT}/processed/averitec/dev.jsonl" \
  --rankings outputs/bm25/dev_bm25_top20_v1/dev_rankings.jsonl \
  --qrels "{DATA_ROOT}/processed/averitec/dev_qrels.jsonl" \
  --model Qwen/Qwen3-4B --variant classify_top3 --max-claims 20 \
  --output-dir outputs/local_llm --run-name dev_qwen3_4b_top3_v1
```

After checking the pilot outputs, rerun without `--max-claims` using the same
run name. The configuration records the model ID, resolved model revision,
prompt hash, 4-bit inference mode, disabled thinking mode, and generation
settings. Keep the same model, prompt, and settings for the full run.

The same local runner supports all three verifier comparisons. Use a distinct
run name for each mode. All runs retain one completed claim at a time and can
therefore be restarted with the exact same command after a Colab interruption.

**1. Claim-only diagnostic (no evidence):**

```python
!python scripts/run_local_llm_baseline.py \
  --claims "{DATA_ROOT}/processed/averitec/dev.jsonl" \
  --model Qwen/Qwen3-4B --evidence-mode claim_only --max-claims 20 \
  --output-dir outputs/local_llm --run-name dev_qwen3_4b_claim_only_v1
```

**2. Primary end-to-end LLM baseline (BM25 top 3):**

```python
!python scripts/run_local_llm_baseline.py \
  --claims "{DATA_ROOT}/processed/averitec/dev.jsonl" \
  --rankings outputs/bm25/dev_bm25_top20_v1/dev_rankings.jsonl \
  --qrels "{DATA_ROOT}/processed/averitec/dev_qrels.jsonl" \
  --model Qwen/Qwen3-4B --evidence-mode bm25_top3 --max-claims 20 \
  --output-dir outputs/local_llm --run-name dev_qwen3_4b_top3_v1
```

**3. Gold-evidence oracle diagnostic:**

```python
!python scripts/run_local_llm_baseline.py \
  --claims "{DATA_ROOT}/processed/averitec/dev.jsonl" \
  --model Qwen/Qwen3-4B --evidence-mode gold_top3 --max-evidence 3 --max-claims 20 \
  --output-dir outputs/local_llm --run-name dev_qwen3_4b_gold_top3_v1
```

`gold_top3` gives Qwen up to three human-annotated answers and is an oracle
verifier diagnostic only. It is **not** an end-to-end or deployable result;
compare it with BM25 top-3 to estimate the cost of retrieval misses. Claims
without annotated answers are excluded and counted in that run's configuration.

### Hosted API alternative

Use a Colab secret or `getpass`; never place the key in a notebook cell that is
committed to Git.

```python
import os
from getpass import getpass
os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key: ")
```

Start with a small cost-controlled pilot and record an exact model identifier:

```python
!python scripts/run_llm_baseline.py \
  --claims "{DATA_ROOT}/processed/averitec/dev.jsonl" \
  --rankings outputs/bm25/dev_bm25_top20_v1/dev_rankings.jsonl \
  --qrels "{DATA_ROOT}/processed/averitec/dev_qrels.jsonl" \
  --model YOUR_EXACT_MODEL_ID --variant classify_top3 --max-claims 20 \
  --output-dir outputs/llm --run-name dev_llm_top3_v1
```

After verifying the API integration without changing the frozen prompt, rerun
the same named folder without `--max-claims`; successful pilot calls are
reused from its cache:

```python
!python scripts/run_llm_baseline.py \
  --claims "{DATA_ROOT}/processed/averitec/dev.jsonl" \
  --rankings outputs/bm25/dev_bm25_top20_v1/dev_rankings.jsonl \
  --qrels "{DATA_ROOT}/processed/averitec/dev_qrels.jsonl" \
  --model YOUR_EXACT_MODEL_ID --variant classify_top3 \
  --output-dir outputs/llm --run-name dev_llm_top3_v1
```

The stronger second variant uses BM25 top 20, asks the LLM to select exactly
three passages, then predicts the verdict:

```python
!python scripts/run_llm_baseline.py \
  --claims "{DATA_ROOT}/processed/averitec/dev.jsonl" \
  --rankings outputs/bm25/dev_bm25_top20_v1/dev_rankings.jsonl \
  --qrels "{DATA_ROOT}/processed/averitec/dev_qrels.jsonl" \
  --model YOUR_EXACT_MODEL_ID --variant rerank_top20_and_classify \
  --output-dir outputs/llm --run-name dev_llm_rerank_top20_v1
```

The runner stores the exact prompt, structured raw response, token usage,
selected passage IDs, model ID, and errors. It does not treat LLM
self-reported confidence as calibrated probabilities.

## 7. Generate the writeup table

```python
!python scripts/generate_baseline_report.py \
  --run "bm25=outputs/bm25/dev_bm25_top20_v1/dev_metrics.json" \
  --run "llm_rerank=outputs/llm/dev_llm_rerank_top20_v1/rerank_top20_and_classify_metrics.json,outputs/llm/dev_llm_rerank_top20_v1/rerank_top20_and_classify_config.json" \
  --output-dir outputs/reports
```

## 8. Files to share for result analysis

Each `--run-name` creates a separate directory. Download or send the relevant
run directory (not the data archive). In particular, include the following:

- BM25: `*_metrics.json`, `*_rankings.jsonl`, and `*_manifest.json`.
- TF-IDF/logistic regression: `*_metrics.json`, `*_predictions.jsonl`,
  `*_config.json`, `*_manifest.json`, `*_evidence_coverage.jsonl`, and
  `*_retrieval_conditioned_metrics.json`.
- LLM: the same prediction/metric/config/manifest files plus `*_errors.jsonl`;
  include `*_raw.jsonl` only if you are comfortable sharing the model outputs.

The coverage file labels each claim as `evidence_covered`, `evidence_missed`,
or `unjudged`. The conditioned metrics compare verdict quality for the first
two groups; treat the comparison as a diagnostic, not proof that retrieval
caused every classification outcome.
