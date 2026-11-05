# Teammate and agent handoff

Updated November 5, 2026. This guide assumes no access to earlier conversations.
Three teammates own the workstreams below; the project coordinator integrates
their changes and the final report. Names can replace the A/B/C placeholders.
The current assignment is A: browser extension/interface; B: local ML method;
C: hosted API models. The coordinator owns shared evaluation and integration.
The coordinator is preparing the time-based split release and will publish its
path when complete. See [EVALUATION.md](EVALUATION.md) and the pending
[release pointer](../configs/team_evaluation.json); do not infer that those
splits or a new evaluation command already exist.

Root cleanup: current onboarding starts from `README.md` and `AGENTS.md`.
Dependency profiles live in `requirements/`; earlier specifications/runbooks
live in `docs/archive/`. The duplicate root guides were removed. Private draft
notes were preserved in the coordinator's ignored `local/` directory and are
not teammate onboarding sources.

## Start here

We are building a misinformation screening assistant. A user enters a factual
claim; the system retrieves up to three passages, predicts Supported, Refuted,
Not Enough Evidence, or Conflicting Evidence, and lets the user decide whether
to trust or share it. The long-term outcome is better user decisions. Our current
experiments measure retrieval and verdict prediction offline.

The baseline and Initial ML Solution assignments are complete. A stronger
classifier and several retrieval/reading changes have produced modest gains,
but no reliably large improvement. The current work is to understand which
evidence actually resolves a claim and to select models that generalize to later
claims. The product is not ready for autonomous endorsement.

Read these sources in order:

1. [Submitted Implement and Evaluate Baselines report](https://docs.google.com/document/d/1Mg5oxNHiU_7iHtqfiym8EL19Eb0xX3ViJKu4Ium9ufE/edit?usp=sharing):
   the original evaluation setup and baseline results.
2. [Submitted Initial ML Solution report](https://docs.google.com/document/d/1foHMotcyZf8koHD3wpsCdVbcg45mAVq15SVNWQkEYeA/edit):
   the follow-up ML experiments and comparison with the baselines.
3. [Progress and later results](TEAM_PROGRESS.md). The Initial ML Solution report
   does not include the later NLI-reader, class-weighting, QA, and refit-diagnostic
   experiments.
4. [Workstream assignments and deliverables](TEAM_WORKPLAN.md).

The progress document includes the key report results and later findings. If
report access is unavailable, use that local summary to begin setup and ask the
coordinator for document access.

Then read the assigned task brief: [A: interface](tasks/TEAMMATE_A_INTERFACE.md),
[B: main method](tasks/TEAMMATE_B_METHOD.md), or
[C: API models](tasks/TEAMMATE_C_API.md). Use
[the documentation map](README.md) to locate shared contracts and older runbooks.

Source code: <https://github.com/muyishen2040/10718-ML-In-Practice>

Shared artifacts: [10718_averitec](https://drive.google.com/drive/folders/19tuNxxB1y9arSmLnrQcxRuPvDQBf6YeK).
The active data directory is
[v2_source_documents](https://drive.google.com/drive/folders/1CTzmm4C9Qb5HodTHfWPejQk2FQhqCkXC).

## First session: choose your role and check readiness

Give a new agent an explicit assignment, for example:

```text
You are Teammate B, responsible for the local ML method. Read AGENTS.md,
docs/TEAM_HANDOFF.md, docs/TEAM_PROGRESS.md and docs/tasks/TEAMMATE_B_METHOD.md.
Work from this checkout and use the coordinator's evaluation release.
Start with the first deliverable in your brief. Do not assume earlier chat access.
```

Replace B and the task-brief path for A or C. If no role is assigned, check setup
and ask the coordinator which workstream to own before editing shared code.

In the first session, record the Git branch/commit and any local changes, verify
the shared Drive paths using the cell below, and choose a personal output folder.
Check `configs/team_evaluation.json` before training or prompt selection.

| Owner | Work that can start now | Dependency for the next stage |
| --- | --- | --- |
| A: interface | Mock claim-entry/results flow and proposed backend contract | A real adapter from B/C; a retrieval implementation for arbitrary live claims |
| B: local method | Inspect saved predictions/configurations and reader code; prepare one bounded change | Published time-based split and runner support before new comparative fitting |
| C: API models | Provider mocks, parsing, per-call caching and resume behavior | Published split before paid prompt pilots or model selection |
| Coordinator | Arrange Drive access and finish the split manifest and scorer | Publish the evaluation release path/hash and confirm the command with B/C |

End setup with a short update: role, code revision, paths checked, first
deliverable and missing dependencies. If access fails, report the exact missing
path or permission; the shared data does not need to be downloaded again.

## Git publication status: check this before cloning for new experiments

Git remote and local checkout verified November 5, 2026:

- The shared team branch is
  [main](https://github.com/muyishen2040/10718-ML-In-Practice/tree/main).
- The reader feature branch was merged into `main` through commit
  `a7a7b8371306c173fa99c6c550b45c2ac49f56d6`. Create new task branches from
  `origin/main`.
- Tested code and cleanup snapshot:
  `bd328349bc75b4aa1f0c714657f6a0ed48b16195` (106 tests passed). This commit
  includes the follow-up reader, notebooks and handoff documents. Later commits
  on the same branch may update documentation; record the exact revision you use.

The code and handoff are published. Drive access and the new temporal evaluation
release remain separate setup steps. A portable notebook contains its own source
snapshot and can run without a branch checkout, but its code should not be mixed
with imports from another notebook or checkout in the same Python session.

To inspect the remote and start a task branch in a fresh Colab checkout:

```python
!git ls-remote --heads https://github.com/muyishen2040/10718-ML-In-Practice.git
!git clone --branch main https://github.com/muyishen2040/10718-ML-In-Practice.git /content/10718-ML-In-Practice
%cd /content/10718-ML-In-Practice
!git log -1 --oneline
```

Run the clone cell only when that directory is absent. If it already exists,
inspect `git status` and the current branch first; preserve any local work before
switching. Create your own task branch from the published shared base:

```bash
git fetch origin
git switch --create team/teammate-a-interface origin/main
```

Use `team/teammate-b-method` and `team/teammate-c-api` for the other tasks.
Submit a small PR with the experiment setup, result links and limitations.
Avoid concurrent work on the same branch or output directory.

## Add shared data to your own Colab

1. Open the shared `10718_averitec` folder using your own Google account. Ask the
   coordinator for access if needed. Reading existing data needs Viewer access.
2. In Drive, select the folder and use **Organize > Add shortcut > My Drive**.
   Keep the shortcut name `10718_averitec`. This points to the existing files;
   it does not copy tens of GB into a second dataset.
3. In your own Colab session, mount your own Drive. A shared folder visible in
   Drive's website must also resolve through the shortcut after mounting.
4. Keep shared inputs and personal outputs as different paths. Most existing
   notebooks default to writing under the shared data root; change their `OUT`
   or output-directory setting before running them with read-only shared access.

```python
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')

SHARED_ROOT = Path('/content/drive/MyDrive/10718_averitec')
DATA_ROOT = SHARED_ROOT / 'v2_source_documents'
OWNER = 'teammate_a'  # use teammate_b or teammate_c as appropriate
OUT = Path('/content/drive/MyDrive/10718_runs') / OWNER / 'setup_check_v1'
OUT.mkdir(parents=True, exist_ok=True)

required = [
    'processed/averitec/train.jsonl',
    'processed/averitec/dev.jsonl',
    'processed/averitec/dev_qrels.jsonl',
    'outputs/bm25/train_bm25_top20_v2/train_rankings.jsonl',
    'outputs/bm25/dev_bm25_top20_v2/dev_rankings.jsonl',
]
missing = []
for relative in required:
    path = DATA_ROOT / relative
    present = path.is_file()
    print('FOUND' if present else 'MISSING', relative)
    if not present:
        missing.append(relative)
if missing:
    raise FileNotFoundError(
        'Check Drive access, shortcut name, and DATA_ROOT: ' + ', '.join(missing)
    )
print('Personal output directory:', OUT)
```

If a shortcut was added after mounting, reconnect/remount Drive and check again.
Do not redownload the dataset to work around a wrong path. Match the existing
source-document v2 inputs; the sibling `10718_averitec/processed`, `raw`, and
`outputs` directories contain earlier artifacts and are not interchangeable
with `v2_source_documents`.

Store new experiment checkpoints in a personal, uniquely named directory. To
exchange results, share the completed run folder with teammates and record its
stable Drive link in the Git update. A separate writable team-results folder
would make collection easier, but has not been created by this handoff. Active
training processes must not append to the same checkpoint or prediction journal.

## Data and saved retrieval artifacts

All paths below are relative to `DATA_ROOT`, unless explicitly stated otherwise.
Directory listings and the latest result JSONs were read directly from Drive on
November 4, 2026. Some deeper leaf paths are established by the run scripts and
earlier runs rather than a fresh file-content read; verify them before use.

| Artifact | Path | Use and verification |
| --- | --- | --- |
| Usable official train claims | `processed/averitec/train.jsonl` | Listed live; 7,033,809 bytes. Reported 3,067 usable rows after one empty official claim is excluded. Includes labels and annotated answers, which must be kept out of prediction inputs. |
| Official dev claims | `processed/averitec/dev.jsonl` | Listed live; 1,193,729 bytes. All 500 are used for primary verdict evaluation. |
| Dev URL-proxy judgments | `processed/averitec/dev_qrels.jsonl` | Listed live; 386 claims have matched relevance judgments. These are source-URL matches, not human decisive-passage labels. |
| Full dev candidate passages | `processed/averitec/dev_evidence_corpus.jsonl` | Listed live; 22,850,999,135 bytes, about 22.85 GB decimal. The report records 16,050,740 passages from 146,307 source documents. Usually unnecessary for training on saved rankings. |
| Corpus settings/provenance | `processed/averitec/dev_corpus_manifest.json`, `dev_corpus_audit.json` | Manifest read live: 160-word chunks, 40-word overlap. Old absolute runtime paths in manifests describe provenance; resolve current files through your own `DATA_ROOT`. |
| Training source archives | `raw/averitec/train_0_999.zip`, `train_1000_1999.zip`, `train_2000_3067.zip` | All three listed live, totaling about 63.5 GB decimal. A separate `.zip.partial` is incomplete and must not be used as the completed archive. |
| Train BM25 top 20 | `outputs/bm25/train_bm25_top20_v2/train_rankings.jsonl` | Existing train retrieval results; leaf path established by runs. Training can read the retrieved text directly. |
| Dev BM25 top 20 | `outputs/bm25/dev_bm25_top20_v2/dev_rankings.jsonl` | Existing dev rankings, same source-document v2 protocol. |
| Train BM25 top 100 | `outputs/initial_ml_solution_answer_aligned_top100_v1/bm25/train_bm25_top100_v2/train_rankings.jsonl` | Larger saved training candidate pool for evidence-selection experiments. Parent run directory listed live. |
| Dev BM25 top 100 | `outputs/initial_ml_solution_answer_aligned_top100_v1/bm25/dev_bm25_top100_v2/dev_rankings.jsonl` | Matching dev candidate pool; no full-corpus search is needed to rerank these passages. |
| Dev BGE-small top 20 | `outputs/initial_ml_solution_rerun_bs16/dense/dev_bge_small_v15/dev_rankings.jsonl` | Saved dense rankings; not an embedding index. No completed full-train dense rankings are established by the report. |
| Dev BM25+BGE fusion | `outputs/initial_ml_solution_rerun_bs16/hybrid/dev_rrf60/hybrid_rankings.jsonl` | Saved reciprocal-rank fusion, constant 60. |
| Original BM25-trained DistilBERT | `outputs/initial_ml_solution_rerun_bs16/transformer/bm25_top3/` | `final_model`, model selection, predictions and metrics. Source of the 0.3469 baseline. |
| Claim-only and gold-answer DistilBERT controls | `outputs/initial_ml_solution_followup_v1/transformer/{claim_only,gold_oracle}/` | Diagnostic controls, not retrieved-evidence systems. |
| Initial evidence-reader study | `outputs/evidence_aware_reader_v1/` | Parent listed live. Prepared raw/clean/NLI-selected rankings, cached NLI scores, trained pair readers and matched controls. |
| Cached reader inputs | `outputs/evidence_aware_reader_v1/prepared/{raw,clean,clean_nli}/{train,dev}_rankings.jsonl` | Reuse these for reader ablations. `prepared/{train,dev}.partial.jsonl` also stores scored candidate details. |
| Latest class-weighting and QA study | `outputs/evidence_reader_followup_v2/` | Parent and result files listed live; main result and selection JSONs read live. |
| Latest metrics | `outputs/evidence_reader_followup_v2/evidence_reader_results.json` | Full metrics for the latest experiment. |
| Recorded selected configuration | `outputs/evidence_reader_followup_v2/chosen_before_dev.json` | Train-holdout choice recorded before official-dev scoring. |
| Refit/generalization diagnostic | `outputs/evidence_reader_followup_v2/refit_diagnostic_v1/diagnostic_summary.json` | Read live. Compares the same selected checkpoint on holdout/dev with a separately refitted model on dev. |
| Questions and compact QA inputs | `outputs/evidence_reader_followup_v2/questions/`, `prepared/` | Include per-split questions, extraction reports, rankings and `prepared/train_fit_answer_audit.csv`. |

**What we mean by an "index":** the reusable objects verified here are candidate
corpus files, saved ranked passages, scored evidence, and model checkpoints. The
current dense runner embeds a claim's candidates in batches and saves top-ranked
passages; it does not establish that a reusable global FAISS/vector index was
saved. The inspected v2 `cache` directory contained a `huggingface` directory.
Do not assume model-download caches are a retrieval index. Verify and document
any additional vector index before building an experiment around it.

Saved ranking JSONL rows contain `claim_id`, `claim`, and a `retrieved` list with
passage IDs, source text, URLs and scores. They let a classifier train without
embedding or searching millions of passages again. They do not support arbitrary
new queries against the full corpus on their own. The full train passage corpus
need not exist as one combined file: previous BM25 runs streamed the three train
archives one claim pool at a time.

### Inspect the saved input format without running a model

JSONL means one JSON object per line. Normalized claim records have `claim_id`,
`claim`, `label`, `source_urls`, `evidence` and `metadata`. Claim IDs such as
`averitec:train:00000` are join keys; join by ID rather than row position. The
claim's `evidence` and `source_urls` fields contain gold annotations. Prediction
code should receive only the claim text and the chosen retrieved passages,
never the whole annotated record. See [the record schema](../src/data/schema.py)
and [validated evidence joins](../src/data/evidence.py).

After defining `DATA_ROOT` in the setup cell, this reads one training row from
each small input file and shows its field names. It does not load the passage
corpus, call a model or write shared files.

```python
import json

def first_jsonl_row(path):
    with path.open(encoding='utf-8') as handle:
        return next(json.loads(line) for line in handle if line.strip())

claim_row = first_jsonl_row(DATA_ROOT / 'processed/averitec/train.jsonl')
ranking_row = first_jsonl_row(
    DATA_ROOT / 'outputs/bm25/train_bm25_top20_v2/train_rankings.jsonl'
)
print('Claim fields:', sorted(claim_row))
print('Ranking fields:', sorted(ranking_row))
print('Example ranking claim ID:', ranking_row['claim_id'])
passages = ranking_row.get('retrieved') or []
print('Saved candidate count for this claim:', len(passages))
print('Passage fields:', sorted(passages[0]) if passages else [])
```

## Colab code and runtime choices

Choose one source route per fresh session:

- **Repository route:** checkout the verified published commit, work from the
  repository root, and confirm the required script exists before running it.
- **Portable-notebook route:** upload the supplied `.ipynb`, let its first cell
  restore the bundled source, and run in order. Do not import an older checkout's
  `src` modules into that session.

Published notebook references:

The two reader reference notebooks remain directly under `notebooks/`.
Older `initial_ml_solution_*` workflows are under `notebooks/archive/` and
preserved for intentional reproduction. See [the catalog](../notebooks/README.md)
and [script map](../scripts/README.md) before starting an expensive older run.

| Notebook | Purpose |
| --- | --- |
| `archive/initial_ml_solution_colab_rerun.ipynb` | Earlier dense retrieval, fusion and DistilBERT runs; expensive stages are already complete. |
| `archive/initial_ml_solution_fast_bge_verifier_swap_colab.ipynb` | Frozen BM25-trained verifier with BGE/fusion dev evidence. |
| `archive/initial_ml_solution_limitation_study_colab.ipynb` | Evidence diagnostics, auditing and paired analyses. |
| `evidence_aware_reader_colab.ipynb` | Initial NLI scoring, source cleaning and learned three-passage reader. |
| `evidence_reader_followup_colab.ipynb` | Corrected weighting, distillation ablation, QA packing and small-pool selection. Latest completed training study. |

These paths are available on the published team branch above. Baseline
`requirements/colab.txt`, local-LLM requirements and
`requirements/reader-colab.txt` serve different workflows. Use the
chosen notebook's pinned setup. The reader uses Transformers 4.57.1 and Colab's
CUDA-enabled PyTorch, and avoids importing TensorFlow or sentence-transformers.
For the repository route, install `requirements/reader-colab.txt` and set
`USE_TF=0` and `USE_FLAX=0` before importing Transformers. Portable notebooks do
this in their setup cells. Confirm `torch.cuda.is_available()` before GPU work;
a TensorFlow startup warning alone does not determine whether PyTorch has a GPU.
Restart the session when switching incompatible source/dependency snapshots.

CPU is sufficient for artifact checks, split construction, metric aggregation,
TF-IDF/logistic regression and BM25. Neural reader fitting and QA/NLI inference
use a GPU. Full dense retrieval over the dev corpus previously took roughly
12 hours; do not make it an onboarding step. Run a small smoke test first and
reuse finalized artifacts. Keep run settings and input hashes fixed when
resuming; a changed experiment needs a new output directory.

## Shared evaluation contract

- AVeriTeC has 3,068 official train records, 3,067 usable nonempty train claims,
  500 official dev claims, and 1,000 official test claims. Test is untouched.
- Official dev is later in time than train. Initial DistilBERT tuning used a
  307-claim stratified train holdout; newer readers used a 303-claim holdout
  grouped by normalized claim text. Neither internal holdout was chronological.
- Every primary verdict row uses the same 500 dev claims, including the claim
  with no evidence pool. Macro-F1 averages all four labels equally. Report
  accuracy, per-class metrics, confusion matrix, calibration when valid, and
  false reassurance/false alarm counts and rates.
- False reassurance means non-Supported gold predicted Supported. False alarm
  means Supported gold predicted another label. Historical rates divide by all
  500 claims; conditional rates require different, explicitly stated denominators.
- Retrieval Hit@3 uses the original 386 URL-judged claims. The other 114 are
  unjudged. Matching a cited URL does not prove the returned chunk is decisive.
- The report's MRR is over saved top-20 rankings. User-facing MRR@3 is a separate
  metric; recompute every comparator with the same cutoff before comparing it.
- A gold-answer verifier is a diagnostic condition, not an end-to-end system.
  Training-only annotations may supervise a new selector, but prediction inputs
  must be computable without the new claim's gold label, gold answers or qrels.
- Selection is recorded using training-only validation before dev prediction.
  Dev has been inspected repeatedly, so later dev results are exploratory.
  Preserve the recorded choice even when another ablation later scores higher
  on dev; do not retroactively describe that as train-only selection.

## Where the project is stuck

The best reported retrieved-evidence results remain below the claim-only
DistilBERT control, while human-annotated answers produce a much stronger
diagnostic result. This suggests evidence selection and/or evidence use are
limiting factors; it does not prove that retrieval is the sole cause.

The latest study identified a useful class-weighting correction. QA packing
looked strongest on the mixed-period train holdout but did not transfer to
official dev. Evaluating the same saved checkpoint confirmed that most of this
gap precedes refitting; refitting also changed rare-class behavior sharply.
An answer-like span may omit qualifications or fail to resolve the claim, but
that remains a hypothesis requiring audit.

The proposed next work is chronological validation, better decisive-evidence
labels from training data, a stable full-context reader, a browser extension,
and a stronger hosted-model comparison. The API comparison first holds the
three supplied passages fixed; live web retrieval is a separate setting. The
extension initially supports a labeled benchmark demo because cached claim pools
do not provide retrieval for arbitrary browser text. These are planned directions,
not completed improvements. See the work plan for owners and integration tasks.
