"""Build a portable, resumable follow-up to the completed evidence-reader study."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.notebooks import NotebookBuilder, emit_notebook, source_snapshot

TARGET = ROOT / "notebooks/evidence_reader_followup_colab.ipynb"


def build():
    payload, digest = source_snapshot(ROOT, ["scripts/run_answer_followup.py", "scripts/run_evidence_reader.py",
        "scripts/train_transformer_verifier.py", "scripts/train_tfidf_logreg.py", "requirements/reader-colab.txt"])
    notebook = NotebookBuilder(TARGET.name)
    md, code = notebook.markdown, notebook.code
    md("""
# Follow-up: rare-class learning and answer-focused evidence

The previous NLI-initialized pair reader predicted only Supported and Refuted. This notebook tests a concrete training issue before changing evidence: weighted cross-entropy averaged within tiny batches can dilute class weighting. It then asks whether compact, question-conditioned source spans help the same reader.

Five controlled variants share the encoder, train holdout, learning rate, seed and three-item display limit:

| Variant | What changes | What it tests |
| --- | --- | --- |
| `raw_legacy_d01` | Reproduce old loss; distillation 0.1 | Controlled reference, using this notebook's code |
| `raw_sample_d01` | Average weighted losses by sample count | Does effective rare-class weighting help? |
| `raw_sample_d0` | Also remove NLI distillation | Is preserving generic NLI behavior constraining learning? |
| `qa_same3_d0` | Same original three passages, compact questions/answers/source context | Does answer-focused reading help without changing retrieved sources? |
| `qa_selected3_d0` | Select up to three from the union of cached raw and cleaned/NLI top-three passages | Does answerability and question coverage improve selection within this small pool? |

**Upload this notebook to Colab and run in order on a GPU. It includes its source; no repository checkout or push is needed.** It reuses completed preparation from `evidence_aware_reader_v1`. There is no full-corpus embedding, BM25 run or repeat of the 24-candidate NLI scoring pass. New QA work is bounded to six passages and two questions per claim. A timed four-claim smoke test precedes the full pass. Cached extraction resumes per claim; training resumes from complete epoch checkpoints.

The claim-only question generator is [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), used only when previous generated questions are unavailable. The frozen [RoBERTa SQuAD2 extractor](https://huggingface.co/deepset/roberta-base-squad2) proposes exact source spans and can score no-answer cases. Neither is trained here on AVeriTeC gold answers. The four-way verdict reader is trained on official train labels. Answerability scores do not establish truth. Dev has already informed project development, so new dev results remain exploratory; test is untouched.
""")
    md("## 1. Restore the included source")
    code(f"""
from pathlib import Path
import os, sys, json, hashlib, subprocess, time, runpy, gc, traceback
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HOME'] = '/content/huggingface_cache'
SOURCES = json.loads({payload!r})
SOURCE_SHA256 = {digest!r}
assert hashlib.sha256(json.dumps(SOURCES, sort_keys=True, ensure_ascii=True).encode()).hexdigest() == SOURCE_SHA256
RUNTIME = Path('/content') / ('reader_followup_' + SOURCE_SHA256[:12])
for name, module in list(sys.modules.items()):
    if name == 'src' or name.startswith('src.') or name == 'scripts' or name.startswith('scripts.'):
        location = getattr(module, '__file__', None)
        if location and not Path(location).resolve().is_relative_to(RUNTIME):
            raise RuntimeError('A previous project notebook is still loaded. Restart the Colab session, then run this notebook from the top.')
for name, content in SOURCES.items():
    path = RUNTIME / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
os.chdir(RUNTIME)
sys.path.insert(0, str(RUNTIME))
print('Portable source:', SOURCE_SHA256)
""", hidden=True)
    md("## 2. Dependencies and Drive\n\nStart a fresh GPU session. Only package installation uses a subprocess. All experiments run directly with visible progress bars and tracebacks.")
    code("""
if 'transformers' in sys.modules and sys.modules['transformers'].__version__ != '4.57.1':
    raise RuntimeError('Restart the Colab session and run this notebook first.')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(RUNTIME / 'requirements/reader-colab.txt')], check=True)
from google.colab import drive
drive.mount('/content/drive')
import torch
from transformers import Trainer
if not torch.cuda.is_available():
    raise RuntimeError('Choose Runtime > Change runtime type > GPU, then reconnect.')
print('GPU:', torch.cuda.get_device_name(0), '| PyTorch:', torch.__version__)
""")
    md("## 3. Paths and budget\n\nKeep settings unchanged when resuming. This writes a new directory and preserves the old results. Each variant uses one fixed learning rate from the previous raw reader's **train holdout**, with at most three epochs; this isolates the new changes without another large sweep. Keep `RUN_QA=True` for all five variants. `REFIT_ALL=False` evaluates only the train-holdout winner and saves time, but leaves some planned dev contrasts unavailable.")
    code("""
from src.utils.checkpoints import bind_run
from src.utils.io import read_jsonl
from src.utils.runs import sha256_file
from src.verification.pair_training import selection_key
from src.data.evidence import index_claims
from src.data.averitec import CANONICAL_LABELS

DATA_ROOT = Path('/content/drive/MyDrive/10718_averitec/v2_source_documents')
OLD = DATA_ROOT / 'outputs/evidence_aware_reader_v1'
OLD_PREPARED = OLD / 'prepared'
OUT = DATA_ROOT / 'outputs/evidence_reader_followup_v2'
PREPARED = OUT / 'prepared'
QUESTIONS = OUT / 'questions'
RUN_QA = True
REFIT_ALL = True
EPOCHS = 3
TRAIN_BATCH = 2
ACCUMULATION = 8
QA_BATCH = 8
QUESTION_MODEL = 'Qwen/Qwen3-0.6B'
QA_MODEL = 'deepset/roberta-base-squad2'
QROOT = DATA_ROOT / 'outputs/initial_ml_solution_question_guided_v1/retrieval'
QUESTION_CACHES = {s: QROOT / (s + '_qwen_ce32_v1') / (s + '_question_guided.partial.jsonl') for s in ('train', 'dev')}

required = [OLD_PREPARED / 'preparation.json', OLD_PREPARED / 'preparation_report.json']
required += [OLD_PREPARED / mode / (s + '_rankings.jsonl') for mode in ('raw', 'clean_nli') for s in ('train', 'dev')]
required += [DATA_ROOT / 'processed/averitec' / (s + '.jsonl') for s in ('train', 'dev')]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    raise FileNotFoundError('Update OLD/DATA_ROOT to your completed reader run:\\n' + '\\n'.join(missing))
prep = json.loads((OLD_PREPARED / 'preparation.json').read_text())
assert prep['limit'] == 0
assert json.loads((OLD_PREPARED / 'preparation_report.json').read_text())['finalized']
for s, count in [('train', 3067), ('dev', 500)]:
    path = DATA_ROOT / 'processed/averitec' / (s + '.jsonl')
    records = read_jsonl(path)
    assert len(records) == count
    assert sha256_file(path) == prep['input_hashes'][s + '_claims']
    for mode in ('raw', 'clean_nli'):
        index_claims(read_jsonl(OLD_PREPARED / mode / (s + '_rankings.jsonl')), expected={r['claim_id'] for r in records})
previous_selection = OLD / 'pair/raw/model_selection.json'
LR = json.loads(previous_selection.read_text())['selected']['learning_rate'] if previous_selection.is_file() else 3e-5
print('Fixed learning rate:', LR, '| source:', previous_selection if previous_selection.is_file() else 'preset fallback')
OUT.mkdir(parents=True, exist_ok=True)
bind_run(OUT / 'notebook.json', {'source_sha256': SOURCE_SHA256, 'old_preparation_sha256': sha256_file(OLD_PREPARED / 'preparation.json'),
    'lr': LR, 'epochs': EPOCHS, 'train_batch': TRAIN_BATCH, 'accumulation': ACCUMULATION, 'qa_batch': QA_BATCH,
    'question_model': QUESTION_MODEL, 'qa_model': QA_MODEL, 'run_qa': RUN_QA, 'refit_all': REFIT_ALL,
    'question_cache_hashes': {s: sha256_file(p) for s, p in QUESTION_CACHES.items() if p.is_file()}})
print('New outputs:', OUT)

def run(label, script, args):
    old_argv, started = sys.argv[:], time.monotonic()
    print('\\n===', label, '===', flush=True)
    try:
        sys.argv = [str(RUNTIME / 'scripts' / script), *map(str, args)]
        runpy.run_path(sys.argv[0], run_name='__main__')
    except BaseException:
        (OUT / 'last_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        sys.argv = old_argv
        gc.collect()
        torch.cuda.empty_cache()
        print(f'{label}: {(time.monotonic() - started)/60:.1f} minutes this invocation', flush=True)

VARIANTS = {
    'raw_legacy_d01': ('raw', 'batch_weight_mean', .1),
    'raw_sample_d01': ('raw', 'sample_mean', .1),
    'raw_sample_d0': ('raw', 'sample_mean', 0),
}
if RUN_QA:
    VARIANTS.update({'qa_same3_d0': ('qa_same3', 'sample_mean', 0), 'qa_selected3_d0': ('qa_selected3', 'sample_mean', 0)})

def reader_args(name, selection_only):
    mode, loss, distillation = VARIANTS[name]
    prepared = OLD_PREPARED if mode == 'raw' else PREPARED
    args = ['train', '--data-root', DATA_ROOT, '--prepared', prepared, '--mode', mode, '--output', OUT / 'pair' / name,
            '--validation-policy', 'grouped', '--validation-seed', 718, '--seed', 718,
            '--batch-size', TRAIN_BATCH, '--accumulation', ACCUMULATION, '--learning-rates', LR, '--epochs', EPOCHS,
            '--loss-normalization', loss, '--distillation-weight', distillation]
    return args + (['--selection-only'] if selection_only else [])
""")
    md("## 4. Verify the loss issue, then run the three training controls\n\nA single-example batch shows the issue clearly: the legacy reduction divides out that example's class weight. Accumulating gradients across batches does not restore it. This does not prove the fix will improve macro-F1, so we measure the effect with everything else held fixed. Holdout reports include all four F1 scores and prediction counts. No dev inference runs in this cell.")
    code("""
from src.verification.pair_reader import weighted_verdict_loss
toy_logits = torch.zeros((1, 4))
toy_y = torch.tensor([3])
toy_weights = torch.tensor([.5, .5, 2., 4.])
print('Singleton legacy loss:', weighted_verdict_loss(toy_logits, toy_y, toy_weights, 'batch_weight_mean').item())
print('Singleton sample-weighted loss:', weighted_verdict_loss(toy_logits, toy_y, toy_weights, 'sample_mean').item())
for name in ['raw_legacy_d01', 'raw_sample_d01', 'raw_sample_d0']:
    run('Train-only control / ' + name, 'run_evidence_reader.py', reader_args(name, True))
""")
    md("## 5. Generate or reuse two questions per claim\n\nReuse earlier generated questions when the original claim file hash, prompt and claim text match. If no compatible cache exists, the small Qwen generator receives only the claim. It never sees gold answers, labels or annotated sources. Malformed generations are recorded as fallbacks. These are hypotheses about what to check, not factual evidence.")
    code("""
if RUN_QA:
    args = ['questions', '--data-root', DATA_ROOT, '--output', QUESTIONS, '--model', QUESTION_MODEL]
    for split, path in QUESTION_CACHES.items():
        if path.is_file():
            args += ['--' + split + '-question-cache', path]
    run('Claim-only question cache', 'run_answer_followup.py', args)
    print((QUESTIONS / 'questions_report.json').read_text())
""")
    md("## 6. Four-claim QA smoke test\n\nThe extractor checks both questions against each candidate and compares the best source span with its no-answer score. Up to six cached passages are read: the union of raw top three and cleaned/NLI top three. Both variants preserve source IDs and exact context for audit. The selection policy uses fixed answerability, question coverage, NLI and source-diversity terms; none are tuned on dev. This is a small-pool reading test, not a search over the entire candidate corpus.")
    code("""
def prepare_args(destination):
    return ['prepare', '--data-root', DATA_ROOT, '--parent-prepared', OLD_PREPARED, '--questions', QUESTIONS,
            '--output', destination, '--model', QA_MODEL, '--batch-size', QA_BATCH, '--max-length', 384]
if RUN_QA:
    started = time.monotonic()
    run('Eight-claim QA smoke test (four per split)', 'run_answer_followup.py', prepare_args(OUT / 'qa_smoke') + ['--limit', 4])
    elapsed = time.monotonic() - started
    print('Smoke seconds including first model download:', round(elapsed, 1))
    sample = read_jsonl(OUT / 'qa_smoke/qa_same3/train_rankings.jsonl')[0]
    print('Example claim:', sample['claim'])
    if sample['retrieved']:
        print('Packed input:\\n', sample['retrieved'][0]['text'])
        print('Source URL:', sample['retrieved'][0].get('url'))
""")
    md("## 7. Prepare full answer contexts and inspect token coverage\n\nThe full pass saves each completed claim and resumes automatically. No old NLI model is loaded here. Teacher probabilities from the original passage cannot supervise changed QA text, so the two QA variants require distillation weight zero. Review the training-fit audit CSV before treating automatic spans as decisive evidence.")
    code("""
if RUN_QA:
    run('Prepare answer contexts / all claims', 'run_answer_followup.py', prepare_args(PREPARED))
    print((PREPARED / 'preparation_report.json').read_text())
    from transformers import AutoTokenizer
    from src.retrieval.evidence_quality import pair_premise
    tokenizer = AutoTokenizer.from_pretrained(prep['nli_model'], revision=prep['revision'], use_fast=True)
    for mode in ('qa_same3', 'qa_selected3'):
        rows = read_jsonl(PREPARED / mode / 'train_rankings.jsonl')
        # Training input audit only; do not tune packing on dev scores.
        lengths = [len(tokenizer.encode(pair_premise(p), add_special_tokens=False)) for r in rows for p in r['retrieved']]
        budget = prep['max_length'] - prep['claim_budget'] - 4
        print(mode, 'passages:', len(lengths), '| above conservative evidence budget:', sum(n > budget for n in lengths), '/', len(lengths))
    print('Training-fit audit:', PREPARED / 'train_fit_answer_audit.csv')
""")
    md("## 8. Train the two answer-focused variants\n\nBoth use corrected class weighting, no distillation, the same grouped train holdout and the same model/learning-rate budget as the raw controls. The source extractor stays frozen. Only the NLI-initialized verdict encoder and aggregation head are trained; vocabulary embeddings remain frozen to limit GPU memory.")
    code("""
if RUN_QA:
    for name in ['qa_same3_d0', 'qa_selected3_d0']:
        run('Train-only answer-focused reader / ' + name, 'run_evidence_reader.py', reader_args(name, True))
""")
    md("## 9. Record the train-only choice, then refit and evaluate\n\nChoose by macro-F1 on the grouped official-train holdout. Safety-error counts break ties. All variants must use identical holdout IDs. Save the choice before running any new dev inference. With `REFIT_ALL=True`, every planned variant is refit on all 3,067 train claims and evaluated on the same 500 dev claims. Per-class and safety results matter more than a high accuracy from predicting only the majority class.")
    code("""
summaries, holdout_ids = {}, None
for name in VARIANTS:
    path = OUT / 'pair' / name / 'model_selection.json'
    result = json.loads(path.read_text())
    ids = set(result['validation_claim_ids'])
    if holdout_ids is not None:
        assert holdout_ids == ids, 'Holdout changed between experiments'
    holdout_ids = ids
    summaries[name] = result['selected']
    counts = result['selected'].get('predicted_label_counts', {})
    print(name, '| holdout macro-F1:', round(summaries[name]['macro_f1'], 4), '| predictions:', counts)
chosen = sorted(summaries, key=lambda name: (selection_key(summaries[name]), name))[0]
bind_run(OUT / 'chosen_before_dev.json', {'chosen': chosen, 'holdout_claim_ids': sorted(holdout_ids),
    'criterion': 'Grouped official-train holdout macro-F1; FR/FA counts break ties', 'metrics': summaries})
print('Recorded choice:', chosen)
for name in (list(VARIANTS) if REFIT_ALL else [chosen]):
    run('Refit + dev / ' + name, 'run_evidence_reader.py', reader_args(name, False))
""")
    md("## 10. Export the comparisons\n\nResults include per-class F1, predicted rare-class counts, calibration, FR/FA, the original URL-proxy denominator, a text-overlap subset, and planned paired bootstrap comparisons. Previous results are read from saved predictions. Intervals describe uncertainty on these claims; they do not account for retraining randomness or repeated dev inspection.")
    code("""
args = ['compare', '--data-root', DATA_ROOT, '--prepared', PREPARED if RUN_QA else OLD_PREPARED,
        '--output', OUT, '--reference-root', OLD]
qrels = DATA_ROOT / 'processed/averitec/dev_qrels.jsonl'
if qrels.is_file():
    args += ['--qrels', qrels]
run('Export follow-up comparisons', 'run_evidence_reader.py', args)
from IPython.display import Markdown, display
display(Markdown((OUT / 'evidence_reader_results.md').read_text()))
print('Share these files:')
for path in [OUT / 'evidence_reader_results.json', OUT / 'chosen_before_dev.json',
             PREPARED / 'preparation_report.json', QUESTIONS / 'questions_report.json',
             PREPARED / 'train_fit_answer_audit.csv']:
    if path.is_file():
        print(path)
""")
    md("""
## What the results will tell us

- Legacy vs corrected weighting, with distillation unchanged: whether the old reduction was limiting rare-class learning.
- Corrected weighting with vs without distillation: whether generic NLI preservation helps or restricts task adaptation.
- Raw vs QA on the same three sources: whether focused reading helps, independently of retrieved source identity. Packing also changes context length, so it does not isolate question wording alone.
- Same-three QA vs QA selected from up to six: whether answerability-based selection adds value within the available cached pool.

An answer span can be confidently extracted from a misleading or irrelevant source. Zero/no-answer scores do not imply a Not Enough Evidence gold label, and one matching URL does not prove decisive passage relevance. If answer-focused inputs still fail, inspect the 80 training-fit audit rows before investing in another larger model: useful next supervision would be human-checked decisive spans and hard negatives that mention the topic without resolving it.

To resume, reconnect Drive and rerun in order with the same settings and this notebook version. Completed question/extraction rows and training summaries are reused. An interrupted epoch repeats work since the latest complete epoch checkpoint. Changed settings or source hashes require a new `OUT` directory. Nothing deletes prior data or checkpoints. This notebook cannot recover useful evidence outside its small candidate union, certify historical publication dates, or establish a real-world user benefit from offline F1 alone.
""")
    return notebook.document(source_sha256=digest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", action="store_true")
    args = parser.parse_args()
    emit_notebook(build(), TARGET, patch=args.patch)
