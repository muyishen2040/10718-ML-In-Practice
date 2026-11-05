"""Emit the self-contained Colab notebook for the feature-branch main method."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.notebooks import NotebookBuilder, emit_notebook, source_snapshot

TARGET = ROOT / "notebooks/evidence_aware_reader_colab.ipynb"


def build():
    payload, digest = source_snapshot(ROOT, ["scripts/run_evidence_reader.py", "scripts/train_transformer_verifier.py",
        "scripts/train_tfidf_logreg.py", "requirements/reader-colab.txt"])
    notebook = NotebookBuilder(TARGET.name)
    md, code = notebook.markdown, notebook.code
    md("""
# Evidence-aware reader: source cleaning + three separate passage judgments

This is the main-method candidate on branch `feat/evidence-aware-reader`. It reuses existing BM25 top-100 rankings, cleans sources, and uses an NLI-pretrained DeBERTa to read each claim–passage pair separately. A learned attention/aggregation layer predicts the four-way verdict from up to three passages. A small distillation loss preserves the pretrained support/contradiction/neutral behavior without inventing passage labels from claim verdicts.

The notebook embeds its source files. Upload this file to Colab; **no git pull or push is required**. Use a fresh GPU runtime and run in order. Existing data/results are preserved in Drive; all new artifacts use a new output directory.

The default NLI model is [cross-encoder/nli-deberta-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small), pretrained on SNLI/MultiNLI. Its three-way outputs are features, not four-way fact-check verdicts. Source filtering is an experimental policy developed after dev inspection. Dev findings remain exploratory; the test split stays untouched.

Expect an initial scoring pass over roughly 24 candidate passages per claim, followed by compact training runs. This is bounded work, but on a T4 the full ablation can still take hours. Keeping every trial/refit checkpoint can consume tens of GB of Drive space. There is a four-claim GPU smoke test before the full pass. Each cell reports its time, and every completed claim/training epoch is saved. No full-corpus embedding or retrieval index is rebuilt.
""")
    md("## 1. Restore the portable source snapshot")
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
RUNTIME = Path('/content') / ('evidence_reader_' + SOURCE_SHA256[:12])
for name, text in SOURCES.items():
    path = RUNTIME / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
os.chdir(RUNTIME)
sys.path.insert(0, str(RUNTIME))
print('Source snapshot:', SOURCE_SHA256)
""", hidden=True)
    md("## 2. Install text-model dependencies and connect Drive\n\nUse a fresh runtime if a previous notebook loaded a different Transformers version. Package installation is the only subprocess step; experiments run directly so progress bars and errors stay visible.")
    code("""
if 'transformers' in sys.modules and sys.modules['transformers'].__version__ != '4.57.1':
    raise RuntimeError('Restart the Colab session, then run this notebook first.')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(RUNTIME / 'requirements/reader-colab.txt')], check=True)
from google.colab import drive
drive.mount('/content/drive')
import torch
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification
if not torch.cuda.is_available():
    raise RuntimeError('Choose Runtime > Change runtime type > GPU, reconnect and rerun.')
print('GPU:', torch.cuda.get_device_name(0), '| PyTorch:', torch.__version__)
""")
    md("## 3. Paths and experiment budget\n\nBoth train and dev must use the same candidate depth. The notebook uses completed top-100 files when both exist, otherwise completed top-20 files for both. The raw reference must reproduce the original top-three dev passages. Keep all settings unchanged when resuming; a changed source snapshot or settings needs a new `RUN_NAME`.")
    code("""
DATA_ROOT = Path('/content/drive/MyDrive/10718_averitec/v2_source_documents')
RUN_NAME = 'evidence_aware_reader_v1'
OUT = DATA_ROOT / 'outputs' / RUN_NAME
PREPARED = OUT / 'prepared'
TOP100 = DATA_ROOT / 'outputs/initial_ml_solution_answer_aligned_top100_v1/bm25'
TRAIN100 = TOP100 / 'train_bm25_top100_v2/train_rankings.jsonl'
DEV100 = TOP100 / 'dev_bm25_top100_v2/dev_rankings.jsonl'
TRAIN20 = DATA_ROOT / 'outputs/bm25/train_bm25_top20_v2/train_rankings.jsonl'
DEV20 = DATA_ROOT / 'outputs/bm25/dev_bm25_top20_v2/dev_rankings.jsonl'
BASELINE = DATA_ROOT / 'outputs/initial_ml_solution_rerun_bs16/transformer/bm25_top3'
TRAIN_CLAIMS = DATA_ROOT / 'processed/averitec/train.jsonl'
DEV_CLAIMS = DATA_ROOT / 'processed/averitec/dev.jsonl'
QRELS = DATA_ROOT / 'processed/averitec/dev_qrels.jsonl'
NLI_MODEL = 'cross-encoder/nli-deberta-v3-small'
NLI_LABEL_ORDER = 'contradiction,entailment,neutral'
MAX_CANDIDATES = 24
PAIR_LENGTH = 384
SCORING_BATCH = 16
TRAIN_BATCH = 2
RUN_PAIR_TRAINING = True
RUN_MATCHED_DISTIL = True
REFIT_ALL_PAIR_MODES = True  # False refits only the holdout-selected pair pipeline.
RUN_SECOND_SEED = False
MODES = ['raw', 'clean', 'clean_nli']
VALIDATION_POLICY = 'grouped'  # Keeps normalized duplicate claims in the same train/holdout group.
from src.utils.io import read_jsonl, write_json
from src.utils.checkpoints import bind_run
from src.data.evidence import index_claims
from src.verification.pair_training import selection_key
if TRAIN100.is_file() and DEV100.is_file():
    TRAIN_RANKINGS, DEV_RANKINGS, CANDIDATE_DEPTH = TRAIN100, DEV100, 100
else:
    TRAIN_RANKINGS, DEV_RANKINGS, CANDIDATE_DEPTH = TRAIN20, DEV20, 20
required = [TRAIN_CLAIMS, DEV_CLAIMS, TRAIN_RANKINGS, DEV_RANKINGS, DEV20,
            BASELINE / 'transformer_run_config.json', BASELINE / 'final_model/config.json']
missing = [str(p) for p in required if not p.is_file()]
if missing:
    raise FileNotFoundError('Update the paths above to your completed files:\\n' + '\\n'.join(missing))
train, dev = read_jsonl(TRAIN_CLAIMS), read_jsonl(DEV_CLAIMS)
assert len(train) == 3067 and len(dev) == 500
for path, records in ((TRAIN_RANKINGS, train), (DEV_RANKINGS, dev)):
    index_claims(read_jsonl(path), expected={r['claim_id'] for r in records})
old_rankings, new_rankings = index_claims(read_jsonl(DEV20)), index_claims(read_jsonl(DEV_RANKINGS))
for cid in old_rankings:
    original = [(p['passage_id'], p['text']) for p in old_rankings[cid]['retrieved'][:3]]
    supplied = [(p['passage_id'], p['text']) for p in new_rankings[cid]['retrieved'][:3]]
    assert original == supplied, f'Top-three reference differs for {cid}; inspect corpus/ranking version.'
OUT.mkdir(parents=True, exist_ok=True)
bind_run(OUT / 'notebook.json', {'source_sha256': SOURCE_SHA256, 'model': NLI_MODEL,
    'candidate_depth': CANDIDATE_DEPTH, 'max_candidates': MAX_CANDIDATES, 'pair_length': PAIR_LENGTH})
print('Shared candidate depth:', CANDIDATE_DEPTH)
print('New outputs:', OUT)

def run(label, script, args):
    old_argv = sys.argv[:]
    started = time.monotonic()
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
        print(f'{label}: {(time.monotonic()-started)/60:.1f} minutes this invocation', flush=True)

def prepare_args(destination):
    return ['prepare', '--train-claims', TRAIN_CLAIMS, '--dev-claims', DEV_CLAIMS,
            '--train-rankings', TRAIN_RANKINGS, '--dev-rankings', DEV_RANKINGS,
            '--output', destination, '--model', NLI_MODEL, '--nli-label-order', NLI_LABEL_ORDER,
            '--max-candidates', MAX_CANDIDATES, '--candidate-depth', CANDIDATE_DEPTH,
            '--max-length', PAIR_LENGTH, '--batch-size', SCORING_BATCH]

def experiment_args(command, mode, output):
    return [command, '--data-root', DATA_ROOT, '--prepared', PREPARED, '--mode', mode, '--output', output]
""")
    md("## 4. GPU smoke test, then full evidence preparation\n\nThe scorer reads passage as premise and claim as hypothesis. Both support and contradiction can improve a passage's rank. Search-result pages and near-identical copies are excluded from cleaned variants. Publication dates are filtered only when explicit metadata establishes them; archive capture dates and future projections are not publication dates. Unknown dates remain an unresolved limitation.")
    code("""
run('Four-claim GPU smoke test', 'run_evidence_reader.py', prepare_args(OUT / 'smoke') + ['--limit', 4])
smoke = json.loads((OUT / 'smoke/preparation_report.json').read_text())
print('Smoke-test scored pairs:', {k: v['scored_pairs'] for k, v in smoke['totals'].items()})
run('Prepare all train/dev evidence', 'run_evidence_reader.py', prepare_args(PREPARED))
report = json.loads((PREPARED / 'preparation_report.json').read_text())
print(json.dumps(report['totals'], indent=2))
""")
    md("## 5. Fast learned aggregation and frozen DistilBERT controls\n\nFirst select a small classifier over frozen NLI/quality features using the grouped train holdout. Its raw/clean/clean_nli rows are fully matched pipelines. The frozen DistilBERT rows keep the old verifier fixed while passages change, isolating an input intervention. These are different experiments and are labeled separately in the exported table.")
    code("""
for mode in MODES:
    run('Fast holdout / ' + mode, 'run_evidence_reader.py',
        experiment_args('fast', mode, OUT / 'fast' / mode) + ['--selection-only', '--validation-policy', VALIDATION_POLICY])
for mode in MODES:
    run('Frozen DistilBERT / ' + mode, 'run_evidence_reader.py',
        experiment_args('frozen', mode, OUT / 'frozen' / mode) + ['--baseline', BASELINE])
""")
    md("## 6. Train the pair reader on official train only\n\nThe encoder starts from the frozen scorer's exact model revision. Transformer layers and the aggregation head are fine-tuned; vocabulary embeddings stay frozen to limit optimizer memory. Each passage gets up to a 384-token claim/source pair, rather than sharing one 512-token input across all three. This increases reading cost; the displayed evidence budget remains three. The learned aggregator is invariant to passage order. No gold answers, qrels or dev verdicts are used for fitting or selection. Two learning rates are tried, with at most three epochs each; the best epoch comes from the grouped train holdout. Missing evidence remains in the evaluation and is not automatically relabeled Not Enough Evidence.")
    code("""
def pair_arguments(mode, *, selection_only=False, seed=718, name=None):
    args = experiment_args('train', mode, OUT / 'pair' / (name or mode)) + [
        '--validation-policy', VALIDATION_POLICY, '--validation-seed', 718, '--seed', seed,
        '--batch-size', TRAIN_BATCH, '--accumulation', 8, '--learning-rates', '1e-5', '3e-5', '--epochs', 3]
    return args + (['--selection-only'] if selection_only else [])

if RUN_PAIR_TRAINING:
    for mode in MODES:
        run('Pair reader train holdout / ' + mode, 'run_evidence_reader.py', pair_arguments(mode, selection_only=True))
""")
    md("## 7. Matched DistilBERT controls\n\nThese optional controls use the same new grouped holdout and raw/clean evidence as the new method, retaining the old 96/110/512 token budgets. They help distinguish effects of source cleaning from a change of encoder and input length. They are new trained baselines, distinct from the frozen original model.")
    code("""
def distil_arguments(mode, selection_only=False):
    args = ['--data-root', DATA_ROOT, '--train-rankings', PREPARED / mode / 'train_rankings.jsonl',
            '--eval-rankings', PREPARED / mode / 'dev_rankings.jsonl', '--output-dir', OUT,
            '--run-name', 'distil_' + mode, '--learning-rates', '5e-5', '--epoch-grid', '3',
            '--validation-policy', VALIDATION_POLICY, '--refit-epoch-policy', 'best_checkpoint', '--resume']
    return args + (['--selection-only'] if selection_only else [])
if RUN_MATCHED_DISTIL:
    for mode in ['raw', 'clean']:
        run('Matched DistilBERT holdout / ' + mode, 'train_transformer_verifier.py', distil_arguments(mode, True))
""")
    md("## 8. Freeze the train-holdout choice, then refit and evaluate\n\nThe chosen configuration is saved before new dev scores are computed for trained models. All dev comparisons remain exploratory because we designed this iteration after inspecting earlier dev results. Existing original baselines remain untouched. The paired comparisons and full classification metrics are exported later.")
    code("""
candidates = []
for mode in MODES:
    fast = json.loads((OUT / 'fast' / mode / 'model_selection.json').read_text())['selected']
    candidates.append({'family': 'fast', 'mode': mode, 'metrics': fast})
    if RUN_PAIR_TRAINING:
        pair = json.loads((OUT / 'pair' / mode / 'model_selection.json').read_text())['selected']
        candidates.append({'family': 'pair', 'mode': mode, 'metrics': pair})
if RUN_MATCHED_DISTIL:
    for mode in ['raw', 'clean']:
        metric = json.loads((OUT / ('distil_' + mode) / 'model_selection.json').read_text())['selected_trial']
        candidates.append({'family': 'distil', 'mode': mode, 'metrics': metric})
chosen = sorted(candidates, key=lambda row: selection_key(row['metrics']))[0]
bind_run(OUT / 'chosen_before_trained_dev.json', {'chosen': chosen, 'criterion': 'grouped official-train holdout macro-F1; safety errors break ties'})
for row in candidates:
    print(row['family'], row['mode'], 'holdout macro-F1:', round(row['metrics']['macro_f1'], 4))
print('Recorded choice:', chosen['family'], chosen['mode'])

for mode in MODES:
    run('Fast refit and dev / ' + mode, 'run_evidence_reader.py',
        experiment_args('fast', mode, OUT / 'fast' / mode) + ['--validation-policy', VALIDATION_POLICY])
if RUN_PAIR_TRAINING:
    pair_modes = MODES if REFIT_ALL_PAIR_MODES else [max((r for r in candidates if r['family']=='pair'), key=lambda r: r['metrics']['macro_f1'])['mode']]
    for mode in pair_modes:
        run('Pair reader refit and dev / ' + mode, 'run_evidence_reader.py', pair_arguments(mode))
if RUN_MATCHED_DISTIL:
    for mode in ['raw', 'clean']:
        run('Matched DistilBERT refit and dev / ' + mode, 'train_transformer_verifier.py', distil_arguments(mode))
""")
    md("## 9. Optional second training seed\n\nA promising single-seed score is insufficient to establish a large improvement. This uses the same grouped holdout while changing only model initialization/training randomness. It runs only the best pair-reader configuration selected on train holdout.")
    code("""
if RUN_PAIR_TRAINING and RUN_SECOND_SEED:
    mode = sorted((r for r in candidates if r['family']=='pair'), key=lambda r: selection_key(r['metrics']))[0]['mode']
    run('Pair reader / second seed', 'run_evidence_reader.py', pair_arguments(mode, seed=719, name=mode + '_seed719'))
""")
    md("## 10. Export results\n\nAll primary verdict rows keep all 500 official dev claims. An additional subset excludes normalized train/dev claim-text overlaps, without changing the primary population. Retrieval uses the original fixed URL-proxy judgments; removing a bad source does not silently remove its claim from the denominator. Lower proxy scores can coexist with better evidence. Consult the previous passage audit when interpreting them.")
    code("""
args = ['compare', '--data-root', DATA_ROOT, '--prepared', PREPARED, '--output', OUT]
if QRELS.is_file():
    args += ['--qrels', QRELS]
run('Export comparisons', 'run_evidence_reader.py', args)
from IPython.display import Markdown, display
display(Markdown((OUT / 'evidence_reader_results.md').read_text(encoding='utf-8')))
print('Share these files:')
for p in [OUT / 'evidence_reader_results.json', PREPARED / 'preparation_report.json', OUT / 'chosen_before_trained_dev.json']:
    print(p)
""")
    md("""
## Resume and interpretation

Reconnect Drive and rerun from setup with the same settings. NLI scoring resumes per claim; training resumes from complete optimizer checkpoints; dev prediction resumes per batch. Source/settings mismatches raise an error instead of silently reusing stale results. Completed epochs consume Drive space; nothing in this notebook deletes old checkpoints.

Compare `frozen/raw` with `frozen/clean` to inspect source filtering using the same original verifier. Compare `pair/raw` with `pair/clean` for matched training. Compare `pair/clean` with `pair/clean_nli` for the selection policy, and compare the matched DistilBERT controls with the pair reader. The scorer's neutral/entailment/contradiction probabilities are not calibrated judgments of claim truth or source credibility. Preserve per-class and safety reporting.

Source hygiene does not certify every page's publication time, authenticate quotations, or solve multi-hop verification. There is no promise of a large gain. Use the train-only choice, multiple seeds, the paired intervals and a frozen final test to decide whether the method improved. Human trust/share outcomes still require a user study.
""")
    return notebook.document(source_sha256=digest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", action="store_true")
    args = parser.parse_args()
    emit_notebook(build(), TARGET, patch=args.patch)
