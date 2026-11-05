"""Build the portable limitation-study Colab with an embedded source snapshot.

Use --patch to emit an apply_patch Add File patch; this keeps artifact creation
reviewable in environments that require apply_patch for workspace edits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = "notebooks/archive/initial_ml_solution_limitation_study_colab.ipynb"


def build() -> dict:
    files = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "scripts" / name for name in (
        "run_limitation_diagnostics.py", "train_transformer_verifier.py", "run_evidence_ablation.py", "train_tfidf_logreg.py")]
    sources = {str(p.relative_to(ROOT)).replace("\\", "/"): p.read_text(encoding="utf-8") for p in files}
    sources["scripts/__init__.py"] = ""
    payload = json.dumps(sources, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    cells = []

    def md(text: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)})

    def code(text: str, hidden: bool = False) -> None:
        cells.append({"cell_type": "code", "execution_count": None,
                      "metadata": {"jupyter": {"source_hidden": True}} if hidden else {},
                      "outputs": [], "source": (text.strip() + "\n").splitlines(keepends=True)})

    md("""
# Why is evidence not helping? A bounded AVeriTeC limitation study

Run this notebook in a fresh **Colab GPU runtime**, from top to bottom. It embeds its Python helpers, so there is no repository pull or missing-script step. All checkpoints and results go to a new Google Drive directory. The large passage corpus is never re-embedded or re-indexed. The test split is never opened.

Our starting observations: BM25 URL-proxy Hit@3 = 0.1969; BM25-trained DistilBERT macro-F1 = 0.3469; supplying fusion dev passages gives 0.3808; claim-only = 0.3970; annotated answers = 0.5284. These observations do not establish whether retrieval or reading is the main bottleneck.

| Question | Controlled check | What it can tell us |
| --- | --- | --- |
| Is useful material present but ranked too low? | Existing top-20/top-100 coverage; URL-oracle raw-passage selection | Candidate/ranking opportunity, subject to the URL proxy |
| Are three results wasting space on the same source? | Distinct-source selection from existing top 20 | Whether simple diversity helps |
| Does the classifier lose useful details? | Same passages, first 110 tokens versus query-selected source sentences | Sensitivity to input packing, then matched training |
| Does it respond to evidence? | Remove it, give another claim's passages, reverse their order | Evidence sensitivity; changes alone do not prove reasoning |
| Does class weighting cause a bad tradeoff? | Balanced/unweighted loss with identical train holdout and text | Minority-label, calibration and safety tradeoffs |
| Did the trained reranker have enough trustworthy supervision? | Inspect existing train-only silver alignment coverage by verdict | Whether supervision is sparse or biased; audit quality separately |
| Does training overshoot its best checkpoint? | Record the selected checkpoint's epoch; refit for that duration | A possible selection/refit mismatch |
| Do earlier-to-later claims behave differently? | Additional temporal holdout inside official train | A limited temporal-shift check |
| Would stronger pretraining help? | Optional DeBERTa-v3-small with the selected text format | Whether encoder choice helps with the same evidence budget |

The fast audit/inference section should take minutes to tens of minutes. Seven train-holdout trials plus final refits may take tens of minutes to a few hours, depending on GPU and Drive. These are planning estimates; each cell prints its actual time. Optional stronger-model and seed checks add work. Start with the defaults.

Gold URL selection and annotated answers are **oracle diagnostics**. They are saved separately and cannot enter train-only pipeline selection. Existing dev has been inspected many times; new dev scores remain exploratory. A final model still needs untouched evaluation and a user study.
""")
    md("## 1. Install the text-only runtime and mount Drive\n\nUse a fresh runtime if previous notebooks imported different Transformers versions. Experiments run in the notebook process so progress bars and tracebacks remain visible.")
    code("""
from pathlib import Path
import gc, hashlib, json, os, runpy, subprocess, sys, time, traceback
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HOME'] = '/content/huggingface_cache'
loaded_transformers = sys.modules.get('transformers')
if loaded_transformers is not None and loaded_transformers.__version__ != '4.57.1':
    raise RuntimeError('Restart the runtime, then run this notebook first (Transformers version differs).')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'transformers==4.57.1', 'accelerate>=1,<2', 'numpy>=1.26,<3',
                'scikit-learn>=1.5,<2', 'joblib>=1.4,<2', 'tqdm>=4,<5',
                'sentencepiece>=0.2,<0.3', 'protobuf>=4,<7'], check=True)
from google.colab import drive
drive.mount('/content/drive')
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
if not torch.cuda.is_available():
    raise RuntimeError('Choose Runtime > Change runtime type > GPU, then reconnect and rerun setup.')
print('Python:', sys.version.split()[0], '| PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0))
""")
    md("## 2. Load the bundled study code\n\nThis cell contains the readable source snapshot used for this notebook. Its hash names an isolated runtime directory. It does not overwrite your project checkout.")
    code(f"""
STUDY_SOURCES = json.loads({payload!r})
SOURCE_SHA256 = {digest!r}
assert hashlib.sha256(json.dumps(STUDY_SOURCES, sort_keys=True, ensure_ascii=True).encode()).hexdigest() == SOURCE_SHA256
RUNTIME = Path('/content') / ('averitec_limitation_study_' + SOURCE_SHA256[:12])
for relative, source in STUDY_SOURCES.items():
    destination = RUNTIME / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding='utf-8')
os.chdir(RUNTIME)
sys.path.insert(0, str(RUNTIME))
print('Study source SHA256:', SOURCE_SHA256)
print('Runtime:', RUNTIME)
""", hidden=True)
    md("## 3. Paths and cost controls\n\nCheck the printed paths. Only the completed BM25 rankings and saved BM25-trained DistilBERT are required. Existing dev BGE/fusion/top-100 files are optional. No dense training rankings are required. Keep these settings unchanged to resume; use a new `STUDY_NAME` if you change inputs.")
    code("""
DATA_ROOT = Path('/content/drive/MyDrive/10718_averitec/v2_source_documents')
STUDY_NAME = 'initial_ml_solution_limitation_study_v1'
OUT = DATA_ROOT / 'outputs' / STUDY_NAME
TRAIN_CLAIMS = DATA_ROOT / 'processed/averitec/train.jsonl'
DEV_CLAIMS = DATA_ROOT / 'processed/averitec/dev.jsonl'
DEV_QRELS = DATA_ROOT / 'processed/averitec/dev_qrels.jsonl'
TRAIN_BM25 = DATA_ROOT / 'outputs/bm25/train_bm25_top20_v2/train_rankings.jsonl'
DEV_BM25 = DATA_ROOT / 'outputs/bm25/dev_bm25_top20_v2/dev_rankings.jsonl'
OLD = DATA_ROOT / 'outputs/initial_ml_solution_rerun_bs16'
OLD_VERIFIER = OLD / 'transformer/bm25_top3'
MODEL_DIR = OLD_VERIFIER / 'final_model'
MODEL_CONFIG = OLD_VERIFIER / 'transformer_run_config.json'
DEV100 = DATA_ROOT / 'outputs/initial_ml_solution_answer_aligned_top100_v1/bm25/dev_bm25_top100_v2/dev_rankings.jsonl'
EXTRA_DEV = {
    'bge': OLD / 'dense/dev_bge_small_v15/dev_rankings.jsonl',
    'fusion': OLD / 'hybrid/dev_rrf60/hybrid_rankings.jsonl',
}
RUN_TRAINING = True
REFIT_ALL_VARIANTS = True   # False refits only the holdout-selected evidence model and claim-only control.
RUN_TEMPORAL_CHECK = True  # Train-only; skipped visibly if dates are incomplete.
RUN_STRONGER_ENCODER = False  # Optional later step, using the same evidence passages.
RUN_SEED_CHECK = False        # Optional extra train-holdout seed, not a dev-based search.
RUN_REFIT_DURATION_CHECK = True  # Reuse the head/balanced trial to compare original full-epoch refitting.
AUDIT_CLAIMS = 24             # About 72-96 passage judgments; use 8 for a short first audit.
BATCH_SIZE = 8
from src.utils.io import read_jsonl, write_json
from src.utils.runs import sha256_file
required = [TRAIN_CLAIMS, DEV_CLAIMS, DEV_QRELS, TRAIN_BM25, DEV_BM25,
            MODEL_CONFIG, MODEL_DIR / 'config.json']
missing = [str(p) for p in required if not p.is_file()]
if missing:
    raise FileNotFoundError('Update the paths above to your completed artifacts:\\n' + '\\n'.join(missing))
if not [*MODEL_DIR.glob('*.safetensors'), *MODEL_DIR.glob('pytorch_model*.bin')]:
    raise FileNotFoundError(f'Missing trained model weights: {MODEL_DIR}')
train_records, dev_records = read_jsonl(TRAIN_CLAIMS), read_jsonl(DEV_CLAIMS)
assert len(train_records) == 3067 and len(dev_records) == 500, 'Expected 3067 usable train and 500 official dev claims.'
settings = json.loads(MODEL_CONFIG.read_text(encoding='utf-8'))
expected_budgets = {'evidence_token_budget_per_passage': 110, 'claim_token_budget': 96, 'max_length': 512}
assert all(settings.get(k) == v for k, v in expected_budgets.items()), 'This study assumes the original 96/110/512 token budgets.'
OUT.mkdir(parents=True, exist_ok=True)
bundle_config = {'source_sha256': SOURCE_SHA256, 'study_name': STUDY_NAME}
from scripts.run_limitation_diagnostics import lock_config
lock_config(OUT / 'notebook_source.json', bundle_config)
for path in required:
    print('Required:', path)
print('Top-100 available:', DEV100.is_file())
print('Optional completed rankings:', {k: p.is_file() for k, p in EXTRA_DEV.items()})
print('All new results/checkpoints:', OUT)

def run_script(label, script, arguments):
    previous_argv = sys.argv[:]
    started = time.monotonic()
    print('\\n===', label, '===', flush=True)
    try:
        sys.argv = [str(RUNTIME / 'scripts' / script), *map(str, arguments)]
        runpy.run_path(sys.argv[0], run_name='__main__')
    except BaseException:
        (OUT / 'last_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        sys.argv = previous_argv
        gc.collect()
        torch.cuda.empty_cache()
        print(f'{label}: {(time.monotonic()-started)/60:.1f} minutes this invocation', flush=True)
""")
    md("## 4. Audit data, retrieval opportunity, redundancy and truncation\n\nThis prepares the same top-three passages in two text formats. Sentence packing uses the claim to select verbatim source sentences within 110 tokens; it does not use labels or annotated answers. Source diversity chooses three distinct URLs from the existing top 20. The oracle chooses URL-matched raw passages from the available dev list and remains a source-level diagnostic.")
    code("""
arguments = ['prepare', '--train-claims', TRAIN_CLAIMS, '--dev-claims', DEV_CLAIMS,
             '--train-rankings', TRAIN_BM25, '--dev-rankings', DEV_BM25, '--qrels', DEV_QRELS,
             '--model-dir', MODEL_DIR, '--output', OUT, '--budget', 110, '--audit-size', AUDIT_CLAIMS]
if DEV100.is_file():
    arguments += ['--dev-top100', DEV100]
for name, path in EXTRA_DEV.items():
    if path.is_file():
        arguments += ['--extra', f'{name}={path}']
run_script('Data and candidate audit', 'run_limitation_diagnostics.py', arguments)
audit = json.loads((OUT / 'data_audit.json').read_text(encoding='utf-8'))
print('Saved-pool URL coverage:', audit['candidate_coverage'])
print('Passages truncated:', audit['passages_over_budget'], '/', audit['passages_measured'])
print('Claims with repeated source in top three:', audit['claims_with_repeated_top3_source'])
print('Lexical term retention (not decisive relevance):', audit['truncation'])
print('Gold-source coverage by label:', audit['coverage_by_gold_label'])
SILVER = DATA_ROOT / 'outputs/initial_ml_solution_answer_aligned_top100_v1/alignment/gold_answer_top100_v1/train_alignment.jsonl'
if SILVER.is_file():
    from scripts.run_limitation_diagnostics import index_rows
    aligned = index_rows(read_jsonl(SILVER), expected={r['claim_id'] for r in train_records})
    silver_report = {'source_sha256': sha256_file(SILVER), 'by_label': {}}
    for label in ['Supported', 'Refuted', 'Not Enough Evidence', 'Conflicting Evidence']:
        group = [r for r in train_records if r['label'] == label]
        silver_report['by_label'][label] = {
            'train_claims': len(group),
            'accepted_positive_claims': sum(bool(aligned[r['claim_id']].get('positive_passage_id')) for r in group),
            'matched_source_claims': sum(bool(aligned[r['claim_id']].get('matched_source_candidate_count')) for r in group),
        }
    silver_report['warning'] = 'Automatic lexical answer alignment is silver supervision; coverage does not establish label quality.'
    write_json(silver_report, OUT / 'silver_supervision_audit.json')
    print('Prior reranker supervision:', json.dumps(silver_report, indent=2))
else:
    print('Optional silver-label audit skipped: previous alignment file was not found.')
""")
    md("## 5. Does the existing verifier use the evidence?\n\nThe classifier stays fixed in this section. Blank, unrelated and reversed evidence test sensitivity; sentence packing tests a new input format. Differences may reflect unfamiliar inputs as well as evidence use. URL-oracle and annotated-answer rows use information unavailable for a new claim and cannot be chosen as a deployable method. Every batch is saved; rerunning resumes safely.")
    code("""
arguments = ['frozen', '--dev-claims', DEV_CLAIMS, '--model-dir', MODEL_DIR,
             '--model-config', MODEL_CONFIG, '--output', OUT, '--batch-size', 16, '--device', 'cuda']
reference = OLD_VERIFIER / 'final_dev_predictions.jsonl'
if reference.is_file():
    arguments += ['--reference-predictions', reference]
run_script('Frozen-model evidence interventions', 'run_limitation_diagnostics.py', arguments)
if (OUT / 'frozen/baseline_reproduction.json').is_file():
    print('Original versus recomputed baseline:', (OUT / 'frozen/baseline_reproduction.json').read_text())
""")
    md("## 6. Read a small passage sample\n\nOpen `manual_passage_audit.csv` in Drive. Fill `relevance` with `decisive_support`, `decisive_refute`, `partial`, `irrelevant`, or `unclear`, plus optional notes. Judge whether the passage establishes or contradicts a material part of the claim; mere topic overlap is insufficient. Multi-part claims may require more than one passage. Rank, gold verdict and proxy-match status are hidden to reduce anchoring. Avoid opening the key until after judging. This balanced sample diagnoses failure types; it is not a population estimate. Rerunning preparation preserves your CSV judgments.")
    code("""
print('Annotate this CSV:', OUT / 'manual_passage_audit.csv')
print('You can continue training before completing the audit; rerun the final summary after entering judgments.')
""")
    md("## 7. Train controlled alternatives on the same train holdout\n\nThe seven comparisons share the original stratified 307-claim holdout, seed, optimizer and three-passage limit. Each has one fixed learning rate (5e-5) and a three-epoch maximum, with checkpoint selection by holdout macro-F1. This tests text packing and class weighting without a large hyperparameter sweep. Claim-only is a control. No dev metrics are computed in this cell. Resume uses Trainer checkpoints; completed trials are reused.")
    code("""
VARIANTS = {
    'head_balanced': ('head', 'balanced'),
    'head_unweighted': ('head', 'none'),
    'sentences_balanced': ('sentences', 'balanced'),
    'sentences_unweighted': ('sentences', 'none'),
    'diverse_sentences_balanced': ('diverse_sentences', 'balanced'),
    'claim_only_balanced': ('claim_only', 'balanced'),
    'claim_only_unweighted': ('claim_only', 'none'),
}
DISTIL_REVISION = '12040accade4e8a0f71eabdb258fecc2e7e948be'

def train_arguments(name, mode, weighting, *, seed=718, selection_only=False,
                    model='distilbert-base-uncased', revision=DISTIL_REVISION,
                    rates='5e-5', policy='stratified', refit_policy='best_checkpoint'):
    arguments = ['--data-root', DATA_ROOT,
                 '--train-rankings', OUT / 'prepared' / mode / 'train_rankings.jsonl',
                 '--eval-rankings', OUT / 'prepared' / mode / 'dev_rankings.jsonl',
                 '--output-dir', OUT / 'training', '--run-name', name, '--model', model,
                 '--learning-rates', rates, '--epoch-grid', '3', '--batch-size', BATCH_SIZE,
                 '--gradient-accumulation-steps', 2, '--checkpoint-steps', 100,
                 '--class-weighting', weighting, '--validation-policy', policy,
                 '--refit-epoch-policy', refit_policy, '--seed', seed, '--validation-seed', 718, '--resume']
    if revision:
        arguments += ['--revision', revision]
    if selection_only:
        arguments += ['--selection-only']
    return arguments

if RUN_TRAINING:
    for name, (mode, weighting) in VARIANTS.items():
        run_script('Train holdout / ' + name, 'train_transformer_verifier.py',
                   train_arguments(name, mode, weighting, selection_only=True))
else:
    print('Training disabled; audit and frozen-model diagnostics are still complete.')
""")
    md("## 8. Select using train holdout, then refit and evaluate\n\nWe record the holdout-selected evidence pipeline before computing its dev result. Refitting uses the selected checkpoint's epoch rather than automatically repeating the full three epochs. With `REFIT_ALL_VARIANTS=True`, the other trained variants also get dev rows for diagnosis. Their dev scores do not change the recorded choice. New training runs include this refit-duration change, so compare the frozen experiments to isolate text effects and the matched trials to assess the new training procedure.")
    code("""
def read_selection(name):
    return json.loads((OUT / 'training' / name / 'model_selection.json').read_text(encoding='utf-8'))['selected_trial']

if RUN_TRAINING:
    ranked = sorted((name for name in VARIANTS if not name.startswith('claim_only')),
                    key=lambda name: (-read_selection(name)['macro_f1'],
                                      read_selection(name)['decision_error_rates']['false_reassurance_rate'],
                                      read_selection(name)['decision_error_rates']['false_alarm_rate'], name))
    CHOSEN = ranked[0]
    selection = {'name': CHOSEN, 'selection': read_selection(CHOSEN),
                 'rule': 'train-holdout macro-F1, then fewer false reassurances and false alarms',
                 'oracle_inputs_allowed': False}
    lock_config(OUT / 'chosen_pipeline_before_dev.json', selection)
    for name in VARIANTS:
        chosen = read_selection(name)
        print(name, 'holdout macro-F1=', round(chosen['macro_f1'], 4),
              'requested epochs=', chosen['epochs'], 'selected checkpoint epoch=', chosen['best_checkpoint_epoch'])
    print('Chosen evidence pipeline:', CHOSEN)
    names = list(VARIANTS) if REFIT_ALL_VARIANTS else [CHOSEN, 'claim_only_unweighted']
    for name in names:
        mode, weighting = VARIANTS[name]
        run_script('Refit and dev / ' + name, 'train_transformer_verifier.py', train_arguments(name, mode, weighting))

    if RUN_REFIT_DURATION_CHECK:
        source_run = OUT / 'training/head_balanced'
        original_selection = read_selection('head_balanced')
        epoch_gap = abs(original_selection['epochs'] - original_selection['best_checkpoint_epoch'])
        if epoch_gap < 0.05:
            print('Refit-duration check skipped: selected epoch already matches the requested epoch.')
        else:
            # Only small completed trial artifacts are copied; optimizer/model checkpoints stay in place.
            import shutil
            name = 'head_balanced_requested_epochs'
            target_run = OUT / 'training' / name
            target_run.mkdir(parents=True, exist_ok=True)
            original_config = json.loads((source_run / 'transformer_run_config.json').read_text())
            original_config['refit_epoch_policy'] = 'requested'
            lock_config(target_run / 'transformer_run_config.json', original_config)
            for summary_file in (source_run / 'trials').glob('*/trial_summary.json'):
                target_trial = target_run / 'trials' / summary_file.parent.name
                target_trial.mkdir(parents=True, exist_ok=True)
                summary = json.loads(summary_file.read_text())
                source_predictions = Path(summary['predictions_path'])
                target_predictions = target_trial / 'validation_predictions.jsonl'
                shutil.copy2(source_predictions, target_predictions)
                summary['predictions_path'] = str(target_predictions)
                write_json(summary, target_trial / 'trial_summary.json')
            run_script('Original versus selected refit duration', 'train_transformer_verifier.py',
                       train_arguments(name, 'head', 'balanced', refit_policy='requested'))
""")
    md("## 9. Check temporal shift within official train\n\nThis extra run holds out the latest roughly 10% of train claims, keeping every claim from the boundary date together. It may hold out more than 307 claims, so its score is not directly comparable to the stratified holdout. It never uses official dev for selection. Incomplete dates are reported rather than silently discarded.")
    code("""
if RUN_TRAINING and RUN_TEMPORAL_CHECK:
    if audit['claim_date_coverage']['train'] != len(train_records):
        print('Temporal check skipped: only', audit['claim_date_coverage']['train'], 'train claims have parseable dates.')
    else:
        run_script('Temporal train holdout / BM25 head', 'train_transformer_verifier.py',
                   train_arguments('temporal_head_balanced', 'head', 'balanced', selection_only=True, policy='temporal'))
        report = json.loads((OUT / 'training/temporal_head_balanced/model_selection.json').read_text())
        print('Temporal holdout claims:', report['selection_validation_claim_count'])
        print('Temporal macro-F1:', report['selected_trial']['macro_f1'])
""")
    md("## 10. Optional: test a stronger encoder and a second seed\n\nEnable the flags in section 3, then run this cell. [Microsoft's DeBERTa-v3-small](https://huggingface.co/microsoft/deberta-v3-small) provides a different pretrained encoder without rerunning retrieval. We test two learning rates on train holdout with the chosen passage format. The model revision is resolved once and recorded. Text excerpts still come from the same 110-token DistilBERT-based preparation; any gain is an encoder/pipeline comparison, not a pure tokenizer-controlled architecture test. A second seed checks sensitivity on the same stratified holdout; it does not establish stability from two runs.")
    code("""
if RUN_TRAINING and RUN_STRONGER_ENCODER:
    from huggingface_hub import model_info
    revision_path = OUT / 'deberta_revision.json'
    if revision_path.is_file():
        revision = json.loads(revision_path.read_text())['revision']
    else:
        revision = model_info('microsoft/deberta-v3-small').sha
        write_json({'model': 'microsoft/deberta-v3-small', 'revision': revision}, revision_path)
    mode, weighting = VARIANTS[CHOSEN]
    arguments = train_arguments('deberta_selected_format', mode, weighting, model='microsoft/deberta-v3-small',
                                revision=revision, rates='2e-5,5e-5')
    run_script('DeBERTa train-only selection', 'train_transformer_verifier.py', arguments + ['--selection-only'])
    print('DeBERTa holdout:', read_selection('deberta_selected_format'))
    print('DistilBERT holdout:', read_selection(CHOSEN))
    run_script('DeBERTa refit and exploratory dev', 'train_transformer_verifier.py', arguments)
if RUN_TRAINING and RUN_SEED_CHECK:
    # Same validation seed/partition: vary model training randomness only.
    mode, weighting = VARIANTS[CHOSEN]
    run_script('Extra training seed, same holdout', 'train_transformer_verifier.py',
               train_arguments('sensitivity_seed_719', mode, weighting, seed=719, selection_only=True))
""")
    md("## 11. Export the results to share\n\nThis exports full metrics, per-class precision/recall/F1, confusion matrices, calibration, safety counts with both denominators, descriptive subgroups, and paired bootstrap intervals. The intervals condition on the current label counts and omit training randomness, repeated dev inspection and claim dependence. They cannot turn exploratory dev findings into a final test result.")
    code("""
run_script('Summarize limitation study', 'run_limitation_diagnostics.py',
           ['summarize', '--output', OUT, '--dev-claims', DEV_CLAIMS, '--bootstrap-trials', 1000])
from IPython.display import Markdown, display
display(Markdown((OUT / 'limitation_results.md').read_text(encoding='utf-8')))
print('Share these artifacts:')
for name in ['limitation_results.json', 'limitation_results.md', 'data_audit.json',
             'chosen_pipeline_before_dev.json', 'manual_passage_audit.csv']:
    print(OUT / name, 'exists=', (OUT / name).is_file())
""")
    md("""
## What to send back and how we will use it

Send `limitation_results.json`, `data_audit.json`, and any completed manual audit. The small Markdown summary is useful too. If a run interrupts, reconnect Drive and rerun from setup with the same study name; completed inference batches, trials and training checkpoints are reused. A changed input or source bundle requires a new study name. Checkpoints can consume several GB; this notebook never deletes existing runs.

We will use the results to choose the next main method:

- A large gap between saved-pool coverage and top-three coverage, plus useful lower-ranked raw passages in the audit, supports training a passage reranker with decisive-passage supervision and hard negatives.
- If the model benefits from oracle answers but cannot use relevant raw passages, prioritize passage-level support/refutation reasoning and an evidence aggregator rather than another expensive embedding index.
- If short query-selected excerpts help in matched training, carry that format into the next reader. Keep exact-source provenance.
- If weighting or refit duration explains the gap, fix training before changing retrieval. Inspect each verdict class and both safety errors.
- If the stronger encoder helps on train holdout and later claims, test it with a frozen pipeline and untouched evaluation. If gains vary across seeds or temporal windows, collect that evidence before claiming a large improvement.

We still cannot measure actual trust/share decisions, global open-web search quality, exact publication-time availability, or foundation-model training contamination with these artifacts. Those require additional data or a user study. No notebook can guarantee a large metric gain; this iteration is designed to identify the most defensible next experiment.

Training API reference: [Transformers 4.57.1 Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer).
""")
    return {"cells": cells, "metadata": {"colab": {"name": Path(TARGET).name},
             "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
             "language_info": {"name": "python"}, "accelerator": "GPU", "source_sha256": digest},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", action="store_true")
    args = parser.parse_args()
    notebook = build()
    for i, cell in enumerate(notebook["cells"]):
        cell["id"] = f"limitation-{i:03d}"
    content = json.dumps(notebook, indent=1, ensure_ascii=True) + "\n"
    if args.patch:
        print("*** Begin Patch")
        if (ROOT / TARGET).exists():
            print("*** Update File: " + str(ROOT / TARGET).replace("\\", "/"))
            print("@@")
            print("\n".join("-" + line for line in (ROOT / TARGET).read_text(encoding="utf-8").splitlines()))
        else:
            print("*** Add File: " + str(ROOT / TARGET).replace("\\", "/"))
        print("\n".join("+" + line for line in content.splitlines()))
        print("*** End Patch")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
