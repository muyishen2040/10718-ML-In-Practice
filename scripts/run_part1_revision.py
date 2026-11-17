"""Reproduce the report's trust metrics and train two chronological comparisons.

Stages are individually resumable. Existing shared claims/rankings/checkpoints
are read-only. Each selected policy is written before future-window prediction.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from src.data.averitec import CANONICAL_LABELS
from src.data.evidence import attach_evidence, index_claims
from src.evaluation.classification import evaluate_predictions
from src.evaluation.temporal_protocol import PROTOCOL_ID, build_manifest, validate_manifest
from src.evaluation.trust_decisions import apply_policy, decision_metrics, from_error_counts, selection_key, threshold_study
from src.utils.checkpoints import atomic_jsonl, bind_run, latest_complete_trainer_checkpoint
from src.utils.io import read_jsonl, write_json
from src.utils.runs import sha256_file

TRAIN_RANKINGS = 'outputs/bm25/train_bm25_top20_v2/train_rankings.jsonl'
MODEL = 'distilbert-base-uncased'
REVISION = '12040accade4e8a0f71eabdb258fecc2e7e948be'
SEED = 718


def snapshot_hash():
    paths = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__),
        ROOT/'scripts/train_transformer_verifier.py', ROOT/'scripts/train_tfidf_logreg.py']
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_inputs(args):
    claims_path = args.data_root / 'processed/averitec/train.jsonl'
    rows = read_jsonl(claims_path)
    manifest = json.loads((args.output / 'split_manifest.json').read_text())
    validate_manifest(manifest, rows, sha256_file(claims_path))
    return rows, manifest


def prepare(args):
    path = args.data_root / 'processed/averitec/train.jsonl'
    rows = read_jsonl(path)
    manifest = build_manifest(rows, sha256_file(path))
    target = args.shared_root / 'processed/averitec/team_splits' / PROTOCOL_ID / 'split_manifest.json'
    bind_run(target, manifest)
    bind_run(args.output / 'split_manifest.json', manifest)
    source = {'protocol_id': PROTOCOL_ID, 'manifest_sha256': sha256_file(target),
              'relative_path': target.relative_to(args.shared_root).as_posix(),
              'fold_ids': [fold['fold_id'] for fold in manifest['folds']],
              'source_snapshot_sha256': snapshot_hash()}
    write_json(source, args.output / 'protocol_release.json')
    print(json.dumps({'release': source, 'folds': [{'fold_id': f['fold_id'], **f['summary'],
        'purged_earlier_duplicates_of_evaluation': len(f['purged_earlier_duplicates_of_evaluation']),
        'purged_fit_duplicates_of_validation': len(f['purged_fit_duplicates_of_validation'])} for f in manifest['folds']],
        'date_exclusions': len(manifest['excluded'])}, indent=2), flush=True)


def historical(args):
    root = args.shared_root
    entries = [
        ('Qwen3-4B / BM25 top 3', root/'outputs/local_llm/dev_qwen3_4b_bm25_top3_v2/classify_top3_predictions.jsonl', 'end_to_end'),
        ('Qwen3-4B / claim only', root.parent/'outputs/local_llm/dev_qwen3_4b_claim_only_v1/claim_only_predictions.jsonl', 'claim_only_control'),
        ('Qwen3-4B / annotated answers', root.parent/'outputs/local_llm/dev_qwen3_4b_gold_top3_v1/gold_top3_predictions.jsonl', 'oracle_control'),
    ]
    for mode, folder, description in [('retrieved','retrieved_train_to_dev_v2','BM25 top 3'),
                                      ('claim_only','claim_only_train_to_dev_v2','claim only'),
                                      ('gold','gold_train_to_dev_v2','annotated answers')]:
        entries.append((f'TF-IDF + logistic regression / {description}',
            root/f'outputs/tfidf_logreg/{folder}/{mode}_train_to_dev_predictions.jsonl',
            'end_to_end' if mode=='retrieved' else 'oracle_control' if mode=='gold' else 'claim_only_control'))
    entries += [('DistilBERT / BM25 top 3',root/'outputs/initial_ml_solution_rerun_bs16/transformer/bm25_top3/final_dev_predictions.jsonl','end_to_end')]
    for mode, label in [('dense','BGE top 3'),('fusion','BM25+BGE fusion top 3')]:
        entries.append((f'BM25-trained DistilBERT / {label}',
            root/f'outputs/initial_ml_solution_dense_fusion_verifiers_v1/frozen_bm25_trained_evidence_swap/{mode}/predictions.jsonl',
            'frozen_classifier_evidence_intervention'))
    gold = index_claims(read_jsonl(args.data_root/'processed/averitec/dev.jsonl'))
    summaries,missing=[],[]
    for name,path,condition in entries:
        if not path.is_file():
            missing.append({'method':name,'path':str(path)})
            continue
        rows=read_jsonl(path)
        index_claims(rows, expected=set(gold))
        for row in rows:
            label = row.get('predicted_label',row.get('label'))
            row['predicted_label']=label
            if row.get('true_label') is not None and row['true_label']!=gold[row['claim_id']]['label']:
                raise ValueError('Saved gold labels disagree with official dev')
            row['true_label']=gold[row['claim_id']]['label']
        metrics=evaluate_predictions(rows)
        summaries.append({'method':name,'condition':condition,'source_path':str(path),
            'source_sha256':sha256_file(path),**decision_metrics(rows),
            'four_class_macro_f1':metrics['macro_f1'],'four_class_accuracy':metrics['accuracy']})
    # Locate the two controls created by the later dense/fusion-comparison workflow.
    for marker, target_f1, label in [('claim',.3970,'claim only'),('gold',.5284,'annotated answers')]:
        candidates=[]
        for directory, subdirs, files in os.walk(root/'outputs'):
            relative=Path(directory).relative_to(root/'outputs')
            subdirs[:]=[name for name in subdirs if not name.startswith('checkpoint-') and name not in
                ('trials','final_model','cache','bm25','prepared','retrieval','alignment','training') and len(relative.parts)<5]
            if marker not in str(relative).lower():
                continue
            for filename in files:
                if filename not in ('final_dev_metrics.json','dev_metrics.json','metrics.json'):
                    continue
                path=Path(directory)/filename
                data=json.loads(path.read_text())
                if data.get('claim_count')==500 and abs(data.get('macro_f1',-1)-target_f1)<.00005:
                    candidates.append((path,data))
        if candidates:
            path,data=sorted(candidates,key=lambda pair:str(pair[0]))[0]
            errors=data['decision_error_rates']
            supported=int(data['per_class']['Supported']['support'])
            summaries.append({'method':f'DistilBERT / {label}','condition':'oracle_control' if marker=='gold' else 'claim_only_control',
                'source_path':str(path),'source_sha256':sha256_file(path),
                'calculation_source':'saved full-population confusion/error counts',
                **from_error_counts(supported,500-supported,errors['false_reassurance_count'],errors['false_alarm_count']),
                'four_class_macro_f1':data['macro_f1'],'four_class_accuracy':data['accuracy']})
        else:
            missing.append({'method':f'DistilBERT / {label}','reason':'No matching saved control metrics located'})
    write_json({'evaluation':'500 official dev claims; retrospective metric revision',
                'primary_metric':'binary_balanced_accuracy','results':summaries,'missing':missing},args.output/'historical_metrics.json')
    write_csv(summaries,args.output/'historical_metrics.csv')
    print(json.dumps({'historical_rows':len(summaries),'missing':missing,'results':summaries},indent=2),flush=True)


def write_csv(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(key for row in rows for key in row if not isinstance(row[key],(dict,list))))
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key:value for key,value in row.items() if key in fields})


def save_trial(rows,config,path):
    atomic_jsonl(rows,path/'validation_predictions.jsonl')
    curve,selected=threshold_study(rows)
    report={'configuration':config,'selected_policy':selected,'threshold_curve':curve,
            'argmax_metrics':evaluate_predictions(rows)}
    write_json(report,path/'trial_complete.json')
    return report


def load_trial(path,validation):
    marker=path/'trial_complete.json'
    if not marker.is_file():
        return None
    rows=read_jsonl(path/'validation_predictions.jsonl')
    indexed=index_claims(rows,expected={r['claim_id'] for r in validation})
    if any(indexed[r['claim_id']]['true_label']!=r['label'] for r in validation):
        raise ValueError('Trial validation labels changed')
    return json.loads(marker.read_text())


def logistic_trial(fit,validation,path,c):
    import joblib
    from src.verification.tfidf_logreg import TfidfLogRegVerifier
    config={'family':'tfidf_logreg','C':c,'class_weight':'balanced','max_iter':2000,'seed':SEED,
            'ngram_range':[1,2],'min_df':1,'max_evidence':3}
    bind_run(path/'config.json',config)
    completed=load_trial(path,validation)
    if completed:
        return completed
    started=time.monotonic()
    model=TfidfLogRegVerifier(evidence_mode='retrieved',random_state=SEED)
    model.pipeline.set_params(logreg__C=c)
    model.fit(fit)
    joblib.dump(model,path/'model.joblib')
    report=save_trial(model.predict_records(validation),config,path)
    print('Completed logistic trial',c,'in',round(time.monotonic()-started,1),'seconds',flush=True)
    return report


def transformer_predictions(model,tokenizer,records,device='cuda'):
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    from scripts.train_transformer_verifier import EncodedRecords
    dataset=EncodedRecords(records,tokenizer,512,96,110)
    loader=DataLoader(dataset,batch_size=16,collate_fn=DataCollatorWithPadding(tokenizer))
    values=[]
    model.to(device).eval()
    with torch.inference_mode():
        for batch in loader:
            inputs={k:v.to(device) for k,v in batch.items() if k!='labels'}
            values.extend(model(**inputs).logits.float().softmax(-1).cpu().tolist())
    return [{'claim_id':r['claim_id'],'true_label':r['label'],
             'predicted_label':CANONICAL_LABELS[max(range(4),key=lambda i:score[i])],
             'probabilities':dict(zip(CANONICAL_LABELS,score)),
             'evidence_passage_ids':[p['passage_id'] for p in r.get('retrieved_evidence',[])[:3]]}
            for r,score in zip(records,values,strict=True)]


def distilbert_trial(fit,validation,path,lr,epochs,fold_id):
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding,Trainer,TrainingArguments,set_seed
    from scripts.train_transformer_verifier import EncodedRecords,LABEL_TO_ID,ID_TO_LABEL
    config={'family':'distilbert','model':MODEL,'revision':REVISION,'learning_rate':lr,'epochs':epochs,
            'max_length':512,'claim_token_budget':96,'evidence_token_budget':110,'batch_size':8,
            'gradient_accumulation':2,'class_weight':'balanced','weight_decay':.01,'warmup_ratio':.06,'seed':SEED}
    bind_run(path/'config.json',config)
    completed=load_trial(path,validation)
    if completed:
        return completed
    if not torch.cuda.is_available():
        raise RuntimeError('A GPU is required for the bounded transformer training stage')
    torch.set_num_threads(2)
    set_seed(SEED)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,use_fast=True)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,revision=REVISION,num_labels=4,
        label2id=LABEL_TO_ID,id2label=ID_TO_LABEL)
    counts=np.bincount([LABEL_TO_ID[r['label']] for r in fit],minlength=4)
    if np.any(counts==0):
        raise ValueError('Temporal fit partition lacks a verdict class')
    weights=torch.tensor(len(fit)/(4*counts),dtype=torch.float32)
    class WeightedTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            labels=inputs['labels']
            output=model(**{k:v for k,v in inputs.items() if k!='labels'})
            loss=torch.nn.functional.cross_entropy(output.logits,labels,weight=weights.to(output.logits.device))
            return (loss,output) if return_outputs else loss
    checkpoint_root=Path('/content/misinfo_part1/checkpoints')/fold_id/path.name
    training_args=TrainingArguments(output_dir=str(checkpoint_root),learning_rate=lr,num_train_epochs=epochs,
        per_device_train_batch_size=8,per_device_eval_batch_size=16,gradient_accumulation_steps=2,
        weight_decay=.01,warmup_ratio=.06,logging_steps=25,save_strategy='epoch',save_total_limit=2,
        eval_strategy='no',fp16=True,report_to='none',seed=SEED,dataloader_num_workers=0,
        save_safetensors=True,disable_tqdm=False)
    trainer=WeightedTrainer(model=model,args=training_args,train_dataset=EncodedRecords(fit,tokenizer,512,96,110),
        data_collator=DataCollatorWithPadding(tokenizer))
    checkpoint=latest_complete_trainer_checkpoint(checkpoint_root)
    trainer.train(resume_from_checkpoint=checkpoint)
    model=trainer.model
    model.save_pretrained(path/'model')
    tokenizer.save_pretrained(path/'model')
    predictions=transformer_predictions(model,tokenizer,validation)
    report=save_trial(predictions,config,path)
    del trainer,model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def train(args):
    import importlib.metadata
    records,manifest=load_inputs(args)
    joined=attach_evidence(records,args.data_root/TRAIN_RANKINGS,exact=True)
    indexed=index_claims(joined)
    config={'protocol_sha256':sha256_file(args.output/'split_manifest.json'),
            'source_snapshot_sha256':snapshot_hash(),'train_rankings_sha256':sha256_file(args.data_root/TRAIN_RANKINGS),
            'logistic_C_grid':[.1,1,10],'learning_rate_grid':[2e-5,5e-5],'epoch_grid':[2,3],
            'seed':SEED,'primary_metric':'binary_balanced_accuracy','threshold_grid':[i/100 for i in range(10,91,5)],
            'versions':{name:importlib.metadata.version(name) for name in ('torch','transformers','scikit-learn','numpy')},
            'python':platform.python_version(),
            'resume':'Completed trials and final weights persist in Drive. Current-trial epoch checkpoints are runtime-local; VM replacement repeats only an unfinished trial.'}
    bind_run(args.output/'experiment_config.json',config)
    for fold in manifest['folds']:
        if args.fold and fold['fold_id']!=args.fold:
            continue
        fit,validation,evaluation=([indexed[cid] for cid in fold['ids'][name]] for name in ('fit','validation','evaluation'))
        print('FOLD',fold['fold_id'],json.dumps(fold['summary']),flush=True)
        for family in ('logistic','distilbert'):
            if args.family and args.family!=family:
                continue
            directory=args.output/'temporal'/fold['fold_id']/family
            if (directory/'evaluation_complete.json').is_file():
                print('Reusing completed evaluation',directory,flush=True)
                continue
            trials=[]
            if family=='logistic':
                for c in config['logistic_C_grid']:
                    path=directory/'trials'/f'C_{c:g}'
                    trials.append((path,logistic_trial(fit,validation,path,c)))
            else:
                for lr in config['learning_rate_grid']:
                    for epochs in config['epoch_grid']:
                        path=directory/'trials'/f'lr_{lr:g}_epochs_{epochs}'
                        trials.append((path,distilbert_trial(fit,validation,path,lr,epochs,fold['fold_id'])))
            path,chosen=min(trials,key=lambda pair:selection_key(pair[1]['selected_policy']))
            frozen={'protocol_sha256':config['protocol_sha256'],'trial':path.name,
                'configuration':chosen['configuration'],'policy':chosen['selected_policy'],
                'fit_claim_count':len(fit),'validation_claim_count':len(validation),
                'trials':[{'trial':p.name,'configuration':v['configuration'],'selected_policy':v['selected_policy']} for p,v in trials],
                'frozen_before_evaluation':True}
            bind_run(directory/'selection_before_evaluation.json',frozen)
            write_csv(chosen['threshold_curve'],directory/'validation_threshold_curve.csv')
            if family=='logistic':
                import joblib
                model=joblib.load(path/'model.joblib')
                predictions=model.predict_records(evaluation)
            else:
                import torch
                from transformers import AutoModelForSequenceClassification,AutoTokenizer
                model=AutoModelForSequenceClassification.from_pretrained(path/'model',local_files_only=True)
                tokenizer=AutoTokenizer.from_pretrained(path/'model',local_files_only=True)
                predictions=transformer_predictions(model,tokenizer,evaluation)
            atomic_jsonl(predictions,directory/'evaluation_predictions.jsonl')
            threshold=frozen['policy']['threshold']
            result={'fold_id':fold['fold_id'],'family':family,'selected_trial':path.name,
                'selected_policy':frozen['policy'],'evaluation':decision_metrics(predictions,threshold),
                'argmax_evaluation':decision_metrics(predictions),
                'four_class_argmax_metrics':evaluate_predictions(predictions),
                'four_class_selected_policy_metrics':evaluate_predictions(apply_policy(predictions,threshold)),
                'evidence':'Saved per-claim BM25 top three; identical inputs for compared classifiers.',
                'evaluation_claim_count':len(evaluation),'selection_frozen_before_evaluation':True}
            write_json(result,directory/'evaluation_complete.json')
            print('EVALUATION',json.dumps(result),flush=True)
            del model
            gc.collect()
            if family=='distilbert':
                torch.cuda.empty_cache()


def summarize(args):
    rows=[]
    for path in sorted((args.output/'temporal').glob('*/*/evaluation_complete.json')):
        result=json.loads(path.read_text())
        for policy,key in [('selected','evaluation'),('argmax','argmax_evaluation')]:
            rows.append({'fold_id':result['fold_id'],'family':result['family'],'policy':policy,
                'selected_trial':result['selected_trial'],
                'threshold':result['selected_policy']['threshold'] if policy=='selected' else None,
                **result[key]})
    write_csv(rows,args.output/'temporal_results.csv')
    write_json({'primary_metric':'binary_balanced_accuracy','results':rows,
        'expected_model_window_runs':4,'completed_model_window_runs':len(rows)//2},args.output/'temporal_results.json')
    print(json.dumps({'results':rows,'completed_model_window_runs':len(rows)//2},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','historical','train','summarize'))
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--shared-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fold',choices=('spring_2020','summer_2020'))
    parser.add_argument('--family',choices=('logistic','distilbert'))
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.shared_root.resolve()):
        raise ValueError('Use a personal output directory outside the shared input root')
    args.output.mkdir(parents=True,exist_ok=True)
    globals()[{'prepare':'prepare','historical':'historical','train':'train','summarize':'summarize'}[args.stage]](args)


if __name__=='__main__':
    main()
