"""Score a complete prediction file against an official split or released temporal partition."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.evidence import index_claims
from src.evaluation.classification import evaluate_predictions
from src.evaluation.temporal_protocol import validate_manifest
from src.evaluation.trust_decisions import apply_policy, decision_metrics
from src.utils.io import read_jsonl, write_json
from src.utils.runs import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--fold')
    parser.add_argument('--partition', choices=('fit', 'validation', 'evaluation'), default='evaluation')
    parser.add_argument('--split', choices=('train', 'dev'), default='dev')
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    claims_path = args.data_root / 'processed/averitec' / ('train.jsonl' if args.manifest else f'{args.split}.jsonl')
    records = read_jsonl(claims_path)
    gold = index_claims(records)
    protocol = None
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        validate_manifest(manifest, records, sha256_file(claims_path))
        fold = next((fold for fold in manifest['folds'] if fold['fold_id'] == args.fold), None)
        if fold is None:
            parser.error('--fold must identify a released temporal window')
        expected = set(fold['ids'][args.partition])
        protocol = manifest['protocol_id']
    else:
        expected = set(gold)
    predictions = read_jsonl(args.predictions)
    index_claims(predictions, expected)
    for row in predictions:
        label = gold[row['claim_id']]['label']
        if row.get('true_label') is not None and row['true_label'] != label:
            raise ValueError('Prediction gold label disagrees with the evaluation claims')
        row['true_label'] = label
    result = {'protocol_id': protocol, 'fold_id': args.fold, 'partition': args.partition if args.manifest else args.split,
              'threshold': args.threshold, **decision_metrics(predictions, args.threshold),
              'four_class_metrics': evaluate_predictions(apply_policy(predictions, args.threshold))}
    if args.output:
        write_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
