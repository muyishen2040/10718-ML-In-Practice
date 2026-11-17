"""Versioned chronological fit/validation/evaluation windows inside official train."""
from __future__ import annotations

from collections import Counter
import re
import unicodedata

from src.data.evidence import index_claims
from src.evaluation.splits import parse_claim_date

PROTOCOL_ID = 'trust_temporal_20261117_v1'
WINDOWS = [('spring_2020', '2020-03-01', '2020-06-04'),
           ('summer_2020', '2020-06-04', '2020-08-26')]


def claim_group(row):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', row['claim']).casefold()))


def date_of(row):
    return parse_claim_date(row.get('metadata', {}).get('claim_date'))


def describe(rows):
    dates = [date_of(row) for row in rows]
    return {'claim_count': len(rows), 'first_date': min(dates), 'last_date': max(dates),
            'class_counts': dict(Counter(row['label'] for row in rows)),
            'normalized_claim_groups': len({claim_group(row) for row in rows})}


def build_manifest(records, source_sha256):
    index_claims(records)
    excluded, usable = [], []
    for row in records:
        date = date_of(row)
        reason = 'missing_or_unparseable_date' if date is None else 'pre_2000_date_quarantined' if date < '2000-01-01' else None
        if reason:
            excluded.append({'claim_id': row['claim_id'], 'raw_date': row.get('metadata', {}).get('claim_date'), 'reason': reason})
        else:
            usable.append(row)
    usable.sort(key=lambda row: (date_of(row), row['claim_id']))
    folds = []
    for fold_id, evaluation_start, evaluation_end in WINDOWS:
        evaluation = [row for row in usable if evaluation_start <= date_of(row) < evaluation_end]
        evaluation_groups = {claim_group(row) for row in evaluation}
        prefix = [row for row in usable if date_of(row) < evaluation_start]
        purged_outer = [row['claim_id'] for row in prefix if claim_group(row) in evaluation_groups]
        prefix = [row for row in prefix if claim_group(row) not in evaluation_groups]
        boundary = date_of(prefix[int(len(prefix) * .8)])
        validation = [row for row in prefix if date_of(row) >= boundary]
        validation_groups = {claim_group(row) for row in validation}
        earlier = [row for row in prefix if date_of(row) < boundary]
        purged_inner = [row['claim_id'] for row in earlier if claim_group(row) in validation_groups]
        fit = [row for row in earlier if claim_group(row) not in validation_groups]
        partitions = {'fit': fit, 'validation': validation, 'evaluation': evaluation}
        folds.append({'fold_id': fold_id, 'evaluation_start': evaluation_start, 'evaluation_end_exclusive': evaluation_end,
                      'validation_start': boundary, 'ids': {key: [row['claim_id'] for row in rows] for key, rows in partitions.items()},
                      'summary': {key: describe(rows) for key, rows in partitions.items()},
                      'purged_earlier_duplicates_of_evaluation': purged_outer,
                      'purged_fit_duplicates_of_validation': purged_inner})
    manifest = {'schema_version': 1, 'protocol_id': PROTOCOL_ID, 'source_train_sha256': source_sha256,
                'primary_metric': 'binary_balanced_accuracy', 'display_k': 3,
                'policy': {'validation_fraction_of_earlier_prefix': .2,
                           'date_ties': 'Keep all records of a boundary date on the later side.',
                           'missing_dates': 'Exclude from chronological experiments and report IDs.',
                           'pre_2000_dates': 'Quarantine without imputing a corrected date.',
                           'duplicates': 'Purge earlier copies when normalized claim text occurs in a later partition.',
                           'refit': 'Evaluate the selected fit-trained checkpoint directly, retaining validation as held out.',
                           'thresholds': [i / 100 for i in range(10, 91, 5)],
                           'selection': 'Validation balanced accuracy, conditional false reassurance, conditional false alarms, then prefer argmax on an exact tie.'},
                'excluded': excluded, 'folds': folds}
    validate_manifest(manifest, records, source_sha256)
    return manifest


def validate_manifest(manifest, records, source_sha256):
    if manifest['source_train_sha256'] != source_sha256:
        raise ValueError('Temporal manifest uses different source claims')
    indexed = index_claims(records)
    for fold in manifest['folds']:
        partitions = {}
        for name, ids in fold['ids'].items():
            if len(ids) != len(set(ids)) or set(ids) - set(indexed):
                raise ValueError('Unknown or duplicate split claim IDs')
            partitions[name] = [indexed[cid] for cid in ids]
            if describe(partitions[name]) != fold['summary'][name]:
                raise ValueError('Fold summary differs from its claim IDs')
        for earlier, later in [('fit', 'validation'), ('fit', 'evaluation'), ('validation', 'evaluation')]:
            a, b = partitions[earlier], partitions[later]
            if max(date_of(row) for row in a) >= min(date_of(row) for row in b):
                raise ValueError('Fold is not strictly chronological')
            if {claim_group(row) for row in a} & {claim_group(row) for row in b}:
                raise ValueError('Duplicate claim group crosses partitions')
