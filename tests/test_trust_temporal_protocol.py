from copy import deepcopy

import pytest

from src.evaluation.temporal_protocol import build_manifest, claim_group, validate_manifest
from src.evaluation.trust_decisions import decision_metrics, from_error_counts, threshold_study


def test_binary_metric_uses_class_denominators_and_flags_both_classes():
    result = from_error_counts(122, 378, 74, 78)
    assert result['binary_balanced_accuracy'] == pytest.approx(.5 * (44 / 122 + 304 / 378))
    assert result['binary_accuracy'] == pytest.approx(.696)
    rows = [{'claim_id':str(i), 'true_label':label, 'predicted_label':'Refuted'}
            for i,label in enumerate(['Supported','Supported','Refuted','Not Enough Evidence'])]
    assert decision_metrics(rows)['binary_balanced_accuracy'] == .5
    with pytest.raises(ValueError):
        decision_metrics(rows[:2])


def test_argmax_and_supported_threshold_are_distinct_and_duplicate_ids_fail():
    rows = [{'claim_id':'a','true_label':'Supported','predicted_label':'Supported',
             'probabilities':{'Supported':.4,'Refuted':.3,'Not Enough Evidence':.2,'Conflicting Evidence':.1}},
            {'claim_id':'b','true_label':'Refuted','predicted_label':'Refuted',
             'probabilities':{'Supported':.1,'Refuted':.7,'Not Enough Evidence':.1,'Conflicting Evidence':.1}}]
    assert decision_metrics(rows)['binary_balanced_accuracy'] == 1
    assert decision_metrics(rows,.5)['binary_balanced_accuracy'] == .5
    curve,chosen = threshold_study(rows)
    assert len(curve) == 18
    assert chosen['policy'] == 'four_way_argmax'
    with pytest.raises(ValueError,match='duplicate'):
        decision_metrics(rows + [rows[0]])


def temporal_rows():
    dates = ['2019-01-01','2019-06-01','2019-12-01','2020-01-01','2020-02-01',
             '2020-03-01','2020-04-01','2020-06-03','2020-06-04','2020-07-01','2020-08-01']
    labels = ['Supported','Refuted','Not Enough Evidence','Conflicting Evidence']
    return [{'claim_id':f'c{i}:{j}','claim':f'claim {i} {j}','label':labels[j % 4],
             'metadata':{'claim_date':date}} for i,date in enumerate(dates) for j in range(8)]


def test_protocol_quarantines_dates_purges_duplicates_and_checks_source_hash():
    rows = temporal_rows()
    rows[0]['claim'] = rows[-1]['claim']
    rows += [{'claim_id':'missing','claim':'no date','label':'Supported','metadata':{}},
             {'claim_id':'old','claim':'old date','label':'Refuted','metadata':{'claim_date':'22-8-1922'}}]
    manifest = build_manifest(rows,'source-hash')
    assert len(manifest['excluded']) == 2
    indexed = {r['claim_id']:r for r in rows}
    for fold in manifest['folds']:
        groups = [{claim_group(indexed[cid]) for cid in fold['ids'][name]}
                  for name in ('fit','validation','evaluation')]
        assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert 'c0:0' in manifest['folds'][1]['purged_earlier_duplicates_of_evaluation']
    with pytest.raises(ValueError,match='different source'):
        validate_manifest(manifest,rows,'changed')
    changed = deepcopy(manifest)
    changed['folds'][0]['ids']['evaluation'].append(changed['folds'][0]['ids']['fit'][0])
    with pytest.raises(ValueError):
        validate_manifest(changed,rows,'source-hash')
