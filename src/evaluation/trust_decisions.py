"""Supported-versus-other decision metrics and validation-only threshold selection."""
from __future__ import annotations

import math

from src.data.averitec import CANONICAL_LABELS
from src.data.evidence import index_claims


def from_error_counts(supported_count, other_count, false_reassurance, false_alarm):
    if min(supported_count, other_count) <= 0:
        raise ValueError('Balanced accuracy requires both decision classes')
    if not 0 <= false_alarm <= supported_count or not 0 <= false_reassurance <= other_count:
        raise ValueError('Decision errors exceed their class denominators')
    total = supported_count + other_count
    return {
        'claim_count': total, 'supported_count': supported_count, 'other_count': other_count,
        'binary_balanced_accuracy': 1 - .5 * (false_alarm / supported_count + false_reassurance / other_count),
        'binary_accuracy': 1 - (false_alarm + false_reassurance) / total,
        'false_reassurance_count': false_reassurance,
        'false_reassurance_conditional_rate': false_reassurance / other_count,
        'false_alarm_count': false_alarm, 'false_alarm_conditional_rate': false_alarm / supported_count,
        'false_reassurance_all_claim_rate': false_reassurance / total,
        'false_alarm_all_claim_rate': false_alarm / total,
    }


def apply_policy(rows, threshold=None):
    output = []
    index_claims(rows)
    for row in rows:
        if row['true_label'] not in CANONICAL_LABELS or row['predicted_label'] not in CANONICAL_LABELS:
            raise ValueError('Unknown verdict label')
        prediction = dict(row)
        if threshold is not None:
            if not 0 <= threshold <= 1:
                raise ValueError('Invalid Supported-score threshold')
            probabilities = row.get('probabilities', {})
            values = [float(probabilities[label]) for label in CANONICAL_LABELS]
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values) or abs(sum(values) - 1) > 1e-4:
                raise ValueError('Expected normalized classifier scores')
            prediction['predicted_label'] = 'Supported' if probabilities['Supported'] >= threshold else max(
                (label for label in CANONICAL_LABELS if label != 'Supported'), key=lambda label: probabilities[label])
        output.append(prediction)
    return output


def decision_metrics(rows, threshold=None):
    predictions = apply_policy(rows, threshold)
    supported = sum(row['true_label'] == 'Supported' for row in predictions)
    fr = sum(row['true_label'] != 'Supported' and row['predicted_label'] == 'Supported' for row in predictions)
    fa = sum(row['true_label'] == 'Supported' and row['predicted_label'] != 'Supported' for row in predictions)
    return from_error_counts(supported, len(predictions) - supported, fr, fa)


def selection_key(report):
    threshold = report.get('threshold')
    return (-report['binary_balanced_accuracy'], report['false_reassurance_conditional_rate'],
            report['false_alarm_conditional_rate'], threshold is not None,
            abs(threshold - .5) if threshold is not None else 0,
            threshold if threshold is not None else 0)


def threshold_study(rows):
    reports = [{'policy': 'four_way_argmax', 'threshold': None, **decision_metrics(rows)}]
    for value in range(10, 91, 5):
        threshold = value / 100
        reports.append({'policy': 'supported_score_threshold', 'threshold': threshold,
                        **decision_metrics(rows, threshold)})
    return reports, min(reports, key=selection_key)
