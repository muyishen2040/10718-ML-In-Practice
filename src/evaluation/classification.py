"""Standardized four-way classification reporting."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data.averitec import CANONICAL_LABELS


def evaluate_predictions(
    predictions: Iterable[dict[str, Any]], labels: Sequence[str] = CANONICAL_LABELS
) -> dict[str, Any]:
    """Compute standardized verdict metrics from prediction artifacts."""
    materialized = list(predictions)
    if not materialized:
        raise ValueError("Cannot evaluate an empty prediction list")
    true_labels = [str(row["true_label"]) for row in materialized]
    predicted_labels = [str(row["predicted_label"]) for row in materialized]
    invalid = (set(true_labels) | set(predicted_labels)) - set(labels)
    if invalid:
        raise ValueError(f"Predictions contain invalid labels: {sorted(invalid)}")
    return {
        "claim_count": len(materialized),
        "macro_f1": float(f1_score(true_labels, predicted_labels, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "per_class": classification_report(
            true_labels,
            predicted_labels,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "labels": list(labels),
            "matrix": confusion_matrix(true_labels, predicted_labels, labels=labels).tolist(),
        },
    }
