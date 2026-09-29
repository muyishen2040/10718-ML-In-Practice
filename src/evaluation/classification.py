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
    report: dict[str, Any] = {
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
    supported = "Supported"
    report["decision_error_rates"] = {
        "false_reassurance_count": sum(true != supported and predicted == supported for true, predicted in zip(true_labels, predicted_labels, strict=True)),
        "false_reassurance_rate": sum(true != supported and predicted == supported for true, predicted in zip(true_labels, predicted_labels, strict=True)) / len(materialized),
        "false_alarm_count": sum(true == supported and predicted != supported for true, predicted in zip(true_labels, predicted_labels, strict=True)),
        "false_alarm_rate": sum(true == supported and predicted != supported for true, predicted in zip(true_labels, predicted_labels, strict=True)) / len(materialized),
    }
    if all(isinstance(row.get("probabilities"), dict) for row in materialized):
        report["calibration"] = calibration_metrics(materialized, labels)
    return report


def calibration_metrics(predictions: Sequence[dict[str, Any]], labels: Sequence[str], bin_count: int = 10) -> dict[str, float | int]:
    """Compute multiclass Brier score and ECE from model probabilities.

    LLM self-reported confidence is intentionally not accepted as a probability
    distribution and therefore does not receive a calibration score.
    """
    brier_total = 0.0
    confidence_bins: list[list[tuple[float, float]]] = [[] for _ in range(bin_count)]
    for row in predictions:
        probabilities = row["probabilities"]
        values = {label: float(probabilities.get(label, 0.0)) for label in labels}
        brier_total += sum((values[label] - float(row["true_label"] == label)) ** 2 for label in labels)
        predicted_label, confidence = max(values.items(), key=lambda item: item[1])
        bin_index = min(int(confidence * bin_count), bin_count - 1)
        confidence_bins[bin_index].append((confidence, float(predicted_label == row["true_label"])))
    ece = 0.0
    for values in confidence_bins:
        if not values:
            continue
        avg_confidence = sum(value[0] for value in values) / len(values)
        avg_accuracy = sum(value[1] for value in values) / len(values)
        ece += len(values) / len(predictions) * abs(avg_confidence - avg_accuracy)
    return {"multiclass_brier_score": brier_total / len(predictions), "expected_calibration_error": ece, "ece_bin_count": bin_count}
