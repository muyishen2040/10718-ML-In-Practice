"""Diagnostic reports joining retrieval coverage and final verdict predictions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.data.averitec import CANONICAL_LABELS
from src.evaluation.classification import evaluate_predictions
from src.retrieval.metrics import evidence_hit_at_k


def evaluate_by_retrieval_coverage(
    predictions: Iterable[dict[str, Any]], rankings: Iterable[dict[str, Any]], qrels: dict[str, list[str]], k: int = 3
) -> dict[str, Any]:
    """Report verdict quality overall and conditioned on URL/passage coverage.

    This is diagnostic conditioning, not a causal claim that retrieval alone
    caused every miss or hit.
    """
    ranking_by_claim = {str(row["claim_id"]): row for row in rankings}
    covered, missed, unjudged = [], [], []
    for prediction in predictions:
        claim_id = str(prediction["claim_id"])
        ranking = ranking_by_claim.get(claim_id)
        if ranking is None or claim_id not in qrels:
            unjudged.append(prediction)
            continue
        retrieved_ids = [str(item["passage_id"]) for item in ranking.get("retrieved") or []]
        if evidence_hit_at_k(retrieved_ids, qrels[claim_id], k):
            covered.append(prediction)
        else:
            missed.append(prediction)
    report: dict[str, Any] = {
        "k": k,
        "judged_prediction_count": len(covered) + len(missed),
        "unjudged_prediction_count": len(unjudged),
        "retrieval_covered_count": len(covered),
        "retrieval_missed_count": len(missed),
        "warning": "Conditioning on retrieval coverage is diagnostic, not causal attribution.",
    }
    if covered:
        report["verdict_metrics_when_evidence_covered"] = evaluate_predictions(covered, CANONICAL_LABELS)
    if missed:
        report["verdict_metrics_when_evidence_missed"] = evaluate_predictions(missed, CANONICAL_LABELS)
    return report
