"""Diagnostic reports joining retrieval coverage and final verdict predictions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.data.averitec import CANONICAL_LABELS
from src.evaluation.classification import evaluate_predictions
from src.retrieval.metrics import evidence_hit_at_k


def annotate_retrieval_coverage(
    predictions: Iterable[dict[str, Any]], rankings: Iterable[dict[str, Any]], qrels: dict[str, list[str]], k: int = 3
) -> list[dict[str, Any]]:
    """Attach an auditable evidence-coverage status to each verdict prediction.

    A prediction may explicitly identify the passages the verifier used.  This
    matters for the LLM reranking baseline: its selected three are the evidence
    set being diagnosed, rather than the original BM25 top three.
    """
    ranking_by_claim = {str(row["claim_id"]): row for row in rankings}
    annotations = []
    for prediction in predictions:
        claim_id = str(prediction["claim_id"])
        ranking = ranking_by_claim.get(claim_id)
        evidence_ids = prediction.get("evidence_passage_ids")
        if evidence_ids is None and ranking is not None:
            evidence_ids = [item["passage_id"] for item in ranking.get("retrieved") or []]
        if evidence_ids is None or claim_id not in qrels:
            status = "unjudged"
        elif evidence_hit_at_k([str(item) for item in evidence_ids], qrels[claim_id], k):
            status = "evidence_covered"
        else:
            status = "evidence_missed"
        annotations.append(
            {
                "claim_id": claim_id,
                "retrieval_coverage": status,
                "evidence_passage_ids_used": [str(item) for item in evidence_ids or []],
            }
        )
    return annotations


def evaluate_by_retrieval_coverage(
    predictions: Iterable[dict[str, Any]], rankings: Iterable[dict[str, Any]], qrels: dict[str, list[str]], k: int = 3
) -> dict[str, Any]:
    """Report verdict quality overall and conditioned on URL/passage coverage.

    This is diagnostic conditioning, not a causal claim that retrieval alone
    caused every miss or hit.
    """
    materialized_predictions = list(predictions)
    coverage_by_claim = {
        row["claim_id"]: row["retrieval_coverage"]
        for row in annotate_retrieval_coverage(materialized_predictions, rankings, qrels, k)
    }
    covered = [row for row in materialized_predictions if coverage_by_claim[row["claim_id"]] == "evidence_covered"]
    missed = [row for row in materialized_predictions if coverage_by_claim[row["claim_id"]] == "evidence_missed"]
    unjudged = [row for row in materialized_predictions if coverage_by_claim[row["claim_id"]] == "unjudged"]
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
