"""Retrieval metrics for passage-level relevance judgments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    for rank, passage_id in enumerate(retrieved_ids, start=1):
        if passage_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Recall is undefined for a claim without relevance judgments")
    return float(bool(set(retrieved_ids[:k]) & relevant))


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]], qrels: Mapping[str, Iterable[str]], k: int = 3
) -> dict[str, float | int]:
    """Evaluate only judged claims and report their number explicitly."""
    judged_ids = sorted(set(rankings) & set(qrels))
    if not judged_ids:
        raise ValueError("No overlap between rankings and relevance judgments")
    recalls = [recall_at_k(rankings[claim_id], qrels[claim_id], k) for claim_id in judged_ids]
    reciprocal_ranks = [reciprocal_rank(rankings[claim_id], qrels[claim_id]) for claim_id in judged_ids]
    return {
        f"recall_at_{k}": sum(recalls) / len(recalls),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "judged_claim_count": len(judged_ids),
    }
