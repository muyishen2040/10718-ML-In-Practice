"""Retrieval metrics for passage-level relevance judgments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    for rank, passage_id in enumerate(retrieved_ids, start=1):
        if passage_id in relevant:
            return 1.0 / rank
    return 0.0


def evidence_hit_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Whether at least one judged evidence unit appears in the first *k* results."""
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Evidence hit rate is undefined without relevance judgments")
    return float(bool(set(retrieved_ids[:k]) & relevant))


def passage_recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of all judged relevant passages returned in the first *k* results."""
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Passage recall is undefined without relevance judgments")
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Relevant retrieved items divided by k; missing ranks count as non-relevant."""
    if k < 1:
        raise ValueError("k must be at least one")
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / k


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Backward-compatible alias for the project-defined evidence hit metric."""
    return evidence_hit_at_k(retrieved_ids, relevant_ids, k)


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]], qrels: Mapping[str, Iterable[str]], k: int = 3
) -> dict[str, float | int]:
    """Evaluate only judged claims and report their number explicitly."""
    judged_ids = sorted(set(rankings) & set(qrels))
    if not judged_ids:
        raise ValueError("No overlap between rankings and relevance judgments")
    hit_rates = [evidence_hit_at_k(rankings[claim_id], qrels[claim_id], k) for claim_id in judged_ids]
    passage_recalls = [passage_recall_at_k(rankings[claim_id], qrels[claim_id], k) for claim_id in judged_ids]
    precisions = [precision_at_k(rankings[claim_id], qrels[claim_id], k) for claim_id in judged_ids]
    reciprocal_ranks = [reciprocal_rank(rankings[claim_id], qrels[claim_id]) for claim_id in judged_ids]
    return {
        f"evidence_hit_at_{k}": sum(hit_rates) / len(hit_rates),
        f"passage_recall_at_{k}": sum(passage_recalls) / len(passage_recalls),
        f"precision_at_{k}": sum(precisions) / len(precisions),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "judged_claim_count": len(judged_ids),
    }
