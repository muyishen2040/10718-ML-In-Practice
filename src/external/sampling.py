"""Deterministic, label-stratified sampling for manageable external corpora."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any


def stratified_sample(
    records: Iterable[dict[str, Any]], labels: Sequence[str], total_size: int, seed: int
) -> list[dict[str, Any]]:
    """Sample as evenly as possible across labels without oversampling records.

    Selection uses labels only to create an evaluation set with meaningful
    coverage of rare classes. Labels remain unavailable to every retriever and
    verifier at inference time.
    """
    if total_size < 1:
        raise ValueError("total_size must be positive")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        label = record.get("label")
        if label in labels:
            buckets[str(label)].append(record)
    unknown = set(labels) - set(buckets)
    if unknown:
        raise ValueError(f"Cannot sample absent labels: {sorted(unknown)}")
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[dict[str, Any]] = []
    remaining = total_size
    active_labels = list(labels)
    while active_labels and remaining:
        quota = max(1, remaining // len(active_labels))
        next_active: list[str] = []
        for label in active_labels:
            take = min(quota, len(buckets[label]))
            selected.extend(buckets[label][:take])
            buckets[label] = buckets[label][take:]
            remaining -= take
            if buckets[label]:
                next_active.append(label)
            if not remaining:
                break
        active_labels = next_active
    if remaining:
        raise ValueError("Requested more records than are available")
    return sorted(selected, key=lambda record: str(record["claim_id"]))


def sample_audit(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(records)
    return {
        "record_count": len(materialized),
        "label_counts": dict(sorted(Counter(str(record["label"]) for record in materialized).items())),
        "source_url_count": sum(len(record.get("source_urls") or []) for record in materialized),
    }
