"""Shared, leakage-conscious helpers for weakly supervised BM25 rerankers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from scripts.run_bm25 import load_qrels
from src.data.corpus import canonical_url
from src.retrieval.metrics import evaluate_rankings
from src.utils.io import read_jsonl


def load_inputs(train_claims: Path, train_rankings: Path, dev_rankings: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    claims = read_jsonl(train_claims)
    train = read_jsonl(train_rankings)
    dev = read_jsonl(dev_rankings)
    by_id = {str(row["claim_id"]): row for row in claims}
    if len(by_id) != len(claims) or len({str(row["claim_id"]) for row in train}) != len(train):
        raise ValueError("Duplicate train claim IDs")
    if set(by_id) != {str(row["claim_id"]) for row in train}:
        raise ValueError("Train claims and train rankings must have identical claim IDs")
    if len({str(row["claim_id"]) for row in dev}) != len(dev):
        raise ValueError("Duplicate dev claim IDs")
    if set(by_id) & {str(row["claim_id"]) for row in dev}:
        raise ValueError("Train and dev claim IDs overlap")
    for row in train + dev:
        ids = [str(item["passage_id"]) for item in row.get("retrieved", [])]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate passage ID in {row['claim_id']}")
    return train, dev, by_id


def weak_labels(row: dict[str, Any], claim: dict[str, Any], top_n: int) -> list[bool]:
    gold = {canonical_url(url) for url in claim.get("source_urls") or []}
    gold.discard(None)
    return [bool(canonical_url(item.get("url")) in gold) for item in row.get("retrieved", [])[:top_n]]


def annotated_train_split(rows: list[dict[str, Any]], claims: dict[str, dict[str, Any]], seed: int, validation_fraction: float) -> tuple[list[int], list[int]]:
    annotated = [index for index, row in enumerate(rows) if any(canonical_url(url) for url in claims[str(row["claim_id"])].get("source_urls") or [])]
    if len(annotated) < 20:
        raise ValueError("Too few train claims have annotated source URLs")
    covered = [any(weak_labels(rows[index], claims[str(rows[index]["claim_id"])], 20)) for index in annotated]
    stratify = covered if min(covered.count(True), covered.count(False)) >= 2 else None
    fit, val = train_test_split(annotated, test_size=validation_fraction, random_state=seed, stratify=stratify)
    return sorted(fit), sorted(val)


def normalized_bm25(row: dict[str, Any], top_n: int) -> list[float]:
    scores = [float(item.get("score") or 0.0) for item in row.get("retrieved", [])[:top_n]]
    if not scores:
        return []
    if not all(math.isfinite(score) for score in scores):
        raise ValueError(f"Non-finite BM25 score in {row['claim_id']}")
    low, high = min(scores), max(scores)
    return [(score - low) / (high - low) if high > low else 0.0 for score in scores]


def rerank(row: dict[str, Any], model_scores: list[float], blend: float, top_n: int, score_key: str) -> dict[str, Any]:
    candidates = row.get("retrieved", [])[:top_n]
    if len(candidates) != len(model_scores):
        raise ValueError("Model score count differs from candidate count")
    if not all(math.isfinite(float(value)) for value in model_scores):
        raise ValueError("Non-finite learned score")
    bm25 = normalized_bm25(row, top_n)
    low, high = (min(model_scores), max(model_scores)) if model_scores else (0.0, 0.0)
    scaled = [(float(score) - low) / (high - low) if high > low else 0.0 for score in model_scores]
    ranked = []
    for item, lexical, learned in zip(candidates, bm25, scaled, strict=True):
        ranked.append({**item, "first_stage_score": item.get("score"), score_key: learned,
                       "score": (1.0 - blend) * lexical + blend * learned})
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["passage_id"])))
    return {"claim_id": str(row["claim_id"]), "claim": str(row["claim"]), "retrieved": ranked}


def proxy_hit(rows: list[dict[str, Any]], claims: dict[str, dict[str, Any]], top_n: int, k: int = 3) -> float:
    if not rows:
        raise ValueError("No validation rows")
    return sum(any(weak_labels(row, claims[str(row["claim_id"])], k)) for row in rows) / len(rows)


def dev_metrics(rows: list[dict[str, Any]], qrels_path: Path, k: int = 3) -> dict[str, float | int]:
    rankings = {str(row["claim_id"]): [str(item["passage_id"]) for item in row["retrieved"]] for row in rows}
    return evaluate_rankings(rankings, load_qrels(qrels_path), k=k)
