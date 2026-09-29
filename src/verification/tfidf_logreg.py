"""Transparent TF-IDF + multinomial logistic-regression verifier baseline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.averitec import CANONICAL_LABELS

EvidenceMode = Literal["claim_only", "gold", "gold_all", "retrieved"]


def evidence_texts(record: dict[str, Any], evidence_mode: EvidenceMode, max_evidence: int = 3) -> list[str]:
    """Return the exact evidence texts exposed to a verifier.

    `gold_all` is an oracle upper bound and deliberately bypasses the user-facing
    three-item cap. Every deployable/end-to-end mode uses at most three items.
    """
    if max_evidence < 1:
        raise ValueError("max_evidence must be at least one")
    if evidence_mode == "claim_only":
        return []
    if evidence_mode in {"gold", "gold_all"}:
        evidence = record.get("evidence") or []
    elif evidence_mode == "retrieved":
        evidence = record.get("retrieved_evidence") or []
    else:
        raise ValueError(f"Unsupported evidence mode: {evidence_mode}")
    texts = []
    for item in evidence:
        value = item.get("text") if isinstance(item, dict) else item
        text = str(value or "").strip()
        if text:
            texts.append(text)
    return texts if evidence_mode == "gold_all" else texts[:max_evidence]


def compose_input(record: dict[str, Any], evidence_mode: EvidenceMode, max_evidence: int = 3) -> str:
    """Format one canonical record for the linear verifier.

    Gold answers are allowed only for the isolated gold-evidence experiment. The
    end-to-end BM25 experiment will provide retrieved passages in a later phase.
    """
    claim = str(record["claim"]).strip()
    passages = evidence_texts(record, evidence_mode, max_evidence)
    evidence_text = "\n".join(f"[EVIDENCE {index}] {text}" for index, text in enumerate(passages, start=1))
    return f"[CLAIM] {claim}\n{evidence_text}" if evidence_text else f"[CLAIM] {claim}"


@dataclass
class TfidfLogRegVerifier:
    """A reproducible supervised baseline with explicit label ordering."""

    evidence_mode: EvidenceMode = "gold"
    max_evidence: int = 3
    min_df: int = 1
    max_features: int | None = None
    class_weight: str | None = "balanced"
    random_state: int = 718
    max_iter: int = 2000

    def __post_init__(self) -> None:
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=self.min_df,
                        max_features=self.max_features,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "logreg",
                    LogisticRegression(
                        max_iter=self.max_iter,
                        class_weight=self.class_weight,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def fit(self, records: Iterable[dict[str, Any]]) -> "TfidfLogRegVerifier":
        materialized = list(records)
        labels = [record.get("label") for record in materialized]
        unknown = set(labels) - set(CANONICAL_LABELS)
        if unknown:
            raise ValueError(f"Training records contain invalid labels: {sorted(unknown)}")
        self.pipeline.fit([compose_input(record, self.evidence_mode, self.max_evidence) for record in materialized], labels)
        return self

    def predict_records(self, records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        materialized = list(records)
        inputs = [compose_input(record, self.evidence_mode, self.max_evidence) for record in materialized]
        predicted = self.pipeline.predict(inputs)
        probabilities = self.pipeline.predict_proba(inputs)
        class_order = list(self.pipeline.named_steps["logreg"].classes_)
        output = []
        for record, label, row_probabilities in zip(materialized, predicted, probabilities, strict=True):
            output.append(
                {
                    "claim_id": record["claim_id"],
                    "true_label": record.get("label"),
                    "predicted_label": str(label),
                    "probabilities": {
                        class_label: float(probability)
                        for class_label, probability in zip(class_order, row_probabilities, strict=True)
                    },
                    "evidence_item_count": len(evidence_texts(record, self.evidence_mode, self.max_evidence)),
                }
            )
        return output
