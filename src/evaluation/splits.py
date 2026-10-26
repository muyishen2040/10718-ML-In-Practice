"""Train-only holdouts for diagnostics; never uses official dev claims."""
from __future__ import annotations

from datetime import datetime
from math import ceil
import re
import unicodedata
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def parse_claim_date(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text[:10] if fmt == "%Y-%m-%d" else text, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def selection_split(records: list[dict], fraction: float, seed: int, policy: str = "stratified") -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(records))
    if policy == "stratified":
        return train_test_split(indices, test_size=fraction, random_state=seed, stratify=[r["label"] for r in records])
    if policy == "grouped":
        groups = [" ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", r["claim"]).casefold())) for r in records]
        labels = [r["label"] for r in records]
        folds = min(max(2, round(1 / fraction)), min(labels.count(label) for label in set(labels)))
        if folds < 2:
            raise ValueError("Too few examples for a grouped train holdout")
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        return next(splitter.split(indices, labels, groups))
    if policy != "temporal":
        raise ValueError(f"Unknown validation policy: {policy}")
    dates = [parse_claim_date(r.get("metadata", {}).get("claim_date")) for r in records]
    if any(date is None for date in dates):
        raise ValueError("Temporal holdout requires a parseable claim_date for every train claim; audit missing dates first")
    ordered = sorted(indices, key=lambda i: (dates[i], str(records[i]["claim_id"])))
    boundary = dates[ordered[max(1, len(records) - ceil(fraction * len(records)))]]
    train = np.asarray([i for i in ordered if dates[i] < boundary], dtype=int)
    holdout = np.asarray([i for i in ordered if dates[i] >= boundary], dtype=int)
    if not len(train) or not len(holdout):
        raise ValueError("Temporal dates do not admit a nonempty earlier/later split")
    return train, holdout
