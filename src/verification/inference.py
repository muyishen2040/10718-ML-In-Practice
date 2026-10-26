"""Shared resumable verdict inference; labels are used only for reporting."""
from __future__ import annotations

import math
from pathlib import Path

from src.data.averitec import CANONICAL_LABELS
from src.data.evidence import index_claims
from src.evaluation.classification import evaluate_predictions
from src.utils.checkpoints import append_journal, atomic_jsonl, read_journal
from src.utils.io import read_jsonl, write_json
from src.utils.progress import progress


def validate_predictions(rows, records):
    found = index_claims(rows, expected={r["claim_id"] for r in records})
    for r in records:
        p = found[r["claim_id"]]
        values = [float(p.get("probabilities", {}).get(k, float("nan"))) for k in CANONICAL_LABELS]
        if p["true_label"] != r["label"] or p["predicted_label"] not in CANONICAL_LABELS:
            raise ValueError("Prediction labels differ from the evaluated claims")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values) or abs(sum(values) - 1) > 1e-4:
            raise ValueError("Invalid class probability vector")
    return [found[r["claim_id"]] for r in records]


def predict_cached(records, predict_batch, output: Path, *, batch_size=8, description="Predict verdicts"):
    output.mkdir(parents=True, exist_ok=True)
    final, partial = output / "predictions.jsonl", output / "predictions.partial.jsonl"
    if final.exists():
        predictions = validate_predictions(read_jsonl(final), records)
    else:
        done = read_journal(partial, {r["claim_id"] for r in records})
        if done:
            subset = [r for r in records if r["claim_id"] in done]
            validate_predictions(list(done.values()), subset)
        pending = [r for r in records if r["claim_id"] not in done]
        with progress(total=len(records), initial=len(done), description=description, unit="claim") as bar:
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                probabilities = predict_batch(batch)
                new = []
                for r, values in zip(batch, probabilities, strict=True):
                    row = {"claim_id": r["claim_id"], "true_label": r["label"],
                           "predicted_label": CANONICAL_LABELS[max(range(4), key=lambda i: values[i])],
                           "probabilities": dict(zip(CANONICAL_LABELS, map(float, values), strict=True)),
                           "evidence_passage_ids": [p["passage_id"] for p in (r.get("retrieved_evidence") or [])[:3]]}
                    new.append(row)
                validate_predictions(new, batch)
                append_journal(partial, new)
                done.update({r["claim_id"]: r for r in new})
                bar.update(len(new))
        predictions = [done[r["claim_id"]] for r in records]
        atomic_jsonl(predictions, final)
    metrics = evaluate_predictions(predictions)
    write_json(metrics, output / "metrics.json")
    return metrics
