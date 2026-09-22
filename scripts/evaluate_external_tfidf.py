"""Run the claim-only three-class AVeriTeC-to-Snopes transfer baseline.

This is deliberately not a full external retrieval evaluation. It establishes a
reproducible distribution-shift result before constructing the cited-source
passage corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.snopes import SNOPES_EXTERNAL_LABELS  # noqa: E402
from src.evaluation.classification import evaluate_predictions  # noqa: E402
from src.verification.tfidf_logreg import TfidfLogRegVerifier  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "external_tfidf")
    parser.add_argument("--random-state", type=int, default=718)
    args = parser.parse_args()

    averitec_path = args.data_root / "processed" / "averitec" / "train.jsonl"
    snopes_path = args.data_root / "external" / "snopes" / "processed" / "records.jsonl"
    averitec_train = read_jsonl(averitec_path)
    training_records = [record for record in averitec_train if record.get("label") in SNOPES_EXTERNAL_LABELS]
    snopes_records = read_jsonl(snopes_path)
    verifier = TfidfLogRegVerifier(
        evidence_mode="claim_only",
        class_weight="balanced",
        random_state=args.random_state,
    ).fit(training_records)
    predictions = verifier.predict_records(snopes_records)
    metrics = evaluate_predictions(predictions, labels=SNOPES_EXTERNAL_LABELS)
    config = {
        "model": "tfidf_logreg",
        "evidence_mode": "claim_only",
        "training_dataset": "AVeriTeC train",
        "training_class_filter": list(SNOPES_EXTERNAL_LABELS),
        "excluded_averitec_conflicting_count": len(averitec_train) - len(training_records),
        "evaluation_dataset": "DisinfoMM English Snopes",
        "evaluation_classes": list(SNOPES_EXTERNAL_LABELS),
        "random_state": args.random_state,
        "warning": "This is a claim-only external transfer baseline, not an evidence-retrieval result.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions, args.output_dir / "averitec_to_snopes_claim_only_predictions.jsonl")
    (args.output_dir / "averitec_to_snopes_claim_only_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "averitec_to_snopes_claim_only_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    joblib.dump(verifier, args.output_dir / "averitec_to_snopes_claim_only_model.joblib")
    print(json.dumps({**config, **metrics}, indent=2))


if __name__ == "__main__":
    main()
