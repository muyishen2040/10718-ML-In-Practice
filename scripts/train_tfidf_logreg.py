"""Train and evaluate the TF-IDF + logistic-regression verifier baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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
    parser.add_argument("--evidence-mode", choices=("claim_only", "gold"), default="gold")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "tfidf_logreg")
    parser.add_argument("--random-state", type=int, default=718)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--unbalanced", action="store_true", help="Disable class-balanced logistic-regression weights.")
    args = parser.parse_args()

    processed = args.data_root / "processed" / "averitec"
    train_records = read_jsonl(processed / f"{args.train_split}.jsonl")
    eval_records = read_jsonl(processed / f"{args.eval_split}.jsonl")
    verifier = TfidfLogRegVerifier(
        evidence_mode=args.evidence_mode,
        min_df=args.min_df,
        max_features=args.max_features,
        class_weight=None if args.unbalanced else "balanced",
        random_state=args.random_state,
    ).fit(train_records)
    predictions = verifier.predict_records(eval_records)
    metrics = evaluate_predictions(predictions)
    run_config = {
        "model": "tfidf_logreg",
        "evidence_mode": args.evidence_mode,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "random_state": args.random_state,
        "min_df": args.min_df,
        "max_features": args.max_features,
        "class_weight": None if args.unbalanced else "balanced",
        "warning": (
            "gold mode is an isolated verifier experiment. It must not be presented as an "
            "end-to-end retrieval result."
            if args.evidence_mode == "gold"
            else None
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.evidence_mode}_{args.train_split}_to_{args.eval_split}"
    write_jsonl(predictions, args.output_dir / f"{prefix}_predictions.jsonl")
    (args.output_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / f"{prefix}_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    joblib.dump(verifier, args.output_dir / f"{prefix}_model.joblib")
    print(json.dumps({**run_config, **metrics}, indent=2))


if __name__ == "__main__":
    main()
