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
from src.utils.io import read_jsonl, write_jsonl  # noqa: E402
from src.utils.runs import write_run_manifest  # noqa: E402


def attach_retrieved_evidence(records: list[dict[str, Any]], rankings_path: Path) -> list[dict[str, Any]]:
    rankings = {str(row["claim_id"]): row.get("retrieved") or [] for row in read_jsonl(rankings_path)}
    missing = [record["claim_id"] for record in records if record["claim_id"] not in rankings]
    if missing:
        raise ValueError(f"Rankings are missing {len(missing)} records, e.g. {missing[:3]}")
    return [{**record, "retrieved_evidence": rankings[str(record["claim_id"])]} for record in records]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-mode", choices=("claim_only", "gold", "gold_all", "retrieved"), default="gold")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "tfidf_logreg")
    parser.add_argument("--random-state", type=int, default=718)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-features", type=int)
    parser.add_argument("--max-evidence", type=int, default=3)
    parser.add_argument("--train-rankings", type=Path, help="Required for --evidence-mode retrieved.")
    parser.add_argument("--eval-rankings", type=Path, help="Required for --evidence-mode retrieved.")
    parser.add_argument("--unbalanced", action="store_true", help="Disable class-balanced logistic-regression weights.")
    args = parser.parse_args()

    processed = args.data_root / "processed" / "averitec"
    train_records = read_jsonl(processed / f"{args.train_split}.jsonl")
    eval_records = read_jsonl(processed / f"{args.eval_split}.jsonl")
    if args.evidence_mode == "retrieved":
        if not args.train_rankings or not args.eval_rankings:
            parser.error("--train-rankings and --eval-rankings are required for retrieved evidence.")
        train_records = attach_retrieved_evidence(train_records, args.train_rankings)
        eval_records = attach_retrieved_evidence(eval_records, args.eval_rankings)
    verifier = TfidfLogRegVerifier(
        evidence_mode=args.evidence_mode,
        max_evidence=args.max_evidence,
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
        "max_evidence": args.max_evidence,
        "class_weight": None if args.unbalanced else "balanced",
        "warning": (
            "gold mode is an isolated verifier experiment capped at three annotated answers. It must not be presented as an "
            "end-to-end retrieval result."
            if args.evidence_mode == "gold"
            else "gold_all is an oracle upper bound with no three-item cap." if args.evidence_mode == "gold_all" else None
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.evidence_mode}_{args.train_split}_to_{args.eval_split}"
    write_jsonl(predictions, args.output_dir / f"{prefix}_predictions.jsonl")
    (args.output_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / f"{prefix}_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    joblib.dump(verifier, args.output_dir / f"{prefix}_model.joblib")
    write_run_manifest(
        args.output_dir / f"{prefix}_manifest.json",
        project_root=PROJECT_ROOT,
        config=run_config,
        input_paths={
            "train_records": processed / f"{args.train_split}.jsonl",
            "eval_records": processed / f"{args.eval_split}.jsonl",
            **({"train_rankings": args.train_rankings, "eval_rankings": args.eval_rankings} if args.evidence_mode == "retrieved" else {}),
        },
    )
    print(json.dumps({**run_config, **metrics}, indent=2))


if __name__ == "__main__":
    main()
