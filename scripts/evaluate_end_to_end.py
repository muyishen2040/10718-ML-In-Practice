"""Diagnose verdict performance conditional on retrieval evidence coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.end_to_end import annotate_retrieval_coverage, evaluate_by_retrieval_coverage  # noqa: E402
from src.utils.io import read_jsonl, write_jsonl  # noqa: E402
from src.utils.runs import write_run_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--rankings", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    qrels = {
        str(row["claim_id"]): [str(value) for value in row.get("relevant_passage_ids") or []]
        for row in read_jsonl(args.qrels)
        if row.get("relevant_passage_ids")
    }
    predictions = read_jsonl(args.predictions)
    rankings = read_jsonl(args.rankings)
    report = evaluate_by_retrieval_coverage(predictions, rankings, qrels, args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    coverage_path = args.output.with_name(f"{args.output.stem}_coverage.jsonl")
    write_jsonl(annotate_retrieval_coverage(predictions, rankings, qrels, args.k), coverage_path)
    write_run_manifest(
        args.output.with_name(f"{args.output.stem}_manifest.json"),
        project_root=PROJECT_ROOT,
        config={"script": "evaluate_end_to_end", "k": args.k},
        input_paths={"predictions": args.predictions, "rankings": args.rankings, "qrels": args.qrels},
    )
    print(json.dumps({**report, "coverage_path": str(coverage_path)}, indent=2))


if __name__ == "__main__":
    main()
