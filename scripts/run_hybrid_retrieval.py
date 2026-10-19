"""Fuse BM25 and dense rankings with reciprocal rank fusion (RRF)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bm25 import load_qrels  # noqa: E402
from src.retrieval.metrics import evaluate_rankings  # noqa: E402
from src.utils.io import read_jsonl, write_jsonl  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.runs import resolve_run_directory, write_run_manifest  # noqa: E402


def load_rankings(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        claim_id = str(row["claim_id"])
        if claim_id in result:
            raise ValueError(f"Duplicate claim ID {claim_id!r} in {path}")
        if not isinstance(row.get("retrieved"), list):
            raise ValueError(f"Malformed retrieved list for claim {claim_id!r} in {path}")
        result[claim_id] = row
    return result


def fuse_rankings(
    bm25_row: dict[str, Any], dense_row: dict[str, Any], *, constant: int, k: int
) -> dict[str, Any]:
    claim_id = str(bm25_row["claim_id"])
    if str(dense_row["claim_id"]) != claim_id:
        raise ValueError("Cannot fuse rankings for different claims")
    item_by_id: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = defaultdict(float)
    for row in (bm25_row, dense_row):
        for rank, item in enumerate(row.get("retrieved") or [], start=1):
            passage_id = str(item["passage_id"])
            item_by_id.setdefault(passage_id, item)
            scores[passage_id] += 1.0 / (constant + rank)
    ordered_ids = sorted(scores, key=lambda passage_id: (-scores[passage_id], passage_id))[:k]
    fused = []
    for passage_id in ordered_ids:
        item = dict(item_by_id[passage_id])
        item["score"] = float(scores[passage_id])
        fused.append(item)
    return {"claim_id": claim_id, "claim": bm25_row.get("claim") or dense_row.get("claim"), "retrieved": fused}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25-rankings", required=True, type=Path)
    parser.add_argument("--dense-rankings", required=True, type=Path)
    parser.add_argument("--qrels", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--ranking-k", type=int, default=20)
    parser.add_argument("--metric-k", type=int, default=3)
    args = parser.parse_args()
    if args.rrf_constant < 1 or args.ranking_k < 1 or args.metric_k < 1:
        parser.error("RRF constant and k values must be positive")
    try:
        output_dir = resolve_run_directory(args.output_dir, args.run_name)
    except ValueError as error:
        parser.error(str(error))
    for path in (args.bm25_rankings, args.dense_rankings):
        if not path.is_file():
            raise FileNotFoundError(path)

    bm25 = load_rankings(args.bm25_rankings)
    dense = load_rankings(args.dense_rankings)
    if set(bm25) != set(dense):
        raise ValueError(
            f"BM25 and dense runs have different claim sets: {len(set(bm25) - set(dense))} BM25-only, "
            f"{len(set(dense) - set(bm25))} dense-only"
        )
    rankings = []
    with progress(total=len(bm25), description="Hybrid RRF fusion", unit="claim") as bar:
        for claim_id in bm25:
            rankings.append(
                fuse_rankings(bm25[claim_id], dense[claim_id], constant=args.rrf_constant, k=args.ranking_k)
            )
            bar.update(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    rankings_path = output_dir / "hybrid_rankings.jsonl"
    temporary_path = output_dir / "hybrid_rankings.jsonl.tmp"
    write_jsonl(rankings, temporary_path)
    temporary_path.replace(rankings_path)
    config = {
        "script": "run_hybrid_retrieval",
        "fusion": "reciprocal_rank_fusion",
        "rrf_constant": args.rrf_constant,
        "ranking_k": args.ranking_k,
        "metric_k": args.metric_k,
        "claim_count": len(rankings),
        "bm25_rankings": str(args.bm25_rankings),
        "dense_rankings": str(args.dense_rankings),
        "rankings_path": str(rankings_path),
    }
    report: dict[str, Any] = dict(config)
    qrels_path = args.qrels
    if qrels_path and qrels_path.is_file():
        qrels = load_qrels(qrels_path)
        report.update(
            evaluate_rankings(
                {
                    str(row["claim_id"]): [str(item["passage_id"]) for item in row["retrieved"]]
                    for row in rankings
                },
                qrels,
                k=args.metric_k,
            )
        )
        report["qrels_path"] = str(qrels_path)
    (output_dir / "hybrid_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_run_manifest(
        output_dir / "hybrid_manifest.json",
        project_root=PROJECT_ROOT,
        config=report,
        input_paths={
            "bm25_rankings": args.bm25_rankings,
            "dense_rankings": args.dense_rankings,
            **({"qrels": qrels_path} if qrels_path and qrels_path.is_file() else {}),
        },
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
