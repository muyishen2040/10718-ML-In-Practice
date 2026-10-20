"""Deterministically prefer distinct source URLs within saved top-N rankings.

The selector does not read labels or relevance judgments. Qrels are used only
for dev evaluation after rankings are frozen. It cannot recover a source that
is missing from the first-stage candidate list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bm25 import load_qrels  # noqa: E402
from src.data.corpus import canonical_url  # noqa: E402
from src.retrieval.metrics import evaluate_rankings  # noqa: E402
from src.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.runs import resolve_run_directory, sha256_file, write_run_manifest  # noqa: E402


def diversify_row(row: dict[str, Any], max_candidates: int = 20) -> dict[str, Any]:
    """Keep the highest-ranked chunk per URL first, then append duplicates in input order.

    An absent/invalid URL is keyed by passage ID so unrelated unknown sources
    are not accidentally collapsed. The output preserves every input candidate.
    """
    candidates = (row.get("retrieved") or [])[:max_candidates]
    ids = [str(item["passage_id"]) for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate passage ID in {row.get('claim_id')}")
    seen: set[str] = set()
    first: list[dict[str, Any]] = []
    later: list[dict[str, Any]] = []
    for item in candidates:
        url = canonical_url(item.get("url"))
        key = f"url:{url}" if url else f"passage:{item['passage_id']}"
        if key in seen:
            later.append(item)
        else:
            seen.add(key)
            first.append(item)
    ranked = [dict(item) for item in first + later]
    if sorted(str(item["passage_id"]) for item in ranked) != sorted(ids):
        raise RuntimeError("Source diversity unexpectedly changed the candidate set")
    return {"claim_id": str(row["claim_id"]), "claim": str(row["claim"]), "retrieved": ranked}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-rankings", required=True, type=Path)
    parser.add_argument("--qrels", type=Path, help="Dev evaluation only; prohibited for train")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--metric-k", type=int, default=3)
    args = parser.parse_args()
    if min(args.max_candidates, args.metric_k) < 1:
        parser.error("Candidate count and metric k must be positive")
    if args.split == "train" and args.qrels:
        parser.error("Do not use relevance judgments to construct train rankings")
    if not args.input_rankings.is_file():
        raise FileNotFoundError(args.input_rankings)
    if args.qrels and not args.qrels.is_file():
        raise FileNotFoundError(args.qrels)
    output = resolve_run_directory(args.output_dir, args.run_name)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "script": "run_source_diversity",
        "split": args.split,
        "input_rankings": str(args.input_rankings.resolve()),
        "input_sha256": sha256_file(args.input_rankings),
        "qrels_sha256": sha256_file(args.qrels) if args.qrels else None,
        "max_candidates": args.max_candidates,
        "metric_k": args.metric_k,
        "policy": "best BM25 chunk per normalized source URL first, then duplicate chunks in original order",
    }
    config_path = output / "diversity_config.json"
    existing = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
    if existing is not None and existing != config:
        raise ValueError("Input/settings changed; use a new run name")
    final_path = output / f"{args.split}_rankings.jsonl"
    metrics_path = output / f"{args.split}_metrics.json"
    manifest_path = output / f"{args.split}_manifest.json"
    if final_path.is_file() and metrics_path.is_file() and manifest_path.is_file():
        print(metrics_path.read_text(encoding="utf-8"))
        return
    write_json(config, config_path)
    rows = read_jsonl(args.input_rankings)
    ids = [str(row["claim_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Input rankings contain duplicate claim IDs")
    ranked = []
    with progress(total=len(rows), description=f"Source-diverse {args.split}", unit="claim") as bar:
        for row in rows:
            ranked.append(diversify_row(row, args.max_candidates))
            bar.update(1)
    temporary = final_path.with_name(f"{final_path.name}.tmp")
    write_jsonl(ranked, temporary)
    temporary.replace(final_path)
    changed_top3 = sum(
        [str(item["passage_id"]) for item in old.get("retrieved", [])[:args.metric_k]]
        != [str(item["passage_id"]) for item in new["retrieved"][:args.metric_k]]
        for old, new in zip(rows, ranked, strict=True)
    )
    report: dict[str, Any] = {
        **config,
        "claim_count": len(rows),
        "changed_top3_claim_count": changed_top3,
        "rankings_path": str(final_path),
        "finalized": True,
        "candidate_scope": "saved first-stage top-N only",
    }
    if args.qrels:
        qrels = load_qrels(args.qrels)
        report["diverse"] = evaluate_rankings(
            {str(row["claim_id"]): [str(item["passage_id"]) for item in row["retrieved"]] for row in ranked},
            qrels, k=args.metric_k,
        )
        report["original_bm25"] = evaluate_rankings(
            {str(row["claim_id"]): [str(item["passage_id"]) for item in row.get("retrieved", [])[:args.max_candidates]] for row in rows},
            qrels, k=args.metric_k,
        )
        report["qrels_path"] = str(args.qrels)
    write_json(report, metrics_path)
    write_run_manifest(
        manifest_path, project_root=PROJECT_ROOT, config=report,
        input_paths={"input_rankings": args.input_rankings, **({"qrels": args.qrels} if args.qrels else {})},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
