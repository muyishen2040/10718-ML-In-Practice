"""Run the deterministic BM25 baseline on a prepared AVeriTeC passage corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import EvidenceItem  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.metrics import evaluate_rankings  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_corpus(path: Path) -> list[EvidenceItem]:
    corpus = []
    for raw in read_jsonl(path):
        if not raw.get("passage_id") or not str(raw.get("text") or "").strip():
            raise ValueError(f"Malformed passage record in {path}: {raw}")
        corpus.append(
            EvidenceItem(
                passage_id=str(raw["passage_id"]),
                text=str(raw["text"]),
                url=raw.get("url"),
                domain=raw.get("domain"),
                metadata=raw.get("metadata") or {},
            )
        )
    return corpus


def load_qrels(path: Path) -> dict[str, list[str]]:
    qrels: dict[str, list[str]] = {}
    for raw in read_jsonl(path):
        claim_id = str(raw["claim_id"])
        passage_ids = raw.get("relevant_passage_ids") or []
        if not passage_ids:
            continue
        qrels[claim_id] = [str(item) for item in passage_ids]
    return qrels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "bm25")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    processed = args.data_root / "processed" / "averitec"
    claims = read_jsonl(processed / f"{args.split}.jsonl")
    corpus = load_corpus(processed / "evidence_corpus.jsonl")
    retriever = BM25Retriever(corpus)
    rankings = []
    for claim in claims:
        results = retriever.retrieve(str(claim["claim"]), k=args.k)
        rankings.append(
            {
                "claim_id": claim["claim_id"],
                "claim": claim["claim"],
                "retrieved": [item.to_dict() for item in results],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = args.output_dir / f"{args.split}_rankings.jsonl"
    with ranking_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rankings:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    qrels_path = processed / f"{args.split}_qrels.jsonl"
    report: dict[str, Any] = {
        "split": args.split,
        "k": args.k,
        "claim_count": len(claims),
        "corpus_passage_count": len(corpus),
        "rankings_path": str(ranking_path),
    }
    if qrels_path.exists():
        qrels = load_qrels(qrels_path)
        report.update(
            evaluate_rankings(
                {
                    row["claim_id"]: [item["passage_id"] for item in row["retrieved"]]
                    for row in rankings
                },
                qrels,
                k=args.k,
            )
        )
    else:
        report["metrics_status"] = f"No qrels found at {qrels_path}; rankings saved but metrics not computed."

    report_path = args.output_dir / f"{args.split}_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
