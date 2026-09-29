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
from src.utils.io import iter_jsonl, read_jsonl, write_jsonl  # noqa: E402
from src.utils.runs import write_run_manifest  # noqa: E402


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


def evidence_from_raw(raw: dict[str, Any], path: Path) -> EvidenceItem:
    if not raw.get("passage_id") or not str(raw.get("text") or "").strip():
        raise ValueError(f"Malformed passage record in {path}: {raw}")
    return EvidenceItem(
        passage_id=str(raw["passage_id"]),
        text=str(raw["text"]),
        url=raw.get("url"),
        domain=raw.get("domain"),
        metadata=raw.get("metadata") or {},
    )


def per_claim_candidate_groups(path: Path):
    """Yield contiguous AVeriTeC candidate pools without global corpus indexing."""
    active_claim_id: str | None = None
    active_items: list[EvidenceItem] = []
    completed_claim_ids: set[str] = set()
    for raw in iter_jsonl(path):
        candidate_claim_id = raw.get("metadata", {}).get("candidate_claim_id")
        if not candidate_claim_id:
            raise ValueError(f"Expected candidate_claim_id in scoped corpus {path}")
        candidate_claim_id = str(candidate_claim_id)
        if active_claim_id is None:
            active_claim_id = candidate_claim_id
        if candidate_claim_id != active_claim_id:
            if candidate_claim_id in completed_claim_ids:
                raise ValueError("Scoped corpus must keep each candidate_claim_id contiguous for streaming BM25.")
            completed_claim_ids.add(active_claim_id)
            yield active_claim_id, active_items
            active_claim_id = candidate_claim_id
            active_items = []
        active_items.append(evidence_from_raw(raw, path))
    if active_claim_id is not None:
        yield active_claim_id, active_items


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
    parser.add_argument("--corpus", type=Path, help="Defaults to data/processed/averitec/<split>_evidence_corpus.jsonl.")
    parser.add_argument("--qrels", type=Path, help="Defaults to data/processed/averitec/<split>_qrels.jsonl.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "bm25")
    parser.add_argument("--k", type=int, default=3, help="Legacy shortcut: use this value for both saved rankings and metrics.")
    parser.add_argument("--ranking-k", type=int, help="Number of passages to save per claim; use 20 for the LLM reranking baseline.")
    parser.add_argument("--metric-k", type=int, help="Rank cutoff for retrieval metrics; normally 3 for the user-facing interface.")
    args = parser.parse_args()
    ranking_k = args.ranking_k if args.ranking_k is not None else args.k
    metric_k = args.metric_k if args.metric_k is not None else args.k
    if ranking_k < 1 or metric_k < 1:
        parser.error("--k, --ranking-k, and --metric-k must be at least one.")

    processed = args.data_root / "processed" / "averitec"
    claims = read_jsonl(processed / f"{args.split}.jsonl")
    corpus_path = args.corpus or processed / f"{args.split}_evidence_corpus.jsonl"
    first_corpus_row = next(iter(iter_jsonl(corpus_path)), None)
    if first_corpus_row is None:
        raise ValueError(f"Corpus is empty: {corpus_path}")
    candidate_pool_scoped = bool(first_corpus_row.get("metadata", {}).get("candidate_claim_id"))
    corpus_passage_count = 0
    missing_candidate_pool_claim_ids: list[str] = []
    if candidate_pool_scoped:
        claim_by_id = {str(claim["claim_id"]): claim for claim in claims}
        retrieved_by_claim_id: dict[str, list[dict[str, Any]]] = {}
        for candidate_claim_id, candidate_items in per_claim_candidate_groups(corpus_path):
            corpus_passage_count += len(candidate_items)
            claim = claim_by_id.get(candidate_claim_id)
            if claim is None:
                continue
            retriever = BM25Retriever(candidate_items)
            retrieved_by_claim_id[candidate_claim_id] = [
                item.to_dict() for item in retriever.retrieve(str(claim["claim"]), k=ranking_k)
            ]
        rankings = []
        for claim in claims:
            claim_id = str(claim["claim_id"])
            retrieved = retrieved_by_claim_id.get(claim_id, [])
            if not retrieved:
                missing_candidate_pool_claim_ids.append(claim_id)
            rankings.append({"claim_id": claim_id, "claim": claim["claim"], "retrieved": retrieved})
        retrieval_scope = "per_claim_candidate_pool"
    else:
        corpus = load_corpus(corpus_path)
        corpus_passage_count = len(corpus)
        retriever = BM25Retriever(corpus)
        rankings = [
            {
                "claim_id": claim["claim_id"],
                "claim": claim["claim"],
                "retrieved": [item.to_dict() for item in retriever.retrieve(str(claim["claim"]), k=ranking_k)],
            }
            for claim in claims
        ]
        retrieval_scope = "global_static_corpus"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = args.output_dir / f"{args.split}_rankings.jsonl"
    write_jsonl(rankings, ranking_path)

    qrels_path = args.qrels or processed / f"{args.split}_qrels.jsonl"
    report: dict[str, Any] = {
        "split": args.split,
        "ranking_k": ranking_k,
        "metric_k": metric_k,
        "claim_count": len(claims),
        "corpus_path": str(corpus_path),
        "corpus_passage_count": corpus_passage_count,
        "retrieval_scope": retrieval_scope,
        "claims_without_candidate_pool_count": len(missing_candidate_pool_claim_ids),
        "claims_without_candidate_pool_samples": missing_candidate_pool_claim_ids[:10],
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
                k=metric_k,
            )
        )
    else:
        report["metrics_status"] = f"No qrels found at {qrels_path}; rankings saved but metrics not computed."

    report_path = args.output_dir / f"{args.split}_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_run_manifest(
        args.output_dir / f"{args.split}_manifest.json",
        project_root=PROJECT_ROOT,
        config={
            "script": "run_bm25",
            "split": args.split,
            "ranking_k": ranking_k,
            "metric_k": metric_k,
            "tokenizer": "unicode_word_casefold",
            "tie_break": "passage_id_ascending",
        },
        input_paths={"claims": processed / f"{args.split}.jsonl", "corpus": corpus_path, "qrels": qrels_path},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
