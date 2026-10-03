"""Run the deterministic BM25 baseline on a prepared AVeriTeC passage corpus."""

from __future__ import annotations

import argparse
import json
import sys
from uuid import uuid4
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import EvidenceItem  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.metrics import evaluate_rankings  # noqa: E402
from src.utils.io import iter_jsonl, read_jsonl, write_jsonl  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.runs import resolve_run_directory, write_run_manifest  # noqa: E402


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


def load_checkpoint_rankings(path: Path, *, known_claim_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Load completed rankings, repairing at most one interrupted final JSONL row.

    A Colab interruption can leave the final append-only checkpoint record only
    partially written.  That record cannot safely be resumed by simply appending
    another JSON object, because both records would share a line.  We therefore
    discard a malformed *last non-empty* line and rerun its claim.  Corruption in
    any earlier record remains a hard error rather than being silently hidden.
    """
    if not path.exists():
        return {}

    contents = path.read_bytes()
    lines = contents.splitlines(keepends=True)
    last_nonempty_line_index = max((index for index, line in enumerate(lines) if line.strip()), default=None)
    completed: dict[str, dict[str, Any]] = {}
    byte_offset = 0
    for line_index, raw_line in enumerate(lines):
        line_start = byte_offset
        byte_offset += len(raw_line)
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if line_index != last_nonempty_line_index:
                raise ValueError(
                    f"Checkpoint {path} is corrupt before its final record at line {line_index + 1}."
                ) from error
            path.write_bytes(contents[:line_start])
            print(
                f"Warning: removed malformed final checkpoint record from {path}; "
                "its claim will be rerun."
            )
            break
        claim_id = str(row.get("claim_id") or "")
        if claim_id not in known_claim_ids:
            raise ValueError(f"Checkpoint {path} contains an unknown claim ID: {claim_id!r}")
        if claim_id in completed:
            raise ValueError(f"Checkpoint {path} contains duplicate claim ID: {claim_id!r}")
        if not isinstance(row.get("retrieved"), list):
            raise ValueError(f"Checkpoint {path} has malformed ranking for claim {claim_id!r}")
        completed[claim_id] = row
    return completed


def repair_checkpoint_rankings(path: Path, *, known_claim_ids: set[str]) -> dict[str, Any]:
    """Back up and rebuild a checkpoint after malformed or duplicate JSONL rows.

    This intentionally repairs only JSON decoding/UTF-8 failures. Parsed rows
    with unknown IDs still raise. Identical duplicate claim rankings are removed;
    conflicting duplicates still raise, because silently choosing between them
    could change the experiment. The original file is retained beside the
    repaired checkpoint before any replacement occurs.
    """
    if not path.exists():
        return {"repaired": False, "valid_row_count": 0, "dropped_line_numbers": [], "backup_path": None}

    valid_rows: list[dict[str, Any]] = []
    dropped_line_numbers: list[int] = []
    dropped_duplicate_line_numbers: list[int] = []
    rows_by_claim_id: dict[str, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                dropped_line_numbers.append(line_number)
                continue
            claim_id = str(row.get("claim_id") or "")
            if claim_id not in known_claim_ids:
                raise ValueError(f"Checkpoint {path} contains an unknown claim ID: {claim_id!r}")
            if not isinstance(row.get("retrieved"), list):
                raise ValueError(f"Checkpoint {path} has malformed ranking for claim {claim_id!r}")
            existing = rows_by_claim_id.get(claim_id)
            if existing is not None:
                if row != existing:
                    raise ValueError(
                        f"Checkpoint {path} contains conflicting duplicate claim ID {claim_id!r} "
                        f"at line {line_number}."
                    )
                dropped_duplicate_line_numbers.append(line_number)
                continue
            rows_by_claim_id[claim_id] = row
            valid_rows.append(row)

    if not dropped_line_numbers and not dropped_duplicate_line_numbers:
        return {
            "repaired": False,
            "valid_row_count": len(valid_rows),
            "dropped_line_numbers": [],
            "dropped_duplicate_line_numbers": [],
            "backup_path": None,
        }

    backup_path = path.with_name(f"{path.name}.before_repair.{uuid4().hex}.bak")
    backup_path.write_bytes(path.read_bytes())
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.repair.tmp")
    try:
        write_jsonl(valid_rows, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    report = {
        "repaired": True,
        "valid_row_count": len(valid_rows),
        "dropped_line_numbers": dropped_line_numbers,
        "dropped_duplicate_line_numbers": dropped_duplicate_line_numbers,
        "backup_path": str(backup_path),
    }
    print(json.dumps({"checkpoint_repair": report}, indent=2))
    return report


def append_checkpoint_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Durably make a small batch of completed rankings available for --resume."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--corpus", type=Path, help="Defaults to data/processed/averitec/<split>_evidence_corpus.jsonl.")
    parser.add_argument("--qrels", type=Path, help="Defaults to data/processed/averitec/<split>_qrels.jsonl.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "bm25")
    parser.add_argument("--run-name", help="Optional name for an isolated, non-overwriting run subdirectory.")
    parser.add_argument("--k", type=int, default=3, help="Legacy shortcut: use this value for both saved rankings and metrics.")
    parser.add_argument("--ranking-k", type=int, help="Number of passages to save per claim; use 20 for the LLM reranking baseline.")
    parser.add_argument("--metric-k", type=int, help="Rank cutoff for retrieval metrics; normally 3 for the user-facing interface.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run from its checkpoint in this run directory.")
    parser.add_argument(
        "--repair-checkpoint",
        action="store_true",
        help="Back up and remove malformed or identical duplicate checkpoint rows before resuming.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save completed rankings every N claims; 1 is safest but slower.")
    args = parser.parse_args()
    try:
        args.output_dir = resolve_run_directory(args.output_dir, args.run_name)
    except ValueError as error:
        parser.error(str(error))
    ranking_k = args.ranking_k if args.ranking_k is not None else args.k
    metric_k = args.metric_k if args.metric_k is not None else args.k
    if ranking_k < 1 or metric_k < 1 or args.checkpoint_every < 1:
        parser.error("--k, --ranking-k, --metric-k, and --checkpoint-every must be at least one.")

    processed = args.data_root / "processed" / "averitec"
    claims = read_jsonl(processed / f"{args.split}.jsonl")
    claim_by_id = {str(claim["claim_id"]): claim for claim in claims}
    if len(claim_by_id) != len(claims):
        raise ValueError(f"Claim file contains duplicate claim IDs: {processed / f'{args.split}.jsonl'}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = args.output_dir / f"{args.split}_rankings.jsonl"
    checkpoint_path = args.output_dir / f"{args.split}_rankings.partial.jsonl"
    if ranking_path.exists():
        parser.error(f"Final rankings already exist at {ranking_path}. Use a new --run-name to preserve this result.")
    if checkpoint_path.exists() and not args.resume:
        parser.error(f"Interrupted-run checkpoint exists at {checkpoint_path}; rerun with --resume or use a new --run-name.")
    if args.repair_checkpoint and not args.resume:
        parser.error("--repair-checkpoint requires --resume.")
    if args.repair_checkpoint:
        repair_checkpoint_rankings(checkpoint_path, known_claim_ids=set(claim_by_id))
    completed_rankings = load_checkpoint_rankings(checkpoint_path, known_claim_ids=set(claim_by_id)) if args.resume else {}
    if completed_rankings:
        print(f"Resuming BM25: reusing {len(completed_rankings)} completed claim rankings.")

    corpus_path = args.corpus or processed / f"{args.split}_evidence_corpus.jsonl"
    first_corpus_row = next(iter(iter_jsonl(corpus_path)), None)
    if first_corpus_row is None:
        raise ValueError(f"Corpus is empty: {corpus_path}")
    candidate_pool_scoped = bool(first_corpus_row.get("metadata", {}).get("candidate_claim_id"))
    corpus_passage_count = 0
    missing_candidate_pool_claim_ids: list[str] = []
    if candidate_pool_scoped:
        retrieved_by_claim_id = dict(completed_rankings)
        pending_checkpoint_rows: list[dict[str, Any]] = []
        with progress(
            total=len(claims), initial=len(completed_rankings), description="Running BM25", unit="claim"
        ) as bar:
            try:
                for candidate_claim_id, candidate_items in per_claim_candidate_groups(corpus_path):
                    corpus_passage_count += len(candidate_items)
                    claim = claim_by_id.get(candidate_claim_id)
                    if claim is None or candidate_claim_id in retrieved_by_claim_id:
                        continue
                    ranking = {
                        "claim_id": candidate_claim_id,
                        "claim": claim["claim"],
                        "retrieved": [item.to_dict() for item in BM25Retriever(candidate_items).retrieve(str(claim["claim"]), k=ranking_k)],
                    }
                    retrieved_by_claim_id[candidate_claim_id] = ranking
                    pending_checkpoint_rows.append(ranking)
                    if len(pending_checkpoint_rows) >= args.checkpoint_every:
                        append_checkpoint_rows(checkpoint_path, pending_checkpoint_rows)
                        pending_checkpoint_rows = []
                    bar.update(1)
            finally:
                append_checkpoint_rows(checkpoint_path, pending_checkpoint_rows)
        rankings = []
        for claim in claims:
            claim_id = str(claim["claim_id"])
            ranking = retrieved_by_claim_id.get(claim_id)
            if ranking is None:
                missing_candidate_pool_claim_ids.append(claim_id)
                ranking = {"claim_id": claim_id, "claim": claim["claim"], "retrieved": []}
            rankings.append(ranking)
        retrieval_scope = "per_claim_candidate_pool"
    else:
        corpus = load_corpus(corpus_path)
        corpus_passage_count = len(corpus)
        retriever = BM25Retriever(corpus)
        rankings_by_claim_id = dict(completed_rankings)
        pending_checkpoint_rows = []
        with progress(
            total=len(claims), initial=len(completed_rankings), description="Running BM25", unit="claim"
        ) as bar:
            try:
                for claim in claims:
                    claim_id = str(claim["claim_id"])
                    if claim_id in rankings_by_claim_id:
                        continue
                    ranking = {
                        "claim_id": claim_id,
                        "claim": claim["claim"],
                        "retrieved": [item.to_dict() for item in retriever.retrieve(str(claim["claim"]), k=ranking_k)],
                    }
                    rankings_by_claim_id[claim_id] = ranking
                    pending_checkpoint_rows.append(ranking)
                    if len(pending_checkpoint_rows) >= args.checkpoint_every:
                        append_checkpoint_rows(checkpoint_path, pending_checkpoint_rows)
                        pending_checkpoint_rows = []
                    bar.update(1)
            finally:
                append_checkpoint_rows(checkpoint_path, pending_checkpoint_rows)
        rankings = [rankings_by_claim_id[str(claim["claim_id"])] for claim in claims]
        retrieval_scope = "global_static_corpus"

    write_jsonl(rankings, ranking_path)
    checkpoint_path.unlink(missing_ok=True)

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
        "resumed_claim_count": len(completed_rankings),
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
            "run_name": args.run_name,
            "split": args.split,
            "ranking_k": ranking_k,
            "metric_k": metric_k,
            "resume": args.resume,
            "checkpoint_every": args.checkpoint_every,
            "tokenizer": "unicode_word_casefold",
            "tie_break": "passage_id_ascending",
        },
        input_paths={"claims": processed / f"{args.split}.jsonl", "corpus": corpus_path, "qrels": qrels_path},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
