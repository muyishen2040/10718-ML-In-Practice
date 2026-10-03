"""Checkpointable train BM25 directly from official AVeriTeC archive shards.

This avoids materializing a huge combined train corpus. Each completed claim
ranking is appended immediately, so rerunning with --resume skips completed
archive members and continues from the checkpoint on Drive.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bm25 import append_checkpoint_rows, load_checkpoint_rankings, repair_checkpoint_rankings  # noqa: E402
from src.data.corpus import chunk_text  # noqa: E402
from src.data.knowledge_store import _candidate_claim_index, _documents_from_value  # noqa: E402
from src.data.schema import EvidenceItem  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.utils.io import read_jsonl, write_jsonl  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.runs import resolve_run_directory, write_run_manifest  # noqa: E402


def documents_from_official_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> Iterable[dict[str, Any]]:
    """Yield source-document records from an official per-claim JSONL member."""
    with archive.open(member) as binary_handle:
        text_handle = io.TextIOWrapper(binary_handle, encoding="utf-8", errors="replace")
        for row_index, line in enumerate(text_handle):
            if line.strip():
                yield from _documents_from_value(
                    json.loads(line), f"{member.filename}#{row_index}", url2text_mode="source_document"
                )


def passages_from_documents(documents: Iterable[dict[str, Any]], claim_id: str, max_words: int, overlap_words: int) -> list[EvidenceItem]:
    passages: list[EvidenceItem] = []
    for document_index, document in enumerate(documents):
        text = str(document.get("text") or "").strip()
        if not text:
            continue
        document_id = str(document.get("document_id") or document.get("url") or f"document:{document_index}")
        for passage_index, passage_text in enumerate(chunk_text(text, max_words, overlap_words)):
            passages.append(
                EvidenceItem(
                    passage_id=f"{document_id}:p{passage_index:04d}",
                    text=passage_text,
                    url=document.get("url"),
                    metadata={
                        "document_id": document_id,
                        "chunk_index": passage_index,
                        "chunk_max_words": max_words,
                        "chunk_overlap_words": overlap_words,
                        "candidate_claim_id": claim_id,
                    },
                )
            )
    return passages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", action="append", type=Path, help="Official archive shard; repeat in source-index order.")
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--ranking-k", type=int, default=20)
    parser.add_argument("--max-words", type=int, default=160)
    parser.add_argument("--overlap-words", type=int, default=40)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--repair-checkpoint",
        action="store_true",
        help="Back up and remove malformed JSONL checkpoint rows before resuming; valid rows are retained.",
    )
    parser.add_argument("--finalize", action="store_true", help="Write final train_rankings.jsonl only after all three archive shards were processed.")
    args = parser.parse_args()
    if not args.archive and not args.finalize:
        parser.error("At least one --archive is required unless using --finalize to materialize an existing checkpoint.")
    if args.ranking_k < 1 or args.checkpoint_every < 1:
        parser.error("--ranking-k and --checkpoint-every must be positive")
    if args.overlap_words < 0 or args.overlap_words >= args.max_words:
        parser.error("Require 0 <= --overlap-words < --max-words")
    try:
        output_dir = resolve_run_directory(args.output_dir, args.run_name)
    except ValueError as error:
        parser.error(str(error))

    claims_path = args.data_root / "processed" / "averitec" / "train.jsonl"
    claims = read_jsonl(claims_path)
    claim_by_source_index = {
        int(row["metadata"]["source_index"]): row
        for row in claims
        if row.get("metadata", {}).get("source_index") is not None
    }
    claim_by_id = {str(row["claim_id"]): row for row in claims}
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / "train_rankings.jsonl"
    checkpoint_path = output_dir / "train_rankings.partial.jsonl"
    if ranking_path.exists():
        parser.error(f"Final rankings already exist at {ranking_path}; use them directly rather than rerunning.")
    if checkpoint_path.exists() and not args.resume:
        parser.error(f"Found {checkpoint_path}; rerun with --resume.")
    if args.repair_checkpoint and not args.resume:
        parser.error("--repair-checkpoint requires --resume.")
    if args.repair_checkpoint:
        repair_checkpoint_rankings(checkpoint_path, known_claim_ids=set(claim_by_id))
    completed = load_checkpoint_rankings(checkpoint_path, known_claim_ids=set(claim_by_id)) if args.resume else {}
    pending: list[dict[str, Any]] = []
    passage_count = 0
    processed_claim_count = 0

    for archive_path in args.archive or []:
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            with progress(total=len(members), description=f"BM25 {archive_path.name}", unit="claim") as bar:
                try:
                    for member in members:
                        source_index = _candidate_claim_index(member.filename)
                        if source_index is None:
                            raise ValueError(f"Could not infer AVeriTeC source index from {member.filename}")
                        claim = claim_by_source_index.get(source_index)
                        if claim is None:
                            # The known empty training claim is absent from the prepared claim file.
                            bar.update(1)
                            continue
                        claim_id = str(claim["claim_id"])
                        if claim_id in completed:
                            bar.update(1)
                            continue
                        passages = passages_from_documents(
                            documents_from_official_member(archive, member), claim_id, args.max_words, args.overlap_words
                        )
                        passage_count += len(passages)
                        retrieved = (
                            [item.to_dict() for item in BM25Retriever(passages).retrieve(str(claim["claim"]), k=args.ranking_k)]
                            if passages
                            else []
                        )
                        ranking = {"claim_id": claim_id, "claim": claim["claim"], "retrieved": retrieved}
                        completed[claim_id] = ranking
                        pending.append(ranking)
                        processed_claim_count += 1
                        if len(pending) >= args.checkpoint_every:
                            append_checkpoint_rows(checkpoint_path, pending)
                            pending = []
                        bar.update(1)
                finally:
                    append_checkpoint_rows(checkpoint_path, pending)

    report: dict[str, Any] = {
        "split": "train",
        "ranking_k": args.ranking_k,
        "max_words": args.max_words,
        "overlap_words": args.overlap_words,
        "completed_train_claim_count": len(completed),
        "processed_this_invocation_count": processed_claim_count,
        "passages_built_this_invocation": passage_count,
        "checkpoint_path": str(checkpoint_path),
        "archive_paths": [str(path) for path in args.archive or []],
        "finalized": False,
    }
    if args.finalize:
        missing = [str(row["claim_id"]) for row in claims if str(row["claim_id"]) not in completed]
        if missing:
            raise RuntimeError(
                f"Cannot finalize: {len(missing)} prepared train claims lack rankings, e.g. {missing[:5]}. "
                "Run all three official archive shards with --resume first."
            )
        write_jsonl((completed[str(row["claim_id"])] for row in claims), ranking_path)
        checkpoint_path.unlink(missing_ok=True)
        report["finalized"] = True
        report["rankings_path"] = str(ranking_path)
    (output_dir / "train_archive_bm25_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_run_manifest(
        output_dir / "train_archive_bm25_manifest.json",
        project_root=PROJECT_ROOT,
        config=report,
        input_paths={"train_claims": claims_path},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
