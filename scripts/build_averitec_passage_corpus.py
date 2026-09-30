"""Chunk normalized AVeriTeC candidate documents and build URL-based qrels.

The official knowledge-store archive must first be inspected and converted to a
JSONL file with `text` plus `url` or `document_id`. This script deliberately
does not guess a large archive's undocumented internal format.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.corpus import build_url_qrels, chunk_text  # noqa: E402
from src.utils.io import read_jsonl, write_jsonl  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.runs import write_run_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents-jsonl", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--max-words", type=int, default=160)
    parser.add_argument("--overlap-words", type=int, default=40)
    args = parser.parse_args()

    processed = args.data_root / "processed" / "averitec"
    claims_path = processed / f"{args.split}.jsonl"
    claims = read_jsonl(claims_path)
    claim_id_by_source_index = {
        int(record["metadata"]["source_index"]): str(record["claim_id"])
        for record in claims
        if record.get("metadata", {}).get("source_index") is not None
    }
    corpus_path = processed / f"{args.split}_evidence_corpus.jsonl"
    qrels_path = processed / f"{args.split}_qrels.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    document_count = 0
    rejected_empty_document_count = 0
    unmatched_candidate_claim_document_count = 0
    candidate_pool_scoped = False
    passage_count = 0
    with (
        args.documents_jsonl.open("rb") as documents,
        corpus_path.open("w", encoding="utf-8", newline="\n") as output,
        progress(
            total=args.documents_jsonl.stat().st_size,
            description="Building passage corpus",
            unit="B",
            unit_scale=True,
        ) as bar,
    ):
        for document_index, line in enumerate(documents):
            bar.update(len(line))
            if not line.strip():
                continue
            document = json.loads(line)
            document_count += 1
            text = str(document.get("text") or "").strip()
            if not text:
                rejected_empty_document_count += 1
                continue
            document_id = str(document.get("document_id") or document.get("url") or f"document:{document_index}")
            url = document.get("url")
            candidate_claim_index = document.get("metadata", {}).get("candidate_claim_index")
            candidate_pool_scoped = candidate_pool_scoped or candidate_claim_index is not None
            candidate_claim_id = (
                claim_id_by_source_index.get(int(candidate_claim_index))
                if candidate_claim_index is not None
                else None
            )
            if candidate_claim_index is not None and candidate_claim_id is None:
                unmatched_candidate_claim_document_count += 1
                continue
            for passage_index, passage_text in enumerate(chunk_text(text, args.max_words, args.overlap_words)):
                output.write(
                    json.dumps(
                        {
                            "passage_id": f"{document_id}:p{passage_index:04d}",
                            "text": passage_text,
                            "url": url,
                            "metadata": {
                                "document_id": document_id,
                                "chunk_index": passage_index,
                                "chunk_max_words": args.max_words,
                                "chunk_overlap_words": args.overlap_words,
                                **({"candidate_claim_id": candidate_claim_id} if candidate_claim_id else {}),
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                passage_count += 1
            if document_count % 1_000 == 0:
                bar.set_postfix(documents=document_count, passages=passage_count)
    corpus_audit = {
        "document_count": document_count,
        "passage_count": passage_count,
        "rejected_empty_document_count": rejected_empty_document_count,
        "unmatched_candidate_claim_document_count": unmatched_candidate_claim_document_count,
        "candidate_pool_scoped": candidate_pool_scoped,
        "streamed": True,
    }
    def passages_with_progress():
        with corpus_path.open("rb") as passages, progress(
            total=corpus_path.stat().st_size,
            description="Building URL qrels",
            unit="B",
            unit_scale=True,
        ) as bar:
            for line in passages:
                bar.update(len(line))
                if line.strip():
                    yield json.loads(line)

    qrels, qrels_audit = build_url_qrels(claims, passages_with_progress())
    write_jsonl(qrels, qrels_path)
    report: dict[str, Any] = {
        "split": args.split,
        "documents_input_count": document_count,
        "corpus": corpus_audit,
        "qrels": qrels_audit,
        "corpus_path": str(corpus_path),
        "qrels_path": str(qrels_path),
        "warning": "Qrels are annotated-source-URL proxies, not exact passage-level judgments.",
    }
    report_path = processed / f"{args.split}_corpus_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_run_manifest(
        processed / f"{args.split}_corpus_manifest.json",
        project_root=PROJECT_ROOT,
        config={"script": "build_averitec_passage_corpus", "split": args.split, "max_words": args.max_words, "overlap_words": args.overlap_words},
        input_paths={"documents_jsonl": args.documents_jsonl, "claims": claims_path},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
