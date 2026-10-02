"""Normalize recognized URL/text records in an AVeriTeC knowledge-store zip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.knowledge_store import normalize_zip_to_jsonl  # noqa: E402
from src.utils.runs import write_run_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-documents", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument(
        "--max-documents",
        type=int,
        help="Optional deterministic cap for a smoke test; capped corpora are not full-benchmark results.",
    )
    parser.add_argument(
        "--url2text-mode",
        choices=("sentence", "source_document"),
        default="sentence",
        help=(
            "How to normalize AVeriTeC URL-plus-sentence rows. 'sentence' preserves the v1 corpus; "
            "'source_document' joins each row's extracted sentences before later passage chunking."
        ),
    )
    args = parser.parse_args()
    audit = normalize_zip_to_jsonl(
        args.archive,
        args.output_documents,
        max_documents=args.max_documents,
        url2text_mode=args.url2text_mode,
    )
    if not audit["normalized_document_count"]:
        raise RuntimeError("No URL/text documents were recognized. Inspect the archive layout and add a dedicated adapter before proceeding.")
    audit_path = args.audit_output or args.output_documents.with_name(f"{args.output_documents.stem}_audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    write_run_manifest(
        args.output_documents.with_name(f"{args.output_documents.stem}_manifest.json"),
        project_root=PROJECT_ROOT,
        config={
            "script": "normalize_averitec_knowledge_store",
            "max_documents": args.max_documents,
            "url2text_mode": args.url2text_mode,
        },
        input_paths={"archive": args.archive},
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
