"""Build claim-only or gold-answer ranking inputs for controlled verifier runs.

Gold answers are an oracle diagnostic: they are never a deployable retrieval result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from src.utils.runs import sha256_file, write_run_manifest  # noqa: E402


def variant_rows(records: list[dict[str, Any]], mode: str) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    with_evidence = 0
    for record in records:
        claim_id = str(record.get("claim_id") or "")
        claim = str(record.get("claim") or "").strip()
        if not claim_id or not claim or claim_id in ids:
            raise ValueError(f"Missing/duplicate claim ID or empty claim: {claim_id!r}")
        ids.add(claim_id)
        evidence: list[dict[str, Any]] = []
        if mode == "gold":
            for item in record.get("evidence") or []:
                if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                    continue
                evidence.append({
                    "passage_id": str(item["passage_id"]),
                    "text": str(item["text"]),
                    "url": item.get("url"),
                    "metadata": {"is_gold_answer": True},
                })
                if len(evidence) == 3:
                    break
        with_evidence += bool(evidence)
        rows.append({"claim_id": claim_id, "claim": claim, "retrieved": evidence})
    return rows, with_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("claim_only", "gold"))
    args = parser.parse_args()

    processed = args.data_root / "processed" / "averitec"
    inputs = {split: processed / f"{split}.jsonl" for split in ("train", "dev")}
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    input_hashes = {split: sha256_file(path) for split, path in inputs.items()}
    config = {"script": "prepare_verifier_variant_rankings", "mode": args.mode, "input_sha256": input_hashes, "max_evidence": 3}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "variant_config.json"
    existing = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
    if existing is not None and any(existing.get(key) != value for key, value in config.items()):
        raise ValueError(f"Variant inputs/settings changed; use a new output directory: {args.output_dir}")

    counts: dict[str, dict[str, int]] = {}
    for split, path in inputs.items():
        destination = args.output_dir / f"{split}_rankings.jsonl"
        rows, with_evidence = variant_rows(read_jsonl(path), args.mode)
        counts[split] = {"claim_count": len(rows), "claims_with_evidence": with_evidence}
        if destination.is_file():
            existing_rows = read_jsonl(destination)
            if existing_rows != rows:
                raise ValueError(f"Existing variant rankings differ; use a new output directory: {destination}")
        else:
            temporary = destination.with_name(f"{destination.name}.tmp")
            write_jsonl(rows, temporary)
            temporary.replace(destination)
    report = {**config, "counts": counts, "warning": "Gold answers are oracle-only, unavailable at prediction time." if args.mode == "gold" else None}
    write_json(report, config_path)
    write_run_manifest(args.output_dir / "variant_manifest.json", project_root=PROJECT_ROOT, config=report, input_paths=inputs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
