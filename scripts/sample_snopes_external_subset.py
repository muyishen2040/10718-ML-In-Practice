"""Create the frozen subset used for evidence-aware external evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.snopes import SNOPES_EXTERNAL_LABELS  # noqa: E402
from src.external.sampling import sample_audit, stratified_sample  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=718)
    args = parser.parse_args()

    processed_dir = args.data_root / "external" / "snopes" / "processed"
    records = read_jsonl(processed_dir / "records.jsonl")
    sampled = stratified_sample(records, SNOPES_EXTERNAL_LABELS, args.size, args.seed)
    audit = {
        "sampling": "stratified_without_replacement",
        "seed": args.seed,
        "requested_size": args.size,
        **sample_audit(sampled),
    }
    write_jsonl(sampled, processed_dir / f"evidence_eval_subset_{args.size}.jsonl")
    (processed_dir / f"evidence_eval_subset_{args.size}_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
