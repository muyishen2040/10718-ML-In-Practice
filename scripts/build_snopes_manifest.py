"""Deduplicate and screen DisinfoMM cited URLs before web-corpus construction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.external.snopes_manifest import build_source_manifest  # noqa: E402


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
    args = parser.parse_args()
    processed_dir = args.data_root / "external" / "snopes" / "processed"
    records = read_jsonl(processed_dir / "records.jsonl")
    manifest, audit = build_source_manifest(records)
    audit["record_count"] = len(records)
    write_jsonl(manifest, processed_dir / "source_manifest.jsonl")
    (processed_dir / "source_manifest_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
