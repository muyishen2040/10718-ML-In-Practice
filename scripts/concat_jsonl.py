"""Concatenate prevalidated JSONL shards in an explicit, reproducible order."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.runs import write_run_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path, help="JSONL shard; repeat in required order.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for source in args.input:
            if not source.is_file():
                raise FileNotFoundError(source)
            with source.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        output.write(line if line.endswith("\n") else f"{line}\n")
                        row_count += 1
    write_run_manifest(
        args.output.with_name(f"{args.output.stem}_manifest.json"),
        project_root=PROJECT_ROOT,
        config={"script": "concat_jsonl", "input_count": len(args.input), "row_count": row_count},
        input_paths={f"input_{index}": path for index, path in enumerate(args.input)},
    )
    print(f"Wrote {row_count} JSONL rows to {args.output}")


if __name__ == "__main__":
    main()
