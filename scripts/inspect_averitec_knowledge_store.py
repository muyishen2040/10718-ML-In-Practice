"""Inspect a downloaded AVeriTeC knowledge-store archive before extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.knowledge_store import inspect_zip_archive  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--sample-members", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_zip_archive(args.archive, args.sample_members)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
