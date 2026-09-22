"""Prepare DisinfoMM's English Snopes subset for external evaluation.

This script only creates claim records and a source-URL manifest. It does not
fetch web pages; that later step must honor site policies and leakage filters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.averitec import write_jsonl  # noqa: E402
from src.data.snopes import load_snopes_records  # noqa: E402

DISINFOMM_URL = "https://huggingface.co/datasets/Syokan/DisinfoMM/resolve/main/Dataset.csv?download=true"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--download", action="store_true", help="Download the immutable DisinfoMM CSV if absent.")
    args = parser.parse_args()

    raw_dir = args.data_root / "external" / "snopes" / "raw"
    processed_dir = args.data_root / "external" / "snopes" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    source_path = raw_dir / "Dataset.csv"
    if args.download and not source_path.exists():
        print("Downloading DisinfoMM CSV…")
        urlretrieve(DISINFOMM_URL, source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing {source_path}. Run with --download or provide the official CSV.")

    records, audit, rejections = load_snopes_records(source_path)
    audit["raw_sha256"] = sha256(source_path)
    audit["source_url"] = DISINFOMM_URL
    audit["rejections"] = rejections
    write_jsonl(records, processed_dir / "records.jsonl")
    (processed_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
