"""Download and canonicalize official AVeriTeC claim splits.

This script intentionally downloads only the small claim files. The official
knowledge store is large and must be acquired/prepared separately before BM25.
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

from src.data.averitec import load_split_with_rejections, split_audit, write_jsonl  # noqa: E402

OFFICIAL_CLAIM_URLS = {
    "train": "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/train.json?download=true",
    "dev": "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/dev.json?download=true",
    "test": "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/test.json?download=true",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--download-claims", action="store_true")
    parser.add_argument("--include-test", action="store_true", help="Prepare labels only if an official labeled test file is available.")
    args = parser.parse_args()

    raw_dir = args.data_root / "raw" / "averitec"
    processed_dir = args.data_root / "processed" / "averitec"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits = ["train", "dev"] + (["test"] if args.include_test else [])

    if args.download_claims:
        for split in splits:
            destination = raw_dir / f"{split}.json"
            if not destination.exists():
                print(f"Downloading {split} claims…")
                urlretrieve(OFFICIAL_CLAIM_URLS[split], destination)

    audit: dict[str, object] = {
        "dataset": "AVeriTeC",
        "splits": {},
        "raw_sha256": {},
        "rejections": {},
    }
    for split in splits:
        source = raw_dir / f"{split}.json"
        if not source.exists():
            raise FileNotFoundError(f"Missing {source}. Run with --download-claims or provide the file.")
        records, rejections = load_split_with_rejections(source, split)
        write_jsonl(records, processed_dir / f"{split}.jsonl")
        audit["splits"][split] = split_audit(records)  # type: ignore[index]
        audit["raw_sha256"][split] = sha256(source)  # type: ignore[index]
        audit["rejections"][split] = rejections  # type: ignore[index]

    (processed_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
