"""Download a large official AVeriTeC archive only after explicit confirmation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.runs import sha256_file  # noqa: E402
from src.utils.progress import progress  # noqa: E402

OFFICIAL_ARCHIVES = {
    "dev": "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/dev_knowledge_store.zip?download=true",
    "train_0_999": "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/train/train_0_999.zip?download=true",
    "train_1000_1999": "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/train/train_1000_1999.zip?download=true",
    "train_2000_3067": "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data_store/knowledge_store/train/train_2000_3067.zip?download=true",
}


def download_with_progress(url: str, destination: Path) -> None:
    """Download in chunks so Colab can show byte-level progress."""
    partial = destination.with_suffix(f"{destination.suffix}.partial")
    if partial.exists():
        raise FileExistsError(
            f"Found incomplete download {partial}. Remove it explicitly before retrying so no partial archive is mistaken for complete."
        )
    with urlopen(url) as response:
        content_length = response.headers.get("Content-Length")
        total_bytes = int(content_length) if content_length and content_length.isdigit() else None
        with partial.open("xb") as output, progress(
            total=total_bytes, description="Downloading AVeriTeC archive", unit="B", unit_scale=True
        ) as bar:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                bar.update(len(chunk))
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", choices=tuple(OFFICIAL_ARCHIVES), default="dev")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--confirm-large-download", action="store_true")
    args = parser.parse_args()
    if not args.confirm_large_download:
        parser.error("Refusing a multi-GB download without --confirm-large-download.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        print(f"Downloading official AVeriTeC {args.archive} knowledge store to {args.output}…")
        download_with_progress(OFFICIAL_ARCHIVES[args.archive], args.output)
    actual = sha256_file(args.output)
    if args.expected_sha256 and actual.casefold() != args.expected_sha256.casefold():
        raise RuntimeError(f"SHA-256 mismatch: expected {args.expected_sha256}, got {actual}")
    print(f"SHA-256: {actual}")


if __name__ == "__main__":
    main()
