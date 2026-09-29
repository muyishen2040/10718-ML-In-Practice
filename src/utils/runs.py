"""Run manifests that make every reported experiment traceable."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_manifest(
    output_path: str | Path,
    *,
    project_root: Path,
    config: dict[str, Any],
    input_paths: dict[str, str | Path],
) -> dict[str, Any]:
    """Write metadata without recording credentials or full dataset contents."""
    inputs: dict[str, dict[str, Any]] = {}
    for name, value in input_paths.items():
        path = Path(value)
        inputs[name] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(project_root),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config": config,
        "inputs": inputs,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
