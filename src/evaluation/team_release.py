"""Resolve one fold of the shared temporal release for B's model runners.

The release (``configs/team_evaluation.json``, protocol
``trust_temporal_20261117_v1``) is built and validated by
``src/evaluation/temporal_protocol.py``. This module only follows the pointer:
it checks the manifest hash recorded there, reuses the release's own
``validate_manifest`` (IDs, summaries, chronology, duplicate groups and the
source train hash), rejects official dev IDs, and returns the fold's three
partitions. It never creates a split.

Each fold has ``fit`` (training), ``validation`` (choose settings and the
Supported-score threshold) and ``evaluation`` (later claims, scored once with
the frozen choice and no refit).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.temporal_protocol import validate_manifest
from src.utils.runs import sha256_file

PARTITIONS = ("fit", "validation", "evaluation")


@dataclass(frozen=True)
class ReleaseFold:
    protocol_id: str
    fold_id: str
    manifest_path: Path
    manifest_sha256: str
    ids: dict[str, tuple[str, ...]]

    @property
    def fit_claim_ids(self) -> tuple[str, ...]:
        return self.ids["fit"]

    def binding(self) -> dict[str, Any]:
        """Run-configuration fields that tie outputs to this exact release fold."""
        return {"protocol_id": self.protocol_id, "fold_id": self.fold_id,
                "split_manifest_sha256": self.manifest_sha256,
                **{f"{name}_claim_count": len(self.ids[name]) for name in PARTITIONS}}


def read_release_pointer(config_path: Path) -> dict[str, Any]:
    release = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if release.get("status") != "ready":
        raise RuntimeError(f"The shared evaluation release is not ready (status={release.get('status')!r}); "
                           "do not substitute a private split. See docs/EVALUATION.md.")
    reference = release.get("split_manifest") or {}
    if not release.get("protocol_id") or not reference.get("relative_path") or not reference.get("sha256"):
        raise ValueError("A ready release must name protocol_id and split_manifest.relative_path/sha256")
    return release


def load_release_fold(config_path: Path, data_root: Path, fold_id: str, train_records: list[dict[str, Any]], *,
                      train_claims_path: Path, forbidden_ids: set[str] = frozenset()) -> ReleaseFold:
    """Resolve the published pointer, verify the manifest and return one fold."""
    release = read_release_pointer(config_path)
    reference = release["split_manifest"]
    path = Path(data_root) / reference["relative_path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if digest != reference["sha256"]:
        raise ValueError("Split manifest hash differs from the published release; do not train on it")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != release["protocol_id"]:
        raise ValueError("Manifest protocol_id differs from the release pointer")
    released_folds = [str(f) for f in release.get("fold_ids") or []]
    if fold_id not in released_folds:
        raise ValueError(f"Fold {fold_id!r} is not among released folds {released_folds}")
    validate_manifest(manifest, train_records, sha256_file(train_claims_path))
    fold = next((f for f in manifest["folds"] if str(f["fold_id"]) == fold_id), None)
    if fold is None:
        raise ValueError(f"Released fold {fold_id!r} is missing from the manifest")
    ids = {name: tuple(map(str, fold["ids"][name])) for name in PARTITIONS}
    leaked = set().union(*ids.values()) & set(forbidden_ids)
    if leaked:
        raise ValueError(f"Fold {fold_id} contains official dev/test claims, e.g. {sorted(leaked)[:3]}")
    return ReleaseFold(protocol_id=release["protocol_id"], fold_id=fold_id, manifest_path=path,
                       manifest_sha256=digest, ids=ids)


def fold_rows(records: list[dict[str, Any]], fold: ReleaseFold) -> dict[str, list[dict[str, Any]]]:
    """Return {partition: rows} in the manifest's ID order."""
    by_id = {str(r["claim_id"]): r for r in records}
    return {name: [by_id[c] for c in fold.ids[name]] for name in PARTITIONS}
