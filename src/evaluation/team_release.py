"""Consume the shared time-based evaluation release (docs/EVALUATION.md).

This is Teammate B's *proposed* loader for the pending release. The manifest
schema below follows the release checklist in EVALUATION.md; it must be agreed
with the release owner and C before scores produced through it are described as
the team's temporal comparison. Nothing here creates a split: it only reads,
verifies and applies a published one.

Proposed ``split_manifest.json`` schema::

    {
      "protocol_id": "averitec_train_time_v1",
      "source_claims_sha256": "<sha256 of processed/averitec/train.jsonl>",
      "folds": [
        {"fold_id": "w1",
         "fit_claim_ids": [...], "validation_claim_ids": [...],
         "fit_date_range": ["2014-01-01", "2019-12-31"],        # optional
         "validation_date_range": ["2020-01-01", "2020-06-30"]}  # optional
      ],
      "exclusions": [{"claim_id": "...", "reason": "missing_date"}]   # optional
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation.splits import parse_claim_date
from src.utils.runs import sha256_file


@dataclass(frozen=True)
class ReleaseFold:
    protocol_id: str
    fold_id: str
    manifest_path: Path
    manifest_sha256: str
    fit_claim_ids: tuple[str, ...]
    validation_claim_ids: tuple[str, ...]

    def binding(self) -> dict[str, Any]:
        """Run-configuration fields that tie outputs to this exact release fold."""
        return {"protocol_id": self.protocol_id, "fold_id": self.fold_id,
                "split_manifest_sha256": self.manifest_sha256,
                "fit_claim_count": len(self.fit_claim_ids),
                "validation_claim_count": len(self.validation_claim_ids)}


def read_release_pointer(config_path: Path) -> dict[str, Any]:
    release = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if release.get("status") != "ready":
        raise RuntimeError(f"The shared evaluation release is not ready (status={release.get('status')!r}); "
                           "do not substitute a private split. See docs/EVALUATION.md.")
    reference = release.get("split_manifest") or {}
    if not release.get("protocol_id") or not reference.get("relative_path") or not reference.get("sha256"):
        raise ValueError("A ready release must name protocol_id and split_manifest.relative_path/sha256")
    return release


def _ids(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty list of claim IDs")
    ids = tuple(str(v) for v in values)
    if any(not v for v in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains blank or duplicate claim IDs")
    return ids


def validate_manifest(manifest: dict[str, Any], train_records: list[dict[str, Any]], *,
                      train_claims_sha256: str | None = None,
                      forbidden_ids: set[str] = frozenset()) -> None:
    """Check IDs, overlap and time order against the actual training claims."""
    if not manifest.get("protocol_id"):
        raise ValueError("Manifest lacks protocol_id")
    expected_hash = manifest.get("source_claims_sha256")
    if expected_hash and train_claims_sha256 and expected_hash != train_claims_sha256:
        raise ValueError("Training claims differ from the claims the split was released for")
    known = {str(r["claim_id"]): r for r in train_records}
    folds = manifest.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("Manifest has no folds")
    fold_ids = [str(f.get("fold_id") or "") for f in folds]
    if any(not f for f in fold_ids) or len(set(fold_ids)) != len(fold_ids):
        raise ValueError("Fold IDs must be nonblank and unique")
    for fold in folds:
        name = fold["fold_id"]
        fit, val = _ids(fold.get("fit_claim_ids"), f"{name}.fit"), _ids(fold.get("validation_claim_ids"), f"{name}.validation")
        if set(fit) & set(val):
            raise ValueError(f"Fold {name} reuses claims in fit and validation")
        unknown = (set(fit) | set(val)) - set(known)
        if unknown:
            raise ValueError(f"Fold {name} names claims absent from training data, e.g. {sorted(unknown)[:3]}")
        if (set(fit) | set(val)) & forbidden_ids:
            raise ValueError(f"Fold {name} contains official dev/test claims")
        fit_dates = [parse_claim_date(known[c].get("metadata", {}).get("claim_date")) for c in fit]
        val_dates = [parse_claim_date(known[c].get("metadata", {}).get("claim_date")) for c in val]
        fit_dates, val_dates = [d for d in fit_dates if d], [d for d in val_dates if d]
        if fit_dates and val_dates and max(fit_dates) > min(val_dates):
            raise ValueError(f"Fold {name} is not earlier-to-later: fit ends {max(fit_dates)}, validation starts {min(val_dates)}")


def load_release_fold(config_path: Path, data_root: Path, fold_id: str, train_records: list[dict[str, Any]], *,
                      train_claims_path: Path | None = None, forbidden_ids: set[str] = frozenset()) -> ReleaseFold:
    """Resolve the published pointer, verify the manifest hash and return one fold."""
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
    validate_manifest(manifest, train_records, forbidden_ids=forbidden_ids,
                      train_claims_sha256=sha256_file(train_claims_path) if train_claims_path else None)
    fold = next((f for f in manifest["folds"] if str(f["fold_id"]) == fold_id), None)
    if fold is None:
        raise ValueError(f"Released fold {fold_id!r} is missing from the manifest")
    return ReleaseFold(protocol_id=release["protocol_id"], fold_id=fold_id, manifest_path=path, manifest_sha256=digest,
                       fit_claim_ids=tuple(map(str, fold["fit_claim_ids"])),
                       validation_claim_ids=tuple(map(str, fold["validation_claim_ids"])))


def fold_rows(records: list[dict[str, Any]], fold: ReleaseFold) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (fit, validation) rows in the manifest's ID order."""
    by_id = {str(r["claim_id"]): r for r in records}
    return [by_id[c] for c in fold.fit_claim_ids], [by_id[c] for c in fold.validation_claim_ids]
