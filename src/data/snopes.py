"""DisinfoMM Snopes filtering and conservative external-label normalization."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from src.data.schema import ClaimRecord

SNOPES_EXTERNAL_LABELS = ("Supported", "Refuted", "Not Enough Evidence")


def normalize_snopes_label(evaluation: str | None) -> str | None:
    """Map DisinfoMM's harmonized verdict to the external three-class task.

    The archival CSV retains spelling/capitalization variants. `None` means the
    source verdict cannot be mapped safely and should be excluded with an audit
    record rather than forced into a project class.
    """
    value = " ".join((evaluation or "").strip().casefold().split())
    value = value.replace("moslty", "mostly").replace("exagerated", "exaggerated")
    if value == "true" or value.startswith("mostly true"):
        return "Supported"
    if value == "false" or value.startswith("mostly false"):
        return "Refuted"
    if value == "incomplete":
        return "Not Enough Evidence"
    return None


def parse_source_urls(value: str | None) -> tuple[str, ...]:
    """Parse the historical comma-separated cited-source field conservatively."""
    urls: list[str] = []
    for candidate in (value or "").split(","):
        url = candidate.strip().rstrip(".,;:)")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if url not in urls:
            urls.append(url)
    return tuple(urls)


def row_to_record(row: dict[str, str], source_index: int) -> ClaimRecord:
    claim = (row.get("Claim") or "").strip()
    if not claim:
        raise ValueError("empty_claim")
    label = normalize_snopes_label(row.get("Evaluation"))
    if label is None:
        raise ValueError("unmappable_evaluation")
    source_id = (row.get("Num") or str(source_index)).strip()
    return ClaimRecord(
        claim_id=f"snopes:{source_id}",
        claim=claim,
        label=label,
        source_urls=parse_source_urls(row.get("Article sources")),
        metadata={
            "dataset": "DisinfoMM",
            "source": "snopes",
            "source_index": source_index,
            "source_id": source_id,
            "date": row.get("Date"),
            "source_evaluation": row.get("Evaluation"),
            "source_label": row.get("Label"),
            "fact_check_url": row.get("Article Link"),
            "declaration_url": row.get("Declaration Link"),
            "keywords": row.get("Keywords"),
            "tags": row.get("Tags"),
            # Deliberately do not retain the post-hoc Explanation as model input.
            "has_explanation": bool((row.get("Explanation") or "").strip()),
        },
    )


def load_snopes_records(
    path: str | Path,
) -> tuple[list[ClaimRecord], dict[str, Any], list[dict[str, Any]]]:
    """Load only English Snopes rows and return records plus an audit trail."""
    source_path = Path(path)
    records: list[ClaimRecord] = []
    rejections: list[dict[str, Any]] = []
    raw_evaluations: Counter[str] = Counter()
    source_rows = 0
    snopes_rows = 0
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Num", "Website", "Claim", "Evaluation", "Article sources"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"DisinfoMM is missing required columns: {sorted(missing)}")
        for source_index, row in enumerate(reader):
            source_rows += 1
            if (row.get("Website") or "").strip().casefold() != "snopes":
                continue
            snopes_rows += 1
            raw_evaluations[(row.get("Evaluation") or "").strip()] += 1
            try:
                records.append(row_to_record(row, source_index))
            except ValueError as error:
                rejections.append(
                    {
                        "source_index": source_index,
                        "source_id": row.get("Num"),
                        "reason": str(error),
                        "source_evaluation": row.get("Evaluation"),
                    }
                )
    claim_ids = [record.claim_id for record in records]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("Duplicate DisinfoMM Snopes record IDs after filtering")
    audit = {
        "dataset": "DisinfoMM",
        "filter": "Website == snopes",
        "source_row_count": source_rows,
        "snopes_row_count": snopes_rows,
        "accepted_record_count": len(records),
        "label_counts": dict(sorted(Counter(record.label for record in records).items())),
        "raw_evaluation_counts": dict(sorted(raw_evaluations.items())),
        "records_with_source_urls": sum(bool(record.source_urls) for record in records),
        "rejection_count": len(rejections),
    }
    return records, audit, rejections
