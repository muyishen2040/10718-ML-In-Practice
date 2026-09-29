"""Data-quality and split-overlap audits for leakage-aware experiments."""

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from src.data.corpus import canonical_url
from src.data.schema import ClaimRecord


def normalized_claim(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def _values_to_ids(records: Sequence[ClaimRecord], value_fn) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for value in value_fn(record):
            if value:
                output[str(value)].add(record.claim_id)
    return output


def split_overlap_audit(split_records: Mapping[str, Sequence[ClaimRecord]], sample_limit: int = 10) -> dict[str, Any]:
    """Report exact overlaps for inspection; it never silently drops examples."""
    claim_values = {
        split: _values_to_ids(records, lambda record: [normalized_claim(record.claim)])
        for split, records in split_records.items()
    }
    article_values = {
        split: _values_to_ids(records, lambda record: [canonical_url(record.metadata.get("fact_checking_article"))])
        for split, records in split_records.items()
    }
    source_values = {
        split: _values_to_ids(records, lambda record: [canonical_url(url) for url in record.source_urls])
        for split, records in split_records.items()
    }
    report: dict[str, Any] = {}
    for left, right in itertools.combinations(sorted(split_records), 2):
        pair = f"{left}__{right}"
        report[pair] = {}
        for name, values in (("normalized_claim", claim_values), ("fact_checking_article_url", article_values), ("annotated_source_url", source_values)):
            overlap = sorted(set(values[left]) & set(values[right]))
            report[pair][name] = {"overlap_count": len(overlap), "samples": overlap[:sample_limit]}
    return report
