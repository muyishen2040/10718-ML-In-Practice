"""Build an auditable cited-source manifest before any web fetching occurs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse


# Snopes itself is always prohibited as evaluation evidence. Other fact-checking
# domains are flagged for review rather than silently discarded: they may be
# useful provenance but often contain verdict leakage.
REJECTED_EVIDENCE_DOMAINS = {"snopes.com", "www.snopes.com"}
FACT_CHECK_REVIEW_DOMAINS = {
    "politifact.com",
    "www.politifact.com",
    "factcheck.org",
    "www.factcheck.org",
    "fullfact.org",
    "fullfact.org",
    "leadstories.com",
    "www.leadstories.com",
    "checkyourfact.com",
    "www.checkyourfact.com",
    "africacheck.org",
    "www.africacheck.org",
}


def domain_for_url(url: str) -> str:
    return urlparse(url).netloc.casefold().removeprefix("www.")


def source_screening_status(domain: str) -> str:
    normalized = domain.casefold().removeprefix("www.")
    if normalized in {item.removeprefix("www.") for item in REJECTED_EVIDENCE_DOMAINS}:
        return "excluded_fact_check_source"
    if normalized in {item.removeprefix("www.") for item in FACT_CHECK_REVIEW_DOMAINS}:
        return "requires_fact_check_review"
    return "unreviewed"


def build_source_manifest(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deduplicate cited URLs while retaining every claim that cited each URL."""
    by_url: dict[str, dict[str, Any]] = {}
    for record in records:
        for url in record.get("source_urls") or []:
            url = str(url)
            domain = domain_for_url(url)
            entry = by_url.setdefault(
                url,
                {
                    "url": url,
                    "domain": domain,
                    "screening_status": source_screening_status(domain),
                    "claim_ids": [],
                },
            )
            claim_id = str(record["claim_id"])
            if claim_id not in entry["claim_ids"]:
                entry["claim_ids"].append(claim_id)
    manifest = sorted(by_url.values(), key=lambda row: row["url"])
    status_counts = Counter(row["screening_status"] for row in manifest)
    domain_counts = Counter(row["domain"] for row in manifest)
    audit = {
        "unique_source_url_count": len(manifest),
        "screening_status_counts": dict(sorted(status_counts.items())),
        "top_domains": [
            {"domain": domain, "url_count": count}
            for domain, count in domain_counts.most_common(25)
        ],
    }
    return manifest, audit
