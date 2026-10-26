"""Validated claim/evidence joins shared by baseline and new pipelines."""
from __future__ import annotations

from typing import Any

from src.utils.io import read_jsonl


def index_claims(rows: list[dict], expected: set[str] | None = None) -> dict[str, dict]:
    result = {}
    for row in rows:
        cid = str(row.get("claim_id") or "")
        if not cid or cid in result:
            raise ValueError(f"Missing or duplicate claim ID: {cid!r}")
        result[cid] = row
    if expected is not None and set(result) != expected:
        raise ValueError(f"Claim coverage differs: missing={sorted(expected-set(result))[:3]}, extra={sorted(set(result)-expected)[:3]}")
    return result


def attach_evidence(records: list[dict[str, Any]], rankings_path, *, exact: bool = False) -> list[dict[str, Any]]:
    claims = index_claims(records)
    rankings = index_claims(read_jsonl(rankings_path), expected=set(claims) if exact else None)
    missing = set(claims) - set(rankings)
    if missing:
        raise ValueError(f"Rankings are missing {len(missing)} records, e.g. {sorted(missing)[:3]}")
    for cid in claims:
        row = rankings[cid]
        if row.get("claim") and str(row["claim"]).strip() != str(claims[cid]["claim"]).strip():
            raise ValueError(f"Claim text differs for {cid}")
        evidence = row.get("retrieved") or []
        ids = [str(p.get("passage_id") or "") for p in evidence]
        if len(ids) != len(set(ids)) or any(not pid for pid in ids):
            raise ValueError(f"Missing or duplicate passage ID for {cid}")
    return [{**row, "retrieved_evidence": rankings[str(row["claim_id"])].get("retrieved") or []} for row in records]
