"""Decisive-evidence pilot labels (Teammate B brief, steps 3-4).

A claim's verdict is *not* copied to its passages. Each reviewed passage gets a
passage-level judgment of whether it decides the claim on its own:

- ``direct_support``: the passage itself establishes the claim as stated.
- ``direct_contradiction``: the passage itself establishes that the claim is false.
- ``related_insufficient``: same topic, but a key fact, date, quantity or
  qualification is missing.
- ``ambiguous``: reviewers cannot decide from the passage text.
- ``unrelated``: not about the claim.

Decisive labels require an exact quote from the passage. An unmatched source URL
is recorded only as annotator context, never as an automatic negative. Gold
answers of *training-fit* claims may be shown to annotators to guide supervision;
they are written to a separate context file and never become selector features.
"""
from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.data.averitec import CANONICAL_LABELS
from src.data.corpus import canonical_url
from src.retrieval.evidence_quality import normalized_text, original_url

DECISIVE_LABELS = ("direct_support", "direct_contradiction", "related_insufficient", "ambiguous", "unrelated")
DECISIVE_POSITIVE = {"direct_support", "direct_contradiction"}
REVIEW_STATUSES = ("unreviewed", "silver", "reviewed")
ANNOTATION_FIELDS = ("decisive_label", "quote", "missing_qualification", "notes", "annotator", "review_status")
SHEET_FIELDS = ("claim_id", "claim", "claim_date", "passage_id", "bm25_rank", "candidate_reason", "url",
                "url_matches_annotated_source", "text", *ANNOTATION_FIELDS)


def _answer_text(record: dict[str, Any]) -> str:
    return " ".join(str(e.get("text") or "") for e in record.get("evidence") or [])


def _overlap(a: str, b: str) -> float:
    left, right = set(normalized_text(a).split()), set(normalized_text(b).split())
    return len(left & right) / max(1, len(left))


def sample_claims(records: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Deterministic label-balanced sample so rare verdicts appear in a small pilot."""
    if count < 1:
        raise ValueError("count must be positive")
    by_label: dict[str, list[dict]] = defaultdict(list)
    for record in sorted(records, key=lambda r: str(r["claim_id"])):
        by_label[record["label"]].append(record)
    rng = random.Random(seed)
    for rows in by_label.values():
        rng.shuffle(rows)
    chosen, labels = [], [label for label in CANONICAL_LABELS if by_label.get(label)]
    while len(chosen) < count and any(by_label[label] for label in labels):
        for label in labels:
            if by_label[label] and len(chosen) < count:
                chosen.append(by_label[label].pop())
    return chosen


def select_candidates(record: dict[str, Any], ranking: dict[str, Any], *, per_claim: int = 10, displayed: int = 3,
                      answer_aligned: int = 3) -> list[dict[str, Any]]:
    """Choose candidates for review from one claim's saved BM25 pool.

    Includes the displayed BM25 top three, passages most aligned with the
    training claim's annotated answers (annotation aid only), and same-topic
    hard negatives spread over the remaining ranks.
    """
    pool = list(ranking.get("retrieved") or [])
    if per_claim < displayed:
        raise ValueError("per_claim must cover the displayed passages")
    picked: dict[str, tuple[int, str]] = {}
    for rank, item in enumerate(pool[:displayed], start=1):
        picked[item["passage_id"]] = (rank, "bm25_displayed")
    answers = _answer_text(record)
    if answers and answer_aligned:
        scored = sorted(((-_overlap(answers, str(p.get("text") or "")), r, p) for r, p in enumerate(pool, start=1)
                         if p["passage_id"] not in picked), key=lambda t: (t[0], t[1]))
        for _, rank, item in scored[:answer_aligned]:
            picked[item["passage_id"]] = (rank, "answer_aligned_aid")
    rest = [(r, p) for r, p in enumerate(pool, start=1) if p["passage_id"] not in picked]
    slots = per_claim - len(picked)
    if slots > 0 and rest:
        step = len(rest) / slots
        for i in range(min(slots, len(rest))):
            rank, item = rest[int(i * step)]
            picked[item["passage_id"]] = (rank, "same_topic_hard_negative_candidate")
    by_id = {p["passage_id"]: p for p in pool}
    sources = {canonical_url(original_url(u)) for u in record.get("source_urls") or []} - {None}
    rows = []
    for pid, (rank, reason) in sorted(picked.items(), key=lambda kv: kv[1][0]):
        item = by_id[pid]
        rows.append({"claim_id": record["claim_id"], "claim": record["claim"],
                     "claim_date": (record.get("metadata") or {}).get("claim_date"), "passage_id": pid, "bm25_rank": rank,
                     "candidate_reason": reason, "url": item.get("url"),
                     "url_matches_annotated_source": canonical_url(original_url(item.get("url"))) in sources,
                     "text": str(item.get("text") or ""), **{f: "" for f in ANNOTATION_FIELDS}, "review_status": "unreviewed"})
    return rows


def build_sheet(records: list[dict[str, Any]], rankings: dict[str, dict[str, Any]], fit_ids: set[str], *, claims: int,
                per_claim: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Return (passage rows to label, claim context rows with gold aids)."""
    eligible = [r for r in records if r["claim_id"] in fit_ids]
    missing = [r["claim_id"] for r in eligible if r["claim_id"] not in rankings]
    if missing:
        raise ValueError(f"Rankings lack {len(missing)} fit claims, e.g. {missing[:3]}")
    chosen = sample_claims(eligible, claims, seed)
    rows, context = [], []
    for record in chosen:
        rows.extend(select_candidates(record, rankings[record["claim_id"]], per_claim=per_claim))
        context.append({"claim_id": record["claim_id"], "claim": record["claim"], "claim_verdict": record["label"],
                        "annotated_answers_for_annotators_only": [
                            {"question": (e.get("metadata") or {}).get("question"), "answer": e.get("text"), "source_url": e.get("url")}
                            for e in record.get("evidence") or []]})
    return rows, context


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_annotations(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    from src.utils.io import read_jsonl
    return read_jsonl(path)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def validate_annotations(rows: Iterable[dict], rankings: dict[str, dict], fit_ids: set[str], *,
                         allow_silver: bool = False) -> tuple[list[dict], dict[str, Any]]:
    """Return usable labeled rows and an audit summary; raise on malformed labels.

    Rows for claims outside ``fit_ids`` are dropped (never used for training),
    so a sheet drawn before the release can be reconciled with its fit folds.
    """
    usable, dropped, seen = [], Counter(), set()
    for row in rows:
        cid, pid = str(row.get("claim_id") or ""), str(row.get("passage_id") or "")
        label = str(row.get("decisive_label") or "").strip()
        status = str(row.get("review_status") or "unreviewed").strip()
        if not label:
            dropped["unlabeled"] += 1
            continue
        if label not in DECISIVE_LABELS:
            raise ValueError(f"Unknown decisive label {label!r} for {cid}/{pid}")
        if status not in REVIEW_STATUSES:
            raise ValueError(f"Unknown review status {status!r} for {cid}/{pid}")
        if (cid, pid) in seen:
            raise ValueError(f"Duplicate annotation for {cid}/{pid}")
        seen.add((cid, pid))
        pool = {p["passage_id"]: p for p in (rankings.get(cid) or {}).get("retrieved") or []}
        if pid not in pool:
            raise ValueError(f"{cid}/{pid} is not in that claim's saved candidate pool")
        if label in DECISIVE_POSITIVE:
            quote = str(row.get("quote") or "")
            if not quote.strip() or _squash(quote) not in _squash(pool[pid].get("text") or ""):
                raise ValueError(f"{cid}/{pid}: a decisive label needs an exact quote from the passage text")
        if cid not in fit_ids:
            dropped["outside_fit_ids"] += 1
            continue
        if status == "unreviewed" or (status == "silver" and not allow_silver):
            dropped[f"status_{status}"] += 1
            continue
        usable.append({**row, "claim_id": cid, "passage_id": pid, "decisive_label": label, "review_status": status})
    per_claim = defaultdict(Counter)
    for row in usable:
        per_claim[row["claim_id"]][row["decisive_label"]] += 1
    summary = {"usable_rows": len(usable), "dropped": dict(dropped), "claims": len(per_claim),
               "label_counts": dict(Counter(r["decisive_label"] for r in usable)),
               "claims_with_decisive_passage": sum(any(c[l] for l in DECISIVE_POSITIVE) for c in per_claim.values()),
               "claims_with_conflicting_decisive_passages": sum(c["direct_support"] > 0 and c["direct_contradiction"] > 0 for c in per_claim.values()),
               "review_status_counts": dict(Counter(r["review_status"] for r in usable))}
    return usable, summary


def selector_targets(rows: Iterable[dict]) -> list[dict]:
    """Binary passage targets; ambiguous rows are excluded, not called negative."""
    return [{"claim_id": r["claim_id"], "passage_id": r["passage_id"], "target": int(r["decisive_label"] in DECISIVE_POSITIVE)}
            for r in rows if r["decisive_label"] != "ambiguous"]
