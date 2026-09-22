"""AVeriTeC claim loading, normalization, and audit helpers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.data.schema import ClaimRecord, EvidenceItem

CANONICAL_LABELS = (
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence",
)

_LABEL_ALIASES = {
    "supported": "Supported",
    "refuted": "Refuted",
    "not enough evidence": "Not Enough Evidence",
    "conflicting evidence/cherrypicking": "Conflicting Evidence",
    "conflicting evidence/cherry-picking": "Conflicting Evidence",
    "conflicting evidence": "Conflicting Evidence",
}


def normalize_label(label: str | None) -> str | None:
    """Return the project label corresponding to an official AVeriTeC label."""
    if label is None:
        return None
    cleaned = " ".join(label.strip().split()).casefold()
    try:
        return _LABEL_ALIASES[cleaned]
    except KeyError as exc:
        raise ValueError(f"Unsupported AVeriTeC label: {label!r}") from exc


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        return []
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError(f"Expected a JSON object or list in {path}")


def _answers_to_evidence(questions: Iterable[dict[str, Any]], claim_id: str) -> tuple[EvidenceItem, ...]:
    """Expose annotated answers for gold-evidence verification only.

    These records must never be used to construct the retrieval corpus.
    """
    evidence: list[EvidenceItem] = []
    for question_index, question in enumerate(questions):
        question_text = str(question.get("question") or "").strip()
        for answer_index, answer in enumerate(question.get("answers") or []):
            answer_text = str(answer.get("answer") or "").strip()
            if not answer_text:
                continue
            source_url = str(answer.get("source_url") or "").strip() or None
            evidence.append(
                EvidenceItem(
                    passage_id=f"{claim_id}:q{question_index}:a{answer_index}",
                    text=answer_text,
                    url=source_url,
                    metadata={
                        "question": question_text,
                        "answer_type": answer.get("answer_type"),
                        "source_medium": answer.get("source_medium"),
                        "is_gold_answer": True,
                    },
                )
            )
    return tuple(evidence)


def raw_to_record(raw: dict[str, Any], split: str, index: int) -> ClaimRecord:
    """Convert one official AVeriTeC object into the project record shape."""
    claim = str(raw.get("claim") or "").strip()
    if not claim:
        raise ValueError(f"{split}[{index}] has an empty claim")
    claim_id = str(raw.get("claim_id") or f"averitec:{split}:{index:05d}")
    label = normalize_label(raw.get("label"))
    questions = raw.get("questions") or []
    if not isinstance(questions, list):
        raise ValueError(f"{claim_id} has a non-list questions field")
    source_urls = tuple(
        item.url for item in _answers_to_evidence(questions, claim_id) if item.url is not None
    )
    return ClaimRecord(
        claim_id=claim_id,
        claim=claim,
        label=label,
        source_urls=tuple(dict.fromkeys(source_urls)),
        evidence=_answers_to_evidence(questions, claim_id),
        metadata={
            "dataset": "AVeriTeC",
            "split": split,
            "claim_date": raw.get("claim_date"),
            "speaker": raw.get("speaker"),
            "reporting_source": raw.get("reporting_source"),
            "location_iso_code": raw.get("location_ISO_code"),
            "required_reannotation": raw.get("required_reannotation"),
            "fact_checking_article": raw.get("fact_checking_article"),
        },
    )


def load_split(path: str | Path, split: str) -> list[ClaimRecord]:
    """Load one official AVeriTeC split from a JSON file."""
    source_path = Path(path)
    records = [raw_to_record(raw, split, index) for index, raw in enumerate(_read_json_or_jsonl(source_path))]
    ids = [record.claim_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate claim IDs found in {source_path}")
    return records


def load_split_with_rejections(
    path: str | Path, split: str
) -> tuple[list[ClaimRecord], list[dict[str, Any]]]:
    """Load a split while preserving an audit trail for unusable source rows.

    An empty claim cannot be evaluated by any project model, so it is excluded
    from prepared output. Any other schema or label error remains a hard failure
    until it is explicitly understood.
    """
    source_path = Path(path)
    records: list[ClaimRecord] = []
    rejections: list[dict[str, Any]] = []
    for index, raw in enumerate(_read_json_or_jsonl(source_path)):
        if not str(raw.get("claim") or "").strip():
            rejections.append(
                {
                    "source_index": index,
                    "reason": "empty_claim",
                    "source_label": raw.get("label"),
                    "fact_checking_article": raw.get("fact_checking_article"),
                }
            )
            continue
        records.append(raw_to_record(raw, split, index))
    ids = [record.claim_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate claim IDs found in {source_path}")
    return records, rejections


def split_audit(records: Iterable[ClaimRecord]) -> dict[str, Any]:
    """Return a small, serializable audit report for a prepared split."""
    materialized = list(records)
    return {
        "record_count": len(materialized),
        "label_counts": dict(sorted(Counter(record.label for record in materialized).items())),
        "records_with_gold_answers": sum(bool(record.evidence) for record in materialized),
        "gold_answer_count": sum(len(record.evidence) for record in materialized),
    }


def write_jsonl(records: Iterable[ClaimRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
