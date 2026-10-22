"""Pure helpers for extractive, question-conditioned evidence selection."""

from __future__ import annotations

import math
from typing import Any

from src.data.corpus import canonical_url


FEATURE_KEYS = (
    "max_answer_margin",
    "mean_answer_margin",
    "bm25_reciprocal_rank",
    "question_guided_top3",
    "answer_word_count",
)


def choose_candidates(
    bm25_row: dict[str, Any], question_row: dict[str, Any], *, limit: int, bm25_keep: int
) -> list[dict[str, Any]]:
    """Use only prediction-time rankings; retain a lexical safety net."""
    if limit < 1 or not 0 <= bm25_keep <= limit:
        raise ValueError("Require limit > 0 and 0 <= bm25_keep <= limit")
    bm25 = bm25_row.get("retrieved") or []
    by_id = {str(item["passage_id"]): item for item in bm25}
    if len(by_id) != len(bm25):
        raise ValueError("Duplicate passage ID in BM25 ranking")
    selected = {str(item["passage_id"]) for item in (question_row.get("retrieved") or [])[:3]}
    ordered_ids = [str(item["passage_id"]) for item in bm25[:bm25_keep]]
    ordered_ids += [str(item["passage_id"]) for item in (question_row.get("retrieved") or [])[:3]]
    ordered_ids += [str(value) for value in question_row.get("shortlist_passage_ids") or []]
    ordered_ids += [str(item["passage_id"]) for item in bm25]
    seen: set[str] = set()
    chosen: list[dict[str, Any]] = []
    bm25_positions = {str(item["passage_id"]): index + 1 for index, item in enumerate(bm25)}
    for passage_id in ordered_ids:
        if passage_id in seen or passage_id not in by_id:
            continue
        seen.add(passage_id)
        item = by_id[passage_id]
        if not str(item.get("text") or "").strip():
            continue
        chosen.append({
            **item,
            "bm25_rank": bm25_positions[passage_id],
            "question_guided_top3": int(passage_id in selected),
        })
        if len(chosen) >= limit:
            break
    return chosen


def best_extractable_span(
    passage: str,
    offsets: list[tuple[int, int]],
    sequence_ids: list[int | None],
    start_logits: list[float],
    end_logits: list[float],
    *,
    cls_index: int,
    max_answer_tokens: int,
    max_logit_candidates: int = 20,
) -> tuple[str, float]:
    """Return a verbatim span and SQuAD2-style answer-vs-null margin."""
    if not (len(offsets) == len(sequence_ids) == len(start_logits) == len(end_logits)):
        raise ValueError("Token arrays must have identical lengths")
    null_score = start_logits[cls_index] + end_logits[cls_index]
    if max_answer_tokens < 1 or max_logit_candidates < 1:
        raise ValueError("Span and logit candidate limits must be positive")
    context_tokens = [index for index, segment in enumerate(sequence_ids)
                      if segment == 1 and offsets[index][1] > offsets[index][0]]
    top_starts = sorted(context_tokens, key=lambda index: -start_logits[index])[:max_logit_candidates]
    top_ends = sorted(context_tokens, key=lambda index: -end_logits[index])[:max_logit_candidates]
    best_score = -math.inf
    best_text = ""
    for start in top_starts:
        for end in top_ends:
            if end < start or end >= start + max_answer_tokens:
                continue
            begin_char, end_char = offsets[start][0], offsets[end][1]
            if not 0 <= begin_char < end_char <= len(passage):
                continue
            score = start_logits[start] + end_logits[end]
            if score > best_score:
                best_score = score
                best_text = passage[begin_char:end_char]
    if not best_text:
        return "", -100.0
    return best_text, float(best_score - null_score)


def source_excerpt(passage: str, answer: str, *, max_chars: int = 330) -> str:
    """Keep the exact source substring around a candidate answer."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    position = passage.find(answer) if answer else -1
    if position < 0:
        return passage[:max_chars].strip()
    start = max(0, position - max(0, (max_chars - len(answer)) // 2))
    end = min(len(passage), start + max_chars)
    start = max(0, end - max_chars)
    return passage[start:end].strip()


def qa_text(question: str, answer: str, passage: str, margin: float) -> str:
    excerpt = source_excerpt(passage, answer if margin > 0 else "")
    if margin > 0 and answer:
        return f"Question: {question}\nCandidate answer (verbatim): {answer}\nSource excerpt: {excerpt}"
    return f"Question: {question}\nNo confident extractive answer. Source excerpt: {excerpt}"


def feature_vector(item: dict[str, Any]) -> list[float]:
    return [float(item[key]) for key in FEATURE_KEYS]


def rank_with_source_diversity(items: list[dict[str, Any]], scores: list[float], *, k: int = 3) -> list[dict[str, Any]]:
    if len(items) != len(scores):
        raise ValueError("One score is required per passage")
    ordered = sorted(range(len(items)), key=lambda index: (-scores[index], str(items[index]["passage_id"])))
    first: list[int] = []
    seen_sources: set[str] = set()
    for index in ordered:
        item = items[index]
        source = canonical_url(item.get("url")) or str(item["passage_id"])
        if source not in seen_sources:
            seen_sources.add(source)
            first.append(index)
            if len(first) == k:
                break
    if len(first) < min(k, len(items)):
        first.extend(index for index in ordered if index not in first)
    indices = first[:k] + [index for index in ordered if index not in first[:k]]
    return [{**items[index], "selector_score": float(scores[index])} for index in indices]
