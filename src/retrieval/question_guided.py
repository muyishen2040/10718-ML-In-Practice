"""Pure, testable parts of question-guided evidence selection.

No labels, gold answers, or qrels are inputs to these functions.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from src.data.corpus import canonical_url


TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'([])")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "has", "have", "how", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "there", "this", "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "would", "could", "should", "claim", "evidence", "supports", "contradicts",
}


def words(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower())) - STOPWORDS


def parse_questions(response: str, claim: str, *, limit: int = 2) -> list[str]:
    """Accept JSON questions only; fall back to a claim-derived question."""
    found: list[str] = []
    decoder = json.JSONDecoder()
    for index, char in enumerate(response):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(response[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("questions"), list):
            for item in value["questions"]:
                if isinstance(item, str):
                    question = " ".join(item.split())[:240]
                    if 8 <= len(question) and question.lower() not in {q.lower() for q in found}:
                        found.append(question)
                if len(found) == limit:
                    break
            break
    if len(found) < limit:
        return [
            f"What evidence supports or contradicts this claim: {claim[:180]}?",
            f"Which source verifies the date, number, or entity in this claim: {claim[:180]}?",
        ][:limit]
    return found


def shortlist_candidates(row: dict[str, Any], questions: list[str], *, limit: int, bm25_keep: int) -> list[dict[str, Any]]:
    """Retain BM25 leaders and candidates matching the question terms."""
    candidates = list(row.get("retrieved") or [])
    ids = [str(item["passage_id"]) for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate candidate passage IDs for {row.get('claim_id')}")
    if limit < 1 or bm25_keep < 0:
        raise ValueError("limit must be positive and bm25_keep nonnegative")
    query_terms = words(" ".join([str(row.get("claim") or ""), *questions]))
    leader_indices = list(range(min(bm25_keep, limit, len(candidates))))
    scored = []
    for index, item in enumerate(candidates):
        overlap = len(query_terms & words(str(item.get("text") or "")[:1600]))
        scored.append((-overlap, index))
    chosen = set(leader_indices)
    for _, index in sorted(scored):
        if len(chosen) >= min(limit, len(candidates)):
            break
        chosen.add(index)
    return [candidates[index] for index in sorted(chosen)]


def reciprocal_rank_scores(score_matrix: list[list[float]], *, claim_weight: float = 1.0) -> list[float]:
    """Fuse per-query cross-encoder ranks without comparing raw logits across queries."""
    if not score_matrix:
        return []
    size = len(score_matrix[0])
    if any(len(row) != size or any(not math.isfinite(float(value)) for value in row) for row in score_matrix):
        raise ValueError("Cross-encoder score matrix is ragged or non-finite")
    result = [0.0] * size
    for query_index, scores in enumerate(score_matrix):
        weight = claim_weight if query_index == 0 else 1.0
        for rank, index in enumerate(sorted(range(size), key=lambda i: (-scores[i], i)), start=1):
            result[index] += weight / (20 + rank)
    return result


def select_diverse(candidates: list[dict[str, Any]], scores: list[float], *, k: int = 3) -> list[dict[str, Any]]:
    """Select at most k unique-source passages, then fill if source count is low."""
    if len(candidates) != len(scores):
        raise ValueError("Candidate and score counts differ")
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], str(candidates[i]["passage_id"])))
    selected: list[int] = []
    seen_sources: set[str] = set()
    for index in order:
        item = candidates[index]
        url = canonical_url(item.get("url"))
        source = f"url:{url}" if url else f"passage:{item['passage_id']}"
        if source not in seen_sources:
            selected.append(index)
            seen_sources.add(source)
        if len(selected) == k:
            break
    if len(selected) < k:
        selected.extend(index for index in order if index not in selected)
    return [{**candidates[index], "score": float(scores[index])} for index in selected[:k]]


def extract_sentences(text: str, claim: str, questions: list[str], *, max_chars: int = 900, max_sentences: int = 3) -> str:
    """Select exact source sentences by term coverage; never generate evidence text."""
    original = str(text).strip()
    if len(original) <= max_chars:
        return original
    sentences = [item.strip() for item in SENTENCE_BOUNDARY.split(original) if item.strip()]
    if len(sentences) < 2:
        return original[:max_chars].rsplit(" ", 1)[0] or original[:max_chars]
    query_sets = [words(claim), *(words(q) for q in questions)]
    scored = []
    for index, sentence in enumerate(sentences):
        tokens = words(sentence)
        coverage = sum(len(tokens & query) / max(1, len(query)) for query in query_sets)
        scored.append((-coverage, index))
    chosen: list[int] = []
    used = 0
    for _, index in sorted(scored):
        sentence = sentences[index]
        if used + len(sentence) + (1 if chosen else 0) > max_chars:
            continue
        chosen.append(index)
        used += len(sentence) + (1 if len(chosen) > 1 else 0)
        if len(chosen) == max_sentences:
            break
    if not chosen:
        return original[:max_chars].rsplit(" ", 1)[0] or original[:max_chars]
    return " ".join(sentences[index] for index in sorted(chosen))
