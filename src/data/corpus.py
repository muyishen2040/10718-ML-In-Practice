"""Candidate-document validation, passage chunking, and URL-based qrels."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def canonical_url(value: str | None) -> str | None:
    """Normalize only URL details that should not change document identity."""
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        return raw.rstrip("/")
    return urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(), parts.path.rstrip("/"), parts.query, ""))


def chunk_text(text: str, max_words: int = 160, overlap_words: int = 40) -> list[str]:
    """Split a document into deterministic overlapping word passages."""
    if max_words < 1 or overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("Require max_words > 0 and 0 <= overlap_words < max_words")
    words = re.findall(r"\S+", text)
    if not words:
        return []
    passages = []
    step = max_words - overlap_words
    for start in range(0, len(words), step):
        passage = " ".join(words[start : start + max_words]).strip()
        if passage:
            passages.append(passage)
        if start + max_words >= len(words):
            break
    return passages


def documents_to_passages(
    documents: Iterable[dict[str, Any]], max_words: int = 160, overlap_words: int = 40
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert normalized candidate documents into provenance-preserving passages.

    Input documents must have a non-empty `text` and should have a stable `url`
    or `document_id`. This function is intentionally format-agnostic; the
    AVeriTeC archive inspector identifies how to create these documents.
    """
    passages: list[dict[str, Any]] = []
    rejected = 0
    for document_index, document in enumerate(documents):
        text = str(document.get("text") or "").strip()
        if not text:
            rejected += 1
            continue
        document_id = str(document.get("document_id") or document.get("url") or f"document:{document_index}")
        url = document.get("url")
        for passage_index, passage_text in enumerate(chunk_text(text, max_words, overlap_words)):
            passages.append(
                {
                    "passage_id": f"{document_id}:p{passage_index:04d}",
                    "text": passage_text,
                    "url": url,
                    "metadata": {
                        "document_id": document_id,
                        "chunk_index": passage_index,
                        "chunk_max_words": max_words,
                        "chunk_overlap_words": overlap_words,
                    },
                }
            )
    return passages, {"passage_count": len(passages), "rejected_empty_document_count": rejected}


def build_url_qrels(
    claims: Iterable[dict[str, Any]], passages: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build *source-URL* relevance judgments from AVeriTeC annotations.

    This is an auditable weak passage-level proxy: every passage from a human
    annotated evidence URL is marked relevant. Reports must call it URL-based
    evidence coverage, not exact human passage relevance.
    """
    # Claims are compact; materialize them once.  The large corpus itself is
    # streamed and only passage IDs for annotated URLs are retained.
    claim_rows = list(claims)
    annotated_urls = {
        canonical_url(url)
        for claim in claim_rows
        for url in (claim.get("source_urls") or [])
    }
    annotated_urls.discard(None)
    passage_ids_by_url: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for passage in passages:
        url = canonical_url(passage.get("url"))
        if url in annotated_urls:
            candidate_claim_id = passage.get("metadata", {}).get("candidate_claim_id")
            passage_ids_by_url[url].append((str(passage["passage_id"]), str(candidate_claim_id) if candidate_claim_id else None))

    qrels: list[dict[str, Any]] = []
    unresolved_url_count = 0
    for claim in claim_rows:
        gold_urls = {canonical_url(url) for url in claim.get("source_urls") or []}
        gold_urls.discard(None)
        relevant_ids = sorted(
            {
                passage_id
                for url in gold_urls
                for passage_id, candidate_claim_id in passage_ids_by_url.get(str(url), [])
                if candidate_claim_id is None or candidate_claim_id == str(claim["claim_id"])
            }
        )
        unresolved = sorted(
            url
            for url in gold_urls
            if not any(
                candidate_claim_id is None or candidate_claim_id == str(claim["claim_id"])
                for _, candidate_claim_id in passage_ids_by_url.get(str(url), [])
            )
        )
        unresolved_url_count += len(unresolved)
        qrels.append(
            {
                "claim_id": claim["claim_id"],
                "relevant_passage_ids": relevant_ids,
                "judgment_type": "annotated_source_url_proxy",
                "annotated_source_url_count": len(gold_urls),
                "unresolved_source_urls": unresolved,
            }
        )
    audit = {
        "judgment_type": "annotated_source_url_proxy",
        "claim_count": len(qrels),
        "judged_claim_count": sum(bool(row["relevant_passage_ids"]) for row in qrels),
        "unresolved_annotated_source_url_count": unresolved_url_count,
        "relevant_passage_count_distribution": dict(
            sorted(Counter(len(row["relevant_passage_ids"]) for row in qrels).items())
        ),
    }
    return qrels, audit
