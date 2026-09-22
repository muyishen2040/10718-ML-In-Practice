"""Deterministic BM25 retrieval over a static passage corpus."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from src.data.schema import EvidenceItem

_TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """A transparent, deterministic tokenizer suitable for a lexical baseline."""
    return _TOKEN_PATTERN.findall(text.casefold())


class BM25Retriever:
    """BM25 retriever with stable tie-breaking by passage ID."""

    def __init__(self, corpus: Sequence[EvidenceItem]) -> None:
        if not corpus:
            raise ValueError("BM25 corpus cannot be empty")
        self.corpus = tuple(corpus)
        self._bm25 = BM25Okapi([tokenize(item.text) for item in self.corpus])

    def retrieve(self, claim: str, k: int = 3) -> list[EvidenceItem]:
        if k < 1:
            raise ValueError("k must be at least 1")
        scores = self._bm25.get_scores(tokenize(claim))
        ranked_indices = sorted(
            range(len(self.corpus)),
            key=lambda index: (-float(scores[index]), self.corpus[index].passage_id),
        )[:k]
        return [
            EvidenceItem(
                passage_id=self.corpus[index].passage_id,
                text=self.corpus[index].text,
                url=self.corpus[index].url,
                domain=self.corpus[index].domain,
                score=float(scores[index]),
                metadata=self.corpus[index].metadata,
            )
            for index in ranked_indices
        ]
