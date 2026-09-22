"""Shared, serializable records used throughout the project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceItem:
    """A retrievable evidence passage with its source provenance."""

    passage_id: str
    text: str
    url: str | None = None
    domain: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimRecord:
    """Canonical claim record emitted by every dataset loader."""

    claim_id: str
    claim: str
    label: str | None
    source_urls: tuple[str, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_urls"] = list(self.source_urls)
        return data
