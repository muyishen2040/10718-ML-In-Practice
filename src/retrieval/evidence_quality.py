"""Prediction-time evidence hygiene and stance-aware selection.

Never consumes verdicts, annotated answers, source_urls, qrels, or the original
fact-check URL. Unknown publication dates stay unknown. Future years in prose
are flagged, not treated as proof of publication after the claim.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

import numpy as np

from src.data.corpus import canonical_url
from src.evaluation.splits import parse_claim_date
from src.retrieval.question_guided import words

QUALITY_NAMES = ("query_coverage", "number_overlap", "rank_prior", "satire_source", "factcheck_source", "publication_known", "future_year_mentioned", "search_results_source")
SATIRE_HOSTS = {"babylonbee.com", "theonion.com"}


@dataclass(frozen=True)
class EvidencePolicy:
    max_candidates: int = 24
    candidate_depth: int = 100
    near_duplicate_threshold: float = 0.90
    remove_search_pages: bool = True
    remove_known_future_publications: bool = True

    def to_dict(self):
        return asdict(self)


def original_url(url: str | None) -> str:
    value = str(url or "")
    # Capture time is deliberately not used as publication time.
    match = re.match(r"https?://web\.archive\.org/web/\d+(?:[a-z_]+)?/(https?://.*)", value, re.I)
    return match.group(1) if match else value


def source_host(url: str | None) -> str:
    return (urlparse(original_url(url)).hostname or "").lower().removeprefix("www.")


def is_search_page(url: str | None) -> bool:
    parsed = urlparse(original_url(url))
    host = (parsed.hostname or "").lower().removeprefix("www.")
    return bool((re.fullmatch(r"google\.[a-z.]+", host) and parsed.path.startswith(("/search", "/url")))
                or (host in {"bing.com", "search.yahoo.com", "duckduckgo.com"} and (parsed.query or "/search" in parsed.path)))


def publication_date(item: dict) -> str | None:
    metadata = item.get("metadata") or {}
    for key in ("publication_date", "published_date", "published_at", "datePublished"):
        date = parse_claim_date(metadata.get(key) or item.get(key))
        if date:
            return date
    # URL paths and Wayback captures do not establish publication dates.
    return None


def normalized_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def shingles(text: str) -> set[tuple[str, ...]]:
    tokens = normalized_text(text).split()
    return {tuple(tokens[i:i + 5]) for i in range(max(1, len(tokens) - 4))}


def quality_features(claim: str, item: dict, rank: int, claim_date: str | None) -> list[float]:
    text = str(item.get("text") or "")
    query, terms = words(claim), words(text)
    numbers = set(re.findall(r"\b\d+(?:[.,]\d+)*\b", claim))
    passage_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)*\b", text))
    host, url = source_host(item.get("url")), original_url(item.get("url")).lower()
    year = int(claim_date[:4]) if claim_date else None
    later = bool(year and any(int(y) > year for y in re.findall(r"\b(20\d{2})\b", text)))
    return [len(query & terms) / max(1, len(query)), len(numbers & passage_numbers) / max(1, len(numbers)),
            1.0 / math.log2(rank + 2), float(host in SATIRE_HOSTS),
            float(bool(re.search(r"fact[-_]?check", url))), float(publication_date(item) is not None), float(later), float(is_search_page(item.get("url")))]


def pair_premise(item: dict) -> str:
    host = source_host(item.get("url")) or "unknown"
    kind = "satire source" if host in SATIRE_HOSTS else "search results" if is_search_page(item.get("url")) else "unspecified"
    return f"Source domain: {host}. Source type: {kind}.\nPassage: {item.get('text') or ''}"


def clean_candidates(claim: str, claim_date: str | None, candidates: list[dict], policy: EvidencePolicy) -> tuple[list[dict], list[dict]]:
    date = parse_claim_date(claim_date)
    selected, rejected, signatures, seen_ids = [], [], [], set()
    if policy.max_candidates < 1 or policy.candidate_depth < 1 or not 0 < policy.near_duplicate_threshold <= 1:
        raise ValueError("Invalid evidence policy")
    for rank, source in enumerate(candidates[:policy.candidate_depth]):
        pid = str(source.get("passage_id") or "")
        if not pid or pid in seen_ids:
            raise ValueError(f"Missing/duplicate candidate passage ID: {pid!r}")
        seen_ids.add(pid)
        text = str(source.get("text") or "").strip()
        reason = "blank" if not text else None
        if policy.remove_search_pages and is_search_page(source.get("url")):
            reason = "search_results_page"
        published = publication_date(source)
        if policy.remove_known_future_publications and date and published and published > date:
            reason = "published_after_claim"
        signature = shingles(text)
        if reason is None:
            for old_pid, old_signature in signatures:
                ratio = len(signature & old_signature) / max(1, len(signature | old_signature))
                if ratio >= policy.near_duplicate_threshold:
                    reason = f"duplicate_of:{old_pid}"
                    break
        if reason:
            rejected.append({"passage_id": pid, "reason": reason, "original_rank": rank + 1})
            continue
        signatures.append((pid, signature))
        # Whitelist only fields that exist before a verdict is annotated.
        item = {"passage_id": pid, "text": text, "url": source.get("url"), "original_rank": rank + 1,
                "publication_date": published, "quality": quality_features(claim, source, rank + 1, date)}
        selected.append(item)
        if len(selected) == policy.max_candidates:
            break
    return selected, rejected


def nli_label_order(id2label: dict, explicit: str | None = None) -> list[str]:
    names = explicit.split(",") if explicit else [str(id2label.get(i, id2label.get(str(i), ""))).lower() for i in range(3)]
    names = [n.strip().lower() for n in names]
    if len(names) != 3 or set(names) != {"entailment", "neutral", "contradiction"}:
        raise ValueError("Cannot identify NLI labels. Supply --nli-label-order in the model's documented logit order.")
    return names


def stance_rank(item: dict) -> float:
    p = item["nli_probabilities"]
    q = item["quality"]
    # Both support and refutation are valuable; neither is a claim verdict.
    return 0.60 * (1 - p["neutral"]) + 0.25 * q[0] + 0.15 * q[2] - 0.20 * q[3]


def select_evidence(candidates: list[dict], *, rerank: bool) -> list[dict]:
    order = sorted(candidates, key=lambda p: (-stance_rank(p), p["original_rank"], p["passage_id"])) if rerank else list(candidates)
    chosen, urls = [], set()
    for p in order:
        source = canonical_url(original_url(p.get("url"))) or p["passage_id"]
        if rerank and source in urls:
            continue
        chosen.append(p)
        urls.add(source)
        if len(chosen) == 3:
            return chosen
    # Low source count: use additional distinct passages, never deleted garbage.
    used = {p["passage_id"] for p in chosen}
    return (chosen + [p for p in order if p["passage_id"] not in used])[:3]


def aggregate_features(row: dict) -> np.ndarray:
    """Cheap learned-aggregation baseline using the same selected three passages."""
    evidence = row.get("retrieved") or []
    p = np.asarray([[item["nli_probabilities"][k] for k in ("entailment", "contradiction", "neutral")] for item in evidence[:3]], dtype=float)
    q = np.asarray([item["quality"] for item in evidence[:3]], dtype=float)
    if not len(p):
        return np.zeros(3 * 2 + len(QUALITY_NAMES) * 2 + 2, dtype=float)
    return np.concatenate([p.mean(0), p.max(0), q.mean(0), q.max(0), [p[:, 0].max() * p[:, 1].max(), len(p) / 3]])
