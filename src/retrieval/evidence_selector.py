"""Small learned evidence selector over saved BM25 candidates (brief, step 5).

Trains on reviewed passage-level decisive labels from training-fit claims and
re-orders each claim's saved top-``depth`` BM25 pool. The output keeps the
rankings schema with three *full* passages so the same frozen verifier can be
fed the new passages, isolating the selection change from verifier training.

Features are prediction-time only: BM25 rank/score, lexical/number overlap,
source-type and date flags, and cached NLI scores when they exist for exactly
that passage text. Claim verdicts, gold answers, annotated source URLs and
qrels are never features.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from src.data.corpus import canonical_url
from src.evaluation.splits import parse_claim_date
from src.retrieval.evidence_quality import QUALITY_NAMES, normalized_text, original_url, publication_date, quality_features

FEATURE_NAMES = ("rank_prior", "log_rank", "bm25_score_ratio", "bm25_score_missing", *QUALITY_NAMES,
                 "claim_bigram_overlap", "log_passage_words", "nli_entailment", "nli_contradiction", "nli_neutral",
                 "nli_missing", "nli_decisiveness")


def cached_nli(item: dict, cache: dict[str, dict] | None) -> dict | None:
    """NLI scores for this passage, only if cached for exactly the same text."""
    if item.get("nli_probabilities"):
        return item["nli_probabilities"]
    entry = (cache or {}).get(item["passage_id"])
    if entry and entry["text"] == str(item.get("text") or "").strip():
        return entry["nli_probabilities"]
    return None


def nli_cache_from_prepared(rows) -> dict[str, dict]:
    """Build a text-checked cache from prepared ``{split}.partial.jsonl`` journal rows."""
    cache = {}
    for row in rows:
        for item in row.get("scored_candidates") or []:
            if item.get("nli_probabilities"):
                cache[item["passage_id"]] = {"text": str(item.get("text") or "").strip(),
                                             "nli_probabilities": item["nli_probabilities"]}
    return cache


def _bigrams(text: str) -> set[tuple[str, str]]:
    tokens = normalized_text(text).split()
    return set(zip(tokens, tokens[1:]))


def candidate_features(claim: str, claim_date: str | None, pool: list[dict], *, depth: int,
                       nli_cache: dict[str, dict] | None = None) -> np.ndarray:
    """Feature matrix for the first ``depth`` candidates of one claim."""
    items = pool[:depth]
    scores = [p.get("score") for p in items]
    finite = [float(s) for s in scores if isinstance(s, (int, float)) and math.isfinite(s)]
    top = max(finite, default=0.0)
    claim_bigrams = _bigrams(claim)
    rows = []
    for rank, item in enumerate(items, start=1):
        text = str(item.get("text") or "")
        score = item.get("score")
        has_score = isinstance(score, (int, float)) and math.isfinite(score) and top > 0
        nli = cached_nli(item, nli_cache)
        probs = [nli[k] for k in ("entailment", "contradiction", "neutral")] if nli else [0.0, 0.0, 0.0]
        rows.append([1 / math.log2(rank + 2), math.log(rank), float(score) / top if has_score else 0.0, float(not has_score),
                     *quality_features(claim, item, rank, claim_date),
                     len(claim_bigrams & _bigrams(text)) / max(1, len(claim_bigrams)), math.log1p(len(text.split())),
                     *probs, float(nli is None), (1 - probs[2]) if nli else 0.0])
    return np.asarray(rows, dtype=float).reshape(len(items), len(FEATURE_NAMES))


def _claim_date(record: dict) -> str | None:
    return parse_claim_date((record.get("metadata") or {}).get("claim_date"))


def training_matrix(targets: list[dict], records: dict[str, dict], rankings: dict[str, dict], *, depth: int,
                    nli_cache: dict[str, dict] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X, y, claim groups and original ranks for labeled candidates."""
    by_claim = defaultdict(dict)
    for t in targets:
        by_claim[t["claim_id"]][t["passage_id"]] = int(t["target"])
    x, y, groups, ranks = [], [], [], []
    for cid in sorted(by_claim):
        pool = (rankings[cid].get("retrieved") or [])[:depth]
        features = candidate_features(records[cid]["claim"], _claim_date(records[cid]), pool, depth=depth, nli_cache=nli_cache)
        positions = {p["passage_id"]: i for i, p in enumerate(pool)}
        for pid, target in by_claim[cid].items():
            if pid not in positions:
                continue  # labeled beyond the selector depth
            x.append(features[positions[pid]])
            y.append(target)
            groups.append(cid)
            ranks.append(positions[pid] + 1)
    if not x:
        raise ValueError("No labeled candidates fall within the selector depth")
    return np.asarray(x), np.asarray(y), np.asarray(groups), np.asarray(ranks)


def ranking_metrics(scores: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int = 3) -> dict[str, float | int]:
    """Hit@k and MRR among *labeled* candidates of claims with a decisive passage."""
    hits, reciprocal = [], []
    for cid in sorted(set(groups)):
        mask = groups == cid
        if not y[mask].any():
            continue
        order = np.argsort(-scores[mask], kind="stable")
        labels = y[mask][order]
        first = int(np.argmax(labels))
        hits.append(float(labels[:k].any()))
        reciprocal.append(1 / (first + 1))
    return {"claims_with_decisive_passage": len(hits), f"hit_at_{k}": float(np.mean(hits)) if hits else float("nan"),
            "mrr_over_labeled_candidates": float(np.mean(reciprocal)) if reciprocal else float("nan")}


def fit_selector(x: np.ndarray, y: np.ndarray, groups: np.ndarray, ranks: np.ndarray, *, c_grid=(0.1, 1.0, 10.0),
                 folds: int = 5, seed: int = 718) -> tuple[Any, dict[str, Any]]:
    """Choose C by claim-grouped CV on fit claims only, then refit on all of them."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(set(y.tolist())) < 2:
        raise ValueError("Selector training needs both decisive and non-decisive passages")
    n_folds = min(folds, len(set(groups.tolist())))
    if n_folds < 2:
        raise ValueError("Need at least two labeled claims for grouped cross-validation")

    def model(c):
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed))

    trials = []
    for c in c_grid:
        out_of_fold = np.zeros(len(y))
        for train, test in GroupKFold(n_splits=n_folds).split(x, y, groups):
            if len(set(y[train].tolist())) < 2:
                out_of_fold[test] = -ranks[test]  # degenerate fold: fall back to BM25 order
                continue
            out_of_fold[test] = model(c).fit(x[train], y[train]).decision_function(x[test])
        trials.append({"C": c, **ranking_metrics(out_of_fold, y, groups)})
    baseline = {"name": "bm25_order", **ranking_metrics(-ranks.astype(float), y, groups)}
    chosen = sorted(trials, key=lambda t: (-t["hit_at_3"], -t["mrr_over_labeled_candidates"], t["C"]))[0]
    report = {"selected_C": chosen["C"], "cv_trials": trials, "bm25_order_same_candidates": baseline, "cv_folds": n_folds,
              "labeled_candidates": int(len(y)), "decisive_candidates": int(y.sum()), "claims": len(set(groups.tolist())),
              "feature_names": list(FEATURE_NAMES),
              "metric_scope": "Hit@3/MRR over reviewed candidates only (not the full pool); grouped by claim"}
    return model(chosen["C"]).fit(x, y), report


def select_passages(model, record: dict, ranking: dict, *, depth: int, k: int = 3, distinct_sources: bool = True,
                    nli_cache: dict[str, dict] | None = None) -> list[dict]:
    """Top-k full passages for one claim in selector order, with reader inputs attached."""
    pool = (ranking.get("retrieved") or [])[:depth]
    if not pool:
        return []
    date = _claim_date(record)
    scores = model.decision_function(candidate_features(record["claim"], date, pool, depth=depth, nli_cache=nli_cache))
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
    chosen, sources = [], set()
    for i in order:
        source = canonical_url(original_url(pool[i].get("url"))) or pool[i]["passage_id"]
        if distinct_sources and source in sources:
            continue
        chosen.append(i)
        sources.add(source)
        if len(chosen) == k:
            break
    chosen += [i for i in order if i not in chosen][:k - len(chosen)]
    result = []
    for i in chosen:
        item = pool[i]
        nli = cached_nli(item, nli_cache)
        entry = {"passage_id": item["passage_id"], "text": str(item.get("text") or ""), "url": item.get("url"),
                 "original_rank": i + 1, "publication_date": publication_date(item), "selector_score": float(scores[i]),
                 "quality": quality_features(record["claim"], item, i + 1, date)}
        if nli:
            entry["nli_probabilities"] = dict(nli)
        result.append(entry)
    return result
