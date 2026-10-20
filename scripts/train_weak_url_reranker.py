"""Train a small reranker from AVeriTeC train source-URL proxy labels.

The model uses BM25 rank/score and claim-passage lexical features, never URLs
or gold labels as prediction-time features. Model selection is on a claim-level
train-only holdout. Dev qrels are read only after dev rankings are frozen.

This script intentionally does NOT produce in-sample train rankings for a
downstream verifier: that would leak train relevance labels into train evidence.
Out-of-fold train rankings would be needed for a matched verifier experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bm25 import load_qrels  # noqa: E402
from src.data.corpus import canonical_url  # noqa: E402
from src.retrieval.metrics import evaluate_rankings  # noqa: E402
from src.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.runs import resolve_run_directory, sha256_file, write_run_manifest  # noqa: E402


WORDS = re.compile(r"[a-z0-9]+")
FEATURE_NAMES = (
    "bm25_normalized", "reciprocal_rank", "claim_token_coverage",
    "token_jaccard", "claim_bigram_coverage", "log_passage_words", "claim_phrase_match",
)


def words(text: str) -> list[str]:
    return WORDS.findall(text.casefold())


def features_for_row(row: dict[str, Any], max_candidates: int) -> np.ndarray:
    candidates = (row.get("retrieved") or [])[:max_candidates]
    if not candidates:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    raw_scores = [float(item.get("score") or 0.0) for item in candidates]
    if not all(math.isfinite(score) for score in raw_scores):
        raise ValueError(f"Non-finite first-stage score for {row['claim_id']}")
    low, high = min(raw_scores), max(raw_scores)
    query_tokens = words(str(row["claim"]))
    query_set = set(query_tokens)
    query_bigrams = set(zip(query_tokens, query_tokens[1:]))
    query_phrase = " ".join(query_tokens)
    vectors = []
    for index, (item, score) in enumerate(zip(candidates, raw_scores, strict=True), start=1):
        passage_tokens = words(str(item.get("text") or ""))
        passage_set = set(passage_tokens)
        passage_bigrams = set(zip(passage_tokens, passage_tokens[1:]))
        overlap = len(query_set & passage_set)
        vectors.append([
            (score - low) / (high - low) if high > low else 0.0,
            1.0 / index,
            overlap / len(query_set) if query_set else 0.0,
            overlap / len(query_set | passage_set) if query_set | passage_set else 0.0,
            len(query_bigrams & passage_bigrams) / len(query_bigrams) if query_bigrams else 0.0,
            math.log1p(len(passage_tokens)),
            float(bool(query_phrase and query_phrase in " ".join(passage_tokens))),
        ])
    return np.asarray(vectors, dtype=np.float64)


def positive_passage_ids(claim: dict[str, Any], ranking: dict[str, Any], max_candidates: int) -> set[str]:
    gold = {canonical_url(url) for url in (claim.get("source_urls") or [])}
    gold.discard(None)
    return {
        str(item["passage_id"])
        for item in (ranking.get("retrieved") or [])[:max_candidates]
        if canonical_url(item.get("url")) in gold
    }


def rerank_row(row: dict[str, Any], model: Any, blend: float, max_candidates: int) -> dict[str, Any]:
    candidates = (row.get("retrieved") or [])[:max_candidates]
    ids = [str(item["passage_id"]) for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate passage IDs for {row['claim_id']}")
    vectors = features_for_row(row, max_candidates)
    logits = model.decision_function(vectors) if len(candidates) and blend else np.zeros(len(candidates))
    if len(candidates) and blend:
        logits = np.asarray(logits).reshape(-1)
    if len(logits) != len(candidates):
        raise ValueError("Model score count differs from candidate count")
    result = []
    for item, vector, logit in zip(candidates, vectors, logits, strict=True):
        if not math.isfinite(float(logit)):
            raise ValueError("Non-finite reranker score")
        score = blend * float(logit) + (1 - blend) * float(vector[0])
        result.append({**item, "first_stage_score": item.get("score"), "score": score, "weak_ranker_logit": float(logit)})
    result.sort(key=lambda item: (-float(item["score"]), str(item["passage_id"])))
    return {"claim_id": str(row["claim_id"]), "claim": str(row["claim"]), "retrieved": result}


def hit_at_three(rows: list[dict[str, Any]], positives: dict[str, set[str]]) -> float:
    if not rows:
        raise ValueError("No validation rows")
    return sum(bool({str(item["passage_id"]) for item in row["retrieved"][:3]} & positives[row["claim_id"]]) for row in rows) / len(rows)


def fit_model(rows: list[dict[str, Any]], claims: dict[str, dict[str, Any]], max_candidates: int, c: float, seed: int) -> Any:
    matrices = []
    labels = []
    for row in rows:
        positives = positive_passage_ids(claims[str(row["claim_id"])], row, max_candidates)
        matrix = features_for_row(row, max_candidates)
        if len(matrix):
            matrices.append(matrix)
            labels.extend(int(str(item["passage_id"]) in positives) for item in row["retrieved"][:max_candidates])
    if not matrices or set(labels) != {0, 1}:
        raise ValueError("Weak URL-proxy training needs both positive and negative candidate pairs")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=seed))
    model.fit(np.concatenate(matrices), np.asarray(labels))
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-claims", required=True, type=Path)
    parser.add_argument("--train-rankings", required=True, type=Path)
    parser.add_argument("--dev-rankings", required=True, type=Path)
    parser.add_argument("--dev-qrels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=718)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if args.max_candidates < 3 or not 0.05 <= args.validation_fraction <= 0.40:
        parser.error("Require at least three candidates and validation fraction in [0.05, 0.40]")
    inputs = (args.train_claims, args.train_rankings, args.dev_rankings, args.dev_qrels)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = resolve_run_directory(args.output_dir, args.run_name)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "script": "train_weak_url_reranker",
        "input_sha256": {name: sha256_file(path) for name, path in zip(
            ("train_claims", "train_rankings", "dev_rankings", "dev_qrels"), inputs, strict=True)},
        "max_candidates": args.max_candidates,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "feature_names": list(FEATURE_NAMES),
        "c_grid": [0.1, 1.0, 10.0],
        "blend_grid": [0.25, 0.5, 0.75, 1.0],
        "selection_scope": "claim-level train-only holdout; URL-proxy hit@3 among claims with annotated source URLs",
        "no_train_ranking_output": "A downstream classifier would require out-of-fold train rerankings to avoid in-sample evidence leakage.",
    }
    config_path = output / "weak_reranker_config.json"
    old_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
    if old_config is not None and old_config != config:
        raise ValueError("Inputs/settings changed; use a new run name")
    final_paths = (output / "model.joblib", output / "model_selection.json", output / "dev_rankings.jsonl", output / "dev_metrics.json", output / "weak_reranker_manifest.json")
    if all(path.is_file() for path in final_paths):
        print((output / "dev_metrics.json").read_text(encoding="utf-8"))
        return
    write_json(config, config_path)

    train_claim_rows = read_jsonl(args.train_claims)
    claims = {str(row["claim_id"]): row for row in train_claim_rows}
    train_rows = read_jsonl(args.train_rankings)
    dev_rows = read_jsonl(args.dev_rankings)
    if len(claims) != len(train_claim_rows) or {str(row["claim_id"]) for row in train_rows} != set(claims) or len(train_rows) != len(claims):
        raise ValueError("Train claims and BM25 rankings must have identical unique claim IDs")
    if len({str(row["claim_id"]) for row in dev_rows}) != len(dev_rows):
        raise ValueError("Duplicate dev claim IDs")
    eligible = [row for row in train_rows if any(canonical_url(url) for url in (claims[str(row["claim_id"])].get("source_urls") or []))]
    if len(eligible) < 20:
        raise ValueError("Too few train claims have annotated source URLs for this experiment")
    positive = {str(row["claim_id"]): positive_passage_ids(claims[str(row["claim_id"])], row, args.max_candidates) for row in eligible}
    coverage = [bool(positive[str(row["claim_id"])]) for row in eligible]
    stratify = coverage if min(coverage.count(True), coverage.count(False)) >= 2 else None
    fit_rows, val_rows = train_test_split(
        eligible, test_size=args.validation_fraction, random_state=args.seed, stratify=stratify,
    )
    val_positive = {str(row["claim_id"]): positive[str(row["claim_id"])] for row in val_rows}
    first_stage_hit = hit_at_three(val_rows, val_positive)
    top20_ceiling = sum(bool(value) for value in val_positive.values()) / len(val_rows)
    trials = []
    with progress(total=12, description="Weak reranker validation", unit="trial") as bar:
        for c in config["c_grid"]:
            model = fit_model(fit_rows, claims, args.max_candidates, c, args.seed)
            for blend in config["blend_grid"]:
                ranked = [rerank_row(row, model, blend, args.max_candidates) for row in val_rows]
                trials.append({"C": c, "blend": blend, "validation_hit_at_3": hit_at_three(ranked, val_positive)})
                bar.update(1)
    selected = sorted(trials, key=lambda row: (-row["validation_hit_at_3"], -row["blend"], row["C"]))[0]
    selection = {
        "validation_claim_count": len(val_rows),
        "validation_first_stage_hit_at_3": first_stage_hit,
        "validation_top20_ceiling": top20_ceiling,
        "selected": selected,
        "all_trials": trials,
        "warning": "Validation labels are weak URL-proxy judgments; a matched verdict model is not evaluated here.",
    }
    write_json(selection, output / "model_selection.json")
    model = fit_model(eligible, claims, args.max_candidates, float(selected["C"]), args.seed)
    temporary_model = output / "model.joblib.tmp"
    joblib.dump(model, temporary_model)
    temporary_model.replace(output / "model.joblib")
    with progress(total=len(dev_rows), description="Weak reranker dev", unit="claim") as bar:
        dev_ranked = []
        for row in dev_rows:
            dev_ranked.append(rerank_row(row, model, float(selected["blend"]), args.max_candidates))
            bar.update(1)
    temporary_rankings = output / "dev_rankings.jsonl.tmp"
    write_jsonl(dev_ranked, temporary_rankings)
    temporary_rankings.replace(output / "dev_rankings.jsonl")
    qrels = load_qrels(args.dev_qrels)
    report = {
        **config,
        "train_eligible_claim_count": len(eligible),
        "train_claims_with_positive_in_top20": sum(coverage),
        "dev_claim_count": len(dev_ranked),
        "selected": selected,
        "dev": evaluate_rankings({str(row["claim_id"]): [str(item["passage_id"]) for item in row["retrieved"]] for row in dev_ranked}, qrels, k=3),
        "bm25_reference": evaluate_rankings({str(row["claim_id"]): [str(item["passage_id"]) for item in row.get("retrieved", [])[:args.max_candidates]] for row in dev_rows}, qrels, k=3),
        "rankings_path": str(output / "dev_rankings.jsonl"),
        "warning": "This is weakly supervised URL-proxy retrieval only. No matched verdict model was trained; do not use in-sample train rerankings for one.",
    }
    write_json(report, output / "dev_metrics.json")
    write_run_manifest(
        output / "weak_reranker_manifest.json", project_root=PROJECT_ROOT, config=report,
        input_paths={"train_claims": args.train_claims, "train_rankings": args.train_rankings,
                     "dev_rankings": args.dev_rankings, "dev_qrels": args.dev_qrels},
    )
    print(json.dumps({"selection": selection, "dev": report["dev"], "bm25_reference": report["bm25_reference"]}, indent=2))


if __name__ == "__main__":
    main()
