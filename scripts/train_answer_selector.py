"""Train a small answer-aware selector from train-only silver passage labels.

Train rankings are five-fold out-of-fold: a claim's own gold answer never fits
the selector that ranks its passages. Official-dev qrels are evaluation-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bm25 import load_qrels  # noqa: E402
from src.retrieval.answer_aware import FEATURE_KEYS, feature_vector, rank_with_source_diversity  # noqa: E402
from src.retrieval.metrics import evaluate_rankings  # noqa: E402
from src.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.runs import resolve_run_directory, sha256_file, write_run_manifest  # noqa: E402


def fit_model(groups: list[tuple[dict[str, Any], str]], c: float):
    features: list[list[float]] = []
    labels: list[int] = []
    for row, positive_id in groups:
        for item in row["retrieved"]:
            features.append(feature_vector(item))
            labels.append(int(str(item["passage_id"]) == positive_id))
    if len(set(labels)) < 2:
        raise ValueError("Selector training requires both silver positives and negatives")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=718))
    model.fit(np.asarray(features, dtype=float), np.asarray(labels, dtype=int))
    return model


def ranked_items(row: dict[str, Any], model: Any, *, ranking_k: int) -> list[dict[str, Any]]:
    items = row["retrieved"]
    if not items:
        return []
    scores = model.decision_function(np.asarray([feature_vector(item) for item in items], dtype=float))
    return rank_with_source_diversity(items, [float(score) for score in np.atleast_1d(scores)])[:ranking_k]


def silver_hit_at_3(groups: list[tuple[dict[str, Any], str]], model: Any) -> float:
    if not groups:
        raise ValueError("Cannot validate on zero claims")
    hits = 0
    for row, positive_id in groups:
        hits += positive_id in {str(item["passage_id"]) for item in ranked_items(row, model, ranking_k=3)}
    return hits / len(groups)


def select_c(groups: list[tuple[dict[str, Any], str]], *, seed: int) -> tuple[float, dict[str, float]]:
    train, validation = train_test_split(groups, test_size=0.2, random_state=seed)
    scores = {}
    for c in (0.1, 1.0, 10.0):
        scores[str(c)] = silver_hit_at_3(validation, fit_model(train, c))
    chosen = min((0.1, 1.0, 10.0), key=lambda c: (-scores[str(c)], c))
    return chosen, scores


def make_rankings(rows: list[dict[str, Any]], model: Any, *, ranking_k: int, qa_variant: bool) -> list[dict[str, Any]]:
    output = []
    with progress(total=len(rows), description="Rank answer-aware evidence", unit="claim") as bar:
        for row in rows:
            selected = ranked_items(row, model, ranking_k=ranking_k)
            retrieved = []
            for item in selected:
                copy = dict(item)
                if qa_variant:
                    copy["text"] = str(copy.pop("qa_text"))
                else:
                    copy.pop("qa_text", None)
                retrieved.append(copy)
            output.append({"claim_id": row["claim_id"], "claim": row["claim"], "retrieved": retrieved})
            bar.update(1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", required=True, type=Path)
    parser.add_argument("--dev-features", required=True, type=Path)
    parser.add_argument("--train-alignment", required=True, type=Path)
    parser.add_argument("--dev-qrels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--ranking-k", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-positive-claims", type=int, default=30)
    parser.add_argument("--seed", type=int, default=718)
    args = parser.parse_args()
    if min(args.ranking_k, args.min_positive_claims) < 1 or args.folds < 2:
        parser.error("ranking-k and min-positive-claims must be positive; folds >= 2")
    for path in (args.train_features, args.dev_features, args.train_alignment, args.dev_qrels):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = resolve_run_directory(args.output_dir, args.run_name)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "script": "train_answer_selector", "train_features_sha256": sha256_file(args.train_features),
        "dev_features_sha256": sha256_file(args.dev_features),
        "train_alignment_sha256": sha256_file(args.train_alignment),
        "dev_qrels_sha256": sha256_file(args.dev_qrels),
        "feature_keys": list(FEATURE_KEYS), "ranking_k": args.ranking_k,
        "folds": args.folds, "min_positive_claims": args.min_positive_claims,
        "seed": args.seed, "c_grid": [0.1, 1.0, 10.0],
        "selection_metric": "train-only silver positive passage hit@3",
        "warning": "Silver answer-aligned passage IDs are noisy and incomplete; URL-proxy dev qrels are evaluation-only.",
    }
    config_path = output / "selector_config.json"
    if config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Inputs/settings changed; choose a new run name")
    raw_train_path = output / "train_raw_rankings.jsonl"
    qa_train_path = output / "train_qa_rankings.jsonl"
    raw_dev_path = output / "dev_raw_rankings.jsonl"
    qa_dev_path = output / "dev_qa_rankings.jsonl"
    report_path = output / "selector_report.json"
    manifest_path = output / "manifest.json"
    model_path = output / "final_selector.joblib"
    finished = (raw_train_path, qa_train_path, raw_dev_path, qa_dev_path, report_path, manifest_path, model_path)
    if all(path.is_file() for path in finished):
        print(report_path.read_text(encoding="utf-8"))
        return
    write_json(config, config_path)
    train_rows = read_jsonl(args.train_features)
    dev_rows = read_jsonl(args.dev_features)
    aligned_rows = read_jsonl(args.train_alignment)
    train_by_id = {str(row["claim_id"]): row for row in train_rows}
    dev_by_id = {str(row["claim_id"]): row for row in dev_rows}
    alignment_by_id = {str(row["claim_id"]): row for row in aligned_rows}
    if len(train_by_id) != len(train_rows) or len(dev_by_id) != len(dev_rows) or len(alignment_by_id) != len(aligned_rows):
        raise ValueError("Duplicate claim ID in features or alignment")
    if set(train_by_id) != set(alignment_by_id) or set(train_by_id) & set(dev_by_id):
        raise ValueError("Train alignment mismatch or overlapping train/dev claim IDs")
    accepted: dict[str, tuple[dict[str, Any], str]] = {}
    for claim_id, row in train_by_id.items():
        positive = alignment_by_id[claim_id].get("positive_passage_id")
        if positive and str(positive) in {str(item["passage_id"]) for item in row["retrieved"]}:
            accepted[claim_id] = (row, str(positive))
    if len(accepted) < args.min_positive_claims:
        raise ValueError(
            f"Only {len(accepted)} silver positives are within scored candidates; "
            "increase candidate-k in the reader or lower min-positive-claims deliberately"
        )
    if len(accepted) < args.folds * 2:
        raise ValueError("Too few accepted silver claims for out-of-fold selection")
    claim_ids = sorted(train_by_id)
    oof: dict[str, list[dict[str, Any]]] = {}
    fold_reports = []
    for fold_number, (fit_indices, predict_indices) in enumerate(KFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(claim_ids), start=1):
        fit_ids = {claim_ids[int(index)] for index in fit_indices}
        predict_ids = [claim_ids[int(index)] for index in predict_indices]
        fit_groups = [accepted[claim_id] for claim_id in sorted(fit_ids & set(accepted))]
        c, validation_scores = select_c(fit_groups, seed=args.seed + fold_number)
        model = fit_model(fit_groups, c)
        for claim_id in predict_ids:
            oof[claim_id] = ranked_items(train_by_id[claim_id], model, ranking_k=args.ranking_k)
        fold_reports.append({"fold": fold_number, "fit_positive_claims": len(fit_groups),
                             "predicted_claims": len(predict_ids), "selected_c": c,
                             "internal_validation_silver_hit_at_3": validation_scores})
        print(f"Finished OOF fold {fold_number}/{args.folds}: C={c}; {len(predict_ids)} claims", flush=True)
    if set(oof) != set(train_by_id):
        raise RuntimeError("Out-of-fold rankings did not cover all train claims")
    final_c, final_validation = select_c(list(accepted.values()), seed=args.seed)
    final_model = fit_model(list(accepted.values()), final_c)
    joblib.dump(final_model, model_path)
    train_scored = [{**row, "retrieved": oof[str(row["claim_id"])]} for row in train_rows]
    # OOF items have already been scored; do not run the all-train model on them.
    def serialize(rows: list[dict[str, Any]], *, qa_variant: bool) -> list[dict[str, Any]]:
        result = []
        for row in rows:
            passages = []
            for item in row["retrieved"]:
                copy = dict(item)
                if qa_variant:
                    copy["text"] = str(copy.pop("qa_text"))
                else:
                    copy.pop("qa_text", None)
                passages.append(copy)
            result.append({"claim_id": row["claim_id"], "claim": row["claim"], "retrieved": passages})
        return result
    raw_train = serialize(train_scored, qa_variant=False)
    qa_train = serialize(train_scored, qa_variant=True)
    raw_dev = make_rankings(dev_rows, final_model, ranking_k=args.ranking_k, qa_variant=False)
    qa_dev = make_rankings(dev_rows, final_model, ranking_k=args.ranking_k, qa_variant=True)
    for path, rows in ((raw_train_path, raw_train), (qa_train_path, qa_train),
                       (raw_dev_path, raw_dev), (qa_dev_path, qa_dev)):
        temporary = path.with_name(path.name + ".tmp")
        write_jsonl(rows, temporary)
        temporary.replace(path)
    qrels = load_qrels(args.dev_qrels)
    dev_metric = evaluate_rankings(
        {str(row["claim_id"]): [str(item["passage_id"]) for item in row["retrieved"]] for row in raw_dev},
        qrels, k=3,
    )
    report = {
        **config, "train_claim_count": len(train_rows), "dev_claim_count": len(dev_rows),
        "aligned_positive_claim_count": sum(bool(row.get("positive_passage_id")) for row in aligned_rows),
        "positive_in_scored_candidates_count": len(accepted), "oof_fold_reports": fold_reports,
        "final_selected_c": final_c, "final_train_validation_silver_hit_at_3": final_validation,
        "dev_url_proxy_metrics": dev_metric,
        "mrr_scope": f"saved_top_{args.ranking_k}", "retrieval_judgments": "source-URL proxy, not exact passage relevance",
        "raw_train_path": str(raw_train_path), "qa_train_path": str(qa_train_path),
        "raw_dev_path": str(raw_dev_path), "qa_dev_path": str(qa_dev_path),
        "model_path": str(model_path),
    }
    write_json(report, report_path)
    write_run_manifest(manifest_path, project_root=PROJECT_ROOT, config=report,
                       input_paths={"train_features": args.train_features, "dev_features": args.dev_features,
                                    "train_alignment": args.train_alignment, "dev_qrels": args.dev_qrels})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
