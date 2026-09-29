"""Run and cache a zero-shot evidence-only LLM baseline on saved rankings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.classification import evaluate_predictions  # noqa: E402
from src.evaluation.end_to_end import annotate_retrieval_coverage, evaluate_by_retrieval_coverage  # noqa: E402
from src.utils.io import read_jsonl, write_jsonl  # noqa: E402
from src.utils.runs import resolve_run_directory, write_run_manifest  # noqa: E402
from src.verification.llm_zero_shot import (  # noqa: E402
    SYSTEM_PROMPT,
    build_prediction,
    build_user_prompt,
    call_openai_structured,
    prompt_sha256,
    validate_llm_verdict,
)


def existing_raw_by_claim(path: Path, *, model: str, variant: str) -> dict[str, dict[str, Any]]:
    """Reuse only calls made with this exact model and baseline variant."""
    if not path.exists():
        return {}
    return {
        str(row["claim_id"]): row
        for row in read_jsonl(path)
        if row.get("status") == "success" and row.get("model") == model and row.get("variant") == variant
    }


def usage_summary(raw_rows: list[dict[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for row in raw_rows:
        usage = row.get("response", {}).get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if isinstance(usage.get(key), int):
                totals[key] += usage[key]
    return dict(totals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--rankings", required=True, type=Path)
    parser.add_argument("--model", required=True, help="Record an exact API model ID or snapshot, never an informal label.")
    parser.add_argument("--variant", choices=("classify_top3", "rerank_top20_and_classify"), default="classify_top3")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "llm")
    parser.add_argument("--run-name", help="Optional name for an isolated, non-overwriting run subdirectory.")
    parser.add_argument("--qrels", type=Path, help="Optional relevance judgments for automatic evidence-coverage diagnostics.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-claims", type=int, default=0, help="0 means all ranked claims; use a small number for a cost-controlled pilot.")
    parser.add_argument("--max-chars-per-passage", type=int, default=1200)
    parser.add_argument("--overwrite", action="store_true", help="Discard cached successful calls for this output directory.")
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is not set. In Colab, set it with getpass; never commit it.")
    try:
        args.output_dir = resolve_run_directory(args.output_dir, args.run_name)
    except ValueError as error:
        parser.error(str(error))
    candidate_limit = 3 if args.variant == "classify_top3" else 20
    claims = {str(row["claim_id"]): row for row in read_jsonl(args.claims)}
    rankings = read_jsonl(args.rankings)
    if args.max_claims:
        rankings = rankings[: args.max_claims]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"{args.variant}_raw.jsonl"
    cached = {} if args.overwrite else existing_raw_by_claim(raw_path, model=args.model, variant=args.variant)
    raw_rows = list(cached.values())
    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for ranking in rankings:
        claim_id = str(ranking["claim_id"])
        claim = claims.get(claim_id)
        if claim is None:
            errors.append({"claim_id": claim_id, "status": "error", "error": "claim_missing_from_claims_file"})
            continue
        passages = list(ranking.get("retrieved") or [])[:candidate_limit]
        if len(passages) != candidate_limit:
            errors.append(
                {
                    "claim_id": claim_id,
                    "status": "error",
                    "error": f"ranking_has_{len(passages)}_passages_but_{args.variant}_requires_{candidate_limit}",
                }
            )
            continue
        if claim_id in cached:
            parsed = cached[claim_id]["parsed"]
            prediction = parsed["prediction"]
            prediction.setdefault(
                "evidence_passage_ids",
                prediction.get("selected_passage_ids") if args.variant == "rerank_top20_and_classify" else [item["passage_id"] for item in passages],
            )
            predictions.append(prediction)
            continue
        try:
            user_prompt = build_user_prompt(claim["claim"], passages, args.variant, args.max_chars_per_passage)
            result, raw_response = call_openai_structured(
                model=args.model, system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, temperature=args.temperature
            )
            validate_llm_verdict(result, passages, args.variant)
            prediction = build_prediction(claim_id, claim.get("label"), result)
            prediction["evidence_passage_ids"] = (
                result.selected_passage_ids
                if args.variant == "rerank_top20_and_classify"
                else [str(item["passage_id"]) for item in passages]
            )
            raw_rows.append(
                {
                    "claim_id": claim_id,
                    "status": "success",
                    "model": args.model,
                    "variant": args.variant,
                    "prompt_sha256": prompt_sha256(SYSTEM_PROMPT, user_prompt),
                    "candidate_passage_ids": [str(item["passage_id"]) for item in passages],
                    "parsed": {"prediction": prediction},
                    "response": raw_response,
                }
            )
            predictions.append(prediction)
        except Exception as error:  # A per-claim failure must not discard prior cached calls.
            errors.append({"claim_id": claim_id, "status": "error", "error": f"{type(error).__name__}: {error}"})
    write_jsonl(raw_rows, raw_path)
    write_jsonl(predictions, args.output_dir / f"{args.variant}_predictions.jsonl")
    write_jsonl(errors, args.output_dir / f"{args.variant}_errors.jsonl")
    metrics = evaluate_predictions(predictions) if predictions else {"claim_count": 0}
    coverage_artifacts: dict[str, str] = {}
    if args.qrels:
        qrels = {
            str(row["claim_id"]): [str(value) for value in row.get("relevant_passage_ids") or []]
            for row in read_jsonl(args.qrels)
            if row.get("relevant_passage_ids")
        }
        coverage = annotate_retrieval_coverage(predictions, rankings, qrels, k=3)
        coverage_path = args.output_dir / f"{args.variant}_evidence_coverage.jsonl"
        conditioned_metrics_path = args.output_dir / f"{args.variant}_retrieval_conditioned_metrics.json"
        write_jsonl(coverage, coverage_path)
        conditioned_metrics_path.write_text(
            json.dumps(evaluate_by_retrieval_coverage(predictions, rankings, qrels, k=3), indent=2) + "\n", encoding="utf-8"
        )
        coverage_artifacts = {
            "evidence_coverage": str(coverage_path),
            "retrieval_conditioned_metrics": str(conditioned_metrics_path),
        }
    config = {
        "script": "run_llm_baseline",
        "provider": "OpenAI",
        "model": args.model,
        "variant": args.variant,
        "temperature": args.temperature,
        "candidate_limit": candidate_limit,
        "max_chars_per_passage": args.max_chars_per_passage,
        "run_name": args.run_name,
        "system_prompt": SYSTEM_PROMPT,
        "successful_claim_count": len(predictions),
        "error_count": len(errors),
        "usage": usage_summary(raw_rows),
        "warning": "LLM reported confidence is not a calibrated probability and is excluded from calibration metrics.",
        "artifacts": coverage_artifacts,
    }
    (args.output_dir / f"{args.variant}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / f"{args.variant}_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    write_run_manifest(
        args.output_dir / f"{args.variant}_manifest.json",
        project_root=PROJECT_ROOT,
        config={key: value for key, value in config.items() if key != "system_prompt"},
        input_paths={"claims": args.claims, "rankings": args.rankings, **({"qrels": args.qrels} if args.qrels else {})},
    )
    print(json.dumps({**config, **metrics}, indent=2))


if __name__ == "__main__":
    main()
