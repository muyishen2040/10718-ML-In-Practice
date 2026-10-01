"""Run a resumable local-GPU, zero-shot evidence-only LLM baseline.

Designed for a Colab GPU. The default Qwen3 4B model is loaded with 4-bit
quantization; each completed response is appended to a cache so an interrupted
session can resume without repeating already completed claims.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.classification import evaluate_predictions  # noqa: E402
from src.evaluation.end_to_end import annotate_retrieval_coverage, evaluate_by_retrieval_coverage  # noqa: E402
from src.utils.io import read_jsonl, write_jsonl  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.runs import resolve_run_directory, write_run_manifest  # noqa: E402
from src.verification.llm_zero_shot import (  # noqa: E402
    CLAIM_ONLY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    ClaimOnlyLLMVerdict,
    LLMVerdict,
    build_claim_only_prompt,
    build_prediction,
    build_user_prompt,
    prompt_sha256,
    validate_llm_verdict,
)


def existing_successes(path: Path, *, model: str, variant: str, evidence_mode: str = "bm25_top3") -> dict[str, dict[str, Any]]:
    """Reuse only successful calls from the exact local model, variant, and evidence mode."""
    if not path.exists():
        return {}
    successes: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if (
            row.get("status") != "success"
            or row.get("model") != model
            or row.get("variant") != variant
            or row.get("evidence_mode", "bm25_top3") != evidence_mode
        ):
            continue
        claim_id = str(row.get("claim_id") or "")
        if claim_id:
            successes[claim_id] = row
    return successes


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    """Persist a result immediately, so a stopped Colab cell remains resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def parse_json_verdict(response_text: str, *, claim_only: bool = False) -> LLMVerdict | ClaimOnlyLLMVerdict:
    """Extract one JSON object, tolerating Markdown fences around an otherwise valid response."""
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            schema = ClaimOnlyLLMVerdict if claim_only else LLMVerdict
            return schema.model_validate(normalize_local_verdict_fields(value))
    raise ValueError("Local model did not return a valid JSON verdict object")


def normalize_local_verdict_fields(value: dict[str, Any]) -> dict[str, Any]:
    """Accept common local-model JSON aliases while retaining the canonical artifact schema.

    This is deliberately limited to field-name and scalar/list normalization; it
    never infers a label, evidence ID, confidence, or rationale that the model
    did not supply.
    """
    normalized = dict(value)
    aliases = {
        "label": ("verdict", "verdict_label", "prediction"),
        "selected_passage_ids": ("selected_passages", "selected_evidence_ids", "evidence_passage_ids", "evidence_ids", "evidence", "passages"),
        "reported_confidence": ("confidence", "score"),
        "brief_rationale": ("rationale", "reason", "reasoning", "explanation"),
    }
    for canonical, alternatives in aliases.items():
        if canonical not in normalized:
            for alternative in alternatives:
                if alternative in normalized:
                    normalized[canonical] = normalized[alternative]
                    break
    selected = normalized.get("selected_passage_ids")
    if isinstance(selected, str):
        normalized["selected_passage_ids"] = [selected]
    elif isinstance(selected, list) and all(isinstance(item, dict) for item in selected):
        passage_ids = []
        for item in selected:
            for key in ("passage_id", "id", "citation"):
                if key in item:
                    passage_ids.append(item[key])
                    break
            else:
                passage_ids.append(item)
        normalized["selected_passage_ids"] = passage_ids
    return normalized


def load_local_model(model_name: str):
    """Load a causal LM in 4-bit mode, failing early with actionable Colab guidance."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError("Install requirements-local-llm-colab.txt before running the local LLM baseline.") from error
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU runtime is required. In Colab: Runtime > Change runtime type > T4 GPU.")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer, torch


def generate_local_verdict(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
) -> tuple[LLMVerdict | ClaimOnlyLLMVerdict, str]:
    """Generate one deterministic non-thinking Qwen-compatible response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        rendered_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError as error:
        raise RuntimeError(
            "This model/tokenizer does not support Qwen3's enable_thinking switch. "
            "Use Qwen/Qwen3-4B with transformers>=4.51."
        ) from error
    model_inputs = tokenizer(rendered_prompt, return_tensors="pt").to(model.device)
    input_length = model_inputs.input_ids.shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_text = tokenizer.decode(generated[0][input_length:], skip_special_tokens=True).strip()
    return parse_json_verdict(response_text, claim_only=system_prompt == CLAIM_ONLY_SYSTEM_PROMPT), response_text


def gold_evidence_passages(claim: dict[str, Any], *, max_evidence: int) -> list[dict[str, Any]]:
    """Return at most three human-annotated answers for the non-deployable oracle diagnostic."""
    return [
        {
            "passage_id": str(item["passage_id"]),
            "text": str(item["text"]),
            "url": item.get("url"),
            "metadata": item.get("metadata") or {},
        }
        for item in list(claim.get("evidence") or [])[:max_evidence]
        if item.get("passage_id") and str(item.get("text") or "").strip()
    ]


def build_cases(
    *,
    claims: list[dict[str, Any]],
    rankings_path: Path | None,
    evidence_mode: str,
    max_evidence: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build an ordered inference set and report claims excluded from the gold oracle."""
    claims_by_id = {str(row["claim_id"]): row for row in claims}
    if evidence_mode == "bm25_top3":
        if rankings_path is None:
            raise ValueError("--rankings is required for --evidence-mode bm25_top3.")
        cases = []
        for ranking in read_jsonl(rankings_path):
            claim_id = str(ranking["claim_id"])
            cases.append({"claim_id": claim_id, "claim": claims_by_id.get(claim_id), "passages": list(ranking.get("retrieved") or [])[:3]})
        return cases, []
    if evidence_mode == "claim_only":
        return [{"claim_id": str(claim["claim_id"]), "claim": claim, "passages": []} for claim in claims], []
    cases = []
    excluded_claim_ids = []
    for claim in claims:
        passages = gold_evidence_passages(claim, max_evidence=max_evidence)
        if not passages:
            excluded_claim_ids.append(str(claim["claim_id"]))
            continue
        cases.append({"claim_id": str(claim["claim_id"]), "claim": claim, "passages": passages})
    return cases, excluded_claim_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--rankings", type=Path, help="Required only for --evidence-mode bm25_top3.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Hugging Face model identifier; record this exact ID in the writeup.")
    parser.add_argument("--variant", choices=("classify_top3", "rerank_top20_and_classify"), default="classify_top3")
    parser.add_argument(
        "--evidence-mode",
        choices=("claim_only", "bm25_top3", "gold_top3"),
        default="bm25_top3",
        help="claim_only is a no-evidence diagnostic; gold_top3 is a non-deployable capped oracle diagnostic.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "local_llm")
    parser.add_argument("--run-name", help="Name for an isolated, resumable run directory.")
    parser.add_argument("--qrels", type=Path, help="Optional URL-proxy relevance judgments for coverage diagnostics.")
    parser.add_argument("--max-claims", type=int, default=0, help="0 means all ranked claims; use 20 for a GPU smoke test.")
    parser.add_argument("--max-evidence", type=int, default=3, help="Gold-answer cap for --evidence-mode gold_top3.")
    parser.add_argument("--max-chars-per-passage", type=int, default=1200)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=718)
    args = parser.parse_args()
    if args.max_claims < 0 or args.max_chars_per_passage < 1 or args.max_new_tokens < 1 or args.max_evidence < 1:
        parser.error("--max-claims must be nonnegative and evidence/text/token limits must be positive.")
    if args.evidence_mode == "bm25_top3" and args.variant != "classify_top3":
        parser.error("The checkpoint BM25 comparison uses --variant classify_top3; rerank_top20 is a separate optional experiment.")
    if args.evidence_mode != "bm25_top3" and args.variant != "classify_top3":
        parser.error("--variant rerank_top20_and_classify is valid only with --evidence-mode bm25_top3.")
    try:
        args.output_dir = resolve_run_directory(args.output_dir, args.run_name)
    except ValueError as error:
        parser.error(str(error))

    random.seed(args.seed)
    claim_rows = read_jsonl(args.claims)
    cases, gold_evidence_missing_claim_ids = build_cases(
        claims=claim_rows,
        rankings_path=args.rankings,
        evidence_mode=args.evidence_mode,
        max_evidence=args.max_evidence,
    )
    if args.max_claims:
        cases = cases[: args.max_claims]
    if not cases:
        parser.error("No claims were available for local LLM inference.")
    candidate_limit = 0 if args.evidence_mode == "claim_only" else 3
    output_stem = "claim_only" if args.evidence_mode == "claim_only" else "gold_top3" if args.evidence_mode == "gold_top3" else args.variant
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / f"{output_stem}_raw.jsonl"
    cached = existing_successes(raw_path, model=args.model, variant=args.variant, evidence_mode=args.evidence_mode)
    model, tokenizer, torch = load_local_model(args.model)
    model_revision = getattr(model.config, "_commit_hash", None)
    predictions_by_claim: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []

    with progress(
        total=len(cases),
        initial=sum(str(row["claim_id"]) in cached for row in cases),
        description=f"Local LLM ({args.evidence_mode})",
        unit="claim",
    ) as bar:
        for case in cases:
            claim_id = str(case["claim_id"])
            cached_row = cached.get(claim_id)
            if cached_row is not None:
                predictions_by_claim[claim_id] = cached_row["parsed"]["prediction"]
                continue
            claim = case["claim"]
            passages = case["passages"]
            expected_passage_count = candidate_limit if args.evidence_mode == "bm25_top3" else None
            if claim is None or (expected_passage_count is not None and len(passages) != expected_passage_count):
                error = {
                    "claim_id": claim_id,
                    "status": "error",
                    "error": "claim_missing_from_claims_file" if claim is None else f"ranking_has_{len(passages)}_passages_but_requires_{expected_passage_count}",
                }
                errors.append(error)
                append_jsonl_row(raw_path, {**error, "model": args.model, "variant": args.variant, "evidence_mode": args.evidence_mode})
                bar.update(1)
                continue
            try:
                if args.evidence_mode == "claim_only":
                    system_prompt = CLAIM_ONLY_SYSTEM_PROMPT
                    user_prompt = build_claim_only_prompt(claim["claim"])
                else:
                    system_prompt = SYSTEM_PROMPT
                    user_prompt = build_user_prompt(
                        claim["claim"],
                        passages,
                        args.variant,
                        args.max_chars_per_passage,
                        evidence_description="human-annotated evidence answers" if args.evidence_mode == "gold_top3" else "retrieved passages",
                    )
                result, response_text = generate_local_verdict(
                    model=model,
                    tokenizer=tokenizer,
                    torch=torch,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                if args.evidence_mode != "claim_only":
                    validate_llm_verdict(result, passages, args.variant)
                prediction = build_prediction(claim_id, claim.get("label"), result)
                prediction["evidence_passage_ids"] = [] if args.evidence_mode == "claim_only" else (
                    result.selected_passage_ids
                    if args.evidence_mode == "gold_top3" or args.variant == "rerank_top20_and_classify"
                    else [str(item["passage_id"]) for item in passages]
                )
                predictions_by_claim[claim_id] = prediction
                append_jsonl_row(
                    raw_path,
                    {
                        "claim_id": claim_id,
                        "status": "success",
                        "provider": "local_transformers_4bit",
                        "model": args.model,
                        "model_revision": model_revision,
                        "variant": args.variant,
                        "evidence_mode": args.evidence_mode,
                        "prompt_sha256": prompt_sha256(system_prompt, user_prompt),
                        "candidate_passage_ids": [str(item["passage_id"]) for item in passages],
                        "parsed": {"prediction": prediction},
                        "response_text": response_text,
                    },
                )
            except Exception as error:  # Preserve successes and continue to other claims after malformed output/OOM.
                error_row = {
                    "claim_id": claim_id,
                    "status": "error",
                    "model": args.model,
                    "variant": args.variant,
                    "evidence_mode": args.evidence_mode,
                    "error": f"{type(error).__name__}: {error}",
                }
                errors.append(error_row)
                append_jsonl_row(raw_path, error_row)
            bar.update(1)

    predictions = [predictions_by_claim[str(row["claim_id"])] for row in cases if str(row["claim_id"]) in predictions_by_claim]
    write_jsonl(predictions, args.output_dir / f"{output_stem}_predictions.jsonl")
    write_jsonl(errors, args.output_dir / f"{output_stem}_errors.jsonl")
    metrics = evaluate_predictions(predictions) if predictions else {"claim_count": 0}
    coverage_artifacts: dict[str, str] = {}
    if args.qrels and predictions and args.evidence_mode == "bm25_top3":
        qrels = {
            str(row["claim_id"]): [str(value) for value in row.get("relevant_passage_ids") or []]
            for row in read_jsonl(args.qrels)
            if row.get("relevant_passage_ids")
        }
        rankings = read_jsonl(args.rankings)
        coverage = annotate_retrieval_coverage(predictions, rankings, qrels, k=3)
        coverage_path = args.output_dir / f"{output_stem}_evidence_coverage.jsonl"
        conditioned_metrics_path = args.output_dir / f"{output_stem}_retrieval_conditioned_metrics.json"
        write_jsonl(coverage, coverage_path)
        conditioned_metrics_path.write_text(
            json.dumps(evaluate_by_retrieval_coverage(predictions, rankings, qrels, k=3), indent=2) + "\n",
            encoding="utf-8",
        )
        coverage_artifacts = {
            "evidence_coverage": str(coverage_path),
            "retrieval_conditioned_metrics": str(conditioned_metrics_path),
        }
    successful_count = len(predictions)
    config = {
        "script": "run_local_llm_baseline",
        "provider": "local_transformers_4bit",
        "model": args.model,
        "model_revision": model_revision,
        "variant": args.variant,
        "evidence_mode": args.evidence_mode,
        "run_name": args.run_name,
        "seed": args.seed,
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens, "thinking": "disabled"},
        "candidate_limit": candidate_limit,
        "gold_evidence_cap": args.max_evidence if args.evidence_mode == "gold_top3" else None,
        "gold_evidence_missing_claim_count": len(gold_evidence_missing_claim_ids),
        "gold_evidence_missing_claim_samples": gold_evidence_missing_claim_ids[:10],
        "max_chars_per_passage": args.max_chars_per_passage,
        "successful_claim_count": successful_count,
        "cached_success_count": len(cached),
        "error_count": len(errors),
        "system_prompt": CLAIM_ONLY_SYSTEM_PROMPT if args.evidence_mode == "claim_only" else SYSTEM_PROMPT,
        "warning": (
            "This is a no-evidence diagnostic, not an evidence-grounded deployment result. "
            "LLM reported confidence is not a calibrated probability and is excluded from calibration metrics."
            if args.evidence_mode == "claim_only"
            else "gold_top3 uses human-annotated answers and is a non-deployable oracle diagnostic. "
            "LLM reported confidence is not a calibrated probability and is excluded from calibration metrics."
            if args.evidence_mode == "gold_top3"
            else "LLM reported confidence is not a calibrated probability and is excluded from calibration metrics."
        ),
        "artifacts": coverage_artifacts,
    }
    (args.output_dir / f"{output_stem}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / f"{output_stem}_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    write_run_manifest(
        args.output_dir / f"{output_stem}_manifest.json",
        project_root=PROJECT_ROOT,
        config={key: value for key, value in config.items() if key != "system_prompt"},
        input_paths={
            "claims": args.claims,
            **({"rankings": args.rankings} if args.evidence_mode == "bm25_top3" and args.rankings else {}),
            **({"qrels": args.qrels} if args.evidence_mode == "bm25_top3" and args.qrels else {}),
        },
    )
    print(json.dumps({**config, **metrics}, indent=2))


if __name__ == "__main__":
    main()
