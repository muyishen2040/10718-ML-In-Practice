"""Run and resume a zero-shot evidence-only LLM benchmark on saved rankings."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
from collections import Counter
from math import isfinite
from pathlib import Path
from random import random
from time import perf_counter, sleep
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.averitec import CANONICAL_LABELS  # noqa: E402
from src.data.evidence import index_claims  # noqa: E402
from src.evaluation.classification import evaluate_predictions  # noqa: E402
from src.evaluation.end_to_end import annotate_retrieval_coverage, evaluate_by_retrieval_coverage  # noqa: E402
from src.utils.checkpoints import append_journal, atomic_jsonl, bind_run, read_journal  # noqa: E402
from src.utils.io import read_jsonl, write_json  # noqa: E402
from src.utils.runs import resolve_run_directory, sha256_file, write_run_manifest  # noqa: E402
from src.verification.hosted_verifier import HOSTED_SYSTEM_PROMPT  # noqa: E402
from src.verification.llm_zero_shot import (  # noqa: E402
    LLMVerdict,
    NO_EVIDENCE_EXPLANATION,
    build_prediction,
    build_user_prompt,
    call_openai_structured,
    prompt_sha256,
    validate_llm_verdict,
)


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def usage_summary(raw_rows) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for row in raw_rows:
        usage = row.get("response", {}).get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if isinstance(usage.get(key), int):
                totals[key] += usage[key]
    return dict(totals)


def openai_client():
    from openai import OpenAI

    return OpenAI(max_retries=0)


class ApiAttemptLimit:
    """Reserve each attempt on disk before a request can incur charges."""

    def __init__(self, path: Path, limit: int) -> None:
        self.path = path
        self.limit = limit
        self.used = len(read_journal(path, key="attempt_id"))

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def reserve(self, identity: dict[str, Any]) -> None:
        if self.exhausted:
            raise ValueError("API attempt limit reached")
        append_journal(self.path, [{**identity, "attempt_id": uuid4().hex}], sync=True)
        self.used += 1


def retry_delay(error: Exception, attempt: int) -> float | None:
    """Retry short-lived transport errors. Defer explicit server waits to a later run."""
    status = getattr(error, "status_code", None)
    if getattr(error, "code", None) in {
        "insufficient_quota", "billing_hard_limit_reached", "credit_balance_exhausted",
        "organization_spend_limit_exceeded", "project_spend_limit_exceeded", "organization_usage_limit_exceeded",
    }:
        return None
    headers = getattr(getattr(error, "response", None), "headers", {})
    if "retry-after" in headers or "retry-after-ms" in headers or headers.get("x-should-retry") == "false":
        return None
    transient_status = status in (408, 409, 429) or (isinstance(status, int) and status >= 500)
    if not transient_status and not isinstance(error, (TimeoutError, ConnectionError)):
        try:
            from openai import APIConnectionError
        except ImportError:
            return None

        if not isinstance(error, APIConnectionError):
            return None
    return min(2 ** min(attempt - 1, 4), 8) + random()


def predict(args, client, claim_id, passages, user_prompt, identity, failed_attempts, limit):
    """Return a successful journal row or the final failed attempt for this claim."""
    if not passages:
        return {
            **identity,
            "status": "success",
            "source": "no_evidence",
            "parsed": {"prediction": {
                "claim_id": claim_id,
                "predicted_label": "Not Enough Evidence",
                "probabilities": None,
                "selected_passage_ids": [],
                "reported_confidence": None,
                "brief_rationale": NO_EVIDENCE_EXPLANATION,
                "evidence_passage_ids": [],
            }},
            "response": {},
        }, None

    for attempt in range(1, args.max_retries + 2):
        if limit is not None:
            limit.reserve(identity)
        start = perf_counter()
        raw_response = {}
        try:
            result, raw_response = call_openai_structured(
                model=args.model,
                system_prompt=HOSTED_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=args.temperature,
                client=client,
                max_completion_tokens=args.max_completion_tokens,
                timeout=args.timeout,
                reasoning_effort=args.reasoning_effort,
                service_tier=args.service_tier,
            )
            validate_llm_verdict(result, passages, args.variant)
            if not result.brief_rationale.strip():
                raise ValueError("Model returned a blank explanation")
        except Exception as error:  # Persist provider/validation failures before going on.
            failure = {
                **identity,
                "attempt_id": uuid4().hex,
                "attempt": attempt,
                "status": "error",
                "error": type(error).__name__,
                "status_code": getattr(error, "status_code", None),
                "provider_error_code": getattr(error, "code", None),
                "request_id": raw_response.get("request_id") or getattr(error, "request_id", None),
                "latency_seconds": perf_counter() - start,
                "response": raw_response,
            }
            headers = getattr(getattr(error, "response", None), "headers", {})
            failure["provider_requested_delay"] = "retry-after" in headers or "retry-after-ms" in headers
            append_journal(failed_attempts, [failure], sync=True)
            if limit is not None and limit.exhausted:
                return None, failure
            delay = retry_delay(error, attempt) if attempt <= args.max_retries else None
            if delay is None:
                return None, failure
            sleep(delay)
        else:
            prediction = build_prediction(claim_id, None, result)
            prediction.pop("true_label")
            prediction["evidence_passage_ids"] = (
                result.selected_passage_ids if args.variant == "rerank_top20_and_classify"
                else identity["candidate_passage_ids"]
            )
            return {
                **identity,
                "status": "success",
                "source": "openai",
                "attempt": attempt,
                "latency_seconds": perf_counter() - start,
                "parsed": {"prediction": prediction},
                "response": raw_response,
            }, None


def prepare_requests(claims, rankings, args, candidate_limit):
    requests, errors = {}, []
    for claim in claims:
        claim_id = str(claim["claim_id"])
        ranking = rankings.get(claim_id)
        if ranking is None:
            errors.append({"claim_id": claim_id, "status": "error", "error": "missing_ranking"})
            continue
        try:
            claim_text = claim["claim"]
            if not isinstance(claim_text, str) or not claim_text.strip():
                raise ValueError("Empty claim text")
            ranked_claim = ranking.get("claim")
            if ranked_claim and str(ranked_claim).strip() != claim_text.strip():
                raise ValueError("Claim text differs from the ranking")
            retrieved = ranking.get("retrieved")
            if retrieved is not None and not isinstance(retrieved, list):
                raise ValueError("Ranked passages must be a list")
            passages = (retrieved or [])[:candidate_limit]
            candidate_ids = [str(passage.get("passage_id") or "") for passage in passages]
            if "" in candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError("Missing or duplicate passage ID")
            if any(not isinstance(passage.get("text"), str) or not passage["text"].strip() for passage in passages):
                raise ValueError("Empty passage text")
            prompt = build_user_prompt(
                claim_text, passages, args.variant, args.max_chars_per_passage
            ) if passages else claim_text.strip()
        except (AttributeError, KeyError, ValueError) as error:
            errors.append({"claim_id": claim_id, "status": "error", "error": "invalid_input", "detail": str(error)})
            continue
        requests[claim_id] = (passages, prompt, candidate_ids)
    return requests, errors


def run(args: argparse.Namespace, prefix: Path) -> dict[str, Any]:
    inputs = {"claims": args.claims, "rankings": args.rankings}
    if args.qrels:
        inputs["qrels"] = args.qrels
    input_hashes = {name: sha256_file(path) for name, path in inputs.items()}
    claims = read_jsonl(args.claims)
    index_claims(claims)
    if args.max_claims:
        claims = claims[:args.max_claims]
    if not claims or any(claim.get("label") not in CANONICAL_LABELS for claim in claims):
        raise ValueError("The claims file must provide at least one claim and canonical gold labels")
    rankings = read_jsonl(args.rankings)
    rankings_by_claim = index_claims(rankings)
    expected = {str(claim["claim_id"]) for claim in claims}
    candidate_limit = 3 if args.variant == "classify_top3" else 20
    requests, errors = prepare_requests(claims, rankings_by_claim, args, candidate_limit)
    prompt_hashes = {
        claim_id: prompt_sha256(HOSTED_SYSTEM_PROMPT, prompt)
        for claim_id, (_, prompt, _) in requests.items()
    }

    config = {
        "journal_version": 1,
        "provider": "OpenAI",
        "provider_endpoint_sha256": json_sha256(os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"),
        "model": args.model,
        "variant": args.variant,
        "temperature": args.temperature,
        "candidate_limit": candidate_limit,
        "max_chars_per_passage": args.max_chars_per_passage,
        "max_completion_tokens": args.max_completion_tokens,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "max_claims": args.max_claims,
        "system_prompt": HOSTED_SYSTEM_PROMPT,
        "output_schema": LLMVerdict.model_json_schema(),
        "no_evidence_explanation": NO_EVIDENCE_EXPLANATION,
        "prompts_sha256": json_sha256(prompt_hashes),
        "input_sha256": input_hashes,
    }
    if args.reasoning_effort is not None:
        config["reasoning_effort"] = args.reasoning_effort
    if args.service_tier is not None:
        config["service_tier"] = args.service_tier
    if args.max_api_attempts is not None:
        config["max_api_attempts"] = args.max_api_attempts
    config_sha256 = json_sha256(config)

    binding, raw_path, failed_attempts = (
        prefix.with_name(prefix.name + suffix)
        for suffix in ("_binding.json", "_raw.jsonl", "_failed_attempts.jsonl")
    )
    attempt_path = prefix.with_name(prefix.name + "_api_attempts.jsonl")
    if attempt_path.exists() and (args.overwrite or args.max_api_attempts is None):
        raise ValueError("This run has reserved API attempts. Keep its settings and omit --overwrite to resume.")
    if args.overwrite:
        for suffix in (
            "_errors.jsonl", "_predictions.jsonl", "_metrics.json", "_config.json", "_manifest.json",
            "_evidence_coverage.jsonl", "_retrieval_conditioned_metrics.json",
        ):
            prefix.with_name(prefix.name + suffix).unlink(missing_ok=True)
        failed_attempts.unlink(missing_ok=True)
        raw_path.unlink(missing_ok=True)
        binding.unlink(missing_ok=True)
    elif not binding.exists() and (raw_path.exists() or failed_attempts.exists() or attempt_path.exists()):
        raise ValueError("Existing journal has no verified configuration. Use a new run directory or --overwrite.")
    bind_run(binding, config)
    cached = read_journal(raw_path, set(requests))
    read_journal(failed_attempts, key="attempt_id")
    raw_path.touch(exist_ok=True)
    failed_attempts.touch(exist_ok=True)
    limit = ApiAttemptLimit(attempt_path, args.max_api_attempts) if args.max_api_attempts is not None else None

    identities = {
        claim_id: {
            "claim_id": claim_id,
            "model": args.model,
            "variant": args.variant,
            "prompt_sha256": prompt_hashes[claim_id],
            "candidate_passage_ids": candidate_ids,
            "request_sha256": json_sha256([config_sha256, claim_id, prompt_hashes[claim_id], candidate_ids]),
        }
        for claim_id, (_, _, candidate_ids) in requests.items()
    }
    for claim_id, record in cached.items():
        if record.get("request_sha256") != identities[claim_id]["request_sha256"] or record.get("status") != "success":
            raise ValueError(f"Saved call differs from the bound request for {claim_id}")

    client = None
    provider_paused = False
    try:
        for claim_id, (passages, user_prompt, _) in requests.items():
            if claim_id in cached or (passages and (provider_paused or (limit is not None and limit.exhausted))):
                continue
            if passages and client is None:
                client = openai_client()
            record, failure = predict(
                args, client, claim_id, passages, user_prompt, identities[claim_id], failed_attempts, limit
            )
            if failure is not None:
                errors.append(failure)
                provider_paused = failure["status_code"] in (401, 403, 429) or failure["provider_requested_delay"]
                continue
            append_journal(raw_path, [record], sync=True)
            cached[claim_id] = record
    finally:
        if client is not None:
            client.close()

    predictions = [
        {**cached[str(claim["claim_id"])]["parsed"]["prediction"], "true_label": claim["label"]}
        for claim in claims if str(claim["claim_id"]) in cached
    ]
    unresolved_ids = sorted(expected - set(cached))
    coverage = {
        "evaluation_status": "partial" if unresolved_ids else "complete",
        "expected_claim_count": len(expected),
        "completed_claim_count": len(predictions),
        "completion_coverage": len(predictions) / len(expected),
        "unresolved_claim_ids": unresolved_ids,
        "unattempted_claim_ids": sorted(set(unresolved_ids) - {row["claim_id"] for row in errors}),
    }
    metrics = evaluate_predictions(predictions) if predictions else {"claim_count": 0}
    if unresolved_ids:
        metrics = {"claim_count": len(predictions), "completed_only_metrics": metrics}
    metrics.update(coverage)
    attempts = read_journal(failed_attempts, key="attempt_id").values()
    config.update(
        coverage,
        script="run_llm_baseline",
        run_name=args.run_name,
        config_sha256=config_sha256,
        successful_claim_count=len(predictions),
        no_evidence_claim_count=sum(row.get("source") == "no_evidence" for row in cached.values()),
        error_count=len(errors),
        failed_attempt_count=len(attempts),
        usage=usage_summary([*cached.values(), *attempts]),
        warning="LLM reported confidence is not a calibrated probability and is excluded from calibration metrics.",
    )
    if limit is not None:
        config["reserved_api_attempt_count"] = limit.used
        config["api_attempt_limit_reached"] = limit.exhausted
    atomic_jsonl(predictions, prefix.with_name(prefix.name + "_predictions.jsonl"))
    atomic_jsonl(errors, prefix.with_name(prefix.name + "_errors.jsonl"))
    if args.qrels:
        qrels = {
            str(row["claim_id"]): [str(value) for value in row["relevant_passage_ids"]]
            for row in read_jsonl(args.qrels) if row.get("relevant_passage_ids")
        }
        evidence_path = prefix.with_name(prefix.name + "_evidence_coverage.jsonl")
        conditioned_path = prefix.with_name(prefix.name + "_retrieval_conditioned_metrics.json")
        atomic_jsonl(annotate_retrieval_coverage(predictions, rankings, qrels, k=3), evidence_path)
        write_json({**coverage, **evaluate_by_retrieval_coverage(predictions, rankings, qrels, k=3)}, conditioned_path)
        config["artifacts"] = {
            "evidence_coverage": str(evidence_path),
            "retrieval_conditioned_metrics": str(conditioned_path),
        }
    else:
        config["artifacts"] = {}
    write_json(metrics, prefix.with_name(prefix.name + "_metrics.json"))
    write_json(config, prefix.with_name(prefix.name + "_config.json"))
    write_run_manifest(
        prefix.with_name(prefix.name + "_manifest.json"), project_root=PROJECT_ROOT,
        config={key: value for key, value in config.items() if key != "system_prompt"}, input_paths=inputs,
    )
    return {**config, **metrics}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--rankings", required=True, type=Path)
    parser.add_argument("--model", required=True, help="Exact provider model ID or snapshot.")
    parser.add_argument("--variant", choices=("classify_top3", "rerank_top20_and_classify"), default="classify_top3")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "llm")
    parser.add_argument("--run-name", help="An isolated subdirectory for a new run or a resume.")
    parser.add_argument("--qrels", type=Path, help="Optional relevance judgments for evidence-coverage diagnostics.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh"))
    parser.add_argument("--service-tier", choices=("auto", "default", "flex", "priority"))
    parser.add_argument("--max-claims", type=int, default=0, help="0 means all rows in the claims file.")
    parser.add_argument("--max-chars-per-passage", type=int, default=1200)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds per API attempt.")
    parser.add_argument("--max-retries", type=int, default=1, help="Additional attempts per claim in this invocation.")
    parser.add_argument("--max-api-attempts", type=int, help="Total API attempts for this run, including retries and resumes.")
    parser.add_argument("--overwrite", action="store_true", help="Discard this variant's journals and outputs before running.")
    args = parser.parse_args(argv)
    if not args.model.strip() or not 0 <= args.temperature <= 2:
        parser.error("Supply a model ID and a temperature between 0 and 2.")
    if (not isfinite(args.timeout) or min(args.max_chars_per_passage, args.max_completion_tokens, args.timeout) <= 0
            or min(args.max_claims, args.max_retries) < 0):
        parser.error("Token, passage and timeout limits must be positive. Claim and retry limits must be nonnegative.")
    if args.max_api_attempts is not None and args.max_api_attempts < 0:
        parser.error("The API attempt limit must be nonnegative.")
    try:
        directory = resolve_run_directory(args.output_dir, args.run_name)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f".{args.variant}.lock").open("a", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error(f"Another process is writing this run: {directory}")
            report = run(args, directory / args.variant)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
