"""Prepare or launch the bounded Teammate C pilot from this checkout."""

import argparse
from decimal import Decimal
from getpass import getpass
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_llm_baseline as runner
from src.data.evidence import index_claims
from src.utils.io import read_jsonl, write_jsonl

PILOT = ROOT / "local/pilot"
DATA = ROOT / "local/v2_source_documents"
MODEL = "gpt-5.4-mini-2026-03-17"
API_BASE = "https://api.openai.com/v1"
API_ATTEMPTS = 10
PASSAGE_CHARS = 1200
COMPLETION_TOKENS = 512
BUDGET = Decimal("0.10")
INPUT_PER_MILLION = Decimal("0.75")
OUTPUT_PER_MILLION = Decimal("4.50")


def inputs(ids_file):
    if ids_file is None:
        fixtures = [
            ("The library closes at six.", "Supported",
             "The library's posted weekday closing time is 6 p.m."),
            ("The visitor parking lot is open on Sunday.", "Refuted",
             "Visitor parking is closed every Sunday."),
            ("The new bus stop opens in June.", "Not Enough Evidence", None),
        ]
        claims, rankings = [], []
        for index, (claim, label, text) in enumerate(fixtures, start=1):
            claim_id = f"synthetic:{index}"
            claims.append({"claim_id": claim_id, "claim": claim, "label": label})
            passages = [{"passage_id": f"{claim_id}:p1", "text": text}] if text else []
            rankings.append({"claim_id": claim_id, "claim": claim, "retrieved": passages})
        rankings_file = PILOT / "synthetic_rankings.jsonl"
        write_jsonl(rankings, rankings_file)
    else:
        ids = [line.strip() for line in ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not 1 <= len(ids) <= API_ATTEMPTS or len(ids) != len(set(ids)):
            raise ValueError(f"Provide 1 to {API_ATTEMPTS} unique training-fit claim IDs, one per line.")
        source_claims = index_claims(read_jsonl(DATA / "processed/averitec/train.jsonl"))
        if set(ids) - set(source_claims):
            raise ValueError("Every pilot claim ID must come from the official training claims.")
        claims = [source_claims[claim_id] for claim_id in ids]
        rankings_file = DATA / "outputs/bm25/train_bm25_top20_v2/train_rankings.jsonl"
        rankings = read_jsonl(rankings_file)

    claims_file = PILOT / f"claims-{runner.json_sha256(claims)[:16]}.jsonl"
    write_jsonl(claims, claims_file)
    requests, errors = runner.prepare_requests(
        claims, index_claims(rankings),
        argparse.Namespace(variant="classify_top3", max_chars_per_passage=PASSAGE_CHARS), 3,
    )
    if errors:
        raise ValueError(f"Fix pilot inputs before running: {errors}")
    return claims_file, rankings_file, requests


def get_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key and sys.stdin.isatty():
        key = getpass("OpenAI project API key (hidden): ").strip()
    if not key:
        raise SystemExit("Set OPENAI_API_KEY securely or run in an interactive terminal.")
    return key


def check_key():
    from openai import APIError, OpenAI

    with OpenAI(api_key=get_key(), base_url=API_BASE, max_retries=0, timeout=15) as client:
        try:
            model = client.models.retrieve(MODEL)
        except APIError as error:
            raise SystemExit(f"Model access check failed ({type(error).__name__}). Check the project key and model access.")
    print(f"Project key can retrieve {model.id}. This check did not generate text.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Send paid requests after preflight succeeds.")
    mode.add_argument("--check-key", action="store_true", help="Check model access without generating text.")
    parser.add_argument("--claim-ids", type=Path, help="Coordinator-designated training-fit IDs, one per line.")
    parser.add_argument("--run-name", help="Reuse the same name to resume within its existing attempt limit.")
    args = parser.parse_args(argv)
    if args.check_key:
        check_key()
        return
    try:
        run_name = args.run_name or ("training-fit-v1" if args.claim_ids else "synthetic-smoke-v1")
        output_root = PILOT / "runs"
        run_directory = runner.resolve_run_directory(output_root, run_name)
        claims, rankings, requests = inputs(args.claim_ids)
        prompt_bytes = max(
            (len(prompt.encode("utf-8")) for passages, prompt, _ in requests.values() if passages),
            default=0,
        )
        schema_bytes = len(json.dumps(runner.LLMVerdict.model_json_schema()).encode("utf-8"))
        # Approximate upper envelope for tokenization, including framing and schema.
        input_estimate = prompt_bytes + len(runner.HOSTED_SYSTEM_PROMPT.encode("utf-8")) + schema_bytes + 1024
        estimate = API_ATTEMPTS * (
            input_estimate * INPUT_PER_MILLION + COMPLETION_TOKENS * OUTPUT_PER_MILLION
        ) / 1_000_000
        if estimate > BUDGET:
            raise ValueError(f"Conservative cost estimate ${estimate:.4f} exceeds the ${BUDGET} pilot budget.")
    except ValueError as error:
        parser.error(str(error))

    print(f"Model {MODEL}, reasoning none, temperature 0.0")
    print(f"{len(requests)} claims, at most {API_ATTEMPTS} paid attempts across restarts, zero automatic retries.")
    print(f"{COMPLETION_TOKENS} completion tokens per request and {PASSAGE_CHARS} characters per passage.")
    print(f"Estimated upper cost ${estimate:.4f} under the ${BUDGET} pilot budget at standard rates.")
    print(f"Run directory {run_directory}")
    if not args.live:
        print("Offline preflight passed. To check the project API key, run with --check-key.")
        return

    os.environ["OPENAI_API_KEY"] = get_key()
    os.environ["OPENAI_BASE_URL"] = API_BASE
    runner.main([
        "--claims", str(claims), "--rankings", str(rankings), "--model", MODEL,
        "--reasoning-effort", "none", "--temperature", "0", "--service-tier", "default",
        "--run-name", run_name, "--output-dir", str(output_root),
        "--max-api-attempts", str(API_ATTEMPTS), "--max-retries", "0",
        "--max-completion-tokens", str(COMPLETION_TOKENS),
        "--max-chars-per-passage", str(PASSAGE_CHARS),
    ])


if __name__ == "__main__":
    main()
