"""Resumable claim-question-guided selection from saved BM25 top-100 passages.

No gold evidence, verdict labels, or qrels are read during selection. Dev qrels
are opened only after every ranking is finalized. The two output rankings have
identical passage IDs: one contains original passages and one exact-source
sentence excerpts for a controlled verifier ablation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bm25 import append_checkpoint_rows, load_checkpoint_rankings, load_qrels  # noqa: E402
from src.retrieval.metrics import evaluate_rankings  # noqa: E402
from src.retrieval.question_guided import (  # noqa: E402
    extract_sentences,
    parse_questions,
    reciprocal_rank_scores,
    select_diverse,
    shortlist_candidates,
)
from src.utils.io import iter_jsonl, read_jsonl, write_json, write_jsonl  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.runs import resolve_run_directory, sha256_file, write_run_manifest  # noqa: E402


QUESTION_SYSTEM_PROMPT = (
    "You break down a factual claim into two short, independently answerable "
    "verification questions. Do not decide if the claim is true. Do not use "
    "outside knowledge. Return JSON only: {\"questions\": [\"...\", \"...\"]}."
)


def load_qwen(model_name: str, revision: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA Colab runtime is required for the question generator")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision or None)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision or None, device_map="auto",
        quantization_config=quantization, torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer, torch


def generate_questions(claim: str, model: Any, tokenizer: Any, torch: Any, *, max_new_tokens: int) -> tuple[list[str], str]:
    messages = [
        {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Claim: {claim[:1000]}"},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError as error:
        raise RuntimeError("The Qwen3 tokenizer must support enable_thinking=False") from error
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0][encoded.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return parse_questions(response, claim), response


def load_cross_encoder(model_name: str, revision: str, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a GPU runtime in Colab")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision or None, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision or None)
    if int(model.config.num_labels) != 1:
        raise ValueError("Expected a cross-encoder with one relevance logit")
    model.to(device).eval()
    return model, tokenizer, torch


def score_query(query: str, candidates: list[dict[str, Any]], model: Any, tokenizer: Any, torch: Any,
                *, device: str, batch_size: int, max_seq_length: int, max_passage_chars: int) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        encoded = tokenizer(
            [query] * len(batch), [str(item.get("text") or "")[:max_passage_chars] for item in batch],
            padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt",
        )
        with torch.inference_mode():
            output = model(**{key: value.to(device) for key, value in encoded.items()})
        scores.extend(float(value) for value in output.logits.reshape(-1).float().cpu().tolist())
    return scores


def prepare_output_row(row: dict[str, Any], *, excerpt: bool) -> dict[str, Any]:
    retrieved = []
    for item in row["retrieved"]:
        copy = dict(item)
        copy["text"] = str(item["excerpt_text"] if excerpt else item["full_text"])
        copy.pop("full_text", None)
        copy.pop("excerpt_text", None)
        retrieved.append(copy)
    return {"claim_id": row["claim_id"], "claim": row["claim"], "retrieved": retrieved}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-rankings", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True, help="Used only to validate claim IDs; labels are not read by the model")
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--qrels", type=Path, help="Dev metrics only, after finalization")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--qwen-revision", default="1cfa9a7208912126459214e8b04321603b3df60c")
    parser.add_argument("--cross-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--cross-revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shortlist-k", type=int, default=32)
    parser.add_argument("--bm25-keep", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--max-passage-chars", type=int, default=1400)
    parser.add_argument("--excerpt-chars", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-claims", type=int, default=0, help="First N claims for a train-only smoke test; 0 means all")
    parser.add_argument("--seed", type=int, default=718)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.split == "train" and args.qrels:
        parser.error("Train selection must not receive qrels")
    if args.split == "dev" and args.max_claims:
        parser.error("Do not inspect a dev subset for method tuning; run all 500 claims once")
    if min(args.shortlist_k, args.batch_size, args.max_seq_length, args.max_passage_chars,
           args.excerpt_chars, args.max_new_tokens, args.checkpoint_every) < 1 or args.max_claims < 0:
        parser.error("Lengths, batch size, and checkpoint interval must be positive")
    if args.bm25_keep < 0 or args.bm25_keep > args.shortlist_k:
        parser.error("bm25-keep must be between zero and shortlist-k")
    for path in (args.input_rankings, args.claims, *([args.qrels] if args.qrels else [])):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = resolve_run_directory(args.output_dir, args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{args.split}_question_guided.partial.jsonl"
    full_path = output_dir / f"{args.split}_full_rankings.jsonl"
    excerpt_path = output_dir / f"{args.split}_excerpt_rankings.jsonl"
    report_path = output_dir / f"{args.split}_metrics.json"
    manifest_path = output_dir / f"{args.split}_manifest.json"
    config_path = output_dir / "question_guided_config.json"
    config = {
        "script": "run_question_guided_retrieval", "split": args.split,
        "input_rankings": str(args.input_rankings.resolve()), "input_sha256": sha256_file(args.input_rankings),
        "claims_sha256": sha256_file(args.claims),
        "qwen_model": args.qwen_model, "qwen_revision": args.qwen_revision,
        "cross_model": args.cross_model, "cross_revision": args.cross_revision,
        "device": args.device, "shortlist_k": args.shortlist_k, "bm25_keep": args.bm25_keep,
        "batch_size": args.batch_size, "max_seq_length": args.max_seq_length,
        "max_passage_chars": args.max_passage_chars, "excerpt_chars": args.excerpt_chars,
        "max_new_tokens": args.max_new_tokens, "max_claims": args.max_claims, "seed": args.seed,
        "question_prompt": QUESTION_SYSTEM_PROMPT,
        "selection": "claim and two generated questions scored by zero-shot cross-encoder; per-query RRF, then unique-source top 3",
        "excerpt": "up to three verbatim source sentences selected by lexical query coverage",
    }
    saved = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    if saved and any(saved.get(key) != value for key, value in config.items()):
        raise ValueError(f"Run inputs or settings changed; choose a new run name: {config_path}")
    if all(path.is_file() for path in (full_path, excerpt_path, report_path, manifest_path)):
        print(report_path.read_text(encoding="utf-8"))
        return
    if checkpoint.exists() and not args.resume:
        parser.error(f"Partial checkpoint exists; rerun unchanged with --resume: {checkpoint}")
    if not saved:
        write_json(config, config_path)

    known_ids = {str(row["claim_id"]) for row in read_jsonl(args.claims)}
    completed = load_checkpoint_rankings(checkpoint, known_claim_ids=known_ids) if args.resume else {}
    total = sum(1 for _ in iter_jsonl(args.input_rankings))
    if args.max_claims:
        total = min(total, args.max_claims)
    if len(completed) > total:
        raise ValueError("Checkpoint has more claims than this run permits")
    random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    pending: list[dict[str, Any]] = []
    if len(completed) < total:
        print(f"Loading Qwen question generator ({args.qwen_model})...", flush=True)
        qwen, qwen_tokenizer, torch = load_qwen(args.qwen_model, args.qwen_revision)
        print(f"Loading relevance cross-encoder ({args.cross_model})...", flush=True)
        cross, cross_tokenizer, _ = load_cross_encoder(args.cross_model, args.cross_revision or saved.get("resolved_cross_revision") or "", args.device)
        cross_revision = getattr(cross.config, "_commit_hash", None) or getattr(cross_tokenizer, "init_kwargs", {}).get("_commit_hash")
        qwen_revision = getattr(qwen.config, "_commit_hash", None) or getattr(qwen_tokenizer, "init_kwargs", {}).get("_commit_hash")
        if saved.get("resolved_cross_revision") and saved["resolved_cross_revision"] != cross_revision:
            raise ValueError("Cross-encoder revision changed since checkpoint")
        if saved.get("resolved_qwen_revision") and saved["resolved_qwen_revision"] != qwen_revision:
            raise ValueError("Qwen revision changed since checkpoint")
        saved = {**config, "resolved_cross_revision": cross_revision, "resolved_qwen_revision": qwen_revision}
        write_json(saved, config_path)

        def flush() -> None:
            nonlocal pending
            batch, pending = pending, []
            append_checkpoint_rows(checkpoint, batch)

        seen: set[str] = set()
        with progress(total=total, initial=len(completed), description=f"Question-guided {args.split}", unit="claim") as bar:
            try:
                for number, source in enumerate(iter_jsonl(args.input_rankings)):
                    if number >= total:
                        break
                    claim_id = str(source["claim_id"])
                    if claim_id in seen or claim_id not in known_ids:
                        raise ValueError(f"Duplicate or unknown claim ID: {claim_id}")
                    seen.add(claim_id)
                    if claim_id in completed:
                        continue
                    claim = str(source["claim"])
                    if not claim.strip():
                        raise ValueError(f"Empty claim in rankings: {claim_id}")
                    questions, raw_response = generate_questions(
                        claim, qwen, qwen_tokenizer, torch, max_new_tokens=args.max_new_tokens
                    )
                    candidates = shortlist_candidates(source, questions, limit=args.shortlist_k, bm25_keep=args.bm25_keep)
                    if candidates:
                        queries = [claim, *questions]
                        matrix = [score_query(
                            query, candidates, cross, cross_tokenizer, torch, device=args.device,
                            batch_size=args.batch_size, max_seq_length=args.max_seq_length,
                            max_passage_chars=args.max_passage_chars,
                        ) for query in queries]
                        fused = reciprocal_rank_scores(matrix)
                        selected = select_diverse(candidates, fused, k=3)
                    else:
                        selected = []
                    evidence = []
                    for item in selected:
                        full_text = str(item.get("text") or "")
                        evidence.append({
                            **item, "full_text": full_text,
                            "excerpt_text": extract_sentences(full_text, claim, questions, max_chars=args.excerpt_chars),
                        })
                    result = {
                        "claim_id": claim_id, "claim": claim, "questions": questions,
                        "question_response": raw_response,
                        "shortlist_passage_ids": [str(item["passage_id"]) for item in candidates],
                        "retrieved": evidence,
                    }
                    completed[claim_id] = result
                    pending.append(result)
                    if len(pending) >= args.checkpoint_every:
                        flush()
                    bar.update(1)
            finally:
                flush()
    if len(completed) != total:
        raise RuntimeError(f"Only {len(completed)}/{total} claims completed; resume first")
    ordered: list[dict[str, Any]] = []
    for number, source in enumerate(iter_jsonl(args.input_rankings)):
        if number >= total:
            break
        claim_id = str(source["claim_id"])
        if claim_id not in completed:
            raise RuntimeError(f"Missing completed ranking: {claim_id}")
        ordered.append(completed[claim_id])
    for destination, excerpt in ((full_path, False), (excerpt_path, True)):
        temporary = destination.with_name(destination.name + ".tmp")
        write_jsonl((prepare_output_row(row, excerpt=excerpt) for row in ordered), temporary)
        temporary.replace(destination)
    report: dict[str, Any] = {
        **(saved or config), "claim_count": total, "full_rankings_path": str(full_path),
        "excerpt_rankings_path": str(excerpt_path), "checkpoint_path": str(checkpoint),
        "missing_candidate_pool_claim_count": sum(not row["retrieved"] for row in ordered),
        "question_fallback_count": sum(row["questions"][0].startswith("What evidence supports or contradicts this claim:") for row in ordered),
        "finalized": True, "metric_k": 3, "candidate_scope": "saved BM25 top 100 only",
    }
    if args.qrels:
        qrels = load_qrels(args.qrels)
        report.update(evaluate_rankings(
            {row["claim_id"]: [str(item["passage_id"]) for item in row["retrieved"]] for row in ordered},
            qrels, k=3,
        ))
        report["mrr_at_3"] = report.pop("mrr")
        report["mrr_scope"] = "top_three_only"
        source_ids = {}
        for number, source in enumerate(iter_jsonl(args.input_rankings)):
            if number >= total:
                break
            source_ids[str(source["claim_id"])] = [str(item["passage_id"]) for item in source.get("retrieved") or []]
        bm25_top3 = evaluate_rankings({key: ids[:3] for key, ids in source_ids.items()}, qrels, k=3)
        bm25_top100 = evaluate_rankings(source_ids, qrels, k=100)
        shortlist_metric = evaluate_rankings(
            {row["claim_id"]: row["shortlist_passage_ids"] for row in ordered}, qrels, k=args.shortlist_k,
        )
        report["bm25_reference_hit_at_3"] = bm25_top3["evidence_hit_at_3"]
        report["bm25_reference_mrr_at_3"] = bm25_top3["mrr"]
        report["bm25_candidate_hit_at_100"] = bm25_top100["evidence_hit_at_100"]
        report[f"shortlist_hit_at_{args.shortlist_k}"] = shortlist_metric[f"evidence_hit_at_{args.shortlist_k}"]
        report["qrels_path"] = str(args.qrels)
        report["qrels_sha256"] = sha256_file(args.qrels)
    write_json(report, report_path)
    write_run_manifest(
        manifest_path, project_root=PROJECT_ROOT, config=report,
        input_paths={"input_rankings": args.input_rankings, "claims": args.claims, **({"qrels": args.qrels} if args.qrels else {})},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
