"""Resumable extractive QA features over saved question-guided/BM25 candidates.

This script never reads verdicts, gold answers, or qrels. Candidate answers are
verbatim substrings of prediction-time passages, not generated factual claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bm25 import append_checkpoint_rows, load_checkpoint_rankings  # noqa: E402
from src.retrieval.answer_aware import (  # noqa: E402
    best_extractable_span, choose_candidates, qa_text,
)
from src.utils.io import iter_jsonl, read_jsonl, write_json, write_jsonl  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.runs import resolve_run_directory, sha256_file, write_run_manifest  # noqa: E402


def load_reader(model_name: str, revision: str | None, device: str):
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime before loading the answer reader")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Answer extraction requires a fast tokenizer with offset mappings")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, revision=revision)
    model.to(device).eval()
    resolved = getattr(model.config, "_commit_hash", None) or tokenizer.init_kwargs.get("_commit_hash")
    return model, tokenizer, torch, resolved


def read_answers(
    questions: list[str], candidates: list[dict[str, Any]], model: Any, tokenizer: Any, torch: Any,
    *, device: str, batch_size: int, max_seq_length: int, max_answer_tokens: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    pairs = [(index, question) for index in range(len(candidates)) for question in questions]
    answers: list[list[tuple[str, str, float]]] = [[] for _ in candidates]
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        passages = [str(candidates[index]["text"]) for index, _ in batch]
        encoded = tokenizer(
            [question for _, question in batch], passages,
            truncation="only_second", max_length=max_seq_length, padding=True,
            return_offsets_mapping=True, return_tensors="pt",
        )
        sequences = [encoded.sequence_ids(index) for index in range(len(batch))]
        offsets = encoded.pop("offset_mapping").tolist()
        input_ids = encoded["input_ids"].tolist()
        with torch.inference_mode():
            result = model(**{key: value.to(device) for key, value in encoded.items()})
        starts = result.start_logits.float().cpu().tolist()
        ends = result.end_logits.float().cpu().tolist()
        for local_index, (candidate_index, question) in enumerate(batch):
            cls_id = tokenizer.cls_token_id
            cls_index = input_ids[local_index].index(cls_id) if cls_id in input_ids[local_index] else 0
            answer, margin = best_extractable_span(
                passages[local_index], offsets[local_index], sequences[local_index],
                starts[local_index], ends[local_index], cls_index=cls_index,
                max_answer_tokens=max_answer_tokens,
            )
            answers[candidate_index].append((question, answer, margin))
    enriched: list[dict[str, Any]] = []
    for item, options in zip(candidates, answers, strict=True):
        best_question, best_answer, best_margin = max(options, key=lambda entry: entry[2])
        passage = str(item["text"])
        enriched.append({
            **item,
            "best_question": best_question,
            "candidate_answer": best_answer if best_margin > 0 else "",
            "max_answer_margin": max(-100.0, min(100.0, best_margin)),
            "mean_answer_margin": sum(max(-100.0, min(100.0, entry[2])) for entry in options) / len(options),
            "answer_word_count": len(best_answer.split()) if best_margin > 0 else 0,
            "bm25_reciprocal_rank": 1.0 / int(item["bm25_rank"]),
            "qa_text": qa_text(best_question, best_answer, passage, best_margin),
        })
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25-rankings", required=True, type=Path)
    parser.add_argument("--question-checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="deepset/tinyroberta-squad2")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--bm25-keep", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--max-claims", type=int, default=0, help="Train-only smoke test; zero means all")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.candidate_k, args.batch_size, args.max_seq_length, args.max_answer_tokens, args.checkpoint_every) < 1:
        parser.error("All lengths and checkpoint interval must be positive")
    if not 0 <= args.bm25_keep <= args.candidate_k or args.max_claims < 0:
        parser.error("Invalid bm25-keep or max-claims")
    if args.split == "dev" and args.max_claims:
        parser.error("Do not use a dev subset to tune this method")
    for path in (args.bm25_rankings, args.question_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = resolve_run_directory(args.output_dir, args.run_name)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "answer_reader_config.json"
    checkpoint = output / f"{args.split}_answer_features.partial.jsonl"
    features_path = output / f"{args.split}_answer_features.jsonl"
    report_path = output / f"{args.split}_answer_reader_report.json"
    manifest_path = output / f"{args.split}_manifest.json"
    config = {
        "script": "run_answer_reader", "split": args.split,
        "bm25_rankings_sha256": sha256_file(args.bm25_rankings),
        "question_checkpoint_sha256": sha256_file(args.question_checkpoint),
        "model": args.model, "requested_revision": args.revision, "device": args.device,
        "candidate_k": args.candidate_k, "bm25_keep": args.bm25_keep,
        "batch_size": args.batch_size, "max_seq_length": args.max_seq_length,
        "max_answer_tokens": args.max_answer_tokens, "max_claims": args.max_claims,
        "checkpoint_every": args.checkpoint_every,
        "warning": "QA answerability is a model score, not proof that the quoted span resolves the claim.",
    }
    saved = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    if saved and any(saved.get(key) != value for key, value in config.items()):
        raise ValueError("Run inputs/settings changed; choose a new run name")
    if all(path.is_file() for path in (features_path, report_path, manifest_path)):
        print(report_path.read_text(encoding="utf-8"))
        return
    if checkpoint.exists() and not args.resume:
        parser.error(f"Partial checkpoint exists; rerun unchanged with --resume: {checkpoint}")
    if not saved:
        write_json(config, config_path)
    question_rows = read_jsonl(args.question_checkpoint)
    question_by_id = {str(row["claim_id"]): row for row in question_rows}
    if len(question_by_id) != len(question_rows):
        raise ValueError("Duplicate claim ID in question checkpoint")
    total = sum(1 for _ in iter_jsonl(args.bm25_rankings))
    if args.max_claims:
        total = min(total, args.max_claims)
    known_ids: set[str] = set()
    for number, row in enumerate(iter_jsonl(args.bm25_rankings)):
        if number >= total:
            break
        claim_id = str(row["claim_id"])
        if claim_id in known_ids or claim_id not in question_by_id:
            raise ValueError(f"Duplicate BM25 or missing question claim ID: {claim_id}")
        known_ids.add(claim_id)
    if not args.max_claims and known_ids != set(question_by_id):
        raise ValueError("Question/BM25 claim IDs do not match exactly")
    completed = load_checkpoint_rankings(checkpoint, known_claim_ids=known_ids) if args.resume else {}
    if len(completed) > total:
        raise ValueError("Checkpoint exceeds this run's claim count")
    pending: list[dict[str, Any]] = []
    if len(completed) < total:
        print(f"Loading answer reader: {args.model}", flush=True)
        model, tokenizer, torch, resolved = load_reader(args.model, args.revision, args.device)
        if saved.get("resolved_revision") and saved["resolved_revision"] != resolved:
            raise ValueError("Answer reader revision differs from the checkpoint")
        saved = {**config, "resolved_revision": resolved}
        write_json(saved, config_path)

        def flush() -> None:
            nonlocal pending
            rows, pending = pending, []
            append_checkpoint_rows(checkpoint, rows)

        with progress(total=total, initial=len(completed), description=f"Extractive QA / {args.split}", unit="claim") as bar:
            try:
                for number, source in enumerate(iter_jsonl(args.bm25_rankings)):
                    if number >= total:
                        break
                    claim_id = str(source["claim_id"])
                    if claim_id in completed:
                        continue
                    question_row = question_by_id[claim_id]
                    if str(source.get("claim")) != str(question_row.get("claim")):
                        raise ValueError(f"Claim text mismatch: {claim_id}")
                    questions = [str(value).strip() for value in question_row.get("questions") or [] if str(value).strip()]
                    questions = questions[:2] or [str(source["claim"])]
                    candidates = choose_candidates(source, question_row, limit=args.candidate_k, bm25_keep=args.bm25_keep)
                    row = {
                        "claim_id": claim_id, "claim": source["claim"], "questions": questions,
                        "retrieved": read_answers(
                            questions, candidates, model, tokenizer, torch, device=args.device,
                            batch_size=args.batch_size, max_seq_length=args.max_seq_length,
                            max_answer_tokens=args.max_answer_tokens,
                        ),
                    }
                    completed[claim_id] = row
                    pending.append(row)
                    if len(pending) >= args.checkpoint_every:
                        flush()
                    bar.update(1)
            finally:
                flush()
    if len(completed) != total:
        raise RuntimeError(f"Only {len(completed)}/{total} claims completed; resume first")
    ordered = []
    for number, source in enumerate(iter_jsonl(args.bm25_rankings)):
        if number >= total:
            break
        ordered.append(completed[str(source["claim_id"])])
    temporary = features_path.with_name(features_path.name + ".tmp")
    write_jsonl(ordered, temporary)
    temporary.replace(features_path)
    report = {
        **(saved or config), "claim_count": total, "finalized": True,
        "candidate_count": sum(len(row["retrieved"]) for row in ordered),
        "positive_margin_candidate_count": sum(
            item["max_answer_margin"] > 0 for row in ordered for item in row["retrieved"]
        ),
        "features_path": str(features_path), "checkpoint_path": str(checkpoint),
    }
    write_json(report, report_path)
    write_run_manifest(
        manifest_path, project_root=PROJECT_ROOT, config=report,
        input_paths={"bm25_rankings": args.bm25_rankings, "question_checkpoint": args.question_checkpoint},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
