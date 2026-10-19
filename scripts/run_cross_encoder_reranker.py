"""Resume-safe zero-shot cross-encoder reranking of saved candidate lists.

Only the supplied top-N candidates are scored. This cannot recover evidence
absent from the first-stage retriever, and gold labels are never model inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bm25 import append_checkpoint_rows, load_checkpoint_rankings, load_qrels  # noqa: E402
from src.retrieval.metrics import evaluate_rankings  # noqa: E402
from src.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.runs import resolve_run_directory, sha256_file, write_run_manifest  # noqa: E402


def reranked_row(source: dict[str, Any], scores: Sequence[float], max_candidates: int) -> dict[str, Any]:
    candidates = (source.get("retrieved") or [])[:max_candidates]
    if len(candidates) != len(scores):
        raise ValueError("Reranker score count differs from candidate count")
    ids = [str(item["passage_id"]) for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate passage IDs for {source.get('claim_id')}")
    ranked = []
    for item, score in zip(candidates, scores, strict=True):
        if not math.isfinite(float(score)):
            raise ValueError("Reranker returned a non-finite score")
        row = dict(item)
        row["first_stage_score"] = item.get("score")
        row["score"] = float(score)
        ranked.append(row)
    ranked.sort(key=lambda row: (-float(row["score"]), str(row["passage_id"])))
    return {"claim_id": str(source["claim_id"]), "claim": str(source["claim"]), "retrieved": ranked}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-rankings", required=True, type=Path)
    parser.add_argument("--qrels", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--model", default="BAAI/bge-reranker-base")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.max_candidates, args.max_seq_length, args.batch_size, args.checkpoint_every) < 1:
        parser.error("Candidate count, sequence length, batch size, and checkpoint interval must be positive")
    if not args.input_rankings.is_file():
        raise FileNotFoundError(args.input_rankings)
    if args.qrels and not args.qrels.is_file():
        raise FileNotFoundError(args.qrels)
    if args.split == "train" and args.qrels:
        parser.error("Do not supply dev qrels for a train reranking run")
    try:
        output_dir = resolve_run_directory(args.output_dir, args.run_name)
    except ValueError as error:
        parser.error(str(error))
    output_dir.mkdir(parents=True, exist_ok=True)
    rankings_path = output_dir / f"{args.split}_rankings.jsonl"
    checkpoint_path = output_dir / f"{args.split}_rankings.partial.jsonl"
    metrics_path = output_dir / f"{args.split}_metrics.json"
    manifest_path = output_dir / f"{args.split}_manifest.json"
    config_path = output_dir / "reranker_config.json"
    if checkpoint_path.exists() and not args.resume:
        parser.error(f"Partial checkpoint exists; rerun unchanged with --resume: {checkpoint_path}")

    # Saved first-stage rankings contain at most 20 passages per claim and are
    # small enough to read in full. Validate identities before any model work.
    source_rows = read_jsonl(args.input_rankings)
    source_by_id = {str(row["claim_id"]): row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("Input rankings contain duplicate claim IDs")
    input_hash = sha256_file(args.input_rankings)
    config = {
        "script": "run_cross_encoder_reranker",
        "split": args.split,
        "model": args.model,
        "requested_revision": args.revision,
        "input_rankings": str(args.input_rankings.resolve()),
        "input_sha256": input_hash,
        "qrels_sha256": sha256_file(args.qrels) if args.qrels else None,
        "max_candidates": args.max_candidates,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "device": args.device,
    }
    saved = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    if any(saved.get(key) != value for key, value in config.items()) and saved:
        raise ValueError(f"Run inputs/settings changed; choose a new --run-name: {config_path}")
    if rankings_path.is_file() and metrics_path.is_file() and manifest_path.is_file():
        print(metrics_path.read_text(encoding="utf-8"))
        return
    if rankings_path.is_file():
        # Recovery if interruption occurred after the atomic rankings write.
        final_rows = read_jsonl(rankings_path)
        if len(final_rows) != len(source_rows):
            raise ValueError("Final rankings are incomplete; manual audit required")
    else:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select a GPU Colab runtime before this reranker run.")
        revision = args.revision or saved.get("resolved_revision")
        print(f"Loading cross-encoder {args.model} on {args.device}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision or None, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, revision=revision or None)
        if int(model.config.num_labels) != 1:
            raise ValueError(f"Expected one relevance logit, got {model.config.num_labels}")
        model.to(args.device).eval()
        resolved_revision = getattr(model.config, "_commit_hash", None) or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        if saved.get("resolved_revision") and resolved_revision != saved["resolved_revision"]:
            raise ValueError("Loaded model revision differs from existing checkpoint configuration")
        saved = {**config, "resolved_revision": resolved_revision}
        write_json(saved, config_path)

        completed = load_checkpoint_rankings(checkpoint_path, known_claim_ids=set(source_by_id)) if args.resume else {}
        pending: list[dict[str, Any]] = []

        def flush_pending() -> None:
            nonlocal pending
            # Clear the in-memory batch *before* appending. If Colab interrupts
            # mid-write, a valid prefix is resumed and a truncated tail is
            # repaired, without finally appending duplicate claim IDs.
            batch = pending
            pending = []
            append_checkpoint_rows(checkpoint_path, batch)

        with progress(total=len(source_rows), initial=len(completed), description=f"Rerank {args.split}", unit="claim") as bar:
            try:
                for row in source_rows:
                    claim_id = str(row["claim_id"])
                    if claim_id in completed:
                        continue
                    candidates = (row.get("retrieved") or [])[:args.max_candidates]
                    scores: list[float] = []
                    for offset in range(0, len(candidates), args.batch_size):
                        batch = candidates[offset : offset + args.batch_size]
                        encoded = tokenizer(
                            [str(row["claim"])] * len(batch),
                            [str(item["text"]) for item in batch],
                            padding=True,
                            truncation=True,
                            max_length=args.max_seq_length,
                            return_tensors="pt",
                        )
                        encoded = {key: value.to(args.device) for key, value in encoded.items()}
                        with torch.inference_mode():
                            logits = model(**encoded).logits.reshape(-1).float().cpu().tolist()
                        if len(logits) != len(batch):
                            raise ValueError("Model returned an unexpected number of scores")
                        scores.extend(float(value) for value in logits)
                    result = reranked_row(row, scores, args.max_candidates)
                    completed[claim_id] = result
                    pending.append(result)
                    if len(pending) >= args.checkpoint_every:
                        flush_pending()
                    bar.update(1)
            finally:
                flush_pending()
        if len(completed) != len(source_rows):
            raise RuntimeError(f"Only {len(completed)}/{len(source_rows)} claims reranked; resume first")
        temporary = rankings_path.with_name(f"{rankings_path.name}.tmp")
        write_jsonl((completed[str(row["claim_id"])] for row in source_rows), temporary)
        temporary.replace(rankings_path)
        final_rows = [completed[str(row["claim_id"])] for row in source_rows]

    report = {
        **saved,
        "claim_count": len(final_rows),
        "rankings_path": str(rankings_path),
        "candidate_pool_scope": "first_stage_top_N_only",
        "finalized": True,
    }
    if args.qrels:
        report.update(evaluate_rankings(
            {str(row["claim_id"]): [str(item["passage_id"]) for item in row["retrieved"]] for row in final_rows},
            load_qrels(args.qrels),
            k=3,
        ))
        report["qrels_path"] = str(args.qrels)
    write_json(report, metrics_path)
    write_run_manifest(
        manifest_path,
        project_root=PROJECT_ROOT,
        config=report,
        input_paths={"first_stage_rankings": args.input_rankings, **({"qrels": args.qrels} if args.qrels else {})},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
