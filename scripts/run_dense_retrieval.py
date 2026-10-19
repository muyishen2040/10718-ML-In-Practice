"""Resumable dense retrieval over AVeriTeC claim-scoped candidate pools.

Development rankings are built by streaming the normalized v2 JSONL corpus one
claim pool at a time. Training rankings can be built directly from the three
official archive shards without materializing a combined corpus. Each finished
claim is appended to a JSONL checkpoint, so a restarted Colab run skips its
embedding work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_averitec_archive_bm25 import (  # noqa: E402
    documents_from_official_member,
    passages_from_documents,
)
from scripts.run_bm25 import append_checkpoint_rows, load_checkpoint_rankings, load_qrels  # noqa: E402
from src.data.knowledge_store import _candidate_claim_index  # noqa: E402
from src.retrieval.metrics import evaluate_rankings  # noqa: E402
from src.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.runs import resolve_run_directory, write_run_manifest  # noqa: E402


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_embedding_model(model_name: str, revision: str | None, device: str | None, max_seq_length: int):
    started = time.monotonic()
    print(
        f"Importing sentence-transformers for {model_name} "
        f"(HF_HOME={os.environ.get('HF_HOME', '<default>')})...",
        flush=True,
    )
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Dense retrieval requires sentence-transformers. Install requirements-colab.txt first."
        ) from error

    print(f"Library import completed in {time.monotonic() - started:.1f}s; loading model weights...", flush=True)
    model = SentenceTransformer(
        model_name,
        revision=revision or None,
        device=device or None,
        trust_remote_code=False,
    )
    model.max_seq_length = max_seq_length
    first_module = model[0] if len(model) else None
    auto_model = getattr(first_module, "auto_model", None)
    config = getattr(auto_model, "config", None)
    resolved_revision = getattr(config, "_commit_hash", None)
    print(
        f"Embedding model ready in {time.monotonic() - started:.1f}s "
        f"(device={getattr(model, 'device', 'unknown')}).",
        flush=True,
    )
    return model, resolved_revision


def dense_top_k(
    model: Any,
    claim: str,
    candidates: Iterable[Any],
    *,
    k: int,
    batch_size: int,
    query_prefix: str,
) -> list[dict[str, Any]]:
    """Score candidates in batches, retaining only the deterministic top k."""
    query = model.encode(
        [f"{query_prefix}{claim}"],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    top_rows: list[dict[str, Any]] = []
    batch: list[Any] = []

    try:
        candidate_count = len(candidates)  # type: ignore[arg-type]
    except TypeError:
        candidate_count = None

    def score_batch(items: list[Any]) -> None:
        if not items:
            return
        vectors = model.encode(
            [item.text for item in items],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = np.asarray(vectors, dtype=np.float32) @ np.asarray(query, dtype=np.float32)
        batch_rows = []
        for item, score in zip(items, scores, strict=True):
            row = item.to_dict()
            row["score"] = float(score)
            batch_rows.append(row)
        top_rows[:] = sorted(
            [*top_rows, *batch_rows],
            key=lambda row: (-float(row["score"]), str(row["passage_id"])),
        )[:k]

    with progress(
        total=candidate_count,
        description="Embedding candidates",
        unit="passage",
        position=1,
        leave=False,
    ) as bar:
        for item in candidates:
            batch.append(item)
            if len(batch) >= batch_size:
                score_batch(batch)
                bar.update(len(batch))
                batch = []
        score_batch(batch)
        bar.update(len(batch))
    return top_rows


def write_jsonl_atomic(rows: Iterable[dict[str, Any]], path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    write_jsonl(rows, temporary)
    temporary.replace(path)


def complete_claim_row(claim: dict[str, Any], retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    return {"claim_id": str(claim["claim_id"]), "claim": str(claim["claim"]), "retrieved": retrieved}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, help="Required for dev; normalized v2 claim-scoped passage JSONL.")
    parser.add_argument("--archive", action="append", type=Path, help="Repeat for train archive shards, in official order.")
    parser.add_argument("--qrels", type=Path, help="Defaults to processed/averitec/<split>_qrels.jsonl when present.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", help="Optional immutable Hugging Face revision for the embedding model.")
    parser.add_argument("--query-prefix", default=DEFAULT_QUERY_PREFIX)
    parser.add_argument("--ranking-k", type=int, default=20)
    parser.add_argument("--metric-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--max-words", type=int, default=160, help="Archive-document passage window size.")
    parser.add_argument("--overlap-words", type=int, default=40, help="Archive-document passage overlap.")
    parser.add_argument("--device", help="For example cuda or cpu; defaults to SentenceTransformers auto-selection.")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize", action="store_true", help="Finalize train rankings after all archive shards.")
    args = parser.parse_args()

    if args.ranking_k < 1 or args.metric_k < 1 or args.batch_size < 1 or args.checkpoint_every < 1:
        parser.error("ranking/metric k, batch size, and checkpoint interval must be positive")
    if args.max_seq_length < 8:
        parser.error("--max-seq-length must be at least 8")
    if args.max_words < 1 or args.overlap_words < 0 or args.overlap_words >= args.max_words:
        parser.error("Require --max-words > 0 and 0 <= --overlap-words < --max-words")
    if args.split == "dev" and (not args.corpus or args.archive):
        parser.error("dev requires --corpus and does not accept --archive")
    if args.split == "train" and (args.corpus or (not args.archive and not args.finalize)):
        parser.error("train requires one or more --archive inputs, unless --finalize is used")

    claims_path = args.data_root / "processed" / "averitec" / f"{args.split}.jsonl"
    if not claims_path.is_file():
        raise FileNotFoundError(claims_path)
    claims = read_jsonl(claims_path)
    claim_by_id = {str(row["claim_id"]): row for row in claims}
    source_index_to_claim = {
        int(row["metadata"]["source_index"]): row
        for row in claims
        if row.get("metadata", {}).get("source_index") is not None
    }

    try:
        output_dir = resolve_run_directory(args.output_dir, args.run_name)
    except ValueError as error:
        parser.error(str(error))
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / f"{args.split}_rankings.jsonl"
    checkpoint_path = output_dir / f"{args.split}_rankings.partial.jsonl"
    metrics_path = output_dir / f"{args.split}_metrics.json"
    run_config_path = output_dir / "dense_run_config.json"
    if checkpoint_path.exists() and not args.resume:
        parser.error(f"Found {checkpoint_path}; rerun the exact command with --resume.")

    saved_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.is_file() else {}
    immutable_config = {
        "split": args.split,
        "model": args.model,
        "requested_revision": args.revision,
        "query_prefix": args.query_prefix,
        "ranking_k": args.ranking_k,
        "metric_k": args.metric_k,
        "max_seq_length": args.max_seq_length,
        "max_words": args.max_words,
        "overlap_words": args.overlap_words,
    }
    for key, value in immutable_config.items():
        if key in saved_config and saved_config[key] != value:
            raise ValueError(f"Dense run configuration differs for {key}; use a new --run-name.")

    # Persist input identity across separate archive-shard invocations. Size
    # and mtime are cheap to inspect on Drive; re-hashing 64-GB archives each
    # time Colab reconnects would be wasteful.
    saved_archive_inputs = saved_config.get("archive_inputs", {})
    current_archive_inputs = dict(saved_archive_inputs)
    for archive_path in args.archive or []:
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        stat = archive_path.stat()
        identity = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        key = str(archive_path.resolve())
        if key in current_archive_inputs and current_archive_inputs[key] != identity:
            raise ValueError(f"Archive input changed since this run started: {archive_path}")
        current_archive_inputs[key] = identity
    if args.corpus:
        if not args.corpus.is_file():
            raise FileNotFoundError(args.corpus)
        stat = args.corpus.stat()
        corpus_identity = {
            "path": str(args.corpus.resolve()),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if "corpus_input" in saved_config and saved_config["corpus_input"] != corpus_identity:
            raise ValueError(f"Dense retrieval corpus changed since this run started: {args.corpus}")
        saved_config["corpus_input"] = corpus_identity
    saved_config["archive_inputs"] = current_archive_inputs

    if ranking_path.is_file():
        if metrics_path.is_file() and (output_dir / f"{args.split}_manifest.json").is_file():
            print(metrics_path.read_text(encoding="utf-8"))
            return
        # Recover bookkeeping if Colab stopped between the final ranking,
        # metrics, and manifest writes. This also covers train runs without qrels.
        final_rankings = read_jsonl(ranking_path)
        qrels_path = args.qrels or args.data_root / "processed" / "averitec" / f"{args.split}_qrels.jsonl"
        report = {
            **saved_config,
            "rankings_path": str(ranking_path),
            "claim_count": len(final_rankings),
            "finalized": True,
            "recovered_after_interruption": True,
        }
        if qrels_path.is_file():
            report.update(
                evaluate_rankings(
                    {
                        str(row["claim_id"]): [str(item["passage_id"]) for item in row.get("retrieved") or []]
                        for row in final_rankings
                    },
                    load_qrels(qrels_path),
                    k=args.metric_k,
                )
            )
            report["qrels_path"] = str(qrels_path)
        write_json(report, metrics_path)
        write_run_manifest(
            output_dir / f"{args.split}_manifest.json",
            project_root=PROJECT_ROOT,
            config=report,
            input_paths={
                "claims": claims_path,
                **({"qrels": qrels_path} if qrels_path.is_file() else {}),
            },
        )
        print(json.dumps(report, indent=2))
        return

    finalizing_only = args.split == "train" and args.finalize and not args.archive
    effective_revision = args.revision or saved_config.get("resolved_revision")
    if finalizing_only:
        model = None
        resolved_revision = effective_revision
    else:
        model, resolved_revision = load_embedding_model(
            args.model, effective_revision, args.device, args.max_seq_length
        )
    saved_config.update(immutable_config)
    saved_config["resolved_revision"] = resolved_revision
    write_json(saved_config, run_config_path)
    completed = load_checkpoint_rankings(checkpoint_path, known_claim_ids=set(claim_by_id)) if args.resume else {}
    pending: list[dict[str, Any]] = []
    processed_claim_count = 0
    processed_passage_count = 0
    missing_pool_ids: list[str] = []

    def checkpoint(row: dict[str, Any]) -> None:
        nonlocal processed_claim_count, pending
        completed[str(row["claim_id"])] = row
        pending.append(row)
        processed_claim_count += 1
        if len(pending) >= args.checkpoint_every:
            append_checkpoint_rows(checkpoint_path, pending)
            pending = []

    if args.split == "dev":
        from scripts.run_bm25 import per_claim_candidate_groups

        if not args.corpus.is_file():
            raise FileNotFoundError(args.corpus)
        # This loop visits every claim again on resume, so do not initialize the bar
        # with completed claims as well as incrementing for them below.
        with progress(total=len(claims), initial=0, description="Dense dev retrieval", unit="claim", position=0) as bar:
            try:
                for candidate_claim_id, candidate_items in per_claim_candidate_groups(args.corpus):
                    claim = claim_by_id.get(candidate_claim_id)
                    if claim is None:
                        continue
                    if candidate_claim_id in completed:
                        bar.update(1)
                        continue
                    bar.set_postfix(claim=candidate_claim_id)
                    processed_passage_count += len(candidate_items)
                    row = complete_claim_row(
                        claim,
                        dense_top_k(
                            model,
                            str(claim["claim"]),
                            candidate_items,
                            k=args.ranking_k,
                            batch_size=args.batch_size,
                            query_prefix=args.query_prefix,
                        ),
                    )
                    checkpoint(row)
                    bar.update(1)
            finally:
                append_checkpoint_rows(checkpoint_path, pending)
                pending = []
            for claim in claims:
                claim_id = str(claim["claim_id"])
                if claim_id not in completed:
                    missing_pool_ids.append(claim_id)
                    checkpoint(complete_claim_row(claim, []))
                    bar.update(1)
        append_checkpoint_rows(checkpoint_path, pending)
        pending = []
        should_finalize = True
    else:
        if args.archive:
            if model is None:
                raise RuntimeError("Embedding model was not loaded for archive processing")
            from scripts.run_averitec_archive_bm25 import passages_from_documents

            with progress(total=len(claims), initial=len(completed), description="Dense train retrieval", unit="claim", position=0) as bar:
                try:
                    for archive_path in args.archive:
                        if not archive_path.is_file():
                            raise FileNotFoundError(archive_path)
                        with zipfile.ZipFile(archive_path) as archive:
                            members = [member for member in archive.infolist() if not member.is_dir()]
                            for member in members:
                                source_index = _candidate_claim_index(member.filename)
                                if source_index is None:
                                    raise ValueError(f"Could not infer claim index from {member.filename}")
                                claim = source_index_to_claim.get(source_index)
                                if claim is None:
                                    continue
                                claim_id = str(claim["claim_id"])
                                if claim_id in completed:
                                    continue
                                bar.set_postfix(claim=claim_id)
                                passages = passages_from_documents(
                                    documents_from_official_member(archive, member),
                                    claim_id,
                                    max_words=args.max_words,
                                    overlap_words=args.overlap_words,
                                )
                                processed_passage_count += len(passages)
                                row = complete_claim_row(
                                    claim,
                                    dense_top_k(
                                        model,
                                        str(claim["claim"]),
                                        passages,
                                        k=args.ranking_k,
                                        batch_size=args.batch_size,
                                        query_prefix=args.query_prefix,
                                    ),
                                )
                                checkpoint(row)
                                bar.update(1)
                finally:
                    append_checkpoint_rows(checkpoint_path, pending)
                    pending = []
        should_finalize = args.finalize

    config = {
        "script": "run_dense_retrieval",
        "split": args.split,
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "query_prefix": args.query_prefix,
        "ranking_k": args.ranking_k,
        "metric_k": args.metric_k,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "claim_count": len(claims),
        "processed_this_invocation_count": processed_claim_count,
        "passages_embedded_this_invocation": processed_passage_count,
        "resumed_claim_count": len(completed) - processed_claim_count,
        "missing_candidate_pool_claim_ids": missing_pool_ids,
        "candidate_pool_scope": "per_claim_candidate_pool",
        "resume": args.resume,
        "checkpoint_path": str(checkpoint_path),
        "archive_paths": list(current_archive_inputs),
        "corpus_path": str(args.corpus) if args.corpus else None,
        "archive_inputs": current_archive_inputs,
    }

    if should_finalize:
        missing_rankings = [str(row["claim_id"]) for row in claims if str(row["claim_id"]) not in completed]
        if missing_rankings:
            raise RuntimeError(
                f"Cannot finalize dense rankings: {len(missing_rankings)} claims are incomplete, "
                f"e.g. {missing_rankings[:5]}. Rerun all required shards with --resume."
            )
        ordered_rankings = (completed[str(row["claim_id"])] for row in claims)
        write_jsonl_atomic(ordered_rankings, ranking_path)
        checkpoint_path.unlink(missing_ok=True)
        config["rankings_path"] = str(ranking_path)
        qrels_path = args.qrels or args.data_root / "processed" / "averitec" / f"{args.split}_qrels.jsonl"
        report: dict[str, Any] = {**config, "finalized": True}
        if qrels_path.is_file():
            qrels = load_qrels(qrels_path)
            report.update(
                evaluate_rankings(
                    {
                        str(row["claim_id"]): [str(item["passage_id"]) for item in row.get("retrieved") or []]
                        for row in completed.values()
                    },
                    qrels,
                    k=args.metric_k,
                )
            )
            report["qrels_path"] = str(qrels_path)
        write_json(report, metrics_path)
        write_run_manifest(
            output_dir / f"{args.split}_manifest.json",
            project_root=PROJECT_ROOT,
            config=report,
            input_paths={
                "claims": claims_path,
                **({"qrels": qrels_path} if qrels_path.is_file() else {}),
            },
        )
    else:
        report = {**config, "finalized": False}
        write_json(report, output_dir / f"{args.split}_progress.json")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
