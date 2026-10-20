"""Train an inexpensive linear projection over frozen BGE embeddings.

Only saved BM25 top-N candidates are embedded. Chunk caches make Colab resume
without repeating finished encoding. Source URLs supply *weak training labels*
and are never features at prediction time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dense_retrieval import DEFAULT_MODEL, DEFAULT_QUERY_PREFIX, load_embedding_model  # noqa: E402
from scripts.supervised_rerank_common import (  # noqa: E402
    annotated_train_split, dev_metrics, load_inputs, proxy_hit, rerank, weak_labels,
)
from src.utils.io import write_json, write_jsonl  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.runs import resolve_run_directory, sha256_file, write_run_manifest  # noqa: E402


def cache_embeddings(model, rows, directory: Path, top_n: int, batch_size: int, chunk_claims: int, prefix: str):
    directory.mkdir(parents=True, exist_ok=True)
    chunks = []
    with progress(total=len(rows), description=f"Embed {directory.name}", unit="claim") as bar:
        for start in range(0, len(rows), chunk_claims):
            end = min(start + chunk_claims, len(rows))
            path = directory / f"{start:05d}_{end:05d}.npz"
            current = rows[start:end]
            if not path.is_file():
                queries = model.encode(
                    [prefix + str(row["claim"]) for row in current], batch_size=batch_size,
                    normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
                ).astype(np.float16)
                texts = [str(item.get("text") or "") for row in current for item in row.get("retrieved", [])[:top_n]]
                passages = model.encode(
                    texts, batch_size=batch_size, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False,
                ).astype(np.float16) if texts else np.empty((0, queries.shape[1]), np.float16)
                padded = np.zeros((len(current), top_n, queries.shape[1]), np.float16)
                mask = np.zeros((len(current), top_n), bool)
                offset = 0
                for index, row in enumerate(current):
                    count = len(row.get("retrieved", [])[:top_n])
                    padded[index, :count] = passages[offset:offset + count]
                    mask[index, :count] = True
                    offset += count
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, queries=queries, passages=padded, mask=mask)
                temporary.replace(path)
            with np.load(path) as saved:
                if saved["queries"].shape[0] != len(current) or saved["passages"].shape[:2] != (len(current), top_n):
                    raise ValueError(f"Embedding cache shape mismatch: {path}")
                if not np.array_equal(saved["mask"].sum(axis=1), [len(row.get("retrieved", [])[:top_n]) for row in current]):
                    raise ValueError(f"Embedding cache candidate counts mismatch: {path}")
            chunks.append(path)
            bar.update(len(current))
    with_arrays = [np.load(path) for path in chunks]
    try:
        return tuple(np.concatenate([saved[key] for saved in with_arrays]) for key in ("queries", "passages", "mask"))
    finally:
        for saved in with_arrays:
            saved.close()


def fit_projection(queries, passages, mask, labels, indices, lr: float, epochs: int, seed: int, device: str):
    import torch

    torch.manual_seed(seed)
    dim = queries.shape[-1]
    layer = torch.nn.Linear(dim, dim, bias=False, device=device)
    torch.nn.init.eye_(layer.weight)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=lr, weight_decay=0.01)
    eligible = np.asarray([index for index in indices if labels[index].any()], dtype=int)
    if not len(eligible):
        raise ValueError("No URL-proxy positives among BM25 top-N training candidates")
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(eligible)
        with progress(total=len(order), description=f"Projection epoch {epoch + 1}/{epochs}", unit="claim", leave=False) as bar:
            for start in range(0, len(order), 64):
                batch = order[start:start + 64]
                q = torch.as_tensor(queries[batch].astype(np.float32), device=device)
                p = torch.as_tensor(passages[batch].astype(np.float32), device=device)
                valid = torch.as_tensor(mask[batch], device=device)
                positive = torch.as_tensor(labels[batch], device=device)
                q_projected = torch.nn.functional.normalize(layer(q), dim=-1)
                p_projected = torch.nn.functional.normalize(layer(p), dim=-1)
                logits = (q_projected[:, None, :] * p_projected).sum(-1) / 0.05
                logits = logits.masked_fill(~valid, -1e4)
                selected = logits.masked_fill(~positive, -1e4)
                loss = (torch.logsumexp(logits, dim=1) - torch.logsumexp(selected, dim=1)).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                bar.update(len(batch))
    return layer.cpu().eval()


def projected_scores(layer, queries, passages, mask):
    import torch

    with torch.inference_mode():
        q = torch.nn.functional.normalize(layer(torch.as_tensor(queries.astype(np.float32))), dim=-1)
        p = torch.nn.functional.normalize(layer(torch.as_tensor(passages.astype(np.float32))), dim=-1)
        scores = (q[:, None, :] * p).sum(-1).numpy()
    return [scores[index, :int(mask[index].sum())].tolist() for index in range(len(scores))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-claims", type=Path, required=True)
    parser.add_argument("--train-rankings", type=Path, required=True)
    parser.add_argument("--dev-rankings", type=Path, required=True)
    parser.add_argument("--dev-qrels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--query-prefix", default=DEFAULT_QUERY_PREFIX)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-claims", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=718)
    args = parser.parse_args()
    if min(args.top_n, args.batch_size, args.chunk_claims, args.max_seq_length) < 1:
        parser.error("Sizes must be positive")
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select a Colab GPU runtime")
    inputs = (args.train_claims, args.train_rankings, args.dev_rankings, args.dev_qrels)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = resolve_run_directory(args.output_dir, args.run_name)
    output.mkdir(parents=True, exist_ok=True)
    config = {"script": "train_projection_reranker", "input_sha256": [sha256_file(path) for path in inputs],
              "model": args.model, "revision": args.revision, "query_prefix": args.query_prefix,
              "top_n": args.top_n, "batch_size": args.batch_size, "chunk_claims": args.chunk_claims,
              "max_seq_length": args.max_seq_length, "seed": args.seed, "device": args.device,
              "lr_grid": [0.001, 0.003], "epoch_grid": [2, 4], "blend_grid": [0.0, 0.25, 0.5, 0.75, 1.0]}
    config_path = output / "config.json"
    if config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Run settings changed; choose a new run name")
    final_path, metrics_path = output / "dev_rankings.jsonl", output / "dev_metrics.json"
    if final_path.is_file() and metrics_path.is_file():
        print(metrics_path.read_text(encoding="utf-8"))
        return
    write_json(config, config_path)
    train, dev, claims = load_inputs(*inputs[:3])
    fit_indices, val_indices = annotated_train_split(train, claims, args.seed, 0.1)
    revision_path = output / "encoder_revision.json"
    prior_revision = json.loads(revision_path.read_text(encoding="utf-8"))["resolved_revision"] if revision_path.is_file() else None
    print(f"Loading frozen encoder; only saved top-{args.top_n} candidates are embedded.", flush=True)
    model, resolved = load_embedding_model(args.model, args.revision or prior_revision, args.device, args.max_seq_length)
    if prior_revision and resolved != prior_revision:
        raise ValueError("Encoder revision changed across resumed embedding cache")
    if resolved:
        write_json({"resolved_revision": resolved}, revision_path)
    tq, tp, tm = cache_embeddings(model, train, output / "cache_train", args.top_n, args.batch_size, args.chunk_claims, args.query_prefix)
    dq, dp, dm = cache_embeddings(model, dev, output / "cache_dev", args.top_n, args.batch_size, args.chunk_claims, args.query_prefix)
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    labels = np.zeros(tm.shape, bool)
    for index, row in enumerate(train):
        weak = weak_labels(row, claims[str(row["claim_id"])], args.top_n)
        labels[index, :len(weak)] = weak
    trials = []
    val_rows = [train[index] for index in val_indices]
    for lr in config["lr_grid"]:
        for epochs in config["epoch_grid"]:
            layer = fit_projection(tq, tp, tm, labels, fit_indices, lr, epochs, args.seed, args.device)
            scores = projected_scores(layer, tq[val_indices], tp[val_indices], tm[val_indices])
            for blend in config["blend_grid"]:
                ranked = [rerank(row, score, blend, args.top_n, "projection_score") for row, score in zip(val_rows, scores, strict=True)]
                trials.append({"lr": lr, "epochs": epochs, "blend": blend,
                               "validation_hit_at_3": proxy_hit(ranked, claims, args.top_n)})
            print(f"Finished projection lr={lr}, epochs={epochs}", flush=True)
    selected = sorted(trials, key=lambda trial: (-trial["validation_hit_at_3"], trial["epochs"], trial["lr"], trial["blend"]))[0]
    write_json({"train_only_validation_claim_count": len(val_indices), "trials": trials, "selected": selected}, output / "selection.json")
    annotated = sorted(fit_indices + val_indices)
    layer = fit_projection(tq, tp, tm, labels, annotated, selected["lr"], selected["epochs"], args.seed, args.device)
    temporary_model = output / "projection.pt.tmp"
    torch.save(layer.state_dict(), temporary_model)
    temporary_model.replace(output / "projection.pt")
    dev_scores = projected_scores(layer, dq, dp, dm)
    ranked = [rerank(row, score, selected["blend"], args.top_n, "projection_score") for row, score in zip(dev, dev_scores, strict=True)]
    temporary = final_path.with_suffix(".jsonl.tmp")
    write_jsonl(ranked, temporary)
    temporary.replace(final_path)
    report = {**config, "selected": selected, "dev": dev_metrics(ranked, args.dev_qrels),
              "bm25_reference": dev_metrics(dev, args.dev_qrels), "rankings_path": str(final_path),
              "warning": "URL-proxy weak labels; BM25 top-N ceiling applies. Dev labels were not used in training or selection."}
    write_json(report, metrics_path)
    write_run_manifest(output / "manifest.json", project_root=PROJECT_ROOT, config=report,
                       input_paths=dict(zip(("train_claims", "train_rankings", "dev_rankings", "dev_qrels"), inputs, strict=True)))
    print(json.dumps({"selected": selected, "dev": report["dev"], "bm25_reference": report["bm25_reference"]}, indent=2))


if __name__ == "__main__":
    main()
