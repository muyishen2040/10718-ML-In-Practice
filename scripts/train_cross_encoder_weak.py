"""Fine-tune a small cross-encoder on train-only source or answer-aligned labels.

Selection uses a train-only claim holdout. The full-fit model reranks dev;
two held-out-fold models build leakage-safe *train* rankings for a matched
verifier. Interrupted training and scoring resume from Drive checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bm25 import append_checkpoint_rows, load_checkpoint_rankings  # noqa: E402
from scripts.supervised_rerank_common import (  # noqa: E402
    annotated_train_split, dev_metrics, load_inputs, proxy_hit, rerank, weak_labels,
)
from src.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.runs import resolve_run_directory, sha256_file, write_run_manifest  # noqa: E402


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def ensure_checkpoint_newline(path: Path) -> None:
    """Allow append after a valid final JSON row that lacks its newline."""
    if not path.is_file() or not path.stat().st_size:
        return
    with path.open("rb+") as handle:
        handle.seek(-1, 2)
        if handle.read(1) != b"\n":
            handle.write(b"\n")


def training_pairs(rows, claims, indices, top_n: int, negatives: int, positive_ids=None):
    """Select one train positive and BM25-hard negatives per eligible claim."""
    pairs = []
    for index in indices:
        row = rows[index]
        candidates = row.get("retrieved", [])[:top_n]
        labels = weak_labels(row, claims[str(row["claim_id"])], top_n)
        if positive_ids is None:
            positive = next((item for item, label in zip(candidates, labels, strict=True) if label), None)
        else:
            selected_id = positive_ids.get(str(row["claim_id"]))
            positive = next((item for item in candidates if str(item["passage_id"]) == selected_id), None)
        if positive is None:
            continue
        pairs.append((str(row["claim"]), str(positive.get("text") or ""), 1.0))
        selected_negatives = 0
        for item, label in zip(candidates, labels, strict=True):
            if not label and str(item.get("text") or "").strip():
                pairs.append((str(row["claim"]), str(item["text"]), 0.0))
                selected_negatives += 1
                if selected_negatives == negatives:
                    break
    if not pairs or {pair[2] for pair in pairs} != {0.0, 1.0}:
        raise ValueError("Need weak positive and negative train pairs")
    return pairs


def load_base(model_name: str, revision: str | None, device: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
    if int(model.config.num_labels) != 1:
        raise ValueError("Expected a one-logit cross-encoder relevance model")
    resolved = getattr(model.config, "_commit_hash", None)
    return tokenizer, model.to(device), resolved


def fit_model(rows, claims, indices, directory: Path, args, revision: str | None, positive_ids=None):
    """Resume exact next minibatch from a periodically saved optimizer state."""
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / "model" / "config.json"
    if complete.is_file():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(directory / "model", use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(directory / "model").to(args.device).eval()
        return tokenizer, model
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    tokenizer, model, _ = load_base(args.model, revision, args.device)
    pairs = training_pairs(rows, claims, indices, args.top_n, args.negatives, positive_ids)
    print(f"Training {directory.name}: {len(indices)} claims, {len(pairs)} labeled pairs", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    checkpoint = directory / "state.pt"
    start_epoch = cursor = step = 0
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # Optimizer tensors must return to the chosen training device.
        for group in optimizer.state.values():
            for key, value in group.items():
                if key != "step" and isinstance(value, torch.Tensor):
                    group[key] = value.to(args.device)
        start_epoch, cursor, step = state["epoch"], state["cursor"], state["step"]
        torch.set_rng_state(state["torch_rng"])
        if args.device.startswith("cuda") and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def save_state(epoch: int, offset: int):
        temporary = directory / "state.pt.tmp"
        torch.save({"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(), "epoch": epoch, "cursor": offset,
                    "step": step, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else None}, temporary)
        temporary.replace(checkpoint)

    for epoch in range(start_epoch, args.epochs):
        order = np.random.default_rng(args.seed + epoch).permutation(len(pairs))
        position = cursor if epoch == start_epoch else 0
        model.train()
        with progress(total=len(order), initial=position, description=f"{directory.name} epoch {epoch + 1}/{args.epochs}", unit="pair") as bar:
            try:
                for offset in range(position, len(order), args.batch_size):
                    batch = [pairs[int(index)] for index in order[offset:offset + args.batch_size]]
                    encoded = tokenizer(
                        [pair[0] for pair in batch], [pair[1] for pair in batch],
                        padding=True, truncation=True, max_length=args.max_seq_length, return_tensors="pt",
                    ).to(args.device)
                    targets = torch.tensor([pair[2] for pair in batch], dtype=torch.float32, device=args.device)
                    logits = model(**encoded).logits.reshape(-1)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, targets, pos_weight=torch.tensor(float(args.negatives), device=args.device),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    step += 1
                    next_offset = min(offset + len(batch), len(order))
                    bar.update(len(batch))
                    if step % args.checkpoint_steps == 0:
                        save_state(epoch, next_offset)
            except BaseException:
                # A KeyboardInterrupt between optimizer and checkpoint still
                # resumes from the previous committed state, not partial files.
                raise
        save_state(epoch + 1, 0)
        cursor = 0
    model.eval()
    temp = directory / "model.tmp"
    temp.mkdir(exist_ok=True)
    model.save_pretrained(temp, safe_serialization=True)
    tokenizer.save_pretrained(temp)
    temp.replace(directory / "model")
    return tokenizer, model


def raw_scores(row, tokenizer, model, args):
    import torch

    candidates = row.get("retrieved", [])[:args.top_n]
    scores = []
    for start in range(0, len(candidates), args.inference_batch_size):
        batch = candidates[start:start + args.inference_batch_size]
        encoded = tokenizer(
            [str(row["claim"])] * len(batch), [str(item.get("text") or "") for item in batch],
            padding=True, truncation=True, max_length=args.max_seq_length, return_tensors="pt",
        ).to(args.device)
        with torch.inference_mode():
            scores.extend(model(**encoded).logits.reshape(-1).float().cpu().tolist())
    return scores


def score_rows(rows, tokenizer, model, args, blend: float, checkpoint: Path, label: str):
    known = {str(row["claim_id"]) for row in rows}
    completed = load_checkpoint_rankings(checkpoint, known_claim_ids=known)
    ensure_checkpoint_newline(checkpoint)
    pending = []
    model.eval()
    with progress(total=len(rows), initial=len(completed), description=label, unit="claim") as bar:
        try:
            for row in rows:
                claim_id = str(row["claim_id"])
                if claim_id in completed:
                    continue
                ranked = rerank(row, raw_scores(row, tokenizer, model, args), blend, args.top_n, "cross_encoder_score")
                completed[claim_id] = ranked
                pending.append(ranked)
                if len(pending) >= args.checkpoint_claims:
                    append_checkpoint_rows(checkpoint, pending)
                    pending = []
                bar.update(1)
        finally:
            if pending:
                append_checkpoint_rows(checkpoint, pending)
    return [completed[str(row["claim_id"])] for row in rows]


def aligned_passage_hit(rows, positive_ids: dict[str, str], k: int = 3) -> tuple[float, int]:
    eligible = [row for row in rows if str(row["claim_id"]) in positive_ids]
    if not eligible:
        raise ValueError("No aligned positive passages in the train-only validation set")
    hits = sum(positive_ids[str(row["claim_id"])] in
               {str(item["passage_id"]) for item in row.get("retrieved", [])[:k]} for row in eligible)
    return hits / len(eligible), len(eligible)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-claims", type=Path, required=True)
    parser.add_argument("--train-rankings", type=Path, required=True)
    parser.add_argument("--dev-rankings", type=Path, required=True)
    parser.add_argument("--dev-qrels", type=Path, required=True)
    parser.add_argument("--positive-labels", type=Path, help="Optional train-only answer-aligned passage IDs; replaces first URL-matched positive.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--negatives", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--checkpoint-steps", type=int, default=250)
    parser.add_argument("--checkpoint-claims", type=int, default=20)
    parser.add_argument("--oof-folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=718)
    parser.add_argument("--dev-only", action="store_true", help="Fit and score dev first; defer expensive out-of-fold train rankings.")
    args = parser.parse_args()
    if min(args.top_n, args.negatives, args.epochs, args.batch_size, args.inference_batch_size,
           args.max_seq_length, args.checkpoint_steps, args.checkpoint_claims, args.oof_folds) < 1:
        parser.error("Sizes must be positive")
    if args.oof_folds < 2:
        parser.error("At least two out-of-fold models are required")
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select a Colab GPU runtime")
    inputs = (args.train_claims, args.train_rankings, args.dev_rankings, args.dev_qrels)
    for path in (*inputs, *((args.positive_labels,) if args.positive_labels else ())):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = resolve_run_directory(args.output_dir, args.run_name)
    output.mkdir(parents=True, exist_ok=True)
    config = {"script": "train_cross_encoder_weak", "input_sha256": [sha256_file(path) for path in inputs],
              "model": args.model, "revision": args.revision, "device": args.device, "top_n": args.top_n,
              "negatives": args.negatives, "learning_rate": args.learning_rate, "epochs": args.epochs,
              "batch_size": args.batch_size, "inference_batch_size": args.inference_batch_size,
              "max_seq_length": args.max_seq_length, "checkpoint_steps": args.checkpoint_steps,
              "checkpoint_claims": args.checkpoint_claims, "oof_folds": args.oof_folds, "seed": args.seed,
              "blend_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
              "positive_labels_sha256": sha256_file(args.positive_labels) if args.positive_labels else None}
    config_path = output / "config.json"
    if config_path.is_file():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        # Existing URL-label runs predate the optional answer-alignment field.
        saved_config.setdefault("positive_labels_sha256", None)
        if saved_config != config:
            raise ValueError("Run settings changed; choose a new run name")
    dev_path, train_path, metrics_path = (output / name for name in ("dev_rankings.jsonl", "train_oof_rankings.jsonl", "dev_metrics.json"))
    if args.dev_only and dev_path.is_file() and metrics_path.is_file():
        print(metrics_path.read_text(encoding="utf-8"))
        return
    if dev_path.is_file() and train_path.is_file() and metrics_path.is_file():
        print(metrics_path.read_text(encoding="utf-8"))
        return
    write_json(config, config_path)
    train, dev, claims = load_inputs(*inputs[:3])
    positive_ids = None
    if args.positive_labels:
        aligned = read_jsonl(args.positive_labels)
        if len({str(row["claim_id"]) for row in aligned}) != len(aligned) or {str(row["claim_id"]) for row in aligned} != {str(row["claim_id"]) for row in train}:
            raise ValueError("Answer-alignment rows must cover the exact unique train claim IDs")
        positive_ids = {str(row["claim_id"]): str(row["positive_passage_id"]) for row in aligned if row.get("positive_passage_id")}
        for row in train:
            selected_id = positive_ids.get(str(row["claim_id"]))
            if selected_id is None:
                continue
            candidates = row.get("retrieved", [])[:args.top_n]
            gold = weak_labels(row, claims[str(row["claim_id"])], args.top_n)
            if not any(str(item["passage_id"]) == selected_id and label for item, label in zip(candidates, gold, strict=True)):
                raise ValueError(f"Aligned positive is absent from annotated source candidates: {row['claim_id']}")
    fit, val = annotated_train_split(train, claims, args.seed, 0.1)
    revision_path = output / "base_revision.json"
    saved_revision = json.loads(revision_path.read_text(encoding="utf-8"))["revision"] if revision_path.is_file() else None
    revision = args.revision or saved_revision
    if not revision:
        _, initial, revision = load_base(args.model, None, args.device)
        del initial
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None
        write_json({"revision": revision}, revision_path)
    selection_path = output / "selection.json"
    if selection_path.is_file():
        selection_report = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_report.get("train_only_validation_claim_count") != len(val):
            raise ValueError("Saved selection set does not match the current train split")
        selected = selection_report["selected"]
    else:
        tokenizer, selector = fit_model(train, claims, fit, output / "selection_fit", args, revision, positive_ids)
        val_rows = [train[index] for index in val]
        val_scores = [raw_scores(row, tokenizer, selector, args) for row in progress_iter(val_rows, "Validate reranker")]
        trials = []
        for blend in config["blend_grid"]:
            ranked = [rerank(row, score, blend, args.top_n, "cross_encoder_score") for row, score in zip(val_rows, val_scores, strict=True)]
            trial = {"blend": blend, "validation_url_proxy_hit_at_3": proxy_hit(ranked, claims, args.top_n)}
            if positive_ids is not None:
                trial["validation_aligned_passage_hit_at_3"], trial["validation_aligned_claim_count"] = aligned_passage_hit(ranked, positive_ids)
            trials.append(trial)
        if positive_ids is None:
            selected = sorted(trials, key=lambda trial: (-trial["validation_url_proxy_hit_at_3"], trial["blend"]))[0]
        else:
            selected = sorted(trials, key=lambda trial: (-trial["validation_aligned_passage_hit_at_3"],
                                                          -trial["validation_url_proxy_hit_at_3"], trial["blend"]))[0]
        write_json({"train_only_validation_claim_count": len(val), "trials": trials, "selected": selected}, selection_path)
        del selector
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    # Run the affordable dev pilot before optional OOF scoring of all train claims.
    if not dev_path.is_file():
        full_indices = sorted(fit + val)
        full_tokenizer, full_model = fit_model(train, claims, full_indices, output / "full_fit", args, revision, positive_ids)
        dev_ranked = score_rows(dev, full_tokenizer, full_model, args, selected["blend"],
                                output / "dev_rankings.partial.jsonl", "Rerank dev")
        temporary = dev_path.with_suffix(".jsonl.tmp")
        write_jsonl(dev_ranked, temporary)
        temporary.replace(dev_path)
        del full_model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        dev_ranked = read_jsonl(dev_path)

    def write_report(oof_complete: bool) -> dict:
        report = {**config, "selected": selected, "dev": dev_metrics(dev_ranked, args.dev_qrels),
                  "bm25_reference": dev_metrics(dev, args.dev_qrels), "dev_rankings": str(dev_path),
                  "train_oof_rankings": str(train_path) if oof_complete else None,
                  "train_oof_fold_count": args.oof_folds if oof_complete else None,
                  "answer_aligned_positive_claim_count": len(positive_ids) if positive_ids is not None else None,
                  "oof_complete": oof_complete,
                  "warning": "Train-only silver passage alignment; dev qrels are URL proxies. "
                             "Dev labels were not used for selection. Matched train evidence requires OOF rankings."}
        write_json(report, metrics_path)
        input_paths = dict(zip(("train_claims", "train_rankings", "dev_rankings", "dev_qrels"), inputs, strict=True))
        if args.positive_labels:
            input_paths["positive_labels"] = args.positive_labels
        write_run_manifest(output / "manifest.json", project_root=PROJECT_ROOT, config=report,
                           input_paths=input_paths)
        return report

    if args.dev_only:
        report = write_report(oof_complete=train_path.is_file())
        print(json.dumps({"selected": selected, "dev": report["dev"], "bm25_reference": report["bm25_reference"],
                          "oof_complete": report["oof_complete"]}, indent=2))
        return
    # Every train claim is scored by a model that was fitted on *other* claims.
    global_oof_checkpoint = output / "train_oof_rankings.partial.jsonl"
    oof_completed = load_checkpoint_rankings(global_oof_checkpoint, known_claim_ids={str(row["claim_id"]) for row in train})
    ensure_checkpoint_newline(global_oof_checkpoint)
    fold_assignment = {}
    for fold, (fit_fold, heldout) in enumerate(KFold(n_splits=args.oof_folds, shuffle=True, random_state=args.seed).split(train)):
        for index in heldout:
            fold_assignment[str(train[int(index)]["claim_id"])] = fold
        heldout_rows = [train[int(index)] for index in heldout]
        if all(str(row["claim_id"]) in oof_completed for row in heldout_rows):
            continue
        fold_tokenizer, fold_model = fit_model(train, claims, list(map(int, fit_fold)), output / f"oof_fold_{fold}", args, revision, positive_ids)
        # Separate fold checkpoints avoid ambiguous claim ownership on resume.
        fold_rows = score_rows(heldout_rows, fold_tokenizer, fold_model, args, selected["blend"],
                               output / f"oof_fold_{fold}.partial.jsonl", f"OOF fold {fold}")
        newly_completed = []
        for row in fold_rows:
            row["oof_fold"] = fold
            if str(row["claim_id"]) not in oof_completed:
                oof_completed[str(row["claim_id"])] = row
                newly_completed.append(row)
        if newly_completed:
            append_checkpoint_rows(global_oof_checkpoint, newly_completed)
        del fold_model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if len(oof_completed) != len(train):
        raise RuntimeError(f"OOF train rankings incomplete: {len(oof_completed)}/{len(train)}")
    for row in train:
        claim_id = str(row["claim_id"])
        if oof_completed[claim_id].get("oof_fold") != fold_assignment[claim_id]:
            raise ValueError(f"OOF fold provenance mismatch for {claim_id}")
    if not train_path.is_file():
        temporary = train_path.with_suffix(".jsonl.tmp")
        write_jsonl([oof_completed[str(row["claim_id"])] for row in train], temporary)
        temporary.replace(train_path)
    report = write_report(oof_complete=True)
    print(json.dumps({"selected": selected, "dev": report["dev"], "bm25_reference": report["bm25_reference"]}, indent=2))


def progress_iter(rows, label):
    with progress(total=len(rows), description=label, unit="claim") as bar:
        for row in rows:
            yield row
            bar.update(1)


if __name__ == "__main__":
    main()
