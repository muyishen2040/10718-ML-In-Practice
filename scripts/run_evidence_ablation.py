"""Test a frozen verifier with original, missing, and mismatched evidence.

This is a diagnostic, not a deployable verifier. The same final model and
official-dev claims are used in each condition; no labels or qrels enter the
model inputs. Interrupted inference can resume from append-only JSONL files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_tfidf_logreg import attach_retrieved_evidence  # noqa: E402
from src.data.averitec import CANONICAL_LABELS  # noqa: E402
from src.evaluation.classification import evaluate_predictions  # noqa: E402
from src.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.runs import resolve_run_directory, sha256_file, write_run_manifest  # noqa: E402


def evidence_variants(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Preserve the missing-pool claim and rotate only nonempty evidence pools."""
    ids = [str(row["claim_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate claim IDs")
    nonempty = sorted((row for row in records if row.get("retrieved_evidence")), key=lambda row: str(row["claim_id"]))
    if len(nonempty) < 2:
        raise ValueError("Need at least two nonempty claim evidence pools for a swap diagnostic")
    donors = {
        str(row["claim_id"]): nonempty[(index + 1) % len(nonempty)]
        for index, row in enumerate(nonempty)
    }
    blank = [{**row, "retrieved_evidence": []} for row in records]
    swapped = []
    for row in records:
        claim_id = str(row["claim_id"])
        donor = donors.get(claim_id)
        swapped.append({
            **row,
            "retrieved_evidence": (donor.get("retrieved_evidence") or [])[:3] if donor else [],
            "evidence_donor_claim_id": str(donor["claim_id"]) if donor else None,
        })
    return {"blank": blank, "swapped": swapped}


def load_partial_predictions(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Discard one torn final line, but never silently accept duplicate IDs."""
    if not path.is_file():
        return {}
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    last = max((index for index, line in enumerate(lines) if line.strip()), default=-1)
    found: dict[str, dict[str, Any]] = {}
    offset = 0
    for index, raw in enumerate(lines):
        start = offset
        offset += len(raw)
        if not raw.strip():
            continue
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if index != last:
                raise ValueError(f"Corrupt checkpoint before final line: {path}") from error
            path.write_bytes(data[:start])
            print(f"Removed interrupted final checkpoint line from {path}", flush=True)
            break
        claim_id = str(row.get("claim_id") or "")
        if claim_id not in expected_ids or claim_id in found:
            raise ValueError(f"Unexpected or duplicate checkpoint claim ID {claim_id!r} in {path}")
        if row.get("predicted_label") not in CANONICAL_LABELS or row.get("true_label") not in CANONICAL_LABELS:
            raise ValueError(f"Malformed checkpoint prediction for {claim_id!r}")
        found[claim_id] = row
    if data and not data.endswith(b"\n") and len(found) == sum(bool(line.strip()) for line in lines):
        # A complete final JSON record can still lack the newline required
        # before the next append. Preserve it and add only the separator.
        with path.open("ab") as handle:
            handle.write(b"\n")
    return found


def comparison(original: list[dict[str, Any]], altered: list[dict[str, Any]]) -> dict[str, int | float]:
    baseline = {str(row["claim_id"]): row for row in original}
    changed = lost = fixed = 0
    for row in altered:
        old = baseline[str(row["claim_id"])]
        if old["true_label"] != row["true_label"]:
            raise ValueError(f"Gold label changed for {row['claim_id']}")
        changed += old["predicted_label"] != row["predicted_label"]
        lost += old["predicted_label"] == old["true_label"] and row["predicted_label"] != row["true_label"]
        fixed += old["predicted_label"] != old["true_label"] and row["predicted_label"] == row["true_label"]
    return {
        "changed_prediction_count": changed,
        "changed_prediction_fraction": changed / len(altered),
        "originally_correct_now_wrong": lost,
        "originally_wrong_now_correct": fixed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--rankings", required=True, type=Path)
    parser.add_argument("--original-predictions", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-run-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.checkpoint_every) < 1:
        parser.error("Batch size and checkpoint interval must be positive")
    for path in (args.claims, args.rankings, args.original_predictions, args.model_run_config, args.model_dir / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    weight_files = sorted((*args.model_dir.glob("*.safetensors"), *args.model_dir.glob("pytorch_model*.bin")))
    if not weight_files:
        raise FileNotFoundError(f"No model weights found in {args.model_dir}")
    output = resolve_run_directory(args.output_dir, args.run_name)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "script": "run_evidence_ablation",
        "claims_sha256": sha256_file(args.claims),
        "rankings_sha256": sha256_file(args.rankings),
        "original_predictions_sha256": sha256_file(args.original_predictions),
        "model_run_config_sha256": sha256_file(args.model_run_config),
        "model_config_sha256": sha256_file(args.model_dir / "config.json"),
        "model_weights_sha256": {path.name: sha256_file(path) for path in weight_files},
        "model_dir": str(args.model_dir.resolve()),
        "batch_size": args.batch_size,
        "checkpoint_every": args.checkpoint_every,
        "device": args.device,
        "swap_policy": "next nonempty claim in sorted claim-ID order; missing-pool claims remain empty",
    }
    config_path = output / "ablation_config.json"
    prior = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
    if prior is not None and prior != config:
        raise ValueError("Ablation inputs/settings changed; use a new run name")
    report_path = output / "ablation_report.json"
    if report_path.is_file() and (output / "ablation_manifest.json").is_file() and all(
        (output / f"{name}_predictions.jsonl").is_file() for name in ("blank", "swapped")
    ):
        print(report_path.read_text(encoding="utf-8"))
        return
    write_json(config, config_path)

    records = attach_retrieved_evidence(read_jsonl(args.claims), args.rankings)
    by_id = {str(row["claim_id"]): row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate dev claim IDs")
    original = read_jsonl(args.original_predictions)
    original_by_id = {str(row["claim_id"]): row for row in original}
    if len(original_by_id) != len(original) or set(original_by_id) != set(by_id):
        raise ValueError("Original predictions must cover the exact dev claim set")
    for row in records:
        if original_by_id[str(row["claim_id"])]["true_label"] != row["label"]:
            raise ValueError(f"Original prediction label mismatch for {row['claim_id']}")
    variants = evidence_variants(records)
    model_settings = json.loads(args.model_run_config.read_text(encoding="utf-8"))

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from scripts.train_transformer_verifier import compose_text

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select a Colab GPU runtime")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(args.device).eval()
    if int(model.config.num_labels) != len(CANONICAL_LABELS):
        raise ValueError("Final model has an unexpected number of verdict labels")
    if any(str(model.config.id2label.get(index)) != label for index, label in enumerate(CANONICAL_LABELS)):
        raise ValueError("Final model label order differs from the original verifier")
    results: dict[str, Any] = {}
    for name, rows in variants.items():
        final_path = output / f"{name}_predictions.jsonl"
        partial_path = output / f"{name}_predictions.partial.jsonl"
        if final_path.is_file():
            predictions = read_jsonl(final_path)
            if len(predictions) != len(rows) or {r["claim_id"] for r in predictions} != set(by_id):
                raise ValueError(f"Incomplete or duplicate final predictions: {final_path}")
        else:
            if partial_path.is_file() and not args.resume:
                raise ValueError(f"Partial checkpoint exists; rerun with --resume: {partial_path}")
            completed = load_partial_predictions(partial_path, set(by_id)) if args.resume else {}
            pending: list[dict[str, Any]] = []

            def flush() -> None:
                nonlocal pending
                batch = pending
                pending = []
                if batch:
                    with partial_path.open("a", encoding="utf-8", newline="\n") as handle:
                        for item in batch:
                            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                        handle.flush()

            with progress(total=len(rows), initial=len(completed), description=f"Ablation {name}", unit="claim") as bar:
                try:
                    for start in range(0, len(rows), args.batch_size):
                        batch_rows = [row for row in rows[start : start + args.batch_size] if row["claim_id"] not in completed]
                        if not batch_rows:
                            continue
                        texts = [compose_text(
                            row, tokenizer,
                            int(model_settings["claim_token_budget"]),
                            int(model_settings["evidence_token_budget_per_passage"]),
                        ) for row in batch_rows]
                        encoded = tokenizer(
                            texts, padding=True, truncation=True,
                            max_length=int(model_settings["max_length"]), return_tensors="pt",
                        )
                        encoded = {key: value.to(args.device) for key, value in encoded.items()}
                        with torch.inference_mode():
                            probabilities = torch.softmax(model(**encoded).logits.float(), dim=-1).cpu().tolist()
                        for row, values in zip(batch_rows, probabilities, strict=True):
                            prediction = {
                                "claim_id": str(row["claim_id"]),
                                "true_label": str(row["label"]),
                                "predicted_label": CANONICAL_LABELS[max(range(len(values)), key=values.__getitem__)],
                                "probabilities": {label: float(values[index]) for index, label in enumerate(CANONICAL_LABELS)},
                                "evidence_item_count": min(3, len(row.get("retrieved_evidence") or [])),
                                "evidence_donor_claim_id": row.get("evidence_donor_claim_id"),
                            }
                            completed[prediction["claim_id"]] = prediction
                            pending.append(prediction)
                        if len(pending) >= args.checkpoint_every:
                            flush()
                        bar.update(len(batch_rows))
                finally:
                    flush()
            if len(completed) != len(rows):
                raise RuntimeError(f"Only {len(completed)}/{len(rows)} {name} predictions completed")
            predictions = [completed[str(row["claim_id"])] for row in rows]
            temporary = final_path.with_name(f"{final_path.name}.tmp")
            write_jsonl(predictions, temporary)
            temporary.replace(final_path)
        results[name] = {
            "metrics": evaluate_predictions(predictions),
            "versus_original": comparison(original, predictions),
            "predictions_path": str(final_path),
        }
    report = {
        **config,
        "claim_count": len(records),
        "original_metrics_recomputed": evaluate_predictions(original),
        "variants": results,
        "interpretation_warning": "Blank/swapped inputs are distribution-shift diagnostics; do not interpret score differences as a causal estimate of retrieval quality.",
    }
    write_json(report, report_path)
    write_run_manifest(
        output / "ablation_manifest.json", project_root=PROJECT_ROOT, config=report,
        input_paths={"claims": args.claims, "rankings": args.rankings, "original_predictions": args.original_predictions,
                     "model_config": args.model_dir / "config.json", "training_run_config": args.model_run_config},
    )
    print(json.dumps({
        "original_macro_f1": report["original_metrics_recomputed"]["macro_f1"],
        "variants": {name: {"macro_f1": item["metrics"]["macro_f1"], **item["versus_original"]} for name, item in results.items()},
        "report_path": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
