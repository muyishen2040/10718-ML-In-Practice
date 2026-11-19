"""Mechanics check for a saved pair-reader run (Teammate B brief, step 1).

Reads a completed run directory without modifying it and records label order,
tokenization settings, weight precision, the selected epoch versus the refit
schedule, evidence IDs and recomputed metrics. Optionally re-runs frozen
inference on a few dev claims and compares probabilities with the saved ones.

This is a reproduction of recorded mechanics, not an independent validation
result: the saved model was selected/refit on all official train claims and
official dev has been inspected repeatedly.
"""
from __future__ import annotations

import json
import math
import struct
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.averitec import CANONICAL_LABELS
from src.evaluation.classification import evaluate_predictions
from src.utils.io import read_jsonl, write_json
from src.utils.runs import sha256_file


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def safetensors_dtypes(path: Path) -> dict[str, int]:
    """Count tensor dtypes from a safetensors header without importing torch."""
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    return dict(Counter(v["dtype"] for k, v in header.items() if k != "__metadata__"))


def _check(checks: list[dict], name: str, passed: bool | None, detail: Any) -> None:
    checks.append({"check": name, "status": "skipped" if passed is None else "pass" if passed else "FAIL", "detail": detail})


def inspect_run(run: Path, *, data_root: Path | None = None, prepared: Path | None = None) -> dict[str, Any]:
    checks: list[dict] = []
    run_config = _json(run / "reader_run.json")
    selection = _json(run / "model_selection.json")
    final_fit = _json(run / "final_fit_complete.json")
    final = run / "final_model"
    model_config = _json(final / "config.json")
    tokenizer_config = _json(final / "tokenizer_config.json")
    _check(checks, "run files present", all(x is not None for x in (run_config, selection, final_fit, model_config)),
           {"reader_run.json": run_config is not None, "model_selection.json": selection is not None,
            "final_fit_complete.json": final_fit is not None, "final_model/config.json": model_config is not None})

    summary: dict[str, Any] = {"run": str(run), "kind": "mechanics_check_not_independent_validation"}
    if run_config:
        summary["run_config"] = {k: run_config.get(k) for k in (
            "mode", "learning_rates", "epochs", "batch_size", "accumulation", "seed", "validation_policy",
            "validation_seed", "distillation_weight", "loss_normalization", "freeze_word_embeddings", "device", "release")}
        summary["mixed_precision_training"] = "fp16 autocast" if str(run_config.get("device", "")).startswith("cuda") else "fp32"
    if model_config:
        id2label = [model_config.get("id2label", {}).get(str(i)) for i in range(4)]
        _check(checks, "verdict label order", id2label == list(CANONICAL_LABELS), id2label)
        nli = model_config.get("nli_labels")
        _check(checks, "NLI label order names all three classes",
               sorted(nli or []) == ["contradiction", "entailment", "neutral"], nli)
        encoder = model_config.get("encoder_config", {})
        summary["model"] = {"encoder_type": encoder.get("model_type"), "encoder_name": encoder.get("_name_or_path"),
                            "hidden_size": encoder.get("hidden_size"), "class_weights": model_config.get("class_weights"),
                            "loss_normalization": model_config.get("loss_normalization"),
                            "distillation_weight": model_config.get("distillation_weight"),
                            "saved_torch_dtype": model_config.get("torch_dtype") or model_config.get("dtype")}
    weights = sorted(final.glob("*.safetensors")) if final.is_dir() else []
    if weights:
        dtypes = Counter()
        for path in weights:
            dtypes.update(safetensors_dtypes(path))
        summary["weights"] = {"files": {p.name: sha256_file(p) for p in weights}, "tensor_dtypes": dict(dtypes)}
        _check(checks, "saved weights are full precision", set(dtypes) <= {"F32", "I64", "BOOL"}, dict(dtypes))
    if tokenizer_config:
        summary["tokenizer"] = {k: tokenizer_config.get(k) for k in ("tokenizer_class", "model_max_length", "do_lower_case")}
    if prepared and (prepared / "preparation.json").is_file():
        preparation = _json(prepared / "preparation.json")
        summary["preparation"] = {k: preparation.get(k) for k in ("nli_model", "revision", "max_length", "claim_budget", "nli_label_order", "limit")}
        if run_config:
            _check(checks, "run bound to this preparation",
                   run_config.get("preparation_sha256") == sha256_file(prepared / "preparation.json"), None)
        if model_config and preparation.get("nli_label_order"):
            _check(checks, "model NLI order matches preparation", model_config.get("nli_labels") == preparation["nli_label_order"],
                   {"model": model_config.get("nli_labels"), "preparation": preparation["nli_label_order"]})

    if selection:
        chosen = selection.get("selected", {})
        summary["selection"] = {"learning_rate": chosen.get("learning_rate"), "best_epoch": chosen.get("best_epoch"),
                                "holdout_macro_f1": chosen.get("macro_f1"), "trial_count": len(selection.get("trials", [])),
                                "fit_claims": len(selection.get("fit_claim_ids", [])),
                                "holdout_claims": len(selection.get("validation_claim_ids", [])),
                                "selection_metric": selection.get("selection_metric")}
        overlap = set(selection.get("fit_claim_ids", [])) & set(selection.get("validation_claim_ids", []))
        _check(checks, "fit/holdout IDs disjoint", not overlap, len(overlap))
    if final_fit and selection:
        refit_epochs = final_fit.get("selection", {}).get("best_epoch")
        states = sorted(final.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[1]))
        last_state = _json(states[-1]) if states else None
        summary["refit"] = {"training_claims": final_fit.get("training_claims"), "scheduled_epochs": refit_epochs,
                            "last_checkpoint_epoch": last_state.get("epoch") if last_state else None,
                            "last_checkpoint_step": last_state.get("global_step") if last_state else None}
        _check(checks, "refit uses the selected epoch count", refit_epochs == selection.get("selected", {}).get("best_epoch"),
               {"refit": refit_epochs, "selected": selection.get("selected", {}).get("best_epoch")})
        if last_state:
            _check(checks, "refit checkpoint reached scheduled epochs",
                   math.isclose(float(last_state["epoch"]), float(refit_epochs), abs_tol=1e-3), summary["refit"])

    predictions_path = run / "dev" / "predictions.jsonl"
    if predictions_path.is_file():
        predictions = read_jsonl(predictions_path)
        ids = [p["claim_id"] for p in predictions]
        _check(checks, "dev prediction IDs unique", len(ids) == len(set(ids)), len(ids))
        sums = [sum(p["probabilities"].values()) for p in predictions]
        _check(checks, "probabilities normalized", all(abs(s - 1) < 1e-4 for s in sums), {"min": min(sums), "max": max(sums)})
        argmax_ok = all(max(p["probabilities"], key=p["probabilities"].get) == p["predicted_label"] for p in predictions)
        _check(checks, "predicted label is probability argmax", argmax_ok, None)
        evidence_counts = Counter(len(p.get("evidence_passage_ids") or []) for p in predictions)
        _check(checks, "at most three evidence IDs per claim", max(evidence_counts) <= 3, dict(evidence_counts))
        recomputed = evaluate_predictions(predictions)
        saved = _json(run / "dev" / "metrics.json")
        summary["dev_metrics_recomputed"] = {"claim_count": recomputed["claim_count"], "macro_f1": recomputed["macro_f1"],
            "accuracy": recomputed["accuracy"], **{k: v for k, v in recomputed["decision_error_rates"].items() if k.endswith("count")},
            "predicted_label_counts": dict(Counter(p["predicted_label"] for p in predictions))}
        if saved:
            _check(checks, "recomputed dev macro-F1 matches saved", math.isclose(saved["macro_f1"], recomputed["macro_f1"], abs_tol=1e-9),
                   {"saved": saved["macro_f1"], "recomputed": recomputed["macro_f1"]})
        if data_root:
            claims = {r["claim_id"]: r for r in read_jsonl(data_root / "processed/averitec/dev.jsonl")}
            _check(checks, "dev predictions cover official dev exactly", set(ids) == set(claims), {"predictions": len(ids), "claims": len(claims)})
            _check(checks, "true labels join to dev claims", all(claims.get(p["claim_id"], {}).get("label") == p["true_label"] for p in predictions), None)
        if prepared and run_config:
            rankings_path = prepared / run_config["mode"] / "dev_rankings.jsonl"
            if rankings_path.is_file():
                expected = {r["claim_id"]: [e["passage_id"] for e in (r.get("retrieved") or [])[:3]] for r in read_jsonl(rankings_path)}
                mismatched = [p["claim_id"] for p in predictions if p.get("evidence_passage_ids") != expected.get(p["claim_id"])]
                _check(checks, "evidence IDs match prepared top three", not mismatched, {"mismatched": len(mismatched), "examples": mismatched[:3]})
    else:
        _check(checks, "dev predictions present", None, "dev/predictions.jsonl not found")
    summary["checks"] = checks
    summary["failed_checks"] = [c["check"] for c in checks if c["status"] == "FAIL"]
    return summary


def rescore(run: Path, data_root: Path, prepared: Path, limit: int, device: str) -> dict[str, Any]:
    """Re-run frozen inference on the first ``limit`` dev claims."""
    import torch
    from transformers import AutoTokenizer

    from src.data.evidence import attach_evidence
    from src.verification.pair_reader import EvidencePairReader, PairCollator, PairDataset

    run_config, preparation = _json(run / "reader_run.json"), _json(prepared / "preparation.json")
    records = attach_evidence(read_jsonl(data_root / "processed/averitec/dev.jsonl"),
                              prepared / run_config["mode"] / "dev_rankings.jsonl", exact=True)[:limit]
    saved = {p["claim_id"]: p for p in read_jsonl(run / "dev" / "predictions.jsonl")}
    tokenizer = AutoTokenizer.from_pretrained(run / "final_model", use_fast=True)
    model = EvidencePairReader.from_pretrained(run / "final_model").to(device).eval()
    dataset = PairDataset(records, tokenizer, max_length=preparation["max_length"], claim_budget=preparation["claim_budget"])
    collator = PairCollator(tokenizer)
    differences, agree = [], 0
    for start in range(0, len(dataset), 8):
        rows = [dataset[i] for i in range(start, min(start + 8, len(dataset)))]
        batch = {k: v.to(device) for k, v in collator(rows).items() if k != "labels"}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            values = model(**batch).logits.float().softmax(-1).cpu().tolist()
        for record, vector in zip(records[start:start + 8], values, strict=True):
            old = saved[record["claim_id"]]
            differences.append(max(abs(vector[i] - old["probabilities"][label]) for i, label in enumerate(CANONICAL_LABELS)))
            agree += CANONICAL_LABELS[max(range(4), key=vector.__getitem__)] == old["predicted_label"]
    return {"claims": len(records), "device": device, "max_abs_probability_difference": max(differences, default=0.0),
            "argmax_agreement": agree / max(1, len(records)),
            "note": "Small differences are expected between fp16 GPU and fp32 CPU inference."}


def mechanics_check(args) -> dict[str, Any]:
    report = inspect_run(args.run, data_root=args.data_root, prepared=args.prepared)
    if getattr(args, "rescore_limit", 0):
        report["rescore"] = rescore(args.run, args.data_root, args.prepared, args.rescore_limit, args.device)
        report["checks"].append({"check": "re-scored argmax agrees with saved predictions",
                                 "status": "pass" if report["rescore"]["argmax_agreement"] == 1 else "FAIL",
                                 "detail": report["rescore"]})
        report["failed_checks"] = [c["check"] for c in report["checks"] if c["status"] == "FAIL"]
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(report, args.output / "mechanics_report.json")
    lines = [f"# Reader mechanics check: {args.run.name}", "",
             "Reproduction of recorded mechanics; not an independent validation result.", "",
             "| Check | Status | Detail |", "| --- | --- | --- |"]
    lines += [f"| {c['check']} | {c['status']} | {json.dumps(c['detail'], default=str)[:160]} |" for c in report["checks"]]
    (args.output / "mechanics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return report
