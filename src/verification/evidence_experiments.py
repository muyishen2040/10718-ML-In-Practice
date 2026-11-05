"""Fast aggregation, frozen-baseline controls and common result reporting."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.averitec import CANONICAL_LABELS
from src.data.evidence import attach_evidence
from src.evaluation.classification import evaluate_predictions
from src.evaluation.paired import paired_delta
from src.evaluation.splits import selection_split
from src.retrieval.evidence_quality import aggregate_features, normalized_text
from src.retrieval.metrics import evaluate_rankings
from src.utils.checkpoints import atomic_jsonl, bind_run
from src.utils.io import read_jsonl, write_json
from src.utils.runs import sha256_file
from src.verification.inference import predict_cached, validate_predictions
from src.verification.pair_training import selection_key


def fast_aggregate(args):
    claims = {s: read_jsonl(args.data_root / "processed/averitec" / f"{s}.jsonl") for s in ("train", "dev")}
    rankings = {s: args.prepared / args.mode / f"{s}_rankings.jsonl" for s in claims}
    rows = {s: attach_evidence(claims[s], rankings[s], exact=True) for s in claims}
    if json.loads((args.prepared / "preparation.json").read_text())["limit"]:
        raise ValueError("Fast training requires full preparation")
    x = {s: np.stack([aggregate_features({"retrieved": r["retrieved_evidence"]}) for r in rows[s]]) for s in rows}
    y = np.array([r["label"] for r in rows["train"]])
    a, b = selection_split(rows["train"], .1, args.validation_seed, args.validation_policy)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    bind_run(output / "fast_run.json", {"mode": args.mode, "rankings_sha256": {s: sha256_file(p) for s, p in rankings.items()},
        "claims_sha256": {s: sha256_file(args.data_root / "processed/averitec" / f"{s}.jsonl") for s in claims},
        "validation_seed": args.validation_seed, "validation_policy": args.validation_policy,
        "regularization": [.1, 1, 10], "implementation_sha256": sha256_file(Path(__file__))})

    def model(c):
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=args.validation_seed))

    def predict(classifier, features, records):
        probabilities = classifier.predict_proba(features)
        indices = [list(classifier.classes_).index(k) for k in CANONICAL_LABELS]
        return [{"claim_id": r["claim_id"], "true_label": r["label"], "predicted_label": CANONICAL_LABELS[int(p[indices].argmax())],
                 "probabilities": dict(zip(CANONICAL_LABELS, map(float, p[indices]), strict=True))}
                for r, p in zip(records, probabilities, strict=True)]

    trials = []
    for c in (.1, 1, 10):
        classifier = model(c).fit(x["train"][a], y[a])
        p = predict(classifier, x["train"][b], [rows["train"][int(i)] for i in b])
        trials.append({**evaluate_predictions(p), "C": c})
    chosen = sorted(trials, key=selection_key)[0]
    write_json({"selected": chosen, "trials": trials, "validation_claim_ids": [rows["train"][int(i)]["claim_id"] for i in b]}, output / "model_selection.json")
    if args.selection_only:
        print("Fast train-only macro-F1:", chosen["macro_f1"])
        return
    classifier = model(chosen["C"]).fit(x["train"], y)
    joblib.dump(classifier, output / "aggregator.joblib")
    predictions = predict(classifier, x["dev"], rows["dev"])
    atomic_jsonl(predictions, output / "dev/predictions.jsonl")
    write_json(evaluate_predictions(predictions), output / "dev/metrics.json")
    print("Fast dev macro-F1:", evaluate_predictions(predictions)["macro_f1"])


def frozen_distil(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from scripts.train_transformer_verifier import compose_text

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime")
    path = args.prepared / args.mode / "dev_rankings.jsonl"
    records = attach_evidence(read_jsonl(args.data_root / "processed/averitec/dev.jsonl"), path, exact=True)
    config_path, model_dir = args.baseline / "transformer_run_config.json", args.baseline / "final_model"
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    bind_run(args.output / "frozen_run.json", {"rankings": sha256_file(path), "claims": sha256_file(args.data_root / "processed/averitec/dev.jsonl"),
        "settings": settings, "weights": {p.name: sha256_file(p) for p in [*model_dir.glob("*.safetensors"), *model_dir.glob("pytorch_model*.bin")]}})
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(args.device).eval()
    if [model.config.id2label.get(i) for i in range(4)] != list(CANONICAL_LABELS):
        raise ValueError("Baseline label order differs")

    def batch(rows):
        text = [compose_text(r, tokenizer, settings["claim_token_budget"], settings["evidence_token_budget_per_passage"]) for r in rows]
        encoded = tokenizer(text, truncation=True, padding=True, max_length=settings["max_length"], return_tensors="pt")
        with torch.inference_mode():
            return model(**{k: v.to(args.device) for k, v in encoded.items()}).logits.float().softmax(-1).cpu().tolist()

    print(json.dumps(predict_cached(records, batch, args.output, batch_size=16, description=f"Frozen DistilBERT / {args.mode}"), indent=2))


def compare_results(args):
    claims = read_jsonl(args.data_root / "processed/averitec/dev.jsonl")
    reports, predictions = {}, {}
    for path in sorted(args.output.glob("**/metrics.json")):
        pred_path = path.with_name("predictions.jsonl")
        if not pred_path.is_file():
            continue
        name = str(path.parent.relative_to(args.output)).replace("\\", "/").removesuffix("/dev")
        predictions[name] = validate_predictions(read_jsonl(pred_path), claims)
        reports[name] = evaluate_predictions(predictions[name])
    for path in sorted(args.output.glob("distil_*/final_dev_metrics.json")):
        name = path.parent.name
        predictions[name] = validate_predictions(read_jsonl(path.with_name("final_dev_predictions.jsonl")), claims)
        reports[name] = evaluate_predictions(predictions[name])
    reference_root = getattr(args, "reference_root", None)
    if reference_root is not None:
        # Read completed old predictions without refitting or rewriting old runs.
        for path in sorted(reference_root.glob("**/metrics.json")):
            pred_path = path.with_name("predictions.jsonl")
            if not pred_path.is_file():
                continue
            name = "previous/" + path.parent.relative_to(reference_root).as_posix().removesuffix("/dev")
            predictions[name] = validate_predictions(read_jsonl(pred_path), claims)
            reports[name] = evaluate_predictions(predictions[name])
        for path in sorted(reference_root.glob("distil_*/final_dev_predictions.jsonl")):
            name = "previous/" + path.parent.name
            predictions[name] = validate_predictions(read_jsonl(path), claims)
            reports[name] = evaluate_predictions(predictions[name])
    reference_name = "frozen/raw" if "frozen/raw" in predictions else "previous/frozen/raw"
    reference = predictions.get(reference_name)
    paired = {name: paired_delta(reference, rows) for name, rows in predictions.items() if reference and name != reference_name}
    contrasts = {}
    for old, new in (("pair/raw_legacy_d01", "pair/raw_sample_d01"),
                     ("pair/raw_sample_d01", "pair/raw_sample_d0"),
                     ("pair/raw_sample_d0", "pair/qa_same3_d0"),
                     ("pair/qa_same3_d0", "pair/qa_selected3_d0")):
        if old in predictions and new in predictions:
            contrasts[new + " minus " + old] = paired_delta(predictions[old], predictions[new])
    normalized_train = {normalized_text(r["claim"]) for r in read_jsonl(args.data_root / "processed/averitec/train.jsonl")}
    keep = {r["claim_id"] for r in claims if normalized_text(r["claim"]) not in normalized_train}
    nonoverlap = {name: evaluate_predictions([r for r in rows if r["claim_id"] in keep]) for name, rows in predictions.items()} if keep else {}
    retrieval = {}
    if args.qrels:
        qrels = {r["claim_id"]: r["relevant_passage_ids"] for r in read_jsonl(args.qrels) if r.get("relevant_passage_ids")}
        for path in sorted(args.prepared.glob("*/dev_rankings.jsonl")):
            mode = path.parent.name
            rows = read_jsonl(path)
            retrieval[mode] = evaluate_rankings({r["claim_id"]: [p["passage_id"] for p in r["retrieved"]] for r in rows}, qrels, k=3)
    counts = {name: {label: sum(r["predicted_label"] == label for r in rows) for label in CANONICAL_LABELS}
              for name, rows in predictions.items()}
    report = {"verdicts": reports, "predicted_label_counts": counts, "paired_vs_frozen_raw": paired,
              "reference_name": reference_name, "planned_paired_contrasts": contrasts, "nonoverlapping_text_subset": nonoverlap,
              "subset_claim_count": len(keep), "retrieval_fixed_original_url_proxy": retrieval,
              "caveats": ["Official dev is exploratory after repeated inspection; test untouched.",
                          "NLI probabilities are not calibrated four-way verdict probabilities.",
                          "Original qrels and all claims are retained even when a source is filtered.",
                          "Unknown publication dates do not establish temporal validity."]}
    write_json(report, args.output / "evidence_reader_results.json")
    lines = ["# Evidence reader experiment", "", f"Same {len(claims)} dev claims; FR/FA are counts.", "",
             "| Method | Macro-F1 | Accuracy | FR | FA |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, m in reports.items():
        errors = m["decision_error_rates"]
        lines.append(f"| {name} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} | {errors['false_reassurance_count']} | {errors['false_alarm_count']} |")
    lines += ["", "| Method | Supported F1 | Refuted F1 | NEI F1 | Conflict F1 | Predicted NEI / conflict |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, m in reports.items():
        f1 = [m["per_class"][label]["f1-score"] for label in CANONICAL_LABELS]
        rare = f"{counts[name]['Not Enough Evidence']} / {counts[name]['Conflicting Evidence']}"
        lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in f1) + f" | {rare} |")
    if contrasts:
        lines += ["", "Planned paired comparisons (macro-F1 delta and 95% interval):"]
        for name, contrast in contrasts.items():
            low, high = contrast["interval_95pct"]
            lines.append(f"- {name}: {contrast['delta_macro_f1']:+.4f} [{low:+.4f}, {high:+.4f}]")
    lines += ["", "Full per-class, calibration, overlap-subset and paired interval results: evidence_reader_results.json.",
              "", "Frozen rows use the original BM25-trained DistilBERT. Pair rows train the new NLI-initialized reader. Fast rows train a classifier on frozen NLI features."]
    (args.output / "evidence_reader_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
