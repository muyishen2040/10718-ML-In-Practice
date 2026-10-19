"""Train a compact evidence-aware transformer with resumable tuning trials.

Model selection uses a stratified holdout carved only from official train
claims. The selected configuration is then refit on all official train claims
and evaluated once on the unchanged official development split.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_tfidf_logreg import attach_retrieved_evidence  # noqa: E402
from src.data.averitec import CANONICAL_LABELS  # noqa: E402
from src.evaluation.classification import evaluate_predictions  # noqa: E402
from src.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from src.utils.runs import resolve_run_directory, sha256_file, write_run_manifest  # noqa: E402


LABEL_TO_ID = {label: index for index, label in enumerate(CANONICAL_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def parse_float_grid(value: str, name: str) -> list[float]:
    try:
        result = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated numbers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(f"{name} values must be positive")
    return list(dict.fromkeys(result))


def parse_int_grid(value: str, name: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(f"{name} values must be positive")
    return list(dict.fromkeys(result))


def attach_rankings(records: list[dict[str, Any]], rankings_path: Path) -> list[dict[str, Any]]:
    return attach_retrieved_evidence(records, rankings_path)


def compose_text(record: dict[str, Any], tokenizer: Any, claim_budget: int, evidence_budget: int) -> str:
    claim_tokens = tokenizer.encode(str(record["claim"]), add_special_tokens=False)[:claim_budget]
    claim_text = tokenizer.decode(claim_tokens, skip_special_tokens=True)
    sections = [f"Claim: {claim_text}"]
    evidence = [item for item in (record.get("retrieved_evidence") or [])[:3] if str(item.get("text") or "").strip()]
    if not evidence:
        sections.append("Evidence: No candidate passage was retrieved.")
    else:
        for index, item in enumerate(evidence, start=1):
            tokens = tokenizer.encode(str(item["text"]), add_special_tokens=False)[:evidence_budget]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            sections.append(f"Evidence {index}: {text}")
    return "\n".join(sections)


class EncodedRecords(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, max_length: int, claim_budget: int, evidence_budget: int):
        self.records = records
        texts = [compose_text(row, tokenizer, claim_budget, evidence_budget) for row in records]
        encoded = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )
        self.features = []
        for index, record in enumerate(records):
            feature = {key: values[index] for key, values in encoded.items()}
            feature["labels"] = LABEL_TO_ID[str(record["label"])]
            self.features.append(feature)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


def prediction_metrics(eval_prediction: Any) -> dict[str, float]:
    logits = eval_prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    gold = np.asarray(eval_prediction.label_ids)
    predicted = np.asarray(logits).argmax(axis=1)
    supported_id = LABEL_TO_ID["Supported"]
    false_reassurance = np.logical_and(gold != supported_id, predicted == supported_id)
    false_alarm = np.logical_and(gold == supported_id, predicted != supported_id)
    return {
        "macro_f1": float(f1_score(gold, predicted, labels=list(range(len(CANONICAL_LABELS))), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(gold, predicted)),
        "false_reassurance_rate": float(false_reassurance.mean()),
        "false_alarm_rate": float(false_alarm.mean()),
    }


def safe_name(value: float) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "p", f"{value:.0e}" if value < 0.001 else f"{value:g}")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best_checkpoint(trial_dir: Path) -> str | None:
    """Return the newest complete Trainer checkpoint, ignoring torn Drive writes."""
    required_state = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
    valid: list[tuple[int, Path]] = []
    for path in trial_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        has_model = any(path.glob("model*.safetensors")) or any(path.glob("pytorch_model*.bin"))
        if has_model and all((path / name).is_file() for name in required_state):
            valid.append((step, path))
    return str(max(valid, default=(0, None), key=lambda pair: pair[0])[1]) if valid else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-rankings", required=True, type=Path)
    parser.add_argument("--eval-rankings", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--revision", help="Optional immutable Hugging Face model revision.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--claim-token-budget", type=int, default=96)
    parser.add_argument("--evidence-token-budget", type=int, default=110)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--learning-rates", default="2e-5,5e-5")
    parser.add_argument("--epoch-grid", default="2,3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--checkpoint-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=718)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.max_length < 32 or args.claim_token_budget < 1 or args.evidence_token_budget < 1:
        parser.error("Token lengths must be positive and max-length must be at least 32")
    if not 0.01 <= args.validation_fraction <= 0.4:
        parser.error("validation-fraction must be between 0.01 and 0.4")
    if min(args.batch_size, args.gradient_accumulation_steps, args.checkpoint_steps) < 1:
        parser.error("batch size, gradient accumulation, and checkpoint steps must be positive")
    try:
        learning_rates = parse_float_grid(args.learning_rates, "learning-rates")
        epochs_grid = parse_int_grid(args.epoch_grid, "epoch-grid")
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))

    try:
        output_dir = resolve_run_directory(args.output_dir, args.run_name)
    except ValueError as error:
        parser.error(str(error))
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = args.data_root / "processed" / "averitec"
    train_path = processed / "train.jsonl"
    eval_path = processed / "dev.jsonl"
    for path in (train_path, eval_path, args.train_rankings, args.eval_rankings):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_records = attach_rankings(read_jsonl(train_path), args.train_rankings)
    eval_records = attach_rankings(read_jsonl(eval_path), args.eval_rankings)
    train_indices, validation_indices = train_test_split(
        np.arange(len(train_records)),
        test_size=args.validation_fraction,
        random_state=args.seed,
        stratify=[row["label"] for row in train_records],
    )
    selection_train_records = [train_records[int(index)] for index in train_indices]
    validation_records = [train_records[int(index)] for index in validation_indices]

    config = {
        "script": "train_transformer_verifier",
        "model": args.model,
        "requested_revision": args.revision,
        "max_length": args.max_length,
        "claim_token_budget": args.claim_token_budget,
        "evidence_token_budget_per_passage": args.evidence_token_budget,
        "max_evidence": 3,
        "evidence_source": str(args.train_rankings),
        "train_rankings_sha256": sha256_file(args.train_rankings),
        "eval_rankings": str(args.eval_rankings),
        "eval_rankings_sha256": sha256_file(args.eval_rankings),
        "validation_fraction": args.validation_fraction,
        "validation_seed": args.seed,
        "learning_rate_grid": learning_rates,
        "epoch_grid": epochs_grid,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "checkpoint_steps": args.checkpoint_steps,
        "loss": "inverse_frequency_weighted_cross_entropy",
        "split_policy": "official train split with internal stratified validation; official dev evaluated after refit",
    }
    config_path = output_dir / "transformer_run_config.json"
    existing_config = read_json(config_path)
    if existing_config is not None and any(existing_config.get(key) != value for key, value in config.items()):
        raise ValueError(f"Run configuration in {config_path} differs; choose a new --run-name.")
    config = {**(existing_config or {}), **config}
    model_revision = args.revision or (existing_config or {}).get("resolved_model_revision")
    if existing_config is None:
        write_json(config, config_path)

    final_metrics_path = output_dir / "final_dev_metrics.json"
    final_predictions_path = output_dir / "final_dev_predictions.jsonl"
    if final_metrics_path.exists() and final_predictions_path.exists():
        manifest_path = output_dir / "transformer_manifest.json"
        if not manifest_path.is_file():
            selection_report = read_json(output_dir / "model_selection.json") or {}
            write_run_manifest(
                manifest_path,
                project_root=PROJECT_ROOT,
                config={**config, "selection": selection_report, "final_dev_metrics_path": str(final_metrics_path)},
                input_paths={
                    "train_claims": train_path,
                    "dev_claims": eval_path,
                    "train_rankings": args.train_rankings,
                    "dev_rankings": args.eval_rankings,
                },
            )
        print(final_metrics_path.read_text(encoding="utf-8"))
        return

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError("Transformer training requires transformers; install requirements-colab.txt first.") from error

    class WeightedTrainer(Trainer):
        def __init__(self, *trainer_args: Any, class_weights: torch.Tensor, **trainer_kwargs: Any):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.class_weights = class_weights

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, torch.Tensor],
            return_outputs: bool = False,
            num_items_in_batch: torch.Tensor | None = None,
        ):
            labels = inputs["labels"]
            model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            outputs = model(**model_inputs)
            weights = self.class_weights.to(outputs.logits.device)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights)
            return (loss, outputs) if return_outputs else loss

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=model_revision or None, use_fast=True)
    resolved_model_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if resolved_model_revision:
        config["resolved_model_revision"] = resolved_model_revision
        write_json(config, config_path)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    selection_train_dataset = EncodedRecords(
        selection_train_records, tokenizer, args.max_length, args.claim_token_budget, args.evidence_token_budget
    )
    validation_dataset = EncodedRecords(
        validation_records, tokenizer, args.max_length, args.claim_token_budget, args.evidence_token_budget
    )
    eval_dataset = EncodedRecords(eval_records, tokenizer, args.max_length, args.claim_token_budget, args.evidence_token_budget)

    counts = np.bincount(
        [LABEL_TO_ID[str(row["label"])] for row in selection_train_records], minlength=len(CANONICAL_LABELS)
    )
    if np.any(counts == 0):
        raise ValueError(f"Internal training fold is missing one or more labels: counts={counts.tolist()}")
    class_weights = torch.tensor(len(selection_train_records) / (len(CANONICAL_LABELS) * counts), dtype=torch.float32)
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    trial_rows: list[dict[str, Any]] = []

    def new_model():
        nonlocal resolved_model_revision
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            revision=model_revision or None,
            num_labels=len(CANONICAL_LABELS),
            label2id=LABEL_TO_ID,
            id2label=ID_TO_LABEL,
        )
        resolved_model_revision = getattr(model.config, "_commit_hash", None) or resolved_model_revision
        if resolved_model_revision and config.get("resolved_model_revision") != resolved_model_revision:
            config["resolved_model_revision"] = resolved_model_revision
            write_json(config, config_path)
        return model

    def train_arguments(path: Path, *, lr: float, epochs: int, select_best: bool) -> TrainingArguments:
        common = {
            "output_dir": str(path),
            "learning_rate": lr,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": args.batch_size,
            "per_device_eval_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "weight_decay": 0.01,
            "warmup_ratio": 0.06,
            "logging_steps": 25,
            "save_strategy": "steps",
            "save_steps": args.checkpoint_steps,
            "save_total_limit": 2,
            "seed": args.seed,
            "fp16": bool(torch.cuda.is_available()),
            "report_to": "none",
            "dataloader_num_workers": 0,
            "save_safetensors": True,
        }
        if select_best:
            common.update(
                {
                    "eval_strategy": "steps",
                    "eval_steps": args.checkpoint_steps,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "macro_f1",
                    "greater_is_better": True,
                }
            )
        else:
            common["eval_strategy"] = "no"
            common["load_best_model_at_end"] = False
        return TrainingArguments(**common)

    for learning_rate in learning_rates:
        for epochs in epochs_grid:
            trial_name = f"lr_{safe_name(learning_rate)}_epochs_{epochs}"
            trial_dir = trials_dir / trial_name
            summary_path = trial_dir / "trial_summary.json"
            summary = read_json(summary_path)
            if summary is None:
                if trial_dir.exists() and any(trial_dir.glob("checkpoint-*")) and not args.resume:
                    raise ValueError(f"Found interrupted trial at {trial_dir}; rerun with --resume.")
                trial_dir.mkdir(parents=True, exist_ok=True)
                set_seed(args.seed)
                trainer = WeightedTrainer(
                    model=new_model(),
                    args=train_arguments(trial_dir, lr=learning_rate, epochs=epochs, select_best=True),
                    train_dataset=selection_train_dataset,
                    eval_dataset=validation_dataset,
                    data_collator=collator,
                    compute_metrics=prediction_metrics,
                    class_weights=class_weights,
                )
                last = best_checkpoint(trial_dir) if args.resume else None
                trainer.train(resume_from_checkpoint=last)
                validation_logits = trainer.predict(validation_dataset).predictions
                if isinstance(validation_logits, tuple):
                    validation_logits = validation_logits[0]
                probabilities = torch.softmax(torch.tensor(validation_logits), dim=1).cpu().numpy()
                predicted_ids = probabilities.argmax(axis=1)
                validation_predictions = []
                for row, predicted_id, probability_row in zip(
                    validation_records, predicted_ids, probabilities, strict=True
                ):
                    validation_predictions.append(
                        {
                            "claim_id": str(row["claim_id"]),
                            "true_label": str(row["label"]),
                            "predicted_label": ID_TO_LABEL[int(predicted_id)],
                            "probabilities": {
                                ID_TO_LABEL[index]: float(probability_row[index])
                                for index in range(len(CANONICAL_LABELS))
                            },
                        }
                    )
                metrics = evaluate_predictions(validation_predictions)
                summary = {
                    "trial": trial_name,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "selection_validation_claim_count": len(validation_records),
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "decision_error_rates": metrics["decision_error_rates"],
                    "calibration": metrics.get("calibration"),
                    "predictions_path": str(trial_dir / "validation_predictions.jsonl"),
                }
                write_jsonl(validation_predictions, trial_dir / "validation_predictions.jsonl")
                write_json(summary, summary_path)
            trial_rows.append(summary)

    selected = sorted(
        trial_rows,
        key=lambda row: (
            -float(row["macro_f1"]),
            float(row["decision_error_rates"]["false_reassurance_rate"]),
            float(row["decision_error_rates"]["false_alarm_rate"]),
            float(row["learning_rate"]),
            int(row["epochs"]),
        ),
    )[0]
    selection_report = {
        "selection_metric": "validation macro-F1; ties broken by lower false reassurance, then lower false alarm",
        "selected_trial": selected,
        "all_trials": trial_rows,
    }
    write_json(selection_report, output_dir / "model_selection.json")

    final_dir = output_dir / "final_model"
    final_complete_path = output_dir / "final_train_complete.json"
    if not final_complete_path.exists():
        if final_dir.exists() and any(final_dir.glob("checkpoint-*")) and not args.resume:
            raise ValueError(f"Found interrupted final training at {final_dir}; rerun with --resume.")
        final_dir.mkdir(parents=True, exist_ok=True)
        full_counts = np.bincount(
            [LABEL_TO_ID[str(row["label"])] for row in train_records], minlength=len(CANONICAL_LABELS)
        )
        full_weights = torch.tensor(len(train_records) / (len(CANONICAL_LABELS) * full_counts), dtype=torch.float32)
        set_seed(args.seed)
        final_trainer = WeightedTrainer(
            model=new_model(),
            args=train_arguments(final_dir, lr=float(selected["learning_rate"]), epochs=int(selected["epochs"]), select_best=False),
            train_dataset=EncodedRecords(
                train_records, tokenizer, args.max_length, args.claim_token_budget, args.evidence_token_budget
            ),
            data_collator=collator,
            class_weights=full_weights,
        )
        last = best_checkpoint(final_dir) if args.resume else None
        final_trainer.train(resume_from_checkpoint=last)
        final_trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(final_dir)
        write_json(
            {
                "selected_trial": selected["trial"],
                "training_claim_count": len(train_records),
                "epochs": selected["epochs"],
                "learning_rate": selected["learning_rate"],
            },
            final_complete_path,
        )

    final_model = AutoModelForSequenceClassification.from_pretrained(final_dir)
    final_trainer = WeightedTrainer(
        model=final_model,
        args=train_arguments(output_dir / "prediction_tmp", lr=float(selected["learning_rate"]), epochs=int(selected["epochs"]), select_best=False),
        data_collator=collator,
        class_weights=torch.ones(len(CANONICAL_LABELS), dtype=torch.float32),
    )
    prediction_output = final_trainer.predict(eval_dataset)
    logits = prediction_output.predictions[0] if isinstance(prediction_output.predictions, tuple) else prediction_output.predictions
    probabilities = torch.softmax(torch.tensor(logits), dim=1).cpu().numpy()
    predicted_ids = probabilities.argmax(axis=1)
    eval_predictions = []
    for row, predicted_id, probability_row in zip(eval_records, predicted_ids, probabilities, strict=True):
        retrieved = row.get("retrieved_evidence") or []
        eval_predictions.append(
            {
                "claim_id": str(row["claim_id"]),
                "true_label": str(row["label"]),
                "predicted_label": ID_TO_LABEL[int(predicted_id)],
                "probabilities": {
                    ID_TO_LABEL[index]: float(probability_row[index])
                    for index in range(len(CANONICAL_LABELS))
                },
                "evidence_item_count": min(3, len(retrieved)),
                "evidence_passage_ids": [str(item["passage_id"]) for item in retrieved[:3]],
            }
        )
    metrics = evaluate_predictions(eval_predictions)
    write_jsonl(eval_predictions, final_predictions_path)
    final_report = {
        **metrics,
        "selection": selected,
        "eval_split": "official dev",
        "eval_claim_count": len(eval_records),
        "evidence_source": str(args.eval_rankings),
        "model": args.model,
        "model_path": str(final_dir),
    }
    write_json(final_report, final_metrics_path)
    write_run_manifest(
        output_dir / "transformer_manifest.json",
        project_root=PROJECT_ROOT,
        config={**config, "selection": selection_report, "final_dev_metrics_path": str(final_metrics_path)},
        input_paths={
            "train_claims": train_path,
            "dev_claims": eval_path,
            "train_rankings": args.train_rankings,
            "dev_rankings": args.eval_rankings,
        },
    )
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
