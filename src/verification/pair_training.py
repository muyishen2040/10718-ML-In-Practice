"""Train-only model selection and checkpointed fitting of the pair reader."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from src.data.averitec import CANONICAL_LABELS
from src.data.evidence import attach_evidence
from src.evaluation.classification import evaluate_predictions
from src.evaluation.splits import selection_split
from src.utils.checkpoints import atomic_jsonl, bind_run, latest_complete_trainer_checkpoint
from src.utils.io import read_jsonl, write_json
from src.utils.runs import sha256_file
from src.verification.inference import predict_cached


def selection_key(report):
    return (-report["macro_f1"], report["decision_error_rates"]["false_reassurance_count"],
            report["decision_error_rates"]["false_alarm_count"])


def train_reader(args):
    import torch
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed
    from src.verification.pair_reader import EvidencePairReader, PairCollator, PairDataset

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A Colab GPU runtime is required")
    preparation = json.loads((args.prepared / "preparation.json").read_text(encoding="utf-8"))
    if preparation["limit"]:
        raise ValueError("Training requires full preparation, not the smoke-test subset")
    if not (args.prepared / "preparation_report.json").is_file():
        raise ValueError("Evidence preparation has not finished")
    paths = {"train": args.data_root / "processed/averitec/train.jsonl", "dev": args.data_root / "processed/averitec/dev.jsonl",
             "train_rankings": args.prepared / args.mode / "train_rankings.jsonl", "dev_rankings": args.prepared / args.mode / "dev_rankings.jsonl"}
    records = {s: attach_evidence(read_jsonl(paths[s]), paths[s + "_rankings"], exact=True) for s in ("train", "dev")}
    a, b = selection_split(records["train"], .1, args.validation_seed, args.validation_policy)
    fit_rows, val_rows = [records["train"][int(i)] for i in a], [records["train"][int(i)] for i in b]
    config = {"version": 1, "mode": args.mode, "input_hashes": {k: sha256_file(v) for k, v in paths.items()},
              "preparation_sha256": sha256_file(args.prepared / "preparation.json"), "learning_rates": args.learning_rates,
              "epochs": args.epochs, "batch_size": args.batch_size, "accumulation": args.accumulation,
              "seed": args.seed, "validation_seed": args.validation_seed, "validation_policy": args.validation_policy,
              "distillation_weight": args.distillation_weight, "device": args.device,
              "freeze_word_embeddings": not getattr(args, "unfreeze_embeddings", False),
              "implementation_sha256": sha256_file(Path(__file__)),
              "architecture_sha256": sha256_file(Path(__file__).with_name("pair_reader.py"))}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    bind_run(output / "reader_run.json", config)
    tokenizer = AutoTokenizer.from_pretrained(preparation["nli_model"], revision=preparation["revision"], use_fast=True)
    collator = PairCollator(tokenizer)
    dataset_kwargs = {"max_length": preparation["max_length"], "claim_budget": preparation["claim_budget"]}
    train_data, val_data = PairDataset(fit_rows, tokenizer, **dataset_kwargs), PairDataset(val_rows, tokenizer, **dataset_kwargs)

    def weights(rows):
        counts = np.array([sum(r["label"] == label for r in rows) for label in CANONICAL_LABELS])
        if np.any(counts == 0):
            raise ValueError(f"Training fold lacks a verdict class: {counts.tolist()}")
        return (len(rows) / (4 * counts)).tolist()

    def new_model(rows):
        set_seed(args.seed)
        return EvidencePairReader.from_nli(preparation["nli_model"], preparation["revision"],
                    class_weights=weights(rows), label_order=preparation.get("nli_label_order"),
                    distillation_weight=args.distillation_weight, freeze_word_embeddings=not getattr(args, "unfreeze_embeddings", False))

    def training_args(directory, lr, epochs, validation):
        return TrainingArguments(output_dir=str(directory), learning_rate=lr, num_train_epochs=epochs,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.accumulation, weight_decay=.01, warmup_ratio=.06,
            eval_strategy="epoch" if validation else "no", save_strategy="epoch", save_total_limit=2,
            load_best_model_at_end=validation, metric_for_best_model="macro_f1" if validation else None,
            greater_is_better=True, remove_unused_columns=False, label_names=["labels"],
            fp16=args.device.startswith("cuda"), use_cpu=args.device == "cpu", report_to="none",
            logging_steps=20, disable_tqdm=False, dataloader_num_workers=0, seed=args.seed)

    def predictions(logits, rows):
        if isinstance(logits, tuple):
            logits = logits[0]
        probabilities = torch.as_tensor(logits).float().softmax(-1).numpy()
        return [{"claim_id": r["claim_id"], "true_label": r["label"],
                 "predicted_label": CANONICAL_LABELS[int(vector.argmax())],
                 "probabilities": dict(zip(CANONICAL_LABELS, map(float, vector), strict=True))} for r, vector in zip(rows, probabilities, strict=True)]

    def metric_fn(evaluation):
        return {k: evaluate_predictions(predictions(evaluation.predictions, val_rows))[k] for k in ("macro_f1", "accuracy")}

    trials = []
    for lr in args.learning_rates:
        directory = output / "trials" / f"lr_{lr:g}"
        summary_path = directory / "summary.json"
        if summary_path.is_file():
            trials.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        trainer = Trainer(model=new_model(fit_rows), args=training_args(directory, lr, args.epochs, True),
                          train_dataset=train_data, eval_dataset=val_data, data_collator=collator,
                          processing_class=tokenizer, compute_metrics=metric_fn)
        checkpoint = latest_complete_trainer_checkpoint(directory)
        trainer.train(resume_from_checkpoint=checkpoint)
        val_predictions = predictions(trainer.predict(val_data).predictions, val_rows)
        checkpoint_state = json.loads((Path(trainer.state.best_model_checkpoint) / "trainer_state.json").read_text())
        report = {**evaluate_predictions(val_predictions), "learning_rate": lr, "best_epoch": checkpoint_state["epoch"]}
        atomic_jsonl(val_predictions, directory / "validation_predictions.jsonl")
        write_json(report, summary_path)
        trials.append(report)
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    chosen = sorted(trials, key=selection_key)[0]
    selection = {"selected": chosen, "trials": trials, "selection_metric": "train-only holdout macro-F1; safety errors break ties",
                 "validation_policy": args.validation_policy, "fit_claim_ids": [r["claim_id"] for r in fit_rows],
                 "validation_claim_ids": [r["claim_id"] for r in val_rows]}
    write_json(selection, output / "model_selection.json")
    print("Selected train-holdout score:", chosen["macro_f1"], "epoch:", chosen["best_epoch"], flush=True)
    if args.selection_only:
        return
    final = output / "final_model"
    if not (output / "final_fit_complete.json").is_file():
        trainer = Trainer(model=new_model(records["train"]),
                          args=training_args(final, chosen["learning_rate"], chosen["best_epoch"], False),
                          train_dataset=PairDataset(records["train"], tokenizer, **dataset_kwargs),
                          data_collator=collator, processing_class=tokenizer)
        checkpoint = latest_complete_trainer_checkpoint(final)
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(final))
        tokenizer.save_pretrained(final)
        write_json({"training_claims": len(records["train"]), "selection": chosen}, output / "final_fit_complete.json")
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model = EvidencePairReader.from_pretrained(final).to(args.device).eval()

    def predict_batch(rows):
        dataset = PairDataset(rows, tokenizer, **dataset_kwargs)
        batch = {k: v.to(args.device) for k, v in collator([dataset[i] for i in range(len(dataset))]).items() if k != "labels"}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
            return model(**batch).logits.float().softmax(-1).cpu().tolist()

    metrics = predict_cached(records["dev"], predict_batch, output / "dev", batch_size=args.batch_size, description=f"Pair reader / {args.mode}")
    write_json({**metrics, "selected_train_holdout": chosen, "model_path": str(final)}, output / "final_dev_metrics.json")
    print(json.dumps(metrics, indent=2))
