"""Teammate B method tooling: release folds, decisive labels, selector, mechanics, adapter."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.data.averitec import CANONICAL_LABELS
from src.evaluation.team_release import fold_rows, load_release_fold, read_release_pointer, validate_manifest
from src.retrieval.decisive_evidence import (build_sheet, read_annotations, sample_claims, select_candidates,
                                             selector_targets, validate_annotations, write_csv)
from src.retrieval.evidence_selector import (FEATURE_NAMES, candidate_features, cached_nli, fit_selector,
                                             select_passages, training_matrix)
from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.runs import sha256_file
from src.verification.reader_mechanics import inspect_run

ROOT = Path(__file__).resolve().parents[1]


def claims(n=40, start_year=2018):
    return [{"claim_id": f"train:{i}", "claim": f"mayor closed {i} schools in city {i % 5}", "label": CANONICAL_LABELS[i % 4],
             "metadata": {"claim_date": f"{start_year + i // 20}-0{1 + i % 9}-1{i % 9}"},
             "source_urls": [f"https://src.test/{i}"],
             "evidence": [{"text": f"records show {i} schools closed", "url": f"https://src.test/{i}",
                           "metadata": {"question": "How many?"}}]} for i in range(n)]


def pool(record, n=12):
    i = record["claim_id"].split(":")[1]
    rows = [{"passage_id": f"{record['claim_id']}:p{r}", "text": f"unrelated weather report number {r} sunny", "score": 20.0 - r,
             "url": f"https://news{r}.test/a"} for r in range(n)]
    # The decisive passage sits low in BM25 order but repeats the claim's phrasing.
    rows[7] = {"passage_id": f"{record['claim_id']}:p7", "text": f"official records show mayor closed {i} schools in city", "score": 13.0,
               "url": f"https://src.test/{i}"}
    return {"claim_id": record["claim_id"], "claim": record["claim"], "retrieved": rows}


def write_release(tmp_path, records, fit, val, *, fold="w1", tamper=False):
    data_root = tmp_path / "data"
    manifest = {"protocol_id": "test_v1", "folds": [{"fold_id": fold, "fit_claim_ids": fit, "validation_claim_ids": val}]}
    path = data_root / "processed/averitec/team_splits/test_v1/split_manifest.json"
    write_json(manifest, path)
    digest = sha256_file(path)
    if tamper:
        manifest["folds"][0]["fit_claim_ids"] = fit[:-1]
        write_json(manifest, path)
    config = tmp_path / "team_evaluation.json"
    write_json({"status": "ready", "protocol_id": "test_v1", "fold_ids": [fold],
                "split_manifest": {"relative_path": "processed/averitec/team_splits/test_v1/split_manifest.json", "sha256": digest}}, config)
    return config, data_root


def test_pending_release_is_refused():
    with pytest.raises(RuntimeError, match="not ready"):
        read_release_pointer(ROOT / "configs/team_evaluation.json")


def test_release_fold_loads_verified_ordered_fold(tmp_path):
    records = claims()
    fit, val = [r["claim_id"] for r in records[:20]], [r["claim_id"] for r in records[20:]]
    config, data_root = write_release(tmp_path, records, fit, val)
    fold = load_release_fold(config, data_root, "w1", records)
    a, b = fold_rows(records, fold)
    assert [r["claim_id"] for r in a] == fit and [r["claim_id"] for r in b] == val
    assert fold.binding()["fit_claim_count"] == 20
    with pytest.raises(ValueError, match="not among released"):
        load_release_fold(config, data_root, "w2", records)
    with pytest.raises(ValueError, match="official dev"):
        load_release_fold(config, data_root, "w1", records, forbidden_ids={fit[0]})
    config, data_root = write_release(tmp_path / "t", records, fit, val, tamper=True)
    with pytest.raises(ValueError, match="hash differs"):
        load_release_fold(config, data_root, "w1", records)


@pytest.mark.parametrize("change, message", [
    (lambda f, v: (f, v + f[:1]), "reuses"),
    (lambda f, v: (v, f), "earlier-to-later"),
    (lambda f, v: (f + ["train:missing"], v), "absent"),
    (lambda f, v: (f + f[:1], v), "duplicate"),
])
def test_manifest_validation_rejects_bad_folds(change, message):
    records = claims()
    fit, val = change([r["claim_id"] for r in records[:20]], [r["claim_id"] for r in records[20:]])
    with pytest.raises(ValueError, match=message):
        validate_manifest({"protocol_id": "x", "folds": [{"fold_id": "w1", "fit_claim_ids": fit, "validation_claim_ids": val}]}, records)


def test_pilot_sheet_balances_labels_and_marks_candidate_reasons(tmp_path):
    records = claims()
    rankings = {r["claim_id"]: pool(r) for r in records}
    sample = sample_claims(records, 8, seed=1)
    assert {r["label"] for r in sample} == set(CANONICAL_LABELS)
    assert sample == sample_claims(records, 8, seed=1)
    rows = select_candidates(records[3], rankings["train:3"], per_claim=10)
    assert len(rows) == 10 and len({r["passage_id"] for r in rows}) == 10
    assert [r["bm25_rank"] for r in rows if r["candidate_reason"] == "bm25_displayed"] == [1, 2, 3]
    assert any(r["candidate_reason"] == "answer_aligned_aid" for r in rows)
    assert all(r["review_status"] == "unreviewed" and r["decisive_label"] == "" for r in rows)
    sheet, context = build_sheet(records, rankings, {r["claim_id"] for r in records[:20]}, claims=4, per_claim=10, seed=0)
    assert {r["claim_id"] for r in sheet} <= {r["claim_id"] for r in records[:20]}
    assert "claim_verdict" in context[0] and "claim_verdict" not in sheet[0]
    write_csv(sheet, tmp_path / "s.csv")
    assert len(read_annotations(tmp_path / "s.csv")) == len(sheet)


def test_annotation_validation_requires_quotes_and_fit_claims():
    records = claims(4)
    rankings = {r["claim_id"]: pool(r) for r in records}
    good = {"claim_id": "train:0", "passage_id": "train:0:p7", "decisive_label": "direct_support",
            "quote": "mayor  closed 0 schools", "review_status": "reviewed"}
    rows = [good,
            {"claim_id": "train:0", "passage_id": "train:0:p0", "decisive_label": "unrelated", "review_status": "reviewed"},
            {"claim_id": "train:0", "passage_id": "train:0:p1", "decisive_label": "ambiguous", "review_status": "reviewed"},
            {"claim_id": "train:1", "passage_id": "train:1:p0", "decisive_label": "related_insufficient", "review_status": "silver"},
            {"claim_id": "train:3", "passage_id": "train:3:p0", "decisive_label": "unrelated", "review_status": "reviewed"},
            {"claim_id": "train:2", "passage_id": "train:2:p0", "decisive_label": "", "review_status": "unreviewed"}]
    usable, summary = validate_annotations(rows, rankings, {"train:0", "train:1"})
    assert summary["usable_rows"] == 3 and summary["dropped"] == {"outside_fit_ids": 1, "status_silver": 1, "unlabeled": 1}
    assert summary["claims_with_decisive_passage"] == 1
    assert [t["target"] for t in selector_targets(usable)] == [1, 0]
    assert validate_annotations(rows, rankings, {"train:0", "train:1"}, allow_silver=True)[1]["usable_rows"] == 4
    for bad, message in ((dict(good, quote="not in passage"), "exact quote"), (dict(good, decisive_label="true"), "Unknown decisive"),
                         (dict(good, passage_id="train:0:zzz"), "candidate pool")):
        with pytest.raises(ValueError, match=message):
            validate_annotations([bad], rankings, {"train:0"})
    with pytest.raises(ValueError, match="Duplicate"):
        validate_annotations([good, good], rankings, {"train:0"})


def test_selector_learns_decisive_passages_and_emits_reader_inputs():
    records = claims()
    by_id = {r["claim_id"]: r for r in records}
    rankings = {r["claim_id"]: pool(r) for r in records}
    targets = []
    for r in records[:30]:
        targets += [{"claim_id": r["claim_id"], "passage_id": f"{r['claim_id']}:p{k}", "target": int(k == 7)} for k in (0, 1, 2, 5, 7, 10)]
    x, y, groups, ranks = training_matrix(targets, by_id, rankings, depth=12)
    assert x.shape == (180, len(FEATURE_NAMES)) and y.sum() == 30
    model, report = fit_selector(x, y, groups, ranks)
    assert report["bm25_order_same_candidates"]["hit_at_3"] == 0.0
    assert max(t["hit_at_3"] for t in report["cv_trials"]) == 1.0
    held_out = records[35]
    chosen = select_passages(model, held_out, rankings[held_out["claim_id"]], depth=12)
    assert chosen[0]["passage_id"] == "train:35:p7" and len(chosen) == 3
    assert len({p["url"] for p in chosen}) == 3 and all(len(p["quality"]) == 8 and p["text"] for p in chosen)
    assert chosen[0]["original_rank"] == 8


def test_nli_cache_requires_matching_text():
    item = {"passage_id": "p", "text": "same text"}
    probs = {"entailment": .7, "contradiction": .1, "neutral": .2}
    assert cached_nli(item, {"p": {"text": "same text", "nli_probabilities": probs}}) == probs
    assert cached_nli(item, {"p": {"text": "edited text", "nli_probabilities": probs}}) is None
    features = candidate_features("claim", None, [item], depth=1, nli_cache={"p": {"text": "other", "nli_probabilities": probs}})
    assert features[0, FEATURE_NAMES.index("nli_missing")] == 1.0


def fake_run(tmp_path, labels=CANONICAL_LABELS, refit_epoch=2.0):
    run = tmp_path / "run"
    write_json({"mode": "raw", "epochs": 3, "device": "cuda", "learning_rates": [1e-5]}, run / "reader_run.json")
    write_json({"selected": {"learning_rate": 1e-5, "best_epoch": 2.0, "macro_f1": .4}, "trials": [{}],
                "fit_claim_ids": ["a", "b"], "validation_claim_ids": ["c"]}, run / "model_selection.json")
    write_json({"training_claims": 3, "selection": {"best_epoch": refit_epoch}}, run / "final_fit_complete.json")
    write_json({"id2label": {str(i): l for i, l in enumerate(labels)}, "nli_labels": ["contradiction", "entailment", "neutral"],
                "encoder_config": {"model_type": "deberta-v2"}}, run / "final_model/config.json")
    write_json({"epoch": refit_epoch, "global_step": 10}, run / "final_model/checkpoint-10/trainer_state.json")
    predictions = [{"claim_id": f"dev:{i}", "true_label": CANONICAL_LABELS[i % 4], "predicted_label": CANONICAL_LABELS[0],
                    "probabilities": dict(zip(CANONICAL_LABELS, [.7, .1, .1, .1])), "evidence_passage_ids": [f"dev:{i}:p0"]} for i in range(4)]
    write_jsonl(predictions, run / "dev/predictions.jsonl")
    from src.evaluation.classification import evaluate_predictions
    write_json(evaluate_predictions(predictions), run / "dev/metrics.json")
    return run


def test_mechanics_inspection_passes_consistent_run_and_flags_bad_label_order(tmp_path):
    report = inspect_run(fake_run(tmp_path / "ok"))
    assert report["failed_checks"] == [] and report["dev_metrics_recomputed"]["false_reassurance_count"] == 3
    assert report["mixed_precision_training"] == "fp16 autocast"
    bad = inspect_run(fake_run(tmp_path / "bad", labels=CANONICAL_LABELS[::-1], refit_epoch=3.0))
    assert "verdict label order" in bad["failed_checks"] and "refit uses the selected epoch count" in bad["failed_checks"]


def test_selector_cli_sheet_and_audit_end_to_end(tmp_path):
    records = claims(12)
    write_jsonl(records, tmp_path / "train.jsonl")
    write_jsonl([pool(r) for r in records], tmp_path / "rankings.jsonl")
    script = ROOT / "scripts/run_evidence_selector.py"
    common = ["--claims", str(tmp_path / "train.jsonl"), "--rankings", str(tmp_path / "rankings.jsonl")]
    subprocess.run([sys.executable, str(script), "sheet", *common, "--output", str(tmp_path / "sheet"),
                    "--claims-count", "4", "--provisional-all-train"], check=True, capture_output=True)
    rows = read_jsonl(tmp_path / "sheet/decisive_sheet.jsonl")
    assert len(rows) == 40 and json.loads((tmp_path / "sheet/sheet_config.json").read_text())["fit_ids"]["provisional"]
    rows[0].update(decisive_label="unrelated", review_status="reviewed")
    write_jsonl(rows, tmp_path / "filled.jsonl")
    subprocess.run([sys.executable, str(script), "audit", *common, "--annotations", str(tmp_path / "filled.jsonl"),
                    "--provisional-all-train", "--output", str(tmp_path / "audit.json")], check=True, capture_output=True)
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["usable_rows"] == 1 and audit["dropped"]["unlabeled"] == 39
    refused = subprocess.run([sys.executable, str(script), "train", *common, "--annotations", str(tmp_path / "filled.jsonl"),
                              "--output", str(tmp_path / "sel")], capture_output=True, text=True)
    assert refused.returncode != 0 and "released fit IDs" in refused.stderr
    for script_name in ("run_evidence_reader.py", "run_evidence_selector.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script_name), "--help"], check=True, capture_output=True)


# ---- Real offline torch tests (skipped without torch/transformers) ----

def _tiny_nli(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoModelForSequenceClassification, BertConfig, BertTokenizerFast

    torch.set_num_threads(1)
    directory = tmp_path / "tiny_nli"
    directory.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "source", "domain", "type", "passage", "claim", "event", "occurred", "not", "a", "b", ".", ":"]
    (directory / "vocab.txt").write_text("\n".join(vocab) + "\n")
    BertTokenizerFast(vocab_file=str(directory / "vocab.txt")).save_pretrained(directory)
    config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=24,
                        num_labels=3, id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
    AutoModelForSequenceClassification.from_config(config).save_pretrained(directory)
    return directory


def _prepared(tmp_path, directory):
    from src.retrieval.evidence_pipeline import prepare_pipeline

    data = tmp_path / "data"
    for split, count in (("train", 40), ("dev", 6)):
        rows = [{"claim_id": f"{split}:{i}", "claim": f"claim event {i}", "label": CANONICAL_LABELS[i % 4],
                 "metadata": {"claim_date": f"20{10 + i // 10}-01-02"}} for i in range(count)]
        write_jsonl(rows, data / "processed/averitec" / f"{split}.jsonl")
        write_jsonl([{**r, "retrieved": [{"passage_id": f"{r['claim_id']}:p{k}", "text": f"event occurred {k}", "url": f"https://a{k}.test"}
                                         for k in range(4)]} for r in rows], tmp_path / f"{split}_rankings.jsonl")
    prepared = tmp_path / "prepared"
    prepare_pipeline(argparse.Namespace(output=prepared, train_claims=data / "processed/averitec/train.jsonl",
        dev_claims=data / "processed/averitec/dev.jsonl", train_rankings=tmp_path / "train_rankings.jsonl",
        dev_rankings=tmp_path / "dev_rankings.jsonl", model=str(directory), revision="local", max_length=64, claim_budget=16,
        max_candidates=3, candidate_depth=20, batch_size=8, device="cpu", limit=0, nli_label_order=None))
    return data, prepared


def _train_args(data, prepared, output, **extra):
    return argparse.Namespace(prepared=prepared, data_root=data, mode="raw", output=output, device="cpu", validation_seed=718,
                              validation_policy="grouped", learning_rates=[.001], epochs=2, batch_size=8, accumulation=1, seed=718,
                              distillation_weight=.1, selection_only=False, **extra)


def test_release_fold_training_reports_fixed_epoch_curve_without_refit(tmp_path):
    from src.verification.pair_training import train_reader

    data, prepared = _prepared(tmp_path, _tiny_nli(tmp_path))
    records = read_jsonl(data / "processed/averitec/train.jsonl")
    fit, val = [r["claim_id"] for r in records[:30]], [r["claim_id"] for r in records[30:]]
    config, release_root = write_release(tmp_path, records, fit, val)
    assert release_root == data
    args = _train_args(data, prepared, tmp_path / "fold", release_config=config, fold_id="w1", fold_epoch_policy="fixed")
    train_reader(args)
    selection = json.loads((args.output / "model_selection.json").read_text())
    assert selection["release"]["fold_id"] == "w1" and selection["validation_claim_ids"] == val
    chosen = selection["selected"]
    assert chosen["reported_epoch"] == 2.0 and [p["epoch"] for p in chosen["validation_curve"]] == [1.0, 2.0]
    assert chosen["claim_count"] == 10
    assert not (args.output / "final_model").exists() and not (args.output / "dev").exists()
    run_config = json.loads((args.output / "reader_run.json").read_text())
    assert run_config["validation_policy"] == "released_fold" and "validation_seed" not in run_config
    train_reader(args)  # resumes from the completed trial summary


def test_saved_reader_mechanics_rescore_frozen_inference_and_adapter(tmp_path):
    from src.verification.local_reader import LocalReaderVerifier, infer_rankings
    from src.verification.pair_training import train_reader
    from src.verification.reader_mechanics import mechanics_check

    data, prepared = _prepared(tmp_path, _tiny_nli(tmp_path))
    run = tmp_path / "run"
    train_reader(_train_args(data, prepared, run))
    report = mechanics_check(argparse.Namespace(run=run, output=tmp_path / "mech", data_root=data, prepared=prepared,
                                                rescore_limit=4, device="cpu"))
    assert report["failed_checks"] == [], report["checks"]
    assert report["rescore"]["max_abs_probability_difference"] < 1e-5
    assert (tmp_path / "mech/mechanics_report.md").is_file()

    # Same frozen reader, different (here reversed) passages: an input intervention.
    rows = read_jsonl(tmp_path / "dev_rankings.jsonl")
    write_jsonl([{**r, "retrieved": r["retrieved"][::-1][:3]} for r in rows], tmp_path / "alt/dev_rankings.jsonl")
    metrics = infer_rankings(argparse.Namespace(model=run / "final_model", claims=data / "processed/averitec/dev.jsonl",
                                                rankings=tmp_path / "alt/dev_rankings.jsonl", output=tmp_path / "alt_out",
                                                preparation=prepared / "preparation.json", device="cpu", batch_size=4))
    assert metrics["claim_count"] == 6
    predictions = read_jsonl(tmp_path / "alt_out/predictions.jsonl")
    assert predictions[0]["evidence_passage_ids"] == ["dev:0:p3", "dev:0:p2", "dev:0:p1"]

    verifier = LocalReaderVerifier(run / "final_model", preparation=prepared / "preparation.json")
    passages = [{"passage_id": f"x{k}", "text": "event occurred", "url": f"https://a{k}.test"} for k in range(5)]
    response = verifier.verify("claim event", passages)
    assert response["status"] == "completed" and response["verdict"] in CANONICAL_LABELS
    assert [e["rank"] for e in response["evidence"]] == [1, 2, 3]
    assert "%" not in response["uncertainty_note"] and "final trust" in response["uncertainty_note"]
    assert verifier.verify("claim event", [])["status"] == "insufficient_sources"
    assert verifier.verify("", passages)["status"] == "error"
