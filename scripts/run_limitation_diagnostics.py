"""Bounded, resumable diagnostics using existing rankings; never scans the corpus.

Prepare separates prediction-time variants from explicitly marked oracle inputs.
Frozen experiments change one saved verifier's inputs. Training and selection
are orchestrated by the companion notebook using official-train holdouts only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from scripts.run_evidence_ablation import comparison, evidence_variants, load_partial_predictions
from src.data.averitec import CANONICAL_LABELS
from src.data.corpus import canonical_url
from src.evaluation.classification import evaluate_predictions
from src.evaluation.splits import parse_claim_date as parse_date
from src.retrieval.metrics import evaluate_rankings
from src.retrieval.question_guided import SENTENCE_BOUNDARY, words
from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.progress import progress
from src.utils.runs import sha256_file

VERSION = 1


def atomic_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    write_jsonl(rows, temp)
    temp.replace(path)


def lock_config(path: Path, config: dict) -> None:
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
        raise ValueError(f"Inputs/settings changed. Choose a new output directory: {path.parent}")
    write_json(config, path)


def index_rows(rows: list[dict], *, expected: set[str] | None = None) -> dict[str, dict]:
    result = {}
    for row in rows:
        key = str(row.get("claim_id") or "")
        if not key or key in result:
            raise ValueError(f"Missing or duplicate claim ID: {key!r}")
        result[key] = row
    if expected is not None and set(result) != expected:
        raise ValueError(f"Claim coverage differs: missing={sorted(expected-set(result))[:3]}, extra={sorted(set(result)-expected)[:3]}")
    return result


def load_rankings(path: Path, records: list[dict]) -> dict[str, dict]:
    found = index_rows(read_jsonl(path), expected={str(r["claim_id"]) for r in records})
    for record in records:
        row = found[str(record["claim_id"])]
        if row.get("claim") and row["claim"].strip() != record["claim"].strip():
            raise ValueError(f"Claim text differs in {path}: {record['claim_id']}")
        candidates = row.get("retrieved")
        if not isinstance(candidates, list):
            raise ValueError(f"Malformed retrieved list: {record['claim_id']}")
        ids = [str(p.get("passage_id") or "") for p in candidates]
        if len(ids) != len(set(ids)) or any(not pid for pid in ids):
            raise ValueError(f"Missing/duplicate passage IDs: {record['claim_id']}")
        if any(not str(p.get("text") or "").strip() for p in candidates):
            raise ValueError(f"Blank passage: {record['claim_id']}")
    return found


def source_key(item: dict) -> str:
    return canonical_url(item.get("url")) or f"passage:{item['passage_id']}"


def diverse_top3(candidates: list[dict]) -> list[dict]:
    chosen, seen = [], set()
    for item in candidates:
        key = source_key(item)
        if key not in seen:
            chosen.append(item)
            seen.add(key)
        if len(chosen) == 3:
            break
    used = {p["passage_id"] for p in chosen}
    return (chosen + [p for p in candidates if p["passage_id"] not in used])[:3]


def query_excerpt(text: str, claim: str, tokenizer: Any, budget: int) -> str:
    """Pack verbatim sentences, or a contiguous word window for a long sentence.

    Only the claim and raw passage are used. Nothing is generated or rewritten.
    Sentence order is restored after selection. The model tokenizer enforces
    the same per-passage budget as the original head-truncation baseline.
    """
    original = text.strip()
    count = lambda value: len(tokenizer.encode(value, add_special_tokens=False))
    if count(original) <= budget:
        return original
    sentences = [s.strip() for s in SENTENCE_BOUNDARY.split(original) if s.strip()]
    query = words(claim)
    order = sorted(range(len(sentences)), key=lambda i: (-len(words(sentences[i]) & query), i))
    selected = []
    for i in order:
        proposed = sorted([*selected, i])
        if count(" ".join(sentences[j] for j in proposed)) <= budget:
            selected = proposed
    if selected:
        return " ".join(sentences[i] for i in selected)
    # Long unpunctuated passages: search bounded, verbatim word windows.
    spans = list(re.finditer(r"\S+", original))
    width = min(max(1, budget // 2), len(spans))
    starts = set(range(0, len(spans), max(1, width // 3)))
    starts.add(max(0, len(spans) - width))
    windows = []
    for start in sorted(starts):
        end = min(len(spans), start + width)
        while end > start and count(original[spans[start].start():spans[end - 1].end()]) > budget:
            end -= 1
        if end > start:
            excerpt = original[spans[start].start():spans[end - 1].end()]
            windows.append((-len(words(excerpt) & query), start, excerpt))
    return min(windows)[2] if windows else original[:1]


def prediction_variant(claim_id: str, claim: str, candidates: list[dict], mode: str, tokenizer: Any, budget: int) -> dict:
    """Whitelist fields: gold labels, answers and URLs from annotations cannot enter."""
    selected = diverse_top3(candidates[:20]) if mode == "diverse_sentences" else candidates[:3]
    if mode == "claim_only":
        selected = []
    passages = []
    for item in selected:
        text = str(item["text"])
        if mode in {"sentences", "diverse_sentences"}:
            text = query_excerpt(text, claim, tokenizer, budget)
        passages.append({"passage_id": str(item["passage_id"]), "text": text, "url": item.get("url")})
    return {"claim_id": claim_id, "claim": claim, "retrieved": passages}


def audit_claim(record: dict, candidates: list[dict], qrels: set[str], tokenizer: Any, budget: int) -> dict:
    top = candidates[:3]
    row = {"claim_id": record["claim_id"], "label": record["label"], "judged": bool(qrels),
           "available_candidate_count": len(candidates), "top3_distinct_sources": len({source_key(p) for p in top}),
           "top3_passage_count": len(top), "claim_date": parse_date(record.get("metadata", {}).get("claim_date"))}
    for k in (3, 20, 100):
        # A short saved list is not a top-100 candidate pool.
        row[f"url_hit_at_available_{k}"] = bool({str(p["passage_id"]) for p in candidates[:k]} & qrels) if qrels else None
    lengths, head_coverage, excerpt_coverage = [], [], []
    query = words(record["claim"])
    for p in top:
        tokens = tokenizer.encode(p["text"], add_special_tokens=False)
        lengths.append(len(tokens))
        full_terms = words(p["text"]) & query
        head_terms = words(tokenizer.decode(tokens[:budget], skip_special_tokens=True)) & query
        excerpt_terms = words(query_excerpt(p["text"], record["claim"], tokenizer, budget)) & query
        if full_terms:
            head_coverage.append(len(head_terms) / len(full_terms))
            excerpt_coverage.append(len(excerpt_terms) / len(full_terms))
    row.update({"passages_over_budget": sum(n > budget for n in lengths), "passage_token_lengths": lengths,
                "head_query_term_retention": float(np.mean(head_coverage)) if head_coverage else None,
                "excerpt_query_term_retention": float(np.mean(excerpt_coverage)) if excerpt_coverage else None})
    return row


def export_manual_audit(records: list[dict], pools: dict[str, dict], qrels: dict[str, set], audits: list[dict], output: Path, size: int, seed: int) -> None:
    groups = defaultdict(list)
    for row in audits:
        group = "unjudged" if not row["judged"] else "top3_hit" if row["url_hit_at_available_3"] else "deeper_hit" if row["url_hit_at_available_100"] else "no_saved_hit"
        groups[group].append(row["claim_id"])
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < size and any(groups.values()):
        for group in sorted(groups):
            if groups[group] and len(selected) < size:
                selected.append(groups[group].pop())
    by_id = index_rows(records)
    exported, key = [], []
    for cid in selected:
        candidates = pools[cid]["retrieved"]
        chosen = list(candidates[:3])
        deeper = next((p for p in candidates[3:] if str(p["passage_id"]) in qrels.get(cid, set())), None)
        if deeper:
            chosen.append(deeper)
        rng.shuffle(chosen)  # Hide rank and URL-match status from the annotator.
        for item in chosen:
            aid = f"audit_{len(exported):04d}"
            exported.append({"audit_id": aid, "claim": by_id[cid]["claim"], "source_url": item.get("url"),
                             "passage_text": item["text"], "relevance": "", "notes": ""})
            key.append({"audit_id": aid, "claim_id": cid, "passage_id": item["passage_id"],
                        "proxy_match": str(item["passage_id"]) in qrels.get(cid, set()), "gold_label": by_id[cid]["label"]})
    path = output / "manual_passage_audit.csv"
    if not path.exists():  # Preserve any judgments entered by the user on a rerun.
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["audit_id", "claim", "source_url", "passage_text", "relevance", "notes"])
            writer.writeheader()
            writer.writerows(exported)
    write_json({"rows": key, "sampling": "Balanced diagnostic strata, not a representative prevalence estimate."}, output / "manual_audit_key.json")


def prepare(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    inputs = {"train": args.train_claims, "dev": args.dev_claims, "train_rankings": args.train_rankings,
              "dev_rankings": args.dev_rankings, "qrels": args.qrels}
    if args.dev_top100:
        inputs["dev_top100"] = args.dev_top100
    extras = {}
    for spec in args.extra:
        name, value = spec.split("=", 1)
        if not re.fullmatch(r"[a-z0-9_]+", name) or name in extras:
            raise ValueError("Extra ranking names must be unique lowercase letters/numbers/underscores")
        extras[name] = Path(value)
        inputs[f"extra_{name}"] = Path(value)
    config = {"version": VERSION, "inputs": {k: sha256_file(v) for k, v in inputs.items()},
              "tokenizer_config_sha256": sha256_file(args.model_dir / "config.json"),
              "budget": args.budget, "audit_size": args.audit_size, "seed": args.seed}
    lock_config(out / "prepare_config.json", config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    records = {"train": read_jsonl(args.train_claims), "dev": read_jsonl(args.dev_claims)}
    for rows in records.values():
        index_rows(rows)
        if any(not str(r.get("claim") or "").strip() or r.get("label") not in CANONICAL_LABELS for r in rows):
            raise ValueError("Missing claim text or invalid verdict label")
    if set(index_rows(records["train"])) & set(index_rows(records["dev"])):
        raise ValueError("Train/dev claim IDs overlap")
    rankings = {"train": load_rankings(args.train_rankings, records["train"]), "dev": load_rankings(args.dev_rankings, records["dev"])}
    pools = load_rankings(args.dev_top100, records["dev"]) if args.dev_top100 else rankings["dev"]
    if args.dev_top100:
        for cid, row in pools.items():
            old = rankings["dev"][cid]["retrieved"][:3]
            new = row["retrieved"][:3]
            if [(p["passage_id"], p["text"]) for p in old] != [(p["passage_id"], p["text"]) for p in new]:
                raise ValueError(f"Top-100 and baseline top-three inputs differ: {cid}")
    qrels = {str(r["claim_id"]): set(map(str, r.get("relevant_passage_ids") or [])) for r in read_jsonl(args.qrels)}
    qrels = {cid: ids for cid, ids in qrels.items() if ids}
    for split, rows in records.items():
        generated = {mode: [] for mode in ("head", "sentences", "diverse_sentences", "claim_only")}
        with progress(total=len(rows), description=f"Prepare {split}", unit="claim") as bar:
            for row in rows:
                cid = str(row["claim_id"])
                candidates = rankings[split][cid]["retrieved"]
                for mode in generated:
                    generated[mode].append(prediction_variant(cid, row["claim"], candidates, mode, tokenizer, args.budget))
                bar.update(1)
        for mode, materialized in generated.items():
            atomic_rows(materialized, out / "prepared" / mode / f"{split}_rankings.jsonl")
    # Gold selection is diagnostic only. No oracle train rankings are created.
    oracles = {"oracle_url_head": [], "oracle_url_sentences": [], "oracle_answers": []}
    audits = []
    for row in records["dev"]:
        cid = str(row["claim_id"])
        candidates = pools[cid]["retrieved"]
        relevant = qrels.get(cid, set())
        selected = [p for p in candidates if str(p["passage_id"]) in relevant][:3]
        selected += [p for p in candidates if p not in selected][:3 - len(selected)]
        for name, mode in (("oracle_url_head", "head"), ("oracle_url_sentences", "sentences")):
            oracles[name].append(prediction_variant(cid, row["claim"], selected, mode, tokenizer, args.budget))
        answers = [p for p in row.get("evidence", []) if str(p.get("text") or "").strip()][:3]
        oracles["oracle_answers"].append(prediction_variant(cid, row["claim"], answers, "head", tokenizer, args.budget))
        audits.append(audit_claim(row, candidates, relevant, tokenizer, args.budget))
    for name, rows in oracles.items():
        atomic_rows(rows, out / "oracle" / name / "dev_rankings.jsonl")
    for name, path in extras.items():
        found = load_rankings(path, records["dev"])
        rows = [prediction_variant(r["claim_id"], r["claim"], found[r["claim_id"]]["retrieved"], "head", tokenizer, args.budget) for r in records["dev"]]
        atomic_rows(rows, out / "prepared" / name / "dev_rankings.jsonl")
    atomic_rows(audits, out / "claim_audit.jsonl")
    export_manual_audit(records["dev"], pools, qrels, audits, out, args.audit_size, args.seed)
    report = {"label_counts": {s: dict(Counter(r["label"] for r in rows)) for s, rows in records.items()},
              "dev_claim_count": len(audits), "judged_dev_claim_count": sum(r["judged"] for r in audits),
              "candidate_depth_range": [min(len(r["retrieved"]) for r in pools.values()), max(len(r["retrieved"]) for r in pools.values())],
              "dev_top100_supplied": bool(args.dev_top100), "candidate_coverage": {},
              "passages_over_budget": sum(r["passages_over_budget"] for r in audits),
              "passages_measured": sum(r["top3_passage_count"] for r in audits),
              "claims_with_repeated_top3_source": sum(r["top3_distinct_sources"] < r["top3_passage_count"] for r in audits),
              "notes": ["URL matches do not establish decisive passage relevance.",
                        "Oracle URL selection uses dev qrels and must never be used for model selection or deployment.",
                        "The supplied per-claim candidate pools simplify deployment; global search and publication-time availability are not tested."]}
    judged = [r for r in audits if r["judged"]]
    for k in (3, 20, 100):
        if k == 100 and not args.dev_top100:
            continue
        report["candidate_coverage"][str(k)] = sum(r[f"url_hit_at_available_{k}"] for r in judged) / len(judged) if judged else None
    # Exact normalization checks surface duplicated claims; they do not prove
    # source independence or eliminate foundation-model pretraining exposure.
    normalized_train = defaultdict(list)
    for r in records["train"]:
        normalized_train[" ".join(re.findall(r"\w+", r["claim"].casefold()))].append(r["claim_id"])
    overlaps = [{"dev_claim_id": r["claim_id"], "train_claim_ids": normalized_train[key]}
                for r in records["dev"] if (key := " ".join(re.findall(r"\w+", r["claim"].casefold()))) in normalized_train]
    report["normalized_exact_train_dev_claim_overlaps"] = overlaps
    report["claim_date_coverage"] = {split: sum(parse_date(r.get("metadata", {}).get("claim_date")) is not None for r in rows) for split, rows in records.items()}
    report["truncation"] = {key: float(np.mean(values)) if values else None for key in
                            ("head_query_term_retention", "excerpt_query_term_retention")
                            if (values := [r[key] for r in audits if r[key] is not None])}
    report["coverage_by_gold_label"] = {}
    for label in CANONICAL_LABELS:
        group = [r for r in audits if r["label"] == label]
        judged_group = [r for r in group if r["judged"]]
        report["coverage_by_gold_label"][label] = {
            "all_claims": len(group), "judged_claims": len(judged_group),
            "hits_at_3": sum(r["url_hit_at_available_3"] for r in judged_group),
            "hits_in_saved_pool": sum(r["url_hit_at_available_100"] for r in judged_group)}
    for name in ("head", "sentences", "diverse_sentences", *extras):
        rows = read_jsonl(out / "prepared" / name / "dev_rankings.jsonl")
        ids = {r["claim_id"]: [p["passage_id"] for p in r["retrieved"]] for r in rows}
        report.setdefault("top3_retrieval", {})[name] = evaluate_rankings(ids, qrels, k=3) if judged else None
    write_json(report, out / "data_audit.json")
    print(json.dumps(report, indent=2))


def checked_predictions(path: Path, records: list[dict]) -> list[dict]:
    rows = read_jsonl(path)
    found = index_rows(rows, expected={str(r["claim_id"]) for r in records})
    for r in records:
        pred = found[str(r["claim_id"])]
        values = [float(pred.get("probabilities", {}).get(label, float("nan"))) for label in CANONICAL_LABELS]
        if pred["true_label"] != r["label"] or pred["predicted_label"] not in CANONICAL_LABELS:
            raise ValueError(f"Prediction label mismatch: {r['claim_id']}")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values) or abs(sum(values) - 1) > 1e-4:
            raise ValueError(f"Invalid probability vector: {r['claim_id']}")
    return [found[str(r["claim_id"])] for r in records]


def frozen(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from scripts.train_transformer_verifier import compose_text

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime before frozen inference")
    records = read_jsonl(args.dev_claims)
    settings = json.loads(args.model_config.read_text(encoding="utf-8"))
    base = load_rankings(args.output / "prepared/head/dev_rankings.jsonl", records)
    attached = [{**r, "retrieved_evidence": base[r["claim_id"]]["retrieved"]} for r in records]
    interventions = evidence_variants(attached)
    conditions = {"head": attached, "missing": interventions["blank"], "wrong_claim": interventions["swapped"],
                  "reverse_order": [{**r, "retrieved_evidence": list(reversed(r["retrieved_evidence"]))} for r in attached]}
    paths = {}
    for directory in (args.output / "prepared", args.output / "oracle"):
        for path in sorted(directory.glob("*/dev_rankings.jsonl")):
            if path.parent.name in {"head", "claim_only"}:
                continue
            found = load_rankings(path, records)
            conditions[path.parent.name] = [{**r, "retrieved_evidence": found[r["claim_id"]]["retrieved"]} for r in records]
            paths[path.parent.name] = sha256_file(path)
    weights = sorted([*args.model_dir.glob("*.safetensors"), *args.model_dir.glob("pytorch_model*.bin")])
    if not weights:
        raise FileNotFoundError(f"No saved model weights in {args.model_dir}")
    config = {"version": VERSION, "dev_sha256": sha256_file(args.dev_claims),
              "model_config_sha256": sha256_file(args.model_dir / "config.json"),
              "weights": {p.name: sha256_file(p) for p in weights}, "settings": settings,
              "prepared_config_sha256": sha256_file(args.output / "prepare_config.json"),
              "variant_sha256": paths, "batch_size": args.batch_size, "device": args.device}
    folder = args.output / "frozen"
    folder.mkdir(parents=True, exist_ok=True)
    lock_config(folder / "config.json", config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True).to(args.device).eval()
    if [model.config.id2label.get(i) for i in range(4)] != list(CANONICAL_LABELS):
        raise ValueError("Saved model label order differs from the evaluator")
    for name, rows in conditions.items():
        run = folder / name
        run.mkdir(parents=True, exist_ok=True)
        final, partial = run / "predictions.jsonl", run / "predictions.partial.jsonl"
        if final.exists():
            predictions = checked_predictions(final, rows)
        else:
            done = load_partial_predictions(partial, {str(r["claim_id"]) for r in rows})
            expected_labels = {r["claim_id"]: r["label"] for r in rows}
            if any(p["true_label"] != expected_labels[cid] for cid, p in done.items()):
                raise ValueError("Partial checkpoint gold labels differ")
            pending = [r for r in rows if r["claim_id"] not in done]
            with progress(total=len(rows), initial=len(done), description=f"Frozen {name}", unit="claim") as bar:
                for start in range(0, len(pending), args.batch_size):
                    batch = pending[start:start + args.batch_size]
                    texts = [compose_text(r, tokenizer, int(settings["claim_token_budget"]), int(settings["evidence_token_budget_per_passage"])) for r in batch]
                    encoded = tokenizer(texts, padding=True, truncation=True, max_length=int(settings["max_length"]), return_tensors="pt")
                    with torch.inference_mode():
                        values = model(**{k: v.to(args.device) for k, v in encoded.items()}).logits.float().softmax(-1).cpu().tolist()
                    new = []
                    for r, vector in zip(batch, values, strict=True):
                        p = {"claim_id": r["claim_id"], "true_label": r["label"],
                             "predicted_label": CANONICAL_LABELS[int(np.argmax(vector))],
                             "probabilities": dict(zip(CANONICAL_LABELS, vector, strict=True)),
                             "evidence_passage_ids": [p["passage_id"] for p in r["retrieved_evidence"][:3]]}
                        done[r["claim_id"]] = p
                        new.append(p)
                    # Commit every GPU batch; an interruption loses at most one batch.
                    with partial.open("a", encoding="utf-8", newline="\n") as handle:
                        for p in new:
                            handle.write(json.dumps(p, ensure_ascii=False) + "\n")
                        handle.flush()
                    bar.update(len(new))
            atomic_rows([done[r["claim_id"]] for r in rows], final)
            predictions = checked_predictions(final, rows)
        write_json({**evaluate_predictions(predictions), "oracle_only": name.startswith("oracle_"),
                    "interpretation": "Frozen-model input intervention; training distribution unchanged."}, run / "metrics.json")
        print(name, "macro-F1", round(evaluate_predictions(predictions)["macro_f1"], 4), flush=True)
    if args.reference_predictions:
        old = checked_predictions(args.reference_predictions, records)
        now = checked_predictions(folder / "head/predictions.jsonl", records)
        write_json(comparison(old, now), folder / "baseline_reproduction.json")


def paired_bootstrap(original: list[dict], altered: list[dict], trials: int = 1000, seed: int = 718) -> dict:
    """Paired, class-stratified intervals: conditional on this dev label mix."""
    old = index_rows(original)
    new = index_rows(altered, expected=set(old))
    ids = sorted(old)
    if any(old[cid]["true_label"] != new[cid]["true_label"] for cid in ids):
        raise ValueError("Paired predictions disagree on gold labels")
    labels = {label: i for i, label in enumerate(CANONICAL_LABELS)}
    y = np.array([labels[old[c]["true_label"]] for c in ids])
    a = np.array([labels[old[c]["predicted_label"]] for c in ids])
    b = np.array([labels[new[c]["predicted_label"]] for c in ids])

    def metrics(gold, pred):
        mat = np.bincount(gold * 4 + pred, minlength=16).reshape(4, 4)
        den = mat.sum(0) + mat.sum(1)
        f1 = np.divide(2 * mat.diagonal(), den, out=np.zeros(4), where=den != 0).mean()
        return np.array([f1, (gold == pred).mean()])

    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == i) for i in range(4)]
    delta = []
    for _ in range(trials):
        indices = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups if len(g)])
        delta.append(metrics(y[indices], b[indices]) - metrics(y[indices], a[indices]))
    bounds = np.quantile(np.asarray(delta), [0.025, 0.975], axis=0)
    return {"trials": trials, "method": "paired class-stratified bootstrap; conditional on dev class counts; training-seed variability excluded",
            "delta_macro_f1": float(metrics(y, b)[0] - metrics(y, a)[0]),
            "delta_macro_f1_95pct": bounds[:, 0].tolist(), "delta_accuracy_95pct": bounds[:, 1].tolist(),
            **comparison(original, altered)}


def summarize(args: argparse.Namespace) -> None:
    out = args.output
    records = read_jsonl(args.dev_claims)
    reports, predictions, kinds = {}, {}, {}
    for path in sorted((out / "frozen").glob("*/predictions.jsonl")):
        name = "frozen/" + path.parent.name
        predictions[name] = checked_predictions(path, records)
        kinds[name] = "oracle diagnostic" if path.parent.name.startswith("oracle_") else "frozen input diagnostic"
    for path in sorted((out / "training").glob("*/final_dev_predictions.jsonl")):
        name = "trained/" + path.parent.name
        predictions[name] = checked_predictions(path, records)
        kinds[name] = "matched train/dev pipeline"
    for name, rows in predictions.items():
        reports[name] = {**evaluate_predictions(rows), "kind": kinds[name]}
        n_non = sum(r["true_label"] != "Supported" for r in rows)
        n_sup = len(rows) - n_non
        rates = reports[name]["decision_error_rates"]
        reports[name]["conditional_errors"] = {"fr_over_non_supported": rates["false_reassurance_count"] / n_non if n_non else None,
                                               "fa_over_supported": rates["false_alarm_count"] / n_sup if n_sup else None,
                                               "non_supported_count": n_non, "supported_count": n_sup}
    reference = predictions.get("frozen/head")
    paired = {name: paired_bootstrap(reference, rows, args.bootstrap_trials) for name, rows in predictions.items()
              if reference is not None and name != "frozen/head"}
    # Compare trained interventions on train holdout and dev; no dev-based winner.
    validation = {}
    for path in sorted((out / "training").glob("*/model_selection.json")):
        full_selection = json.loads(path.read_text(encoding="utf-8"))
        validation[path.parent.name] = {**full_selection["selected_trial"], **{
            k: full_selection.get(k) for k in ("validation_policy", "validation_seed", "training_seed", "selection_train_claim_count")}}
    audits = {r["claim_id"]: r for r in read_jsonl(out / "claim_audit.jsonl")}
    subsets = {}
    for name, rows in predictions.items():
        groups = defaultdict(list)
        for p in rows:
            row = audits[p["claim_id"]]
            group = "unjudged" if not row["judged"] else "url_top3_hit" if row["url_hit_at_available_3"] else "url_top3_miss"
            groups[group].append(p)
            meta = next(r for r in records if r["claim_id"] == p["claim_id"]).get("metadata", {})
            date = row.get("claim_date")
            if date:
                groups["month/" + date[:7]].append(p)
            location = meta.get("location_iso_code")
            if isinstance(location, str) and location:
                groups["location/" + location].append(p)
        subsets[name] = {g: evaluate_predictions(items) for g, items in groups.items() if len(items) >= 20}
    report = {"results": reports, "paired_vs_frozen_head": paired, "train_holdout_selection": validation,
              "subgroups_minimum_n_20": subsets, "cautions": [
                  "Dev was inspected repeatedly. All dev findings are exploratory; reserve untouched data for the frozen final pipeline.",
                  "Bootstrap intervals do not account for repeated dev inspection, related claims, training seeds, or pretraining exposure.",
                  "Oracle URL matches do not certify decisive evidence. Gold-answer scores are not formal ceilings.",
                  "Subset differences are descriptive and do not identify causal effects."]}
    report["planned_contrasts"] = {}
    silver_path = out / "silver_supervision_audit.json"
    if silver_path.is_file():
        report["silver_supervision_audit"] = json.loads(silver_path.read_text(encoding="utf-8"))
    for a, b in (("frozen/sentences", "frozen/diverse_sentences"),
                 ("trained/head_balanced", "trained/head_unweighted"),
                 ("trained/head_balanced", "trained/sentences_balanced"),
                 ("trained/head_unweighted", "trained/sentences_unweighted"),
                 ("trained/sentences_balanced", "trained/diverse_sentences_balanced"),
                 ("trained/claim_only_balanced", "trained/claim_only_unweighted"),
                 ("trained/head_balanced", "trained/head_balanced_requested_epochs")):
        if a in predictions and b in predictions:
            report["planned_contrasts"][a + " -> " + b] = paired_bootstrap(predictions[a], predictions[b], args.bootstrap_trials)
    audit_path = out / "manual_passage_audit.csv"
    if audit_path.exists():
        allowed = {"decisive_support", "decisive_refute", "partial", "irrelevant", "unclear"}
        key = {r["audit_id"]: r for r in json.loads((out / "manual_audit_key.json").read_text(encoding="utf-8"))["rows"]}
        counts = defaultdict(Counter)
        with audit_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                value = row["relevance"].strip().lower()
                if not value:
                    continue
                if value not in allowed or row["audit_id"] not in key:
                    raise ValueError(f"Invalid manual audit judgment: {row['audit_id']}, {value}")
                group = "proxy_match" if key[row["audit_id"]]["proxy_match"] else "no_proxy_match"
                counts[group][value] += 1
        report["manual_audit"] = {"counts": {g: dict(v) for g, v in counts.items()},
                                  "warning": "Diagnostic sample balances strata; these counts do not estimate population relevance prevalence."}
    write_json(report, out / "limitation_results.json")
    lines = ["# Limitation study results", "", "All verdict rows use the same official dev claims. FR and FA are counts.", "",
             "| Condition | Macro-F1 | Accuracy | FR | FA | Kind |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for name, m in reports.items():
        rates = m["decision_error_rates"]
        lines.append(f"| {name} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} | {rates['false_reassurance_count']} | {rates['false_alarm_count']} | {m['kind']} |")
    lines.extend(["", "## Paired changes versus the frozen BM25 baseline", "",
                  "| Condition | Delta macro-F1 | 95% interval | Fixed errors | Lost correct answers |", "| --- | ---: | --- | ---: | ---: |"])
    for name, p in paired.items():
        lo, hi = p["delta_macro_f1_95pct"]
        lines.append(f"| {name} | {p['delta_macro_f1']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p['originally_wrong_now_correct']} | {p['originally_correct_now_wrong']} |")
    lines.extend(["", "## Reading the results", "",
                  "- Missing/wrong evidence barely changes predictions: investigate weak evidence use. Large changes alone do not prove good reasoning.",
                  "- URL-oracle raw passages help: candidate ranking likely matters; check decisive relevance manually before trusting this proxy.",
                  "- Oracle answers help but URL-oracle passages do not: inspect passage decisiveness, input format and reading ability.",
                  "- Same-passage sentence packing helps only after matched training: input distribution and token allocation matter.",
                  "- Unweighted loss changes minority recall and safety errors: class weighting is a tradeoff, not an automatic improvement.",
                  "- Holdout gains vanish on later dev: investigate temporal/source shift and training uncertainty.", "",
                  "Use train holdout to choose configurations. Keep dev diagnostic and the test split untouched.", "",
                  "See limitation_results.json for per-class metrics, confusion matrices, calibration, subgroups and full paired comparisons."])
    (out / "limitation_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("train-claims", "dev-claims", "train-rankings", "dev-rankings", "qrels", "model-dir", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--dev-top100", type=Path)
    prep.add_argument("--extra", action="append", default=[], help="Optional name=dev_rankings path, e.g. bge or fusion")
    prep.add_argument("--budget", type=int, default=110)
    prep.add_argument("--audit-size", type=int, default=24)
    prep.add_argument("--seed", type=int, default=718)
    infer = sub.add_parser("frozen")
    for name in ("dev-claims", "model-dir", "model-config", "output"):
        infer.add_argument("--" + name, type=Path, required=True)
    infer.add_argument("--reference-predictions", type=Path)
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--batch-size", type=int, default=16)
    summ = sub.add_parser("summarize")
    summ.add_argument("--output", type=Path, required=True)
    summ.add_argument("--dev-claims", type=Path, required=True)
    summ.add_argument("--bootstrap-trials", type=int, default=1000)
    args = parser.parse_args()
    for name in ("budget", "batch_size", "bootstrap_trials", "audit_size"):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    {"prepare": prepare, "frozen": frozen, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
