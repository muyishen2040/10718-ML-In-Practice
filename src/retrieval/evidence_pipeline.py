"""Resumable preparation of raw, cleaned and NLI-ranked top-three evidence."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from src.data.evidence import index_claims
from src.evaluation.splits import parse_claim_date
from src.retrieval.evidence_quality import (EvidencePolicy, clean_candidates, nli_label_order, pair_premise,
                                          publication_date, quality_features, select_evidence)
from src.utils.checkpoints import append_journal, atomic_jsonl, bind_run, read_journal
from src.utils.io import iter_jsonl, read_jsonl, write_json
from src.utils.progress import progress
from src.utils.runs import sha256_file


def prepare_pipeline(args):
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Choose a GPU Colab runtime before preparing NLI scores")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    existing_path = output / "preparation.json"
    previous = json.loads(existing_path.read_text(encoding="utf-8")) if existing_path.exists() else {}
    policy = EvidencePolicy(max_candidates=args.max_candidates, candidate_depth=args.candidate_depth)
    revision = args.revision or previous.get("revision") or model_info(args.model).sha
    inputs = {"train_claims": args.train_claims, "dev_claims": args.dev_claims,
              "train_rankings": args.train_rankings, "dev_rankings": args.dev_rankings}
    config = {"version": 1, "nli_model": args.model, "revision": revision, "max_length": args.max_length,
              "claim_budget": args.claim_budget, "policy": policy.to_dict(), "limit": args.limit,
              "batch_size": args.batch_size, "device": args.device, "nli_label_order": args.nli_label_order,
              "input_hashes": {name: sha256_file(path) for name, path in inputs.items()},
              "implementation_sha256": sha256_file(Path(__file__)),
              "quality_policy_sha256": sha256_file(Path(__file__).with_name("evidence_quality.py"))}
    bind_run(existing_path, config)
    records = {s: read_jsonl(inputs[f"{s}_claims"]) for s in ("train", "dev")}
    by_id = {s: index_claims(rows) for s, rows in records.items()}
    if set(by_id["train"]) & set(by_id["dev"]):
        raise ValueError("Official train/dev claim IDs overlap")
    if args.limit:
        records = {s: rows[:args.limit] for s, rows in records.items()}
        by_id = {s: index_claims(rows) for s, rows in records.items()}
    model, tokenizer, order = None, None, None
    totals = {}

    def score(claim, items):
        nonlocal model, tokenizer, order
        if not items:
            return
        if model is None:
            print(f"Loading NLI scorer {args.model}@{revision}", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(args.model, revision=revision).to(args.device).eval()
            if int(model.config.num_labels) != 3:
                raise ValueError("The passage scorer must have exactly three NLI labels")
            order = nli_label_order(model.config.id2label, args.nli_label_order)
        hypothesis = tokenizer.decode(tokenizer.encode(claim, add_special_tokens=False)[:args.claim_budget], skip_special_tokens=True)
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            encoded = tokenizer([pair_premise(p) for p in batch], [hypothesis] * len(batch), padding=True,
                                truncation="only_first", max_length=args.max_length, return_tensors="pt")
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
                values = model(**{k: v.to(args.device) for k, v in encoded.items()}).logits.float().softmax(-1).cpu().tolist()
            for item, vector in zip(batch, values, strict=True):
                if any(not math.isfinite(v) or not 0 <= v <= 1 for v in vector) or abs(sum(vector) - 1) > 1e-4:
                    raise ValueError(f"Non-finite NLI output for {item['passage_id']}; checkpoint preserved")
                item["nli_probabilities"] = dict(zip(order, vector, strict=True))

    for split in ("train", "dev"):
        expected = set(by_id[split])
        journal = output / f"{split}.partial.jsonl"
        done = read_journal(journal, expected)
        seen = set()
        with progress(total=len(expected), initial=len(done), description=f"NLI evidence / {split}", unit="claim") as bar:
            for ranking in iter_jsonl(inputs[f"{split}_rankings"]):
                cid = str(ranking.get("claim_id") or "")
                if args.limit and cid not in expected:
                    continue
                if cid not in expected or cid in seen:
                    raise ValueError(f"Unexpected/duplicate ranking claim ID: {cid}")
                seen.add(cid)
                if cid in done:
                    continue
                record = by_id[split][cid]
                if ranking.get("claim") and ranking["claim"].strip() != record["claim"].strip():
                    raise ValueError(f"Claim text differs for {cid}")
                candidates = ranking.get("retrieved")
                if not isinstance(candidates, list):
                    raise ValueError(f"Malformed retrieved list: {cid}")
                date = parse_claim_date(record.get("metadata", {}).get("claim_date"))
                cleaned, rejected = clean_candidates(record["claim"], date, candidates, policy)
                # Raw reference is the exact original top three, even if hygiene rejects it.
                raw = [{"passage_id": p["passage_id"], "text": str(p.get("text") or ""), "url": p.get("url"),
                        "publication_date": publication_date(p), "original_rank": i + 1,
                        "quality": quality_features(record["claim"], p, i + 1, date)} for i, p in enumerate(candidates[:3])]
                union = {p["passage_id"]: p for p in [*cleaned, *raw]}
                score(record["claim"], list(union.values()))
                for p in [*cleaned, *raw]:
                    p["nli_probabilities"] = union[p["passage_id"]]["nli_probabilities"]
                row = {"claim_id": cid, "claim": record["claim"], "raw": raw,
                       "clean": select_evidence(cleaned, rerank=False), "clean_nli": select_evidence(cleaned, rerank=True),
                       "scored_candidates": [{**p, "eligible_after_cleaning": p["passage_id"] in {q["passage_id"] for q in cleaned}} for p in union.values()],
                       "rejections": rejected, "scored_pair_count": len(union), "missing_date": date is None}
                append_journal(journal, [row])
                done[cid] = row
                bar.update(1)
                bar.set_postfix(pairs=len(union), rejected=len(rejected))
        if seen != expected or set(done) != expected:
            raise ValueError(f"Incomplete {split} rankings: {len(done)} / {len(expected)}")
        ordered = [done[r["claim_id"]] for r in records[split]]
        for mode in ("raw", "clean", "clean_nli"):
            atomic_jsonl([{"claim_id": r["claim_id"], "claim": r["claim"], "retrieved": r[mode]} for r in ordered],
                         output / mode / f"{split}_rankings.jsonl")
        counts = Counter(r["reason"].split(":")[0] for row in ordered for r in row["rejections"])
        totals[split] = {"claim_count": len(ordered), "scored_pairs": sum(r["scored_pair_count"] for r in ordered),
                         "rejections_among_scanned_candidates": dict(counts), "missing_claim_dates": sum(r["missing_date"] for r in ordered),
                         "empty_clean_pools": sum(not r["clean"] for r in ordered)}
    report = {"config": config, "totals": totals, "finalized": True,
              "temporal_caveat": "Only explicit publication metadata is filtered. Unknown/updated pages are not certified historical snapshots.",
              "supervision": "Frozen external NLI model; no AVeriTeC verdicts, gold answers or qrels used in selection."}
    write_json(report, output / "preparation_report.json")
    print(json.dumps(report, indent=2))
