"""Build conservative train-only passage labels from AVeriTeC gold answers.

The *human-written answer* and source URL identify a likely source passage.
They are used only to generate weak training labels and audit rows, never as
prediction-time features or as synthetic passages in the retrieval corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.corpus import canonical_url  # noqa: E402
from src.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.runs import resolve_run_directory, sha256_file, write_run_manifest  # noqa: E402


WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset("a an and are as at be by for from has have in into is it of on or that the their there these this to was were will with yes no true false".split())


def content_words(text: str) -> set[str]:
    return {word for word in WORD.findall(text.casefold()) if word not in STOP and len(word) > 1}


def score_passage(text: str, evidence: list[dict[str, Any]], url: str | None) -> tuple[float, int, str, str]:
    passage_words = content_words(text)
    best = (0.0, 0, "", "")
    if not passage_words:
        return best
    for answer in evidence:
        if canonical_url(answer.get("url")) != url:
            continue
        answer_text = str(answer.get("text") or "")
        question_text = str((answer.get("metadata") or {}).get("question") or "")
        answer_words = content_words(answer_text)
        question_words = content_words(question_text)
        overlap = len(answer_words & passage_words)
        # The answer has the stronger weight. Very short answers are not
        # enough to designate a decisive web chunk by themselves.
        answer_recall = overlap / len(answer_words) if answer_words else 0.0
        question_recall = len(question_words & passage_words) / len(question_words) if question_words else 0.0
        score = 0.75 * answer_recall + 0.25 * question_recall
        candidate = (score, overlap, answer_text, question_text)
        if candidate[:2] > best[:2]:
            best = candidate
    return best


def align_claim(claim: dict[str, Any], ranking: dict[str, Any], top_n: int, min_score: float, min_answer_overlap: int) -> dict[str, Any]:
    evidence = [item for item in claim.get("evidence") or [] if isinstance(item, dict) and item.get("url")]
    gold_urls = {canonical_url(item.get("url")) for item in evidence}
    gold_urls.discard(None)
    candidates = ranking.get("retrieved", [])[:top_n]
    matched = []
    for item in candidates:
        url = canonical_url(item.get("url"))
        if url not in gold_urls:
            continue
        score, overlap, answer, question = score_passage(str(item.get("text") or ""), evidence, url)
        matched.append((score, overlap, item, answer, question))
    matched.sort(key=lambda entry: (-entry[0], -entry[1], str(entry[2]["passage_id"])))
    best = matched[0] if matched else None
    accepted = bool(best and best[0] >= min_score and best[1] >= min_answer_overlap)
    return {
        "claim_id": str(claim["claim_id"]),
        "positive_passage_id": str(best[2]["passage_id"]) if accepted else None,
        "selected_candidate_passage_id": str(best[2]["passage_id"]) if best else None,
        "alignment_score": float(best[0]) if best else 0.0,
        "answer_content_overlap": int(best[1]) if best else 0,
        "matched_source_candidate_count": len(matched),
        "gold_answer_count": len(evidence),
        "selected_passage_text": str(best[2].get("text") or "") if best else "",
        "selected_source_url": best[2].get("url") if best else None,
        "matched_answer_text": best[3] if best else "",
        "matched_question_text": best[4] if best else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-claims", type=Path, required=True)
    parser.add_argument("--train-rankings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--min-score", type=float, default=0.15)
    parser.add_argument("--min-answer-overlap", type=int, default=2)
    parser.add_argument("--audit-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=718)
    args = parser.parse_args()
    if min(args.top_n, args.min_answer_overlap, args.audit_size) < 1 or not 0 <= args.min_score <= 1:
        parser.error("Invalid top-n, alignment threshold, or audit size")
    for path in (args.train_claims, args.train_rankings):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = resolve_run_directory(args.output_dir, args.run_name)
    output.mkdir(parents=True, exist_ok=True)
    config = {"script": "prepare_answer_aligned_labels", "train_claims_sha256": sha256_file(args.train_claims),
              "train_rankings_sha256": sha256_file(args.train_rankings), "top_n": args.top_n,
              "min_score": args.min_score, "min_answer_overlap": args.min_answer_overlap,
              "audit_size": args.audit_size, "seed": args.seed,
              "warning": "Automatic answer-to-passage alignment is silver supervision, not human passage relevance."}
    config_path = output / "config.json"
    if config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Inputs/settings changed; use a new run name")
    labels_path, report_path, audit_path = (output / name for name in ("train_alignment.jsonl", "alignment_report.json", "alignment_audit.csv"))
    if all(path.is_file() for path in (labels_path, report_path, audit_path)):
        print(report_path.read_text(encoding="utf-8"))
        return
    write_json(config, config_path)
    claims = read_jsonl(args.train_claims)
    rankings = read_jsonl(args.train_rankings)
    by_id = {str(row["claim_id"]): row for row in rankings}
    if len(by_id) != len(rankings) or len({str(row["claim_id"]) for row in claims}) != len(claims) or set(by_id) != {str(row["claim_id"]) for row in claims}:
        raise ValueError("Train claim/ranking IDs must match exactly and be unique")
    aligned = []
    with progress(total=len(claims), description="Align gold answers to train chunks", unit="claim") as bar:
        for claim in claims:
            aligned.append(align_claim(claim, by_id[str(claim["claim_id"])], args.top_n, args.min_score, args.min_answer_overlap))
            bar.update(1)
    temporary = labels_path.with_suffix(".jsonl.tmp")
    write_jsonl(aligned, temporary)
    temporary.replace(labels_path)
    rng = random.Random(args.seed)
    accepted = [row for row in aligned if row["positive_passage_id"]]
    rejected = [row for row in aligned if row["selected_candidate_passage_id"] and not row["positive_passage_id"]]
    audit = rng.sample(accepted, min(round(args.audit_size * 0.75), len(accepted))) + rng.sample(rejected, min(args.audit_size // 4, len(rejected)))
    rng.shuffle(audit)
    audit_fields = ["claim_id", "claim", "accepted", "alignment_score", "answer_content_overlap", "matched_question_text", "matched_answer_text", "selected_passage_text", "selected_source_url", "human_passage_relevant"]
    claim_by_id = {str(row["claim_id"]): row for row in claims}
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        for row in audit:
            writer.writerow({**{key: row.get(key) for key in audit_fields}, "claim": claim_by_id[row["claim_id"]]["claim"],
                             "accepted": bool(row["positive_passage_id"]), "human_passage_relevant": ""})
    report = {**config, "claim_count": len(claims), "accepted_claim_count": len(accepted),
              "matched_source_in_top_n_count": sum(bool(row["matched_source_candidate_count"]) for row in aligned),
              "rejected_low_alignment_count": len(rejected), "audit_row_count": len(audit),
              "labels_path": str(labels_path), "audit_path": str(audit_path)}
    write_json(report, report_path)
    write_run_manifest(output / "manifest.json", project_root=PROJECT_ROOT, config=report,
                       input_paths={"train_claims": args.train_claims, "train_rankings": args.train_rankings})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
