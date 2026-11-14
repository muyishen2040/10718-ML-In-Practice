"""Choose claims for the user study from the 500 official dev claims.

Two steps, both deterministic:

  candidates  Sample a few eligible claims per (gold label, model correct/wrong) cell
              with a fixed seed, so a person can review them by eye.
  build       Read the reviewed claim IDs from configs/user_study_claims.json and write
              the static file the study page shows, by running the saved baseline model
              through apps/backend/baseline_adapter.py. Saved predictions are only used
              to check that the adapter reproduces the logged label.

Gold labels are used here to pick and later score claims. They are never given to the
model and are not written to the participant-facing file.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence")
SEED = 718
MIN_PASSAGE_CHARS = 80
CLAIM_CHARS = (40, 220)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load(predictions_path, rankings_path):
    predictions = {r["claim_id"]: r for r in read_jsonl(predictions_path)}
    rankings = {r["claim_id"]: r for r in read_jsonl(rankings_path)}
    return predictions, rankings


def eligible(row, pred):
    """Three readable passages, a short claim, and a logged prediction."""
    passages = row.get("retrieved") or []
    if pred is None or len(passages) < 3:
        return False
    if not CLAIM_CHARS[0] <= len(row["claim"]) <= CLAIM_CHARS[1]:
        return False
    return all(len(str(p.get("text") or "")) >= MIN_PASSAGE_CHARS for p in passages[:3])


def cmd_candidates(args):
    predictions, rankings = load(args.predictions, args.rankings)
    cells = defaultdict(list)
    for cid, row in rankings.items():
        pred = predictions.get(cid)
        if not eligible(row, pred):
            continue
        outcome = "model_correct" if pred["true_label"] == pred["predicted_label"] else "model_wrong"
        cells[(pred["true_label"], pred["predicted_label"], outcome)].append(cid)
    rng = random.Random(SEED)
    out = []
    for key in sorted(cells):
        ids = sorted(cells[key])
        rng.shuffle(ids)
        for cid in ids[:args.per_cell]:
            pred = predictions[cid]
            out.append({"claim_id": cid, "claim": rankings[cid]["claim"], "gold": key[0],
                        "model_label": key[1], "outcome": key[2],
                        "model_top_probability": round(max(pred["probabilities"].values()), 3)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} candidates from {sum(len(v) for v in cells.values())} eligible claims -> {args.out}")


def cmd_build(args):
    sys.path.insert(0, str(ROOT / "apps" / "backend"))
    from baseline_adapter import BaselineAdapter

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    predictions, _ = load(args.predictions, args.rankings)
    adapter = BaselineAdapter.from_paths(args.baseline_dir, args.rankings)
    claims, mismatches = [], []
    for item in selection["claims"]:
        cid = item["claim_id"]
        result = adapter.get(cid)
        if result is None or result["status"] != "completed":
            raise SystemExit(f"{cid}: no completed model result")
        logged = predictions[cid]["predicted_label"]
        if result["verdict"] != logged:
            mismatches.append((cid, result["verdict"], logged))
        claims.append({"claim_id": cid, "set": item["set"], "claim": result["claim"],
                       "verdict": result["verdict"], "explanation": result["explanation"],
                       "uncertainty_note": result["uncertainty_note"], "evidence": result["evidence"]})
    if mismatches:
        raise SystemExit(f"adapter differs from logged predictions: {mismatches}")
    payload = {"source": "Saved baseline model outputs (DistilBERT trained on BM25 top-3), replayed, not live inference.",
               "provenance": adapter.name, "claims": claims}
    text = json.dumps(payload, indent=1, ensure_ascii=False)
    Path(args.out).write_text(text, encoding="utf-8")
    # Same data as a script so the page also works when opened straight from disk (no server).
    Path(args.out).with_suffix(".js").write_text("window.STUDY_DATA = " + text + ";\n", encoding="utf-8")
    print(f"wrote {len(claims)} claims -> {args.out} (+ .js)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", default=str(ROOT / "local/baseline/final_dev_predictions.jsonl"))
    parser.add_argument("--rankings", default=str(ROOT / "local/baseline/dev_rankings.jsonl"))
    sub = parser.add_subparsers(dest="command", required=True)
    cand = sub.add_parser("candidates")
    cand.add_argument("--per-cell", type=int, default=3)
    cand.add_argument("--out", default=str(ROOT / "local/study/candidates.json"))
    cand.set_defaults(func=cmd_candidates)
    build = sub.add_parser("build")
    build.add_argument("--selection", default=str(ROOT / "configs/user_study_claims.json"))
    build.add_argument("--baseline-dir", default=str(ROOT / "local/baseline"))
    build.add_argument("--out", default=str(ROOT / "apps/study/study_claims.json"))
    build.set_defaults(func=cmd_build)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
