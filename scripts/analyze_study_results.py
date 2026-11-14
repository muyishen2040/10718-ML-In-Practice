"""Check and flatten user-study result files (the JSON each participant downloads).

    python3 scripts/analyze_study_results.py RESULTS_DIR [--csv OUT.csv]

For every file it validates the structure (12 trials, 6 assisted and 6 unassisted, groups
match the counterbalancing in docs/USER_STUDY_DESIGN.md, answers and confidence present),
then writes one row per judgement with the gold label joined from the saved dev predictions
and prints a plain summary. With a handful of pilot participants the summary only shows
that the pipeline works; it is not a result.

Gold labels come from local/baseline/final_dev_predictions.jsonl (git-ignored shared data).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANSWERS = {"Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence"}
SETS = {1: "A", 2: "A", 3: "B", 4: "B"}      # set shown with the tool, by group
TOOL_FIRST = {1: True, 2: False, 3: True, 4: False}
FIELDS = ["participant_id", "group", "number", "block", "claim_id", "set", "condition", "gold", "model_verdict_shown",
          "model_verdict", "answer", "confidence", "time_ms", "sources_opened", "decision_correct", "four_way_correct",
          "accepted", "followed_model", "model_wrong_decision"]


def load_gold(path):
    gold = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                gold[row["claim_id"]] = {"gold": row["true_label"], "model": row["predicted_label"]}
    return gold


def check(result):
    """Return a list of problems with one participant's file (empty = fine)."""
    problems = []
    trials = result.get("trials") or []
    group = result.get("group")
    if group not in SETS:
        return [f"unknown group {group!r}"]
    if len(trials) != 12:
        problems.append(f"{len(trials)} trials, expected 12")
    assisted = [t for t in trials if t.get("condition") == "assisted"]
    if len(assisted) != 6:
        problems.append(f"{len(assisted)} assisted trials, expected 6")
    for t in trials:
        if t.get("answer") not in ANSWERS or not isinstance(t.get("confidence"), int) or not 1 <= t["confidence"] <= 5:
            problems.append(f"trial {t.get('number')}: missing or invalid answer/confidence")
        if not isinstance(t.get("time_ms"), (int, float)) or t["time_ms"] <= 0:
            problems.append(f"trial {t.get('number')}: missing time")
        want = "assisted" if (t.get("set") == SETS[group]) else "unassisted"
        if t.get("condition") != want:
            problems.append(f"trial {t.get('number')}: condition {t.get('condition')} but group {group} implies {want}")
    first = [t for t in trials if t.get("block") == 1]
    if first and (first[0].get("condition") == "assisted") != TOOL_FIRST[group]:
        problems.append("block order does not match the group")
    if len({t.get("claim_id") for t in trials}) != len(trials):
        problems.append("a claim appears twice")
    if not (result.get("post") or {}).get("helpful"):
        problems.append("survey missing")
    return problems


def flatten(result, gold):
    rows = []
    for t in result["trials"]:
        g = gold.get(t["claim_id"], {})
        gold_label, model = g.get("gold"), g.get("model")
        accepted = t["answer"] == "Supported"
        row = {"participant_id": result["participant_id"], "group": result["group"], "number": t["number"], "block": t["block"],
               "claim_id": t["claim_id"], "set": t["set"], "condition": t["condition"], "gold": gold_label,
               "model_verdict_shown": t.get("model_verdict_shown"), "model_verdict": model, "answer": t["answer"],
               "confidence": t["confidence"], "time_ms": t["time_ms"], "sources_opened": len(t.get("sources_opened") or []),
               "accepted": int(accepted)}
        if gold_label:
            row["decision_correct"] = int(accepted == (gold_label == "Supported"))
            row["four_way_correct"] = int(t["answer"] == gold_label)
            row["model_wrong_decision"] = int((model == "Supported") != (gold_label == "Supported"))
            if t["condition"] == "assisted":
                row["followed_model"] = int(t["answer"] == t.get("model_verdict_shown"))
        rows.append(row)
    return rows


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def summarize(rows):
    print(f"\n{len({r['participant_id'] for r in rows})} participants, {len(rows)} judgements")
    for cond in ("unassisted", "assisted"):
        sub = [r for r in rows if r["condition"] == cond and r["decision_correct"] != ""]
        if not sub:
            continue
        print(f"\n{cond}: n={len(sub)}")
        print(f"  decision correct (accept only Supported):  {mean([r['decision_correct'] for r in sub]):.2f}")
        print(f"  four-way correct:                         {mean([r['four_way_correct'] for r in sub]):.2f}")
        print(f"  mean confidence {mean([r['confidence'] for r in sub]):.2f}, median time {sorted(r['time_ms'] for r in sub)[len(sub) // 2] / 1000:.1f}s")
        by_gold = defaultdict(list)
        for r in sub:
            by_gold[r["gold"]].append(r["decision_correct"])
        print("  decision correct by gold label: " + ", ".join(f"{k} {mean(v):.2f} (n={len(v)})" for k, v in sorted(by_gold.items())))
    assisted = [r for r in rows if r["condition"] == "assisted" and r.get("followed_model") not in (None, "")]
    wrong = [r for r in assisted if r["model_wrong_decision"] == 1]
    if wrong:
        print(f"\nwhen the model's accept/not-accept decision was wrong (assisted, n={len(wrong)}): "
              f"participant decision correct {mean([r['decision_correct'] for r in wrong]):.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir")
    parser.add_argument("--gold", default=str(ROOT / "local/baseline/final_dev_predictions.jsonl"))
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    gold = load_gold(args.gold)
    rows, bad = [], 0
    for path in sorted(Path(args.results_dir).glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        problems = check(result)
        print(f"{'OK ' if not problems else 'BAD'} {path.name}" + ("" if not problems else ": " + "; ".join(problems)))
        if problems:
            bad += 1
        else:
            rows.extend(flatten(result, gold))
    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
        print(f"\nwrote {args.csv}")
    if rows:
        summarize([{k: r.get(k, "") for k in FIELDS} for r in rows])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
