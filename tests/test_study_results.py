"""Checks for scripts/analyze_study_results.py using made-up participants (no real data)."""
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analyze_study_results", ROOT / "scripts" / "analyze_study_results.py")
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)

SELECTION = json.loads((ROOT / "configs" / "user_study_claims.json").read_text(encoding="utf-8"))["claims"]


def fake_participant(pid, group, answer="Refuted"):
    """Same shape as the page's result file, following the page's counterbalancing."""
    tool_set = analyze.SETS[group]
    assisted = [c for c in SELECTION if c["set"] == tool_set]
    plain = [c for c in SELECTION if c["set"] != tool_set]
    blocks = [assisted, plain] if analyze.TOOL_FIRST[group] else [plain, assisted]
    trials, n = [], 0
    for b, block in enumerate(blocks, start=1):
        for c in block:
            n += 1
            is_assisted = c["set"] == tool_set
            trials.append({"number": n, "block": b, "claim_id": c["claim_id"], "set": c["set"],
                           "condition": "assisted" if is_assisted else "unassisted",
                           "model_verdict_shown": "Refuted" if is_assisted else None, "answer": answer,
                           "confidence": 3, "time_ms": 5000 + n, "sources_opened": [{"source": 1, "t_ms": 900}] if is_assisted else []})
    return {"participant_id": pid, "group": group, "trials": trials,
            "post": {"helpful": 4, "overrelied": 2, "would_use": 3, "comments": ""}}


class StudyResultsTest(unittest.TestCase):
    def test_all_groups_valid(self):
        for group in (1, 2, 3, 4):
            self.assertEqual(analyze.check(fake_participant("p", group)), [], group)

    def test_detects_wrong_condition_and_missing_answers(self):
        bad = fake_participant("p", 1)
        bad["trials"][0]["condition"] = "unassisted"
        self.assertTrue(any("condition" in p for p in analyze.check(bad)))
        bad = fake_participant("p", 2)
        bad["trials"][3]["answer"] = None
        self.assertTrue(any("answer" in p for p in analyze.check(bad)))
        short = fake_participant("p", 3)
        short["trials"].pop()
        self.assertTrue(any("12" in p for p in analyze.check(short)))

    def test_block_order_must_match_group(self):
        swapped = fake_participant("p", 1)
        for t in swapped["trials"]:  # group 1 should start with the tool; make it start without
            t["block"] = 3 - t["block"]
        swapped["trials"].sort(key=lambda t: t["block"])
        for i, t in enumerate(swapped["trials"], start=1):
            t["number"] = i
        self.assertTrue(any("block order" in p for p in analyze.check(swapped)))

    def test_flatten_scores_against_gold(self):
        gold = {c["claim_id"]: {"gold": "Supported" if i % 2 else "Refuted", "model": "Refuted"} for i, c in enumerate(SELECTION)}
        rows = analyze.flatten(fake_participant("p", 1, answer="Supported"), gold)
        self.assertEqual(len(rows), 12)
        for r in rows:
            self.assertEqual(r["accepted"], 1)
            self.assertEqual(r["decision_correct"], int(r["gold"] == "Supported"))
        assisted = [r for r in rows if r["condition"] == "assisted"]
        self.assertTrue(all(r["followed_model"] == 0 for r in assisted))  # answered Supported, model showed Refuted

    def test_end_to_end_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rng = random.Random(1)
            for i, group in enumerate((1, 2, 3, 4)):
                answer = rng.choice(["Supported", "Refuted"])
                (tmp / f"p{i}.json").write_text(json.dumps(fake_participant(f"P{i}", group, answer)), encoding="utf-8")
            gold_path = tmp / "gold.jsonl"
            gold_path.write_text("\n".join(json.dumps({"claim_id": c["claim_id"], "true_label": "Refuted", "predicted_label": "Refuted"})
                                           for c in SELECTION), encoding="utf-8")
            import subprocess, sys
            out = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_study_results.py"), str(tmp),
                                  "--gold", str(gold_path), "--csv", str(tmp / "out.csv")], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertEqual(len((tmp / "out.csv").read_text().strip().splitlines()), 1 + 48)


    def test_reads_sheet_raw_export_and_flags_duplicates(self):
        import csv
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            people = [fake_participant("P01", 1), fake_participant("P02", 4)]
            for p in people:
                p["finished_at"] = "t-" + p["participant_id"]
            with open(tmp / "raw.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["server_time", "participant_id", "finished_at", "json"])
                for p in people + [people[0]]:  # P01 twice, as if a submission were repeated
                    writer.writerow(["now", p["participant_id"], p["finished_at"], json.dumps(p)])
            gold = tmp / "gold.jsonl"
            gold.write_text("\n".join(json.dumps({"claim_id": c["claim_id"], "true_label": "Refuted", "predicted_label": "Refuted"})
                                      for c in SELECTION), encoding="utf-8")
            out = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_study_results.py"), "--raw-csv", str(tmp / "raw.csv"),
                                  "--gold", str(gold), "--csv", str(tmp / "out.csv")], capture_output=True, text=True)
            self.assertEqual(out.returncode, 1, out.stdout)          # the repeated row is reported
            self.assertIn("duplicate submission", out.stdout)
            self.assertEqual(len((tmp / "out.csv").read_text().strip().splitlines()), 1 + 24)

    def test_apps_script_receiver(self):
        import shutil
        import subprocess
        if shutil.which("node") is None:
            self.skipTest("node not installed")
        out = subprocess.run(["node", str(ROOT / "tests" / "apps_script_receiver_check.js")], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)


if __name__ == "__main__":
    unittest.main()
