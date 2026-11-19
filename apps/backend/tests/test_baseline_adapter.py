import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline_adapter as ba  # noqa: E402
import server  # noqa: E402


class FakeTok:
    """Whitespace tokenizer: enough to check budgets and layout."""
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


def passages(n, text="word " * 5):
    return [{"passage_id": f"p{i}", "text": text + str(i), "url": f"https://e.com/{i}"} for i in range(n)]


def rankings():
    return {"c1": {"claim_id": "c1", "claim": "Alpha beta", "retrieved": passages(5)},
            "c2": {"claim_id": "c2", "claim": "No sources here", "retrieved": []},
            "c3": {"claim_id": "c3", "claim": "Gamma", "retrieved": passages(2)}}


class AdapterTests(unittest.TestCase):
    def test_compose_text_matches_baseline_layout(self):
        text = ba.compose_text("one two three", passages(5), FakeTok(), 2, 3)
        lines = text.split("\n")
        self.assertEqual(lines[0], "Claim: one two")
        self.assertEqual(len(lines), 4)  # claim + exactly three evidence lines
        self.assertTrue(lines[1].startswith("Evidence 1: ") and len(lines[1].split()) == 2 + 3)
        self.assertEqual(ba.compose_text("x", [], FakeTok(), 5, 5).split("\n")[1],
                         "Evidence: No candidate passage was retrieved.")

    def test_verdict_is_argmax_in_label_order_and_evidence_capped(self):
        calls = []
        def predict(claim, ps):
            calls.append((claim, len(ps)))
            return [0.1, 0.2, 0.1, 0.6]
        a = ba.BaselineAdapter(rankings(), predict)
        r = a.get("c1")
        self.assertEqual((r["status"], r["verdict"]), ("completed", "Conflicting Evidence"))
        self.assertEqual([e["rank"] for e in r["evidence"]], [1, 2, 3])
        self.assertEqual([e["passage_id"] for e in r["evidence"]], ["p0", "p1", "p2"])
        a.get("c1")
        self.assertEqual(len(calls), 1)  # cached

    def test_no_passages_is_insufficient_not_a_prediction(self):
        a = ba.BaselineAdapter(rankings(), lambda c, p: self.fail("model must not run"))
        r = a.get("c2")
        self.assertEqual((r["status"], r["evidence"]), ("insufficient_sources", []))
        self.assertIsNone(a.get("missing"))

    def test_notes_have_no_percentages_and_carry_baseline_caveat(self):
        for probs in ([0.9, 0.05, 0.03, 0.02], [0.5, 0.3, 0.1, 0.1], [0.3, 0.3, 0.2, 0.2]):
            a = ba.BaselineAdapter(rankings(), lambda c, p, pr=probs: pr)
            note = a.get("c3")["uncertainty_note"]
            self.assertNotIn("%", note)
            self.assertIn("baseline", note)

    def test_search_and_limit(self):
        a = ba.BaselineAdapter(rankings(), lambda c, p: [1, 0, 0, 0])
        self.assertEqual([c["claim_id"] for c in a.list_claims("gamma")], ["c3"])
        self.assertEqual(len(a.list_claims("", limit=2)), 2)

    def test_served_through_verify_with_provenance(self):
        a = ba.BaselineAdapter(rankings(), lambda c, p: [0.7, 0.1, 0.1, 0.1])
        code, body = server.handle_verify({"mode": "benchmark", "claim_id": "c1"}, a)
        self.assertEqual((code, body["verdict"], body["provenance"]), (200, "Supported", a.name))
        self.assertLessEqual(len(body["evidence"]), 3)
        code, body = server.handle_verify({"mode": "benchmark", "claim_id": "c2"}, a)
        self.assertEqual((code, body["status"], body["verdict"]), (200, "insufficient_sources", None))


@unittest.skipUnless(os.environ.get("BASELINE_DIR") and os.environ.get("DEV_RANKINGS"),
                     "set BASELINE_DIR and DEV_RANKINGS to run against the real saved baseline")
class RealModelTests(unittest.TestCase):
    def test_real_model_runs_on_dev_claims(self):
        a = ba.BaselineAdapter.from_paths(os.environ["BASELINE_DIR"], os.environ["DEV_RANKINGS"])
        self.assertGreaterEqual(len(a.rankings), 500)
        r = a.get(a.list_claims("", 1)[0]["claim_id"])
        self.assertIn(r["status"], ("completed", "insufficient_sources"))
        if r["status"] == "completed":
            self.assertIn(r["verdict"], ba.LABELS)


if __name__ == "__main__":
    unittest.main()
