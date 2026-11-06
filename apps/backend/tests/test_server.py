import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server  # noqa: E402


class VerifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = server.FixtureReplayAdapter()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(cls.adapter))
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def post(self, body, raw=None):
        data = raw if raw is not None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + "/verify", data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_every_completed_fixture_has_valid_label_and_evidence(self):
        for c in self.adapter.list_claims():
            code, r = self.post({"mode": "benchmark", "claim_id": c["claim_id"]})
            if r["status"] == "completed":
                self.assertEqual(code, 200)
                self.assertIn(r["verdict"], server.LABELS)
                self.assertLessEqual(len(r["evidence"]), 3)
                self.assertEqual([e["rank"] for e in r["evidence"]],
                                 list(range(1, len(r["evidence"]) + 1)))
                for e in r["evidence"]:
                    self.assertTrue(e["url"].startswith("https://"))
                    self.assertTrue(e["text"])

    def test_three_item_limit_and_order(self):
        items = [{"passage_id": f"p{i}", "text": "t", "url": "https://x", "rank": r}
                 for i, r in enumerate([4, 2, 3, 1, 5])]
        out = server.normalize_evidence(items)
        self.assertEqual(len(out), 3)
        self.assertEqual([e["passage_id"] for e in out], ["p3", "p1", "p2"])

    def test_nei_is_distinct_from_insufficient_and_error(self):
        _, nei = self.post({"mode": "benchmark", "claim_id": "fixture:nei"})
        self.assertEqual((nei["status"], nei["verdict"]), ("completed", "Not Enough Evidence"))
        _, ins = self.post({"mode": "benchmark", "claim_id": "fixture:no-sources"})
        self.assertEqual((ins["status"], ins["verdict"]), ("insufficient_sources", None))
        code, err = self.post({"mode": "benchmark", "claim_id": "fixture:error"})
        self.assertEqual((code, err["status"], err["verdict"]), (503, "error", None))

    def test_live_mode_unavailable_and_attaches_no_evidence(self):
        code, r = self.post({"mode": "live", "claim": "Anything at all", "page_url": None})
        self.assertEqual(code, 200)
        self.assertEqual((r["status"], r["verdict"], r["evidence"]), ("insufficient_sources", None, []))

    def test_bad_requests(self):
        self.assertEqual(self.post({"mode": "live", "claim": "  "})[0], 400)
        self.assertEqual(self.post({"mode": "live", "claim": "x" * 2001})[0], 400)
        self.assertEqual(self.post({"mode": "nope"})[0], 400)
        self.assertEqual(self.post({"mode": "benchmark", "claim_id": "averitec:dev:missing"})[0], 404)
        self.assertEqual(self.post(None, raw=b"{not json")[0], 400)

    def test_claim_list_and_health(self):
        with urllib.request.urlopen(self.base + "/benchmark/claims") as r:
            self.assertGreaterEqual(len(json.loads(r.read())["claims"]), 6)
        with urllib.request.urlopen(self.base + "/health") as r:
            self.assertTrue(json.loads(r.read())["ok"])


if __name__ == "__main__":
    unittest.main()
