"""Development HTTP boundary for the screening assistant (stdlib only).

Routes: GET /health, GET /benchmark/claims, POST /verify.
See docs/INTEGRATION_CONTRACT.md. Supports fixture replay and the saved baseline
adapter; other model adapters plug in through the same boundary.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

LABELS = ("Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence")
MAX_EVIDENCE = 3
MAX_CLAIM_CHARS = 2000
MAX_BODY_BYTES = 64 * 1024
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "demo_claims.json"


class Adapter:
    """Verifier boundary: ordered passages in, verdict fields out."""

    name = "base"

    def verify(self, claim, passages):  # pragma: no cover - interface
        raise NotImplementedError


class FixtureReplayAdapter(Adapter):
    """Replays saved fixture results. This is NOT new model inference."""

    name = "fixture_replay"

    def __init__(self, path=FIXTURE_PATH):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.records = {r["claim_id"]: r for r in data["claims"]}

    def list_claims(self, query="", limit=50):
        q = (query or "").strip().lower()
        rows = [{"claim_id": k, "claim": v["claim"]} for k, v in self.records.items()
                if q in k.lower() or q in v["claim"].lower()]
        return rows[:limit]

    def get(self, claim_id):
        return self.records.get(claim_id)


def normalize_evidence(items):
    """Order by rank, keep at most three, renumber ranks 1..n."""
    ordered = sorted(items, key=lambda e: e.get("rank", 0))[:MAX_EVIDENCE]
    return [
        {"passage_id": e["passage_id"], "text": e["text"], "url": e["url"], "rank": i}
        for i, e in enumerate(ordered, start=1)
    ]


def error_response(mode, message):
    return {"mode": mode, "status": "error", "claim": None, "verdict": None,
            "explanation": message, "uncertainty_note": None, "evidence": []}


def model_catalog(adapter):
    if callable(getattr(adapter, 'list_models', None)):
        models = adapter.list_models()
    else:
        models = [{'id': getattr(adapter, 'model_id', adapter.name),
                   'label': getattr(adapter, 'model_label', 'Synthetic demo fixtures' if adapter.name == 'fixture_replay' else adapter.name),
                   'description': getattr(adapter, 'model_description', ''), 'available': True}]
    return {'default_model_id': getattr(adapter, 'default_model_id', models[0]['id']), 'models': models}


def select_adapter(adapter, model_id):
    if callable(getattr(adapter, 'select_model', None)):
        return adapter.select_model(model_id)
    expected = getattr(adapter, 'model_id', adapter.name)
    if model_id is not None and model_id != expected:
        raise ValueError('Unknown model_id')
    return adapter


def handle_verify(payload, adapter):
    """Return (http_status, response_dict). Pure function so tests need no socket."""
    if not isinstance(payload, dict):
        return 400, error_response(None, "Request body must be a JSON object.")
    mode = payload.get("mode")
    if mode == "live":
        claim = payload.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            return 400, error_response(mode, "A non-empty claim is required.")
        if len(claim) > MAX_CLAIM_CHARS:
            return 400, error_response(mode, f"Claim exceeds {MAX_CLAIM_CHARS} characters.")
        # No retrieval path exists for arbitrary claims. Never attach another
        # claim's evidence; report the limitation instead.
        return 200, {"mode": "live", "status": "insufficient_sources", "claim": claim.strip(),
                     "verdict": None,
                     "explanation": "Live checking is not available yet: no source retrieval is set up for new claims.",
                     "uncertainty_note": "Nothing was checked. Do not treat this as a result.",
                     "evidence": []}
    if mode == "benchmark":
        try:
            selected = select_adapter(adapter, payload.get('model_id'))
        except ValueError:
            return 400, error_response(mode, 'Unknown model selection. Choose an available model.')
        except LookupError as error:
            return 503, error_response(mode, str(error))
        claim_id = payload.get("claim_id")
        record = selected.get(claim_id) if isinstance(claim_id, str) else None
        if record is None:
            return 404, error_response(mode, "Unknown benchmark claim_id.")
        if record.get("simulate") == "error":
            return 503, error_response(mode, "Simulated backend failure (fixture).")
        status = record.get("status", "completed")
        verdict = record.get("verdict") if status == "completed" else None
        if status == "completed" and verdict not in LABELS:
            return 500, error_response(mode, "Fixture has an invalid verdict label.")
        return 200, {"mode": "benchmark", "status": status, "claim": record["claim"],
                     "verdict": verdict, "explanation": record.get("explanation"),
                     "uncertainty_note": record.get("uncertainty_note"),
                     "evidence": normalize_evidence(record.get("evidence", [])),
                     "provenance": selected.name,
                     'model_id': getattr(selected, 'model_id', selected.name),
                     'model_label': getattr(selected, 'model_label', 'Synthetic demo fixtures' if selected.name == 'fixture_replay' else selected.name)}
    return 400, error_response(mode if isinstance(mode, str) else None,
                               "mode must be 'benchmark' or 'live'.")


def make_handler(adapter):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self._send(204, {})

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "adapter": adapter.name,
                    'model_selection': bool(getattr(adapter, 'model_selection', False))})
            elif self.path == '/models':
                self._send(200, model_catalog(adapter))
            elif urlparse(self.path).path == "/benchmark/claims":
                params = parse_qs(urlparse(self.path).query)
                query = (params.get("q") or [""])[0][:200]
                try:
                    limit = max(1, min(1000, int((params.get("limit") or ["50"])[0])))
                except ValueError:
                    limit = 50
                self._send(200, {"claims": adapter.list_claims(query, limit), "provenance": adapter.name})
            else:
                self._send(404, error_response(None, "Not found."))

        def do_POST(self):
            if self.path != "/verify":
                return self._send(404, error_response(None, "Not found."))
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                return self._send(413, error_response(None, "Body missing or too large."))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._send(400, error_response(None, "Invalid JSON."))
            code, body = handle_verify(payload, adapter)
            self._send(code, body)

        def log_message(self, fmt, *args):  # keep claim text out of logs
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--baseline-dir", help="Saved baseline run folder (final_model/, transformer_run_config.json)")
    parser.add_argument("--dev-rankings", help="Saved dev_rankings.jsonl (BM25 top 20)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument('--serving-cache', type=Path, help='Staged cache with serving_manifest.json and selectable models')
    args = parser.parse_args()
    if args.serving_cache:
        if args.baseline_dir or args.dev_rankings:
            parser.error('--serving-cache cannot be combined with single-baseline paths')
        from model_registry import load_registry
        manifest = json.loads((args.serving_cache / 'serving_manifest.json').read_text())
        adapter = load_registry(args.serving_cache, manifest, device=args.device)
    elif args.baseline_dir or args.dev_rankings:
        if not (args.baseline_dir and args.dev_rankings):
            parser.error("--baseline-dir and --dev-rankings must be given together")
        from baseline_adapter import BaselineAdapter
        adapter = BaselineAdapter.from_paths(args.baseline_dir, args.dev_rankings, args.device)
    else:
        adapter = FixtureReplayAdapter()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(adapter))
    print(f"Backend on http://{args.host}:{args.port} (adapter: {adapter.name})")
    server.serve_forever()


if __name__ == "__main__":
    main()
