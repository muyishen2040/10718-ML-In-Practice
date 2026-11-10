"""Private localhost bridge to a benchmark backend in an authenticated Colab session.

Run on the extension user's computer, preferably in WSL on Windows. It sends
bounded HTTP requests through `colab exec`; it does not publish the runtime,
share OAuth credentials with the extension, or require SSH port forwarding.
"""
from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlparse

MARKER = "CLAIM_CHECK_RELAY_JSON:"
MAX_BODY = 64 * 1024


def remote_code(method, path, body, port):
    if method not in ("GET", "POST") or not 1 <= port <= 65535:
        raise ValueError("Invalid remote request")
    route = urlparse(path)
    allowed = (method == "GET" and route.path in ("/health", "/benchmark/claims")) or (method == "POST" and path == "/verify")
    if not allowed or route.netloc or route.scheme or not path.startswith("/"):
        raise ValueError("Only the benchmark backend routes are available")
    request = base64.b64encode(json.dumps({"method": method, "path": path, "body": body}).encode()).decode()
    return f'''import base64, json, urllib.request, urllib.error
_request = json.loads(base64.b64decode({request!r}))
_data = json.dumps(_request['body']).encode() if _request['method'] == 'POST' else None
_http = urllib.request.Request('http://127.0.0.1:{port}' + _request['path'], data=_data,
    method=_request['method'], headers={{'Content-Type': 'application/json'}})
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({{}}))
try:
    with _opener.open(_http, timeout=90) as _response:
        _reply = {{'http_status': _response.status, 'body': json.load(_response)}}
except urllib.error.HTTPError as _error:
    _reply = {{'http_status': _error.code, 'body': json.load(_error)}}
except (OSError, ValueError):
    _reply = {{'http_status': 502, 'body': {{'status': 'error', 'verdict': None, 'evidence': [],
        'explanation': 'The private Colab backend is not ready. Rerun the launch notebook.'}}}}
print({MARKER!r} + json.dumps(_reply, ensure_ascii=True))
'''


def parse_reply(stdout):
    for line in reversed(stdout.splitlines()):
        if MARKER in line:
            value = json.loads(line.split(MARKER, 1)[1])
            if isinstance(value, dict) and type(value.get("http_status")) is int and isinstance(value.get("body"), dict):
                if 100 <= value["http_status"] <= 599:
                    return value["http_status"], value["body"]
            break
    raise ValueError("No valid backend reply from Colab CLI; check the session and CLI version")


class CliBridge:
    def __init__(self, executable, session, *, port=8000, timeout=120, auth=None, runner=subprocess.run):
        self.command = [executable] + (["--auth", auth] if auth else [])
        self.command += ["exec", "--session", session, "--timeout", str(timeout - 5)]
        self.port, self.timeout, self.runner = port, timeout, runner
        self.lock = threading.Lock()
        self.catalog = None

    def request(self, method, path, body=None):
        code = remote_code(method, path, body, self.port)
        route = urlparse(path)
        # Fetch the catalog once; searching from the popup should not execute
        # new remote Python on each keystroke.
        if method == "GET" and route.path == "/benchmark/claims" and self.catalog is not None:
            params = parse_qs(route.query)
            query = (params.get("q") or [""])[0][:200].strip().lower()
            try:
                limit = max(1, min(1000, int((params.get("limit") or ["50"])[0])))
            except ValueError:
                limit = 50
            matches = [row for row in self.catalog["claims"] if query in row["claim_id"].lower() or query in row["claim"].lower()]
            return 200, {"claims": matches[:limit], "provenance": self.catalog.get("provenance")}
        with self.lock:
            result = self.runner(self.command, input=code, capture_output=True, text=True,
                                 encoding="utf-8", timeout=self.timeout)
        if result.returncode:
            # Do not copy raw CLI diagnostics/session metadata into HTTP replies.
            raise RuntimeError(f"Colab CLI exited with code {result.returncode}; run colab status for this session")
        try:
            return parse_reply(result.stdout)
        except ValueError:
            raise RuntimeError('Colab CLI returned an unreadable backend reply') from None

    def initialize(self):
        status, health = self.request("GET", "/health")
        if status != 200 or not health.get("ok"):
            raise RuntimeError("Start the backend inside Colab before starting the local relay")
        status, catalog = self.request("GET", "/benchmark/claims?limit=1000")
        if status != 200 or not isinstance(catalog.get("claims"), list):
            raise RuntimeError("Could not load the benchmark catalog")
        if health.get("adapter") != "baseline_distilbert_bm25_top3":
            raise RuntimeError("Expected the trained baseline adapter, not synthetic fixture replay")
        self.catalog = catalog
        return health


def make_handler(bridge):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, value):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            if code != 204:
                self.send_header("Content-Length", str(len(data)))
            origin = self.headers.get("Origin", "")
            if origin.startswith("chrome-extension://"):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            if code != 204:
                self.wfile.write(data)

        def allowed_origin(self):
            origin = self.headers.get("Origin", "")
            return not origin or origin.startswith("chrome-extension://")

        def dispatch(self, method):
            if not self.allowed_origin():
                return self.send_json(403, {"status": "error", "explanation": "Use the installed extension."})
            body = None
            if method == "POST":
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= MAX_BODY:
                        return self.send_json(413, {"status": "error", "explanation": "Invalid request size."})
                    body = json.loads(self.rfile.read(size))
                except (ValueError, UnicodeDecodeError):
                    return self.send_json(400, {"status": "error", "explanation": "Invalid JSON request."})
            try:
                code, result = bridge.request(method, self.path, body)
            except ValueError:
                code, result = 404, {"status": "error", "explanation": "Unknown backend route."}
            except subprocess.TimeoutExpired:
                code, result = 504, {"status": "error", "explanation": "Colab timed out. Check the session and retry."}
            except (RuntimeError, OSError):
                code, result = 502, {"status": "error", "explanation": "Colab connection failed. Check the relay terminal."}
            self.send_json(code, result)

        def do_GET(self):
            self.dispatch("GET")

        def do_POST(self):
            if self.path == '/_relay/stop':
                token = getattr(self.server, 'stop_token', None)
                if not self.allowed_origin() or not token or self.headers.get('X-Relay-Stop') != token:
                    return self.send_json(403, {'status': 'error', 'explanation': 'Invalid local stop request.'})
                self.send_json(200, {'status': 'stopped'})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            self.dispatch("POST")

        def do_OPTIONS(self):
            self.send_json(204 if self.allowed_origin() else 403, {})

        def log_message(self, *args):
            pass

    return Handler


def stop_local_relay(state_path):
    state_path = Path(state_path)
    if not state_path.is_file():
        print('No managed local relay is running.', flush=True)
        return
    state = json.loads(state_path.read_text())
    port = int(state['port'])
    if not 1 <= port <= 65535:
        raise ValueError('Invalid relay state port')
    request = urllib.request.Request(f'http://127.0.0.1:{port}/_relay/stop', data=b'{}',
        headers={'Content-Type': 'application/json', 'X-Relay-Stop': state['stop_token']})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=5) as response:
            if json.load(response).get('status') != 'stopped':
                raise RuntimeError('The local relay did not confirm shutdown')
    except urllib.error.HTTPError:
        raise RuntimeError('Local stop token was rejected; state file preserved') from None
    except urllib.error.URLError:
        print('Local relay is already unavailable.', flush=True)
    if state_path.is_file() and json.loads(state_path.read_text()).get('stop_token') == state['stop_token']:
        state_path.unlink()
    print('Local relay stopped.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="claim-check-extension")
    parser.add_argument("--colab", default=shutil.which("colab"))
    parser.add_argument("--auth", choices=("oauth2", "adc"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--remote-port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument('--state-file', type=Path)
    parser.add_argument('--stop', action='store_true', help='Stop only the relay identified by --state-file')
    args = parser.parse_args()
    if args.stop:
        if not args.state_file:
            parser.error('--stop requires --state-file')
        stop_local_relay(args.state_file)
        return
    if args.state_file and args.state_file.exists():
        parser.error('A managed relay state file exists; run Stop before starting another relay')
    if not args.colab or args.timeout < 10 or not all(1 <= port <= 65535 for port in (args.port, args.remote_port)):
        parser.error("Install Colab CLI and use valid ports and a timeout of at least ten seconds")
    bridge = CliBridge(args.colab, args.session, port=args.remote_port, timeout=args.timeout, auth=args.auth)
    print("Checking the authenticated Colab backend...", flush=True)
    health = bridge.initialize()
    print("Adapter:", health["adapter"], "| benchmark claims:", len(bridge.catalog["claims"]), flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(bridge))
    server.stop_token = secrets.token_urlsafe(24)
    if args.state_file:
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
        args.state_file.write_text(json.dumps({'port': server.server_port, 'stop_token': server.stop_token}) + '\n')
        args.state_file.chmod(0o600)
    print(f"Extension Backend URL: http://127.0.0.1:{server.server_port}", flush=True)
    print("Keep this terminal open. Ctrl+C stops the relay; stop the Colab session separately.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if args.state_file and args.state_file.is_file():
            if json.loads(args.state_file.read_text()).get('stop_token') == server.stop_token:
                args.state_file.unlink()


if __name__ == "__main__":
    main()
