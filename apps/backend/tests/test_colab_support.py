import contextlib
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import colab_relay as relay
import colab_runtime as runtime


def serving_inputs(root):
    model = root / runtime.RUN_PATH / 'final_model'
    model.mkdir(parents=True)
    (model / 'config.json').write_text('{}')
    (model / 'model.safetensors').write_bytes(b'test model bytes')
    (model / 'tokenizer.json').write_text('{}')
    (model / 'checkpoint-ignore').mkdir()
    (model.parent / 'transformer_run_config.json').write_text('{}')
    rankings = root / runtime.RANKINGS_PATH
    rankings.parent.mkdir(parents=True)
    rankings.write_text('{}\n')


class FakeAdapter:
    name = 'baseline_distilbert_bm25_top3'

    def list_claims(self, query='', limit=50):
        return [{'claim_id': f'claim:{i:03d}', 'claim': f'Example claim {i}'} for i in range(500)][:limit]

    def get(self, claim_id):
        if claim_id != 'claim:000':
            return None
        return {'claim_id': claim_id, 'claim': 'Example claim 0', 'verdict': 'Refuted',
                'evidence': [{'passage_id': 'p0', 'text': 'Source text', 'url': 'https://example.org', 'rank': 1}]}


class ColabSupportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data, self.cache = self.root / 'shared', self.root / 'cache'
        serving_inputs(self.data)

    def tearDown(self):
        runtime.stop_backend()
        self.temp.cleanup()

    def test_cache_resume_repairs_corruption_and_keeps_shared_inputs_unchanged(self):
        baseline, rankings, manifest = runtime.stage_artifacts(self.data, self.cache)
        source = self.data / runtime.RUN_PATH / 'final_model/model.safetensors'
        copied = baseline / 'final_model/model.safetensors'
        self.assertEqual(source.read_bytes(), copied.read_bytes())
        self.assertFalse((baseline / 'final_model/checkpoint-ignore').exists())
        copied.write_bytes(b'broken')
        runtime.stage_artifacts(self.data, self.cache)
        self.assertEqual(source.read_bytes(), copied.read_bytes())
        runtime.stage_artifacts(self.data, self.cache)
        self.assertEqual(rankings.read_text(), '{}\n')
        self.assertEqual(runtime.file_hash(copied), manifest['files']['baseline/final_model/model.safetensors']['sha256'])
        self.assertFalse((self.data / 'serving_manifest.json').exists())

    def test_cache_refuses_writing_under_shared_inputs_and_reports_missing_files(self):
        with self.assertRaises(ValueError):
            runtime.stage_artifacts(self.data, self.data / 'cache')
        with self.assertRaises(FileNotFoundError):
            runtime.stage_artifacts(self.root / 'missing', self.cache)

    def start(self):
        return runtime.start_backend(self.data, self.cache, port=0, adapter_factory=lambda *args: FakeAdapter())

    def test_runtime_is_loopback_only_and_reuses_loaded_adapter(self):
        info = self.start()
        with urllib.request.urlopen(info['url'] + '/health') as response:
            self.assertEqual(json.load(response)['adapter'], FakeAdapter.name)
        self.assertEqual(runtime._ACTIVE['server'].server_address[0], '127.0.0.1')
        again = runtime.start_backend(self.data, self.cache, port=0, adapter_factory=lambda *args: self.fail('Reloaded model'))
        self.assertTrue(again['reused'])
        with self.assertRaises(RuntimeError):
            runtime.start_backend(self.data, self.cache, port=8001)

    def test_cli_bridge_round_trip_and_local_catalog_search(self):
        info = self.start()
        calls = []
        def runner(command, *, input, **kwargs):
            calls.append(command)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(input, {})
            return subprocess.CompletedProcess(command, 0, output.getvalue(), '')
        bridge = relay.CliBridge('/fake/colab', 'chosen-session', port=runtime._ACTIVE['server'].server_port, runner=runner)
        bridge.initialize()
        self.assertEqual(len(bridge.catalog['claims']), 500)
        self.assertEqual(len(calls), 2)
        code, result = bridge.request('GET', '/benchmark/claims?q=claim%20499&limit=5')
        self.assertEqual([row['claim_id'] for row in result['claims']], ['claim:499'])
        self.assertEqual(len(calls), 2)
        code, result = bridge.request('POST', '/verify', {'mode': 'benchmark', 'claim_id': 'claim:000'})
        self.assertEqual((code, result['verdict']), (200, 'Refuted'))
        self.assertEqual(result['provenance'], FakeAdapter.name)
        self.assertEqual(calls[0][:4], ['/fake/colab', 'exec', '--session', 'chosen-session'])
        with self.assertRaises(ValueError):
            bridge.request('GET', 'https://other.example/benchmark/claims')

    def test_remote_request_cannot_choose_an_external_endpoint_or_execute_claim_text(self):
        self.start()
        for path in ('https://example.org/health', '//example.org/health', '/unknown'):
            with self.assertRaises(ValueError):
                relay.remote_code('GET', path, None, 8000)
        marker = self.root / 'must-not-exist'
        claim = f'\"); __import__("pathlib").Path({str(marker)!r}).touch(); #'
        script = relay.remote_code('POST', '/verify', {'mode': 'live', 'claim': claim}, runtime._ACTIVE['server'].server_port)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(script, {})
        code, body = relay.parse_reply(output.getvalue())
        self.assertEqual((code, body['status']), (200, 'insufficient_sources'))
        self.assertFalse(marker.exists())

    def test_parse_reply_rejects_missing_or_invalid_replies_and_cli_errors(self):
        for value in ('', relay.MARKER + '{}', relay.MARKER + '{"http_status": 999, "body": {}}'):
            with self.assertRaises(ValueError):
                relay.parse_reply(value)
        def failed(*args, **kwargs):
            return subprocess.CompletedProcess(args, 2, '', 'private diagnostic')
        bridge = relay.CliBridge('/fake/colab', 'session', runner=failed)
        with self.assertRaisesRegex(RuntimeError, 'code 2'):
            bridge.request('GET', '/health')

    def test_portable_cpu_notebook_contains_current_sources_and_no_saved_outputs(self):
        root = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location('extension_notebook_builder', root / 'scripts/build_extension_backend_notebook.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        notebook = builder.build()
        saved = json.loads((root / 'notebooks/extension_backend_cpu_colab.ipynb').read_text())
        self.assertEqual(notebook, saved)
        self.assertNotIn('accelerator', notebook['metadata'])
        bundled = next(''.join(cell['source']) for cell in notebook['cells'] if 'SOURCES = json.loads' in ''.join(cell['source']))
        tree = ast.parse(bundled)
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SOURCES' for t in node.targets))
        payload = ast.literal_eval(assignment.value.args[0])
        self.assertEqual(hashlib.sha256(payload.encode()).hexdigest(), notebook['metadata']['source_sha256'])
        for name, text in json.loads(payload).items():
            self.assertEqual(text, (root / name).read_text(encoding='utf-8'))
            if name.endswith('.py'):
                ast.parse(text)
        for cell in notebook['cells']:
            self.assertFalse(cell.get('outputs'))
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))

    def test_managed_local_shutdown_requires_its_private_token(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), relay.make_handler(None))
        server.stop_token = 'test-private-token'
        state = self.root / 'relay-state.json'
        state.write_text(json.dumps({'port': server.server_port, 'stop_token': server.stop_token}))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            preflight = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/verify',
                method='OPTIONS', headers={'Origin': 'chrome-extension://test-extension-id'})
            with urllib.request.urlopen(preflight) as response:
                self.assertEqual(response.status, 204)
                self.assertIsNone(response.headers.get('Content-Length'))
                self.assertEqual(response.read(), b'')
                self.assertEqual(response.headers['Access-Control-Allow-Origin'], 'chrome-extension://test-extension-id')
            request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/_relay/stop', data=b'{}')
            with self.assertRaises(urllib.error.HTTPError) as failure:
                urllib.request.urlopen(request)
            self.assertEqual(failure.exception.code, 403)
            self.assertTrue(thread.is_alive())
            relay.stop_local_relay(state)
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
            self.assertFalse(state.exists())
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
