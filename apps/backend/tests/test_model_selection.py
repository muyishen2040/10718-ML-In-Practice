"""Model selection must change both evidence and inference, without cache crossover."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_adapter import BaselineAdapter
from model_registry import ModelRegistry
from server import handle_verify, model_catalog
from colab_runtime import SerializedAdapter
from colab_relay import CliBridge, MARKER
import subprocess


def rows(source):
    return {'c1': {'claim_id': 'c1', 'claim': 'A benchmark claim', 'retrieved': [
        {'passage_id': source, 'text': source, 'url': 'https://example.org/' + source}]}}


class SelectionTests(unittest.TestCase):
    def test_shared_classifier_separate_evidence_and_prediction_caches(self):
        calls = []
        def predict(claim, passages):
            calls.append(passages[0]['passage_id'])
            return [1, 0, 0, 0] if calls[-1] == 'bm25' else [0, 1, 0, 0]
        baseline = BaselineAdapter(rows('bm25'), predict)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rankings.jsonl'
            path.write_text(json.dumps(rows('bge')['c1']) + '\n')
            bge = baseline.with_evidence(path, model_id='distilbert_bge', model_label='DistilBERT + BGE',
                description='Semantic evidence', provenance='distilbert_bge_top3')
        self.assertIs(bge.predict_fn, baseline.predict_fn)
        registry = SerializedAdapter(ModelRegistry({'distilbert_bm25': baseline, 'distilbert_bge': bge}))
        for model_id, verdict, source in [('distilbert_bm25', 'Supported', 'bm25'),
                                           ('distilbert_bge', 'Refuted', 'bge'),
                                           ('distilbert_bm25', 'Supported', 'bm25')]:
            code, response = handle_verify({'mode': 'benchmark', 'claim_id': 'c1', 'model_id': model_id}, registry)
            self.assertEqual((code, response['model_id'], response['verdict']), (200, model_id, verdict))
            self.assertEqual(response['evidence'][0]['passage_id'], source)
        self.assertEqual(calls, ['bm25', 'bge'])
        self.assertEqual(handle_verify({'mode': 'benchmark', 'claim_id': 'c1'}, registry)[1]['model_id'], 'distilbert_bm25')
        self.assertEqual(handle_verify({'mode': 'benchmark', 'claim_id': 'c1', 'model_id': 'unknown'}, registry)[0], 400)
        self.assertEqual(handle_verify({'mode': 'benchmark', 'claim_id': 'c1', 'model_id': 'nli_reader'}, registry)[0], 503)
        catalog = model_catalog(registry)
        self.assertEqual(len(catalog['models']), 4)
        self.assertFalse(next(m for m in catalog['models'] if m['id'] == 'nli_reader')['available'])

    def test_registry_rejects_incompatible_claim_catalogs(self):
        a = BaselineAdapter(rows('bm25'), None)
        b = BaselineAdapter(rows('bge'), None)
        b.rankings['c1']['claim'] = 'Different claim'
        with self.assertRaises(ValueError):
            ModelRegistry({'distilbert_bm25': a, 'distilbert_bge': b})

    def test_relay_loads_and_caches_model_catalog(self):
        requests = []
        models = {'default_model_id': 'distilbert_bm25', 'models': [
            {'id': 'distilbert_bm25', 'label': 'DistilBERT + BM25', 'available': True}]}
        replies = [{'ok': True, 'adapter': 'model_registry', 'model_selection': True}, {'claims': []}, models]
        def runner(command, **kwargs):
            requests.append(kwargs['input'])
            return subprocess.CompletedProcess(command, 0, MARKER + json.dumps({'http_status': 200, 'body': replies.pop(0)}), '')
        bridge = CliBridge('/fake/colab', 'test', runner=runner)
        bridge.initialize()
        self.assertEqual(bridge.request('GET', '/models'), (200, models))
        self.assertEqual(len(requests), 3)


if __name__ == '__main__':
    unittest.main()
