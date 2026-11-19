"""Explicit serving choices; unknown or unavailable choices never silently fall back."""
import json
from pathlib import Path

from baseline_adapter import BaselineAdapter

PROFILES = (
    {'id': 'distilbert_bm25', 'label': 'DistilBERT + BM25', 'description': 'Original classifier with keyword-selected evidence.'},
    {'id': 'distilbert_bge', 'label': 'DistilBERT + BGE', 'description': 'The same classifier with semantic-retrieval evidence.'},
    {'id': 'distilbert_fusion', 'label': 'DistilBERT + BM25/BGE fusion', 'description': 'The same classifier with fused keyword and semantic evidence.'},
    {'id': 'nli_reader', 'label': 'Corrected NLI reader', 'description': 'A trained reader compares each BM25 passage with the claim.'},
)


class ModelRegistry:
    name = 'model_registry'
    model_selection = True
    default_model_id = 'distilbert_bm25'

    def __init__(self, adapters, unavailable=None):
        self.adapters = dict(adapters)
        self.unavailable = dict(unavailable or {})
        if self.default_model_id not in self.adapters:
            raise ValueError('The original baseline must be available')
        baseline = self.adapters[self.default_model_id]
        for adapter in self.adapters.values():
            if set(adapter.rankings) != set(baseline.rankings):
                raise ValueError('Serving choices must cover the same claim IDs')
            if any(str(row['claim']) != str(baseline.rankings[cid]['claim']) for cid, row in adapter.rankings.items()):
                raise ValueError('Serving choices disagree on benchmark claim text')

    def list_models(self):
        return [{**profile, 'available': profile['id'] in self.adapters,
                 'unavailable_reason': self.unavailable.get(profile['id'])} for profile in PROFILES]

    def select_model(self, model_id=None):
        model_id = self.default_model_id if model_id is None else model_id
        if not isinstance(model_id, str) or model_id not in {profile['id'] for profile in PROFILES}:
            raise ValueError('Unknown model_id')
        if model_id not in self.adapters:
            raise LookupError(self.unavailable.get(model_id, 'This model is not loaded'))
        return self.adapters[model_id]

    def get(self, claim_id):
        return self.select_model().get(claim_id)

    def list_claims(self, query='', limit=50):
        return self.select_model().list_claims(query, limit)


def load_registry(cache_root, manifest, *, device='cpu'):
    cache_root = Path(cache_root)
    baseline = BaselineAdapter.from_paths(cache_root / 'baseline', cache_root / 'dev_rankings.jsonl', device)
    adapters, unavailable = {'distilbert_bm25': baseline}, {}
    available = manifest.get('available_pipelines', {'distilbert_bm25': True})
    for model_id, filename, provenance in (
        ('distilbert_bge', 'bge_rankings.jsonl', 'distilbert_bge_top3'),
        ('distilbert_fusion', 'fusion_rankings.jsonl', 'distilbert_fusion_top3'),
    ):
        if available.get(model_id):
            profile = next(item for item in PROFILES if item['id'] == model_id)
            adapters[model_id] = baseline.with_evidence(cache_root / filename, model_id=model_id,
                model_label=profile['label'], description=profile['description'], provenance=provenance)
        else:
            unavailable[model_id] = 'Its saved evidence file was not found in shared Drive.'
    if available.get('nli_reader'):
        try:
            from nli_adapter import NliReaderAdapter
            print('Loading the corrected NLI reader on CPU...', flush=True)
            adapters['nli_reader'] = NliReaderAdapter.from_paths(cache_root / 'nli_reader',
                cache_root / 'nli_rankings.jsonl', cache_root / 'nli_preparation.json', device)
        except Exception as error:
            print('NLI reader unavailable:', type(error).__name__, str(error), flush=True)
            unavailable['nli_reader'] = 'The reader could not be loaded; check the Colab notebook output.'
    else:
        unavailable['nli_reader'] = 'Its checkpoint or prepared evidence was not found in shared Drive.'
    return ModelRegistry(adapters, unavailable)
