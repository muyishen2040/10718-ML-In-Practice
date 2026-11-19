"""Serve the corrected saved pair reader using its original prediction-time inputs."""
import hashlib
import json
from pathlib import Path

from baseline_adapter import BaselineAdapter, LABELS, load_rankings


class NliReaderAdapter(BaselineAdapter):
    @classmethod
    def from_paths(cls, run_dir, rankings_path, preparation_path, device='cpu'):
        import torch
        from transformers import AutoTokenizer
        from src.verification.pair_reader import EvidencePairReader, PairDataset, PairCollator
        from src.retrieval.evidence_quality import QUALITY_NAMES

        run_dir, rankings_path, preparation_path = map(Path, (run_dir, rankings_path, preparation_path))
        run = json.loads((run_dir / 'reader_run.json').read_text())
        preparation = json.loads(preparation_path.read_text())
        if hashlib.sha256(preparation_path.read_bytes()).hexdigest() != run['preparation_sha256']:
            raise ValueError('Reader preparation differs from the saved training run')
        if hashlib.sha256(rankings_path.read_bytes()).hexdigest() != run['input_hashes']['dev_rankings']:
            raise ValueError('Reader evidence differs from the saved evaluation inputs')
        rankings = load_rankings(rankings_path)
        for row in rankings.values():
            for item in (row.get('retrieved') or [])[:3]:
                if len(item.get('quality', [])) != len(QUALITY_NAMES):
                    raise ValueError('Reader evidence lacks its prediction-time quality features')
        model_dir = run_dir / 'final_model'
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
        model = EvidencePairReader.from_pretrained(model_dir, local_files_only=True).to(device).eval()
        if [model.config.id2label.get(i) for i in range(4)] != list(LABELS):
            raise ValueError('Reader label order differs from the canonical labels')
        if model.config.loss_normalization != 'sample_mean':
            raise ValueError('Expected the corrected reader checkpoint')
        collator = PairCollator(tokenizer)

        def predict(claim, passages):
            # No gold label is loaded or supplied, even as a placeholder.
            dataset = PairDataset([{'claim': claim, 'retrieved_evidence': passages}], tokenizer,
                max_length=preparation['max_length'], claim_budget=preparation['claim_budget'], show_progress=False)
            batch = collator([dataset[0]])
            if 'labels' in batch:
                raise ValueError('Serving inputs must be unlabeled')
            with torch.inference_mode():
                return model(**{key: value.to(device) for key, value in batch.items()}).logits.float().softmax(-1)[0].cpu().tolist()

        adapter = cls(rankings, predict, model_id='nli_reader', model_label='Corrected NLI reader',
            model_description='A trained reader checks each BM25 passage against the claim.',
            provenance='corrected_nli_reader_bm25_top3',
            model_note='This trained evidence reader is experimental. Read the sources before deciding.')
        adapter.explanation_text = 'A trained reader compared the claim with each displayed passage and combined them into this verdict. It does not generate reasons.'
        return adapter
