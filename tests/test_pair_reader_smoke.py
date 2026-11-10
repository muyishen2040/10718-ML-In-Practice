"""Real offline torch/Transformers tests; no pretrained downloads or GPU needed."""
import argparse
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import AutoModelForSequenceClassification, BertConfig, BertForSequenceClassification, BertTokenizerFast, DebertaV2Config

from src.data.averitec import CANONICAL_LABELS
from src.retrieval.evidence_pipeline import prepare_pipeline
from src.retrieval.evidence_quality import QUALITY_NAMES
from src.utils.io import read_jsonl, write_jsonl
from src.verification.pair_reader import EvidencePairReader, PairReaderConfig
from src.verification.pair_training import train_reader
from src.verification.evidence_experiments import fast_aggregate, frozen_distil, compare_results


def test_pair_reader_unlabeled_inference_matches_labeled_inputs(tmp_path):
    from src.verification.pair_reader import PairDataset, PairCollator
    directory = tiny_nli(tmp_path)
    tokenizer = BertTokenizerFast.from_pretrained(directory)
    record = {'claim': 'claim event occurred', 'retrieved_evidence': [{'text': 'event occurred'}]}
    dataset = PairDataset([record, {**record, 'label': 'Supported'}], tokenizer, max_length=64, show_progress=False)
    collator = PairCollator(tokenizer)
    unlabeled, labeled = collator([dataset[0]]), collator([dataset[1]])
    assert 'labels' not in unlabeled
    for key, value in unlabeled.items():
        torch.testing.assert_close(value, labeled[key])
    with pytest.raises(ValueError, match='mix labeled'):
        collator([dataset[0], dataset[1]])
    model = EvidencePairReader.from_nli(str(directory)).eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(**unlabeled).logits, model(**labeled).logits)


def tiny_nli(tmp_path, kind="bert"):
    directory = tmp_path / "tiny_nli"
    directory.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "source", "domain", "type", "passage", "claim", "event", "occurred", "not", "a", "b", ".", ":"]
    (directory / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(directory / "vocab.txt"))
    tokenizer.save_pretrained(directory)
    config_class = BertConfig if kind == "bert" else DebertaV2Config
    config = config_class(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=24, num_labels=3, id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
    AutoModelForSequenceClassification.from_config(config).save_pretrained(directory)
    return directory


@pytest.mark.parametrize("kind", ["bert", "deberta"])
def test_pair_reader_gradient_permutation_empty_evidence_and_save_load(tmp_path, kind):
    directory = tiny_nli(tmp_path, kind)
    model = EvidencePairReader.from_nli(str(directory)).eval()
    batch = {"input_ids": torch.randint(0, 16, (2, 3, 16)), "attention_mask": torch.ones(2, 3, 16, dtype=torch.long),
             "evidence_mask": torch.tensor([[1, 1, 1], [0, 0, 0]]), "quality": torch.rand(2, 3, len(QUALITY_NAMES)),
             "teacher_probabilities": torch.tensor([[[.2, .3, .5]] * 3] * 2)}
    first = model(**batch).logits
    reversed_batch = {k: v.flip(1) for k, v in batch.items()}
    torch.testing.assert_close(first, model(**reversed_batch).logits, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(first).all()
    model(**batch, labels=torch.tensor([0, 1])).loss.backward()
    assert model.verdict[-1].weight.grad is not None
    assert model.reader.get_input_embeddings().weight.grad is None
    target = tmp_path / "saved"
    model.save_pretrained(target)
    restored = EvidencePairReader.from_pretrained(target).eval()
    torch.testing.assert_close(first, restored(**batch).logits)


def test_full_offline_prepare_train_select_refit_and_resume(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    directory = tiny_nli(tmp_path)
    data = tmp_path / "data"
    for split, count in (("train", 80), ("dev", 8)):
        rows = [{"claim_id": f"{split}:{i}", "claim": f"claim event {i}", "label": CANONICAL_LABELS[i % 4],
                 "metadata": {"claim_date": "2020-10-02"}} for i in range(count)]
        write_jsonl(rows, data / "processed/averitec" / f"{split}.jsonl")
        write_jsonl([{**r, "retrieved": [{"passage_id": f"{r['claim_id']}:p", "text": "event occurred", "url": "https://a.test"}]} for r in rows], tmp_path / f"{split}_rankings.jsonl")
    prepared = tmp_path / "prepared"
    prep = argparse.Namespace(output=prepared, train_claims=data / "processed/averitec/train.jsonl", dev_claims=data / "processed/averitec/dev.jsonl",
        train_rankings=tmp_path / "train_rankings.jsonl", dev_rankings=tmp_path / "dev_rankings.jsonl", model=str(directory), revision="local",
        max_length=64, claim_budget=16, max_candidates=2, candidate_depth=20, batch_size=8, device="cpu", limit=0, nli_label_order=None)
    prepare_pipeline(prep)
    prepare_pipeline(prep)  # Reuses completed journal rows.
    assert len(read_jsonl(prepared / "train.partial.jsonl")) == 80
    args = argparse.Namespace(prepared=prepared, data_root=data, mode="clean_nli", output=tmp_path / "trained",
         device="cpu", validation_seed=718, validation_policy="grouped", learning_rates=[.001], epochs=2,
         batch_size=8, accumulation=1, seed=718, distillation_weight=.1, selection_only=True)
    from transformers import Trainer as BaseTrainer
    from src.utils.checkpoints import latest_complete_trainer_checkpoint

    original_save = BaseTrainer._save_checkpoint
    interrupted = False

    def save_then_interrupt(self, *positional, **kwargs):
        nonlocal interrupted
        original_save(self, *positional, **kwargs)
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(BaseTrainer, "_save_checkpoint", save_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        train_reader(args)
    assert latest_complete_trainer_checkpoint(args.output / "trials/lr_0.001")
    train_reader(args)
    assert not (args.output / "dev/metrics.json").exists()
    args.selection_only = False
    train_reader(args)
    result = json.loads((args.output / "dev/metrics.json").read_text())
    assert result["claim_count"] == 8
    train_reader(args)
    assert len(read_jsonl(args.output / "dev/predictions.jsonl")) == 8
    fast_args = argparse.Namespace(data_root=data, prepared=prepared, mode="clean_nli", output=tmp_path / "fast",
                                   validation_seed=718, validation_policy="grouped", selection_only=True)
    fast_aggregate(fast_args)
    assert not (fast_args.output / "dev/metrics.json").exists()
    fast_args.selection_only = False
    fast_aggregate(fast_args)
    baseline = tmp_path / "baseline"
    classifier = BertForSequenceClassification(BertConfig(vocab_size=17, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=24, num_labels=4, id2label=dict(enumerate(CANONICAL_LABELS))))
    classifier.save_pretrained(baseline / "final_model")
    BertTokenizerFast.from_pretrained(directory).save_pretrained(baseline / "final_model")
    (baseline / "transformer_run_config.json").write_text(json.dumps({"claim_token_budget": 96,
        "evidence_token_budget_per_passage": 110, "max_length": 512}))
    frozen_distil(argparse.Namespace(data_root=data, prepared=prepared, mode="raw", output=tmp_path / "frozen/raw", baseline=baseline, device="cpu"))
    compare_results(argparse.Namespace(data_root=data, prepared=prepared, output=tmp_path, qrels=None))
    report = json.loads((tmp_path / "evidence_reader_results.json").read_text())
    assert report["verdicts"]["frozen/raw"]["claim_count"] == 8
    assert "trained" in report["verdicts"]
    assert "fast" in report["paired_vs_frozen_raw"]
