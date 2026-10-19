import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from scripts.prepare_verifier_variant_rankings import main as prepare_main, variant_rows
from scripts import run_cross_encoder_reranker as reranker_module
from scripts.run_cross_encoder_reranker import main as rerank_main, reranked_row
from src.utils.io import read_jsonl, write_jsonl


def test_variant_rows_keep_claim_only_empty_and_gold_capped() -> None:
    record = {
        "claim_id": "c1",
        "claim": "The claim",
        "evidence": [
            {"passage_id": f"p{i}", "text": f"answer {i}", "url": "https://example.org"}
            for i in range(4)
        ],
    }
    claim_rows, claim_coverage = variant_rows([record], "claim_only")
    gold_rows, gold_coverage = variant_rows([record], "gold")
    assert claim_rows[0]["retrieved"] == []
    assert claim_coverage == 0
    assert [item["passage_id"] for item in gold_rows[0]["retrieved"]] == ["p0", "p1", "p2"]
    assert gold_coverage == 1


def test_prepare_variants_write_both_splits_and_resume(tmp_path, monkeypatch) -> None:
    processed = tmp_path / "processed" / "averitec"
    for split in ("train", "dev"):
        write_jsonl(
            [{"claim_id": f"{split}:1", "claim": "A fact", "label": "Refuted", "evidence": []}],
            processed / f"{split}.jsonl",
        )
    output = tmp_path / "variants"
    monkeypatch.setattr(sys, "argv", ["prepare", "--data-root", str(tmp_path), "--output-dir", str(output), "--mode", "claim_only"])
    prepare_main()
    prepare_main()
    assert len(read_jsonl(output / "train_rankings.jsonl")) == 1
    assert len(read_jsonl(output / "dev_rankings.jsonl")) == 1
    assert json.loads((output / "variant_config.json").read_text())["mode"] == "claim_only"


def test_rerank_preserves_top_n_and_breaks_ties_by_passage_id() -> None:
    source = {
        "claim_id": "c1",
        "claim": "A fact",
        "retrieved": [
            {"passage_id": "b", "text": "first", "score": 2.0},
            {"passage_id": "a", "text": "second", "score": 1.0},
            {"passage_id": "c", "text": "third", "score": 0.0},
        ],
    }
    ranked = reranked_row(source, [0.7, 0.7], max_candidates=2)
    assert [item["passage_id"] for item in ranked["retrieved"]] == ["a", "b"]
    assert ranked["retrieved"][0]["first_stage_score"] == 1.0
    assert len(source["retrieved"]) == 3
    with pytest.raises(ValueError, match="score count"):
        reranked_row(source, [0.7], max_candidates=2)


def test_reranker_cli_scores_and_resumes_without_real_model(tmp_path, monkeypatch) -> None:
    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def to(self, _device):
            return self

        def reshape(self, *_shape):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class FakeTokenizer:
        init_kwargs = {"_commit_hash": "fake-revision"}

        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeTokenizer()

        def __call__(self, _claims, passages, **_kwargs):
            return {"input_ids": FakeTensor([1.0 if "relevant" in text else 0.0 for text in passages])}

    class FakeModel:
        config = SimpleNamespace(num_labels=1, _commit_hash="fake-revision")

        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeModel()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, **encoded):
            return SimpleNamespace(logits=encoded["input_ids"].reshape(-1, 1))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeTokenizer, AutoModelForSequenceClassification=FakeModel))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), inference_mode=nullcontext))
    input_path = tmp_path / "input.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    output_root = tmp_path / "runs"
    write_jsonl(
        [{"claim_id": "dev:1", "claim": "A fact", "retrieved": [
            {"passage_id": f"p{i}", "text": "relevant answer" if i == 4 else "unrelated", "score": float(5-i)}
            for i in range(5)
        ]}], input_path,
    )
    write_jsonl([{"claim_id": "dev:1", "relevant_passage_ids": ["p4"]}], qrels_path)
    monkeypatch.setattr(sys, "argv", [
        "rerank", "--input-rankings", str(input_path), "--qrels", str(qrels_path),
        "--output-dir", str(output_root), "--run-name", "smoke", "--split", "dev",
        "--model", "fake", "--device", "cpu", "--max-candidates", "5",
        "--batch-size", "2", "--checkpoint-every", "1", "--resume",
    ])
    real_append = reranker_module.append_checkpoint_rows
    interrupted = False

    def interrupt_after_append(path, rows):
        nonlocal interrupted
        real_append(path, rows)
        if rows and not interrupted:
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(reranker_module, "append_checkpoint_rows", interrupt_after_append)
    with pytest.raises(KeyboardInterrupt):
        rerank_main()
    monkeypatch.setattr(reranker_module, "append_checkpoint_rows", real_append)
    rerank_main()
    rerank_main()
    ranked = read_jsonl(output_root / "smoke" / "dev_rankings.jsonl")
    metrics = json.loads((output_root / "smoke" / "dev_metrics.json").read_text())
    assert ranked[0]["retrieved"][0]["passage_id"] == "p4"
    assert metrics["evidence_hit_at_3"] == 1.0
    assert metrics["claim_count"] == 1
