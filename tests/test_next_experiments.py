import json
import math
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from scripts.run_evidence_ablation import main as ablation_main
from scripts.run_evidence_ablation import comparison, evidence_variants, load_partial_predictions
from scripts.run_source_diversity import diversify_row, main as diversity_main
from scripts.train_weak_url_reranker import (
    FEATURE_NAMES,
    features_for_row,
    main as weak_reranker_main,
    positive_passage_ids,
    rerank_row,
)
from src.utils.io import read_jsonl, write_jsonl


def test_evidence_variants_rotate_nonempty_without_same_claim() -> None:
    records = [
        {"claim_id": "b", "label": "Refuted", "retrieved_evidence": [{"passage_id": "pb"}]},
        {"claim_id": "empty", "label": "Supported", "retrieved_evidence": []},
        {"claim_id": "a", "label": "Supported", "retrieved_evidence": [{"passage_id": "pa"}]},
    ]
    variants = evidence_variants(records)
    assert all(not row["retrieved_evidence"] for row in variants["blank"])
    swapped = {row["claim_id"]: row for row in variants["swapped"]}
    assert swapped["a"]["retrieved_evidence"][0]["passage_id"] == "pb"
    assert swapped["b"]["retrieved_evidence"][0]["passage_id"] == "pa"
    assert swapped["empty"]["retrieved_evidence"] == []
    assert swapped["empty"]["evidence_donor_claim_id"] is None
    assert records[0]["retrieved_evidence"][0]["passage_id"] == "pb"


def test_evidence_variants_reject_too_few_nonempty_pools() -> None:
    with pytest.raises(ValueError, match="at least two"):
        evidence_variants([{"claim_id": "one", "retrieved_evidence": []}])


def test_ablation_checkpoint_truncated_tail_and_valid_no_newline(tmp_path) -> None:
    path = tmp_path / "predictions.partial.jsonl"
    valid = {"claim_id": "c1", "true_label": "Refuted", "predicted_label": "Refuted"}
    path.write_bytes((json.dumps(valid) + "\n" + '{"claim_id":"c2"').encode())
    assert load_partial_predictions(path, {"c1", "c2"}) == {"c1": valid}
    assert path.read_text().strip() == json.dumps(valid)
    path.write_bytes(json.dumps(valid).encode())
    assert load_partial_predictions(path, {"c1"}) == {"c1": valid}
    assert path.read_bytes().endswith(b"\n")
    with path.open("ab") as handle:
        handle.write((json.dumps({"claim_id": "c2", "true_label": "Supported", "predicted_label": "Supported"}) + "\n").encode())
    assert set(load_partial_predictions(path, {"c1", "c2"})) == {"c1", "c2"}


def test_ablation_comparison_counts_fixed_and_lost() -> None:
    original = [
        {"claim_id": "a", "true_label": "Supported", "predicted_label": "Supported"},
        {"claim_id": "b", "true_label": "Refuted", "predicted_label": "Supported"},
    ]
    altered = [
        {"claim_id": "a", "true_label": "Supported", "predicted_label": "Refuted"},
        {"claim_id": "b", "true_label": "Refuted", "predicted_label": "Refuted"},
    ]
    result = comparison(original, altered)
    assert result == {
        "changed_prediction_count": 2,
        "changed_prediction_fraction": 1.0,
        "originally_correct_now_wrong": 1,
        "originally_wrong_now_correct": 1,
    }


def test_ablation_cli_runs_and_reuses_artifacts_without_real_model(tmp_path, monkeypatch) -> None:
    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def to(self, _device):
            return self

        def float(self):
            return self


        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class FakeTorch:
        utils = SimpleNamespace(data=SimpleNamespace(Dataset=object))
        inference_mode = staticmethod(nullcontext)

        @staticmethod
        def softmax(logits, dim):
            assert dim == -1
            probabilities = []
            for row in logits.values:
                exp = [math.exp(value - max(row)) for value in row]
                probabilities.append([value / sum(exp) for value in exp])
            return FakeTensor(probabilities)

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeTokenizer()

        def encode(self, text, **_kwargs):
            return [ord(character) for character in text]

        def decode(self, tokens, **_kwargs):
            return "".join(chr(token) for token in tokens)

        def __call__(self, texts, **_kwargs):
            return {"input_ids": FakeTensor([len(text) % 10 for text in texts])}

    class FakeModel:
        config = SimpleNamespace(num_labels=4, id2label={0: "Supported", 1: "Refuted", 2: "Not Enough Evidence", 3: "Conflicting Evidence"})

        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeModel()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, **encoded):
            scores = encoded["input_ids"].values
            return SimpleNamespace(logits=FakeTensor([[score, 10 - score, 0, 0] for score in scores]))

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=FakeTokenizer, AutoModelForSequenceClassification=FakeModel,
    ))
    claims = tmp_path / "dev.jsonl"
    rankings = tmp_path / "rankings.jsonl"
    original = tmp_path / "original.jsonl"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"fake test weights")
    run_config = tmp_path / "run_config.json"
    run_config.write_text(json.dumps({
        "claim_token_budget": 96, "evidence_token_budget_per_passage": 110, "max_length": 512,
    }), encoding="utf-8")
    write_jsonl([
        {"claim_id": "a", "claim": "First", "label": "Supported"},
        {"claim_id": "b", "claim": "Second", "label": "Refuted"},
    ], claims)
    write_jsonl([
        {"claim_id": "a", "retrieved": [{"passage_id": "pa", "text": "alpha"}]},
        {"claim_id": "b", "retrieved": [{"passage_id": "pb", "text": "beta"}]},
    ], rankings)
    write_jsonl([
        {"claim_id": "a", "true_label": "Supported", "predicted_label": "Supported"},
        {"claim_id": "b", "true_label": "Refuted", "predicted_label": "Refuted"},
    ], original)
    monkeypatch.setattr(sys, "argv", [
        "ablation", "--claims", str(claims), "--rankings", str(rankings),
        "--original-predictions", str(original), "--model-dir", str(model_dir),
        "--model-run-config", str(run_config), "--output-dir", str(tmp_path / "out"),
        "--run-name", "smoke", "--device", "cpu", "--batch-size", "1",
        "--checkpoint-every", "1", "--resume",
    ])
    ablation_main()
    ablation_main()
    output = tmp_path / "out" / "smoke"
    (output / "ablation_manifest.json").unlink()
    ablation_main()
    assert (output / "ablation_manifest.json").is_file()
    report = json.loads((output / "ablation_report.json").read_text())
    assert report["claim_count"] == 2
    assert set(report["variants"]) == {"blank", "swapped"}
    assert len(read_jsonl(output / "blank_predictions.jsonl")) == 2


def test_source_diversity_keeps_best_per_url_and_all_candidates() -> None:
    row = {"claim_id": "c1", "claim": "Claim", "retrieved": [
        {"passage_id": "a1", "url": "https://a.example/x/", "score": 4.0},
        {"passage_id": "a2", "url": "https://a.example/x", "score": 3.0},
        {"passage_id": "b", "url": "https://b.example/y", "score": 2.0},
        {"passage_id": "unknown", "url": None, "score": 1.0},
    ]}
    output = diversify_row(row)
    assert [item["passage_id"] for item in output["retrieved"]] == ["a1", "b", "unknown", "a2"]
    assert [item["passage_id"] for item in row["retrieved"]] == ["a1", "a2", "b", "unknown"]


def test_source_diversity_cli_evaluates_dev_and_reuses_artifacts(tmp_path, monkeypatch) -> None:
    rankings = tmp_path / "bm25.jsonl"
    qrels = tmp_path / "qrels.jsonl"
    output = tmp_path / "runs"
    write_jsonl([{"claim_id": "dev:1", "claim": "Claim", "retrieved": [
        {"passage_id": "a1", "url": "https://a.example", "score": 3.0},
        {"passage_id": "a2", "url": "https://a.example", "score": 2.0},
        {"passage_id": "a3", "url": "https://a.example", "score": 1.0},
        {"passage_id": "b", "url": "https://b.example", "score": 0.5},
    ]}], rankings)
    write_jsonl([{"claim_id": "dev:1", "relevant_passage_ids": ["b"]}], qrels)
    monkeypatch.setattr(sys, "argv", [
        "source-diversity", "--input-rankings", str(rankings), "--qrels", str(qrels),
        "--output-dir", str(output), "--run-name", "smoke", "--split", "dev",
    ])
    diversity_main()
    diversity_main()
    final = read_jsonl(output / "smoke" / "dev_rankings.jsonl")
    metrics = json.loads((output / "smoke" / "dev_metrics.json").read_text())
    assert [item["passage_id"] for item in final[0]["retrieved"]] == ["a1", "b", "a2", "a3"]
    assert metrics["original_bm25"]["evidence_hit_at_3"] == 0.0
    assert metrics["diverse"]["evidence_hit_at_3"] == 1.0
    assert metrics["changed_top3_claim_count"] == 1


def test_source_diversity_rejects_train_qrels(tmp_path, monkeypatch) -> None:
    rankings = tmp_path / "bm25.jsonl"
    qrels = tmp_path / "qrels.jsonl"
    write_jsonl([], rankings)
    write_jsonl([], qrels)
    monkeypatch.setattr(sys, "argv", [
        "source-diversity", "--input-rankings", str(rankings), "--qrels", str(qrels),
        "--output-dir", str(tmp_path / "out"), "--run-name", "smoke", "--split", "train",
    ])
    with pytest.raises(SystemExit):
        diversity_main()


def test_weak_reranker_uses_url_only_as_train_label() -> None:
    row = {"claim_id": "c1", "claim": "red moon", "retrieved": [
        {"passage_id": "p1", "text": "blue sky", "url": "https://bad.example", "score": 4.0},
        {"passage_id": "p2", "text": "red moon tonight", "url": "https://good.example/", "score": 1.0},
    ]}
    claim = {"source_urls": ["https://good.example"]}
    assert positive_passage_ids(claim, row, 20) == {"p2"}
    assert "url" not in " ".join(FEATURE_NAMES)
    assert features_for_row(row, 20)[1, 2] == 1.0

    class LexicalModel:
        @staticmethod
        def decision_function(features):
            return features[:, 2]

    ranked = rerank_row(row, LexicalModel(), 1.0, 20)
    assert [item["passage_id"] for item in ranked["retrieved"]] == ["p2", "p1"]


def test_weak_reranker_cli_train_only_selection_and_idempotence(tmp_path, monkeypatch) -> None:
    claims_path = tmp_path / "train.jsonl"
    train_path = tmp_path / "train_rankings.jsonl"
    dev_path = tmp_path / "dev_rankings.jsonl"
    qrels_path = tmp_path / "dev_qrels.jsonl"
    train_claims = []
    train_rankings = []
    for number in range(30):
        claim_id = f"train:{number:03d}"
        train_claims.append({"claim_id": claim_id, "source_urls": [f"https://good.example/{number}"]})
        train_rankings.append({"claim_id": claim_id, "claim": f"red moon {number}", "retrieved": [
            {"passage_id": f"{claim_id}:a", "url": "https://bad.example/a", "text": "blue sky", "score": 4.0},
            {"passage_id": f"{claim_id}:b", "url": "https://bad.example/b", "text": "blue sea", "score": 3.0},
            {"passage_id": f"{claim_id}:c", "url": "https://bad.example/c", "text": "green field", "score": 2.0},
            {"passage_id": f"{claim_id}:good", "url": f"https://good.example/{number}/", "text": f"red moon {number} tonight", "score": 1.0},
        ]})
    dev_rankings = [{"claim_id": "dev:1", "claim": "red moon 999", "retrieved": [
        {"passage_id": "dev:a", "url": "https://bad.example/a", "text": "blue sky", "score": 4.0},
        {"passage_id": "dev:b", "url": "https://bad.example/b", "text": "blue sea", "score": 3.0},
        {"passage_id": "dev:c", "url": "https://bad.example/c", "text": "green field", "score": 2.0},
        {"passage_id": "dev:good", "url": "https://good.example/dev", "text": "red moon 999 tonight", "score": 1.0},
    ]}]
    write_jsonl(train_claims, claims_path)
    write_jsonl(train_rankings, train_path)
    write_jsonl(dev_rankings, dev_path)
    write_jsonl([{"claim_id": "dev:1", "relevant_passage_ids": ["dev:good"]}], qrels_path)
    monkeypatch.setattr(sys, "argv", [
        "weak-reranker", "--train-claims", str(claims_path), "--train-rankings", str(train_path),
        "--dev-rankings", str(dev_path), "--dev-qrels", str(qrels_path),
        "--output-dir", str(tmp_path / "out"), "--run-name", "smoke", "--max-candidates", "4",
    ])
    weak_reranker_main()
    weak_reranker_main()
    output = tmp_path / "out" / "smoke"
    report = json.loads((output / "dev_metrics.json").read_text())
    selection = json.loads((output / "model_selection.json").read_text())
    assert report["dev_claim_count"] == 1
    assert report["train_eligible_claim_count"] == 30
    assert selection["validation_claim_count"] == 3
    assert report["bm25_reference"]["evidence_hit_at_3"] == 0.0
    assert report["dev"]["evidence_hit_at_3"] == 1.0
    assert len(read_jsonl(output / "dev_rankings.jsonl")) == 1
    assert not (output / "train_rankings.jsonl").exists()
