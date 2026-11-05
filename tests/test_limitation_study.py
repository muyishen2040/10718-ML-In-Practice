import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_limitation_diagnostics as study
from src.evaluation.splits import parse_claim_date, selection_split
from src.utils.io import read_jsonl, write_json, write_jsonl


class WordTokenizer:
    @staticmethod
    def from_pretrained(*_args, **_kwargs):
        return WordTokenizer()

    def encode(self, text, **_kwargs):
        return text.split()

    def decode(self, tokens, **_kwargs):
        return " ".join(tokens)


def test_excerpt_finds_later_decisive_terms_and_stays_verbatim():
    text = "Unrelated background appears here. The comet arrived in 2020. Another unrelated sentence."
    got = study.query_excerpt(text, "The comet arrived in 2020", WordTokenizer(), 6)
    assert got == "The comet arrived in 2020."
    assert len(got.split()) <= 6
    assert got in text


def test_excerpt_handles_long_unpunctuated_text_with_fixed_budget():
    text = " ".join(["filler"] * 80 + ["comet", "arrived", "2020"] + ["filler"] * 30)
    got = study.query_excerpt(text, "comet arrived 2020", WordTokenizer(), 12)
    assert got in text
    assert len(got.split()) <= 12
    assert "comet" in got


def test_prediction_inputs_whitelist_gold_and_diversity_does_not_use_labels():
    candidates = [{"passage_id": str(i), "text": "source text", "url": url,
                   "label": "Refuted", "gold_answer": "secret"}
                  for i, url in enumerate(["https://a.test/x#fragment", "https://a.test/x/", "https://b.test", "https://c.test"])]
    row = study.prediction_variant("x", "claim", candidates, "diverse_sentences", WordTokenizer(), 20)
    assert [p["passage_id"] for p in row["retrieved"]] == ["0", "2", "3"]
    assert "secret" not in json.dumps(row)
    assert "label" not in json.dumps(row)
    assert not study.prediction_variant("x", "claim", candidates, "claim_only", WordTokenizer(), 20)["retrieved"]


def test_temporal_split_keeps_boundary_dates_together_and_rejects_missing():
    rows = [{"claim_id": str(i), "label": "Refuted", "metadata": {"claim_date": date}}
            for i, date in enumerate(["2018-01-01", "2019-01-01", "2020-01-01", "2020-01-01"])]
    before, after = selection_split(rows, 0.25, 718, "temporal")
    assert list(before) == [0, 1]
    assert list(after) == [2, 3]
    rows[0]["metadata"]["claim_date"] = None
    with pytest.raises(ValueError, match="parseable"):
        selection_split(rows, 0.25, 718, "temporal")
    assert parse_claim_date("25-08-2020") == "2020-08-25"


def test_bootstrap_pairs_by_id_not_file_order():
    old = [{"claim_id": str(i), "true_label": label, "predicted_label": label}
           for i, label in enumerate(study.CANONICAL_LABELS)]
    new = list(reversed(old))
    result = study.paired_bootstrap(old, new, 20)
    assert result["delta_macro_f1"] == 0
    assert result["delta_macro_f1_95pct"] == [0, 0]
    assert result["changed_prediction_count"] == 0
    with pytest.raises(ValueError, match="coverage"):
        study.paired_bootstrap(old, new[:-1], 20)


def prepared_fixture(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=WordTokenizer))
    paths = {}
    for split in ("train", "dev"):
        records = [{"claim_id": f"{split}:{i}", "claim": f"A comet arrived {i}", "label": study.CANONICAL_LABELS[i],
                    "metadata": {"claim_date": f"2020-01-0{i+1}"},
                    "evidence": [{"passage_id": f"answer:{split}:{i}", "text": "ANNOTATED GOLD", "url": "https://gold.test"}]}
                   for i in range(3)]
        rows = [{"claim_id": r["claim_id"], "claim": r["claim"], "retrieved": [
                    {"passage_id": f"{r['claim_id']}:p{j}", "text": f"source {j} comet arrived", "url": f"https://source{j}.test"}
                    for j in range(5)] if i < 2 else []} for i, r in enumerate(records)]
        paths[split] = tmp_path / f"{split}.jsonl"
        paths[f"{split}_rankings"] = tmp_path / f"{split}_rankings.jsonl"
        write_jsonl(records, paths[split])
        write_jsonl(rows, paths[f"{split}_rankings"])
    paths["qrels"] = tmp_path / "qrels.jsonl"
    write_jsonl([{"claim_id": "dev:0", "relevant_passage_ids": ["dev:0:p4"]},
                 {"claim_id": "dev:1", "relevant_passage_ids": ["dev:1:p0"]}], paths["qrels"])
    model = tmp_path / "model"
    model.mkdir()
    write_json({"dummy": True}, model / "config.json")
    args = argparse.Namespace(output=tmp_path / "out", train_claims=paths["train"], dev_claims=paths["dev"],
        train_rankings=paths["train_rankings"], dev_rankings=paths["dev_rankings"], qrels=paths["qrels"],
        dev_top100=None, model_dir=model, extra=[], budget=3, audit_size=3, seed=718)
    study.prepare(args)
    return args, paths


def test_prepare_keeps_oracles_separate_and_preserves_manual_judgments(tmp_path, monkeypatch):
    args, paths = prepared_fixture(tmp_path, monkeypatch)
    head = read_jsonl(args.output / "prepared/head/dev_rankings.jsonl")
    assert head[0]["retrieved"][0]["passage_id"] == "dev:0:p0"
    oracle = read_jsonl(args.output / "oracle/oracle_url_head/dev_rankings.jsonl")
    assert oracle[0]["retrieved"][0]["passage_id"] == "dev:0:p4"
    for path in (args.output / "prepared").glob("*/*.jsonl"):
        assert "ANNOTATED GOLD" not in path.read_text()
    assert not list((args.output / "oracle").glob("*/train_rankings.jsonl"))
    audit = args.output / "manual_passage_audit.csv"
    edited = audit.read_text(encoding="utf-8-sig") + "\n"
    audit.write_text(edited, encoding="utf-8-sig")
    study.prepare(args)
    assert audit.read_text(encoding="utf-8-sig") == edited
    report = json.loads((args.output / "data_audit.json").read_text())
    assert report["judged_dev_claim_count"] == 2
    assert "100" not in report["candidate_coverage"]
    assert head[2]["retrieved"] == []


def test_rankings_reject_duplicate_ids_and_config_changes(tmp_path):
    path = tmp_path / "config.json"
    study.lock_config(path, {"x": 1})
    with pytest.raises(ValueError, match="changed"):
        study.lock_config(path, {"x": 2})
    with pytest.raises(ValueError, match="duplicate"):
        study.index_rows([{"claim_id": "x"}, {"claim_id": "x"}])


def test_frozen_inference_resumes_after_interrupt_and_summarizes(tmp_path, monkeypatch):
    prep, paths = prepared_fixture(tmp_path, monkeypatch)
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def to(self, *_args):
            return self

        def float(self):
            return self

        def softmax(self, _axis):
            values = np.exp(self.values)
            return Tensor(values / values.sum(axis=1, keepdims=True))

        def cpu(self):
            return self

        def tolist(self):
            return self.values.tolist()

    class Tokenizer(WordTokenizer):
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return Tokenizer()

        def __call__(self, texts, **_kwargs):
            return {"input_ids": Tensor([[len(t)] for t in texts])}

    class Model:
        calls = 0
        interrupt = True
        config = SimpleNamespace(id2label=dict(enumerate(study.CANONICAL_LABELS)))

        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return Model()

        def to(self, *_args):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids):
            Model.calls += 1
            if Model.interrupt and Model.calls == 2:
                raise KeyboardInterrupt
            return SimpleNamespace(logits=Tensor([[1, 2, 0, 0]] * len(input_ids.values)))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=Tokenizer, AutoModelForSequenceClassification=Model))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "scripts.train_transformer_verifier", SimpleNamespace(compose_text=lambda r, *_args: r["claim"]))
    (prep.model_dir / "model.safetensors").write_text("fake weights")
    config = tmp_path / "model_run.json"
    write_json({"claim_token_budget": 96, "evidence_token_budget_per_passage": 3, "max_length": 512}, config)
    args = argparse.Namespace(output=prep.output, dev_claims=paths["dev"], model_dir=prep.model_dir,
                              model_config=config, reference_predictions=None, device="cpu", batch_size=1)
    with pytest.raises(KeyboardInterrupt):
        study.frozen(args)
    partial = prep.output / "frozen/head/predictions.partial.jsonl"
    assert len(read_jsonl(partial)) == 1
    Model.interrupt = False
    study.frozen(args)
    assert len(read_jsonl(partial)) == 3  # The completed first row was not appended again.
    study.summarize(argparse.Namespace(output=prep.output, dev_claims=paths["dev"], bootstrap_trials=20))
    result = json.loads((prep.output / "limitation_results.json").read_text())
    assert result["results"]["frozen/oracle_answers"]["kind"] == "oracle diagnostic"
    assert result["results"]["frozen/head"]["claim_count"] == 3
    study.frozen(args)
    assert len(read_jsonl(partial)) == 3


def test_trainer_selection_only_defers_dev_and_refit_uses_best_epoch(tmp_path, monkeypatch):
    """Exercise real training control flow without downloading torch or a model."""
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    torch_stub = SimpleNamespace(utils=SimpleNamespace(data=SimpleNamespace(Dataset=object)), float32="float32",
        tensor=lambda values, **kw: Tensor(values), ones=lambda n, **kw: Tensor(np.ones(n)),
        softmax=lambda tensor, dim: Tensor(np.exp(tensor.values) / np.exp(tensor.values).sum(axis=dim, keepdims=True)),
        cuda=SimpleNamespace(is_available=lambda: False))

    class Tokenizer(WordTokenizer):
        init_kwargs = {"_commit_hash": "immutable-test-revision"}

        @staticmethod
        def from_pretrained(*args, **kwargs):
            return Tokenizer()

        def __call__(self, texts, **kwargs):
            return {"input_ids": [list(range(len(t.split()))) for t in texts], "attention_mask": [[1] * len(t.split()) for t in texts]}

        def save_pretrained(self, directory):
            write_json({"tokenizer": "stub"}, directory / "tokenizer_config.json")

    class Model:
        config = SimpleNamespace(_commit_hash="immutable-test-revision")

        @staticmethod
        def from_pretrained(*args, **kwargs):
            return Model()

    class FakeTrainer:
        epochs_trained = []
        prediction_ids = []

        def __init__(self, **kwargs):
            self.args = kwargs["args"]
            self.state = SimpleNamespace(best_model_checkpoint=None)

        def train(self, **kwargs):
            FakeTrainer.epochs_trained.append(self.args.num_train_epochs)
            checkpoint = study.Path(self.args.output_dir) / "checkpoint-100"
            write_json({"epoch": 1.25}, checkpoint / "trainer_state.json")
            self.state.best_model_checkpoint = str(checkpoint)

        def predict(self, dataset):
            FakeTrainer.prediction_ids.extend(r["claim_id"] for r in dataset.records)
            logits = np.zeros((len(dataset.records), 4))
            for i, row in enumerate(dataset.records):
                logits[i, study.CANONICAL_LABELS.index(row["label"])] = 3
            return SimpleNamespace(predictions=logits)

        def save_model(self, directory):
            write_json({"model": "stub"}, study.Path(directory) / "config.json")

    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForSequenceClassification=Model, AutoTokenizer=Tokenizer, Trainer=FakeTrainer,
        TrainingArguments=lambda **kwargs: SimpleNamespace(**kwargs), DataCollatorWithPadding=lambda **kwargs: None,
        set_seed=lambda *_args: None))
    trainer_path = study.PROJECT_ROOT / "scripts/train_transformer_verifier.py"
    spec = importlib.util.spec_from_file_location("limitation_test_training_driver", trainer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = tmp_path / "data"
    for split, count in (("train", 40), ("dev", 4)):
        rows = [{"claim_id": f"{split}:{i}", "claim": f"claim {i}", "label": study.CANONICAL_LABELS[i % 4]} for i in range(count)]
        write_jsonl(rows, data / "processed/averitec" / f"{split}.jsonl")
        write_jsonl([{**r, "retrieved": []} for r in rows], tmp_path / f"{split}_ranks.jsonl")
    argv = ["train", "--data-root", str(data), "--train-rankings", str(tmp_path / "train_ranks.jsonl"),
            "--eval-rankings", str(tmp_path / "dev_ranks.jsonl"), "--output-dir", str(tmp_path / "runs"),
            "--run-name", "study", "--learning-rates", "5e-5", "--epoch-grid", "3", "--class-weighting", "none",
            "--refit-epoch-policy", "best_checkpoint", "--resume"]
    monkeypatch.setattr(sys, "argv", argv + ["--selection-only"])
    module.main()
    assert FakeTrainer.epochs_trained == [3]
    assert all(cid.startswith("train:") for cid in FakeTrainer.prediction_ids)
    assert not (tmp_path / "runs/study/final_dev_predictions.jsonl").exists()
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert FakeTrainer.epochs_trained == [3, 1.25]
    assert (tmp_path / "runs/study/final_dev_predictions.jsonl").is_file()


def test_portable_notebook_code_and_source_snapshot_are_valid():
    from scripts.build_limitation_notebook import build

    notebook = build()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    bundled_cell = next(c for c in notebook["cells"] if "STUDY_SOURCES =" in "".join(c["source"]))
    node = ast.parse("".join(bundled_cell["source"])).body[0]
    payload = ast.literal_eval(node.value.args[0])
    assert hashlib.sha256(payload.encode()).hexdigest() == notebook["metadata"]["source_sha256"]
    sources = json.loads(payload)
    assert "scripts/run_limitation_diagnostics.py" in sources
    assert "src/evaluation/splits.py" in sources
    for name, source in sources.items():
        ast.parse(source, filename=name)
