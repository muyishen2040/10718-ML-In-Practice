"""The long dense-training run must recover from a torn Drive checkpoint."""

import json
import sys

from scripts import run_dense_retrieval


def test_finalize_repairs_duplicate_and_torn_checkpoint(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    claims_path = data_root / "processed" / "averitec" / "train.jsonl"
    claims_path.parent.mkdir(parents=True)
    claim = {"claim_id": "averitec:train:00001", "claim": "A test claim", "label": "Supported"}
    claims_path.write_text(json.dumps(claim) + "\n", encoding="utf-8")

    output_root = tmp_path / "outputs"
    run_dir = output_root / "dense_test"
    run_dir.mkdir(parents=True)
    ranking = {"claim_id": claim["claim_id"], "claim": claim["claim"], "retrieved": []}
    checkpoint = run_dir / "train_rankings.partial.jsonl"
    checkpoint.write_text(
        json.dumps(ranking) + "\n" + json.dumps(ranking) + "\n" + '{"claim_id": ',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dense_retrieval.py", "--split", "train", "--data-root", str(data_root),
            "--output-dir", str(output_root), "--run-name", "dense_test",
            "--resume", "--repair-checkpoint", "--finalize",
        ],
    )
    run_dense_retrieval.main()

    final = run_dir / "train_rankings.jsonl"
    assert [json.loads(line) for line in final.read_text(encoding="utf-8").splitlines()] == [ranking]
    assert not checkpoint.exists()
    assert len(list(run_dir.glob("train_rankings.partial.jsonl.before_repair.*.bak"))) == 1
    assert json.loads((run_dir / "train_metrics.json").read_text(encoding="utf-8"))["finalized"]
