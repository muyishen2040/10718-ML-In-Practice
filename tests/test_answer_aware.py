"""Small synthetic checks for the answer-aware notebook components."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from scripts import run_answer_reader, train_answer_selector
from src.retrieval.answer_aware import (
    best_extractable_span, choose_candidates, qa_text, rank_with_source_diversity,
)
from src.utils.io import read_jsonl, write_jsonl


def test_candidate_choice_and_verbatim_answer_span() -> None:
    bm25 = {"retrieved": [
        {"passage_id": "p1", "text": "other source", "url": "https://a.test"},
        {"passage_id": "p2", "text": "The bridge opened in 2010.", "url": "https://b.test"},
        {"passage_id": "p3", "text": "another source", "url": "https://c.test"},
    ]}
    qg = {"retrieved": [{"passage_id": "p2"}], "shortlist_passage_ids": ["p3", "p2"]}
    selected = choose_candidates(bm25, qg, limit=2, bm25_keep=1)
    assert [item["passage_id"] for item in selected] == ["p1", "p2"]
    assert selected[1]["question_guided_top3"] == 1
    passage = "The bridge opened in 2010."
    answer, margin = best_extractable_span(
        passage, [(0, 0), (0, 3), (4, 10), (11, 17), (18, 20), (21, 25)],
        [None, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 5], [0, 0, 0, 0, 0, 6],
        cls_index=0, max_answer_tokens=4,
    )
    assert answer == "2010"
    assert margin == 11
    assert "2010" in qa_text("When?", answer, passage, margin)
    assert "No confident" in qa_text("When?", answer, passage, -1)


def test_source_diversity_keeps_ranked_remainder() -> None:
    items = [
        {"passage_id": "a", "url": "https://a.test/"},
        {"passage_id": "b", "url": "https://a.test"},
        {"passage_id": "c", "url": "https://b.test"},
    ]
    ranked = rank_with_source_diversity(items, [3, 2, 1], k=2)
    assert [item["passage_id"] for item in ranked] == ["a", "c", "b"]


def test_reader_checkpoint_is_reused_without_model_reload(tmp_path, monkeypatch) -> None:
    bm25 = tmp_path / "bm25.jsonl"
    questions = tmp_path / "questions.jsonl"
    write_jsonl([{"claim_id": "c1", "claim": "Bridge opened in 2010", "retrieved": [
        {"passage_id": "p1", "url": "https://a.test", "text": "The bridge opened in 2010."},
    ]}], bm25)
    write_jsonl([{"claim_id": "c1", "claim": "Bridge opened in 2010", "questions": ["When did it open?"],
                  "shortlist_passage_ids": ["p1"], "retrieved": [{"passage_id": "p1"}]}], questions)
    monkeypatch.setattr(run_answer_reader, "load_reader", lambda *_: (object(), object(), object(), "test"))
    monkeypatch.setattr(run_answer_reader, "read_answers", lambda qs, cs, *_, **__: [
        {**item, "best_question": qs[0], "candidate_answer": "2010", "max_answer_margin": 3.0,
         "mean_answer_margin": 3.0, "answer_word_count": 1,
         "bm25_reciprocal_rank": 1.0, "qa_text": "Question: When? Answer: 2010"} for item in cs
    ])
    monkeypatch.setattr(sys, "argv", ["run_answer_reader.py", "--bm25-rankings", str(bm25),
                                     "--question-checkpoint", str(questions), "--split", "train",
                                     "--output-dir", str(tmp_path), "--run-name", "reader", "--resume"])
    run_answer_reader.main()
    assert read_jsonl(tmp_path / "reader/train_answer_features.jsonl")[0]["retrieved"][0]["candidate_answer"] == "2010"
    monkeypatch.setattr(run_answer_reader, "load_reader", lambda *_: (_ for _ in ()).throw(AssertionError("model reloaded")))
    run_answer_reader.main()


def test_selector_uses_train_only_and_preserves_passage_ids(tmp_path, monkeypatch) -> None:
    train, dev, aligned, qrels = (tmp_path / name for name in ("train.jsonl", "dev.jsonl", "aligned.jsonl", "qrels.jsonl"))

    def candidate(claim_id: str, letter: str, positive: bool) -> dict:
        return {"passage_id": claim_id + letter, "text": "raw " + letter,
                "qa_text": "quoted " + letter, "url": f"https://{claim_id}{letter}.test",
                "max_answer_margin": 5.0 if positive else -5.0,
                "mean_answer_margin": 4.0 if positive else -4.0,
                "bm25_reciprocal_rank": 1.0 if positive else 0.5,
                "question_guided_top3": int(positive), "answer_word_count": int(positive)}

    train_rows = [{"claim_id": f"t{i:02d}", "claim": f"claim {i}", "retrieved": [
        candidate(f"t{i:02d}", "a", True), candidate(f"t{i:02d}", "b", False)]} for i in range(40)]
    dev_rows = [{"claim_id": "d1", "claim": "dev claim", "retrieved": [
        candidate("d1", "a", True), candidate("d1", "b", False)]}]
    write_jsonl(train_rows, train)
    write_jsonl(dev_rows, dev)
    write_jsonl([{"claim_id": row["claim_id"], "positive_passage_id": row["claim_id"] + "a"} for row in train_rows], aligned)
    write_jsonl([{"claim_id": "d1", "relevant_passage_ids": ["d1a"]}], qrels)
    monkeypatch.setattr(sys, "argv", ["train_answer_selector.py", "--train-features", str(train),
                                     "--dev-features", str(dev), "--train-alignment", str(aligned),
                                     "--dev-qrels", str(qrels), "--output-dir", str(tmp_path),
                                     "--run-name", "selector", "--min-positive-claims", "30"])
    train_answer_selector.main()
    output = tmp_path / "selector"
    report = json.loads((output / "selector_report.json").read_text(encoding="utf-8"))
    assert report["positive_in_scored_candidates_count"] == 40
    assert report["dev_url_proxy_metrics"]["evidence_hit_at_3"] == pytest.approx(1.0)
    raw_train = read_jsonl(output / "train_raw_rankings.jsonl")
    qa_train = read_jsonl(output / "train_qa_rankings.jsonl")
    assert len(raw_train) == 40
    assert raw_train[0]["retrieved"][0]["passage_id"] == qa_train[0]["retrieved"][0]["passage_id"]
    assert qa_train[0]["retrieved"][0]["text"].startswith("quoted")
