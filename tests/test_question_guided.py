"""No-model tests for the question-guided retrieval experiment."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from scripts import run_question_guided_retrieval as runner
from scripts.run_question_guided_retrieval import prepare_output_row
from src.retrieval.question_guided import (
    extract_sentences,
    parse_questions,
    reciprocal_rank_scores,
    select_diverse,
    shortlist_candidates,
)
from src.utils.io import read_jsonl, write_jsonl


def test_questions_are_parsed_or_fall_back_to_claim() -> None:
    response = 'text {"questions": ["When did this happen?", "Who confirmed it?", "extra"]}'
    assert parse_questions(response, "A claim") == ["When did this happen?", "Who confirmed it?"]
    assert "A claim" in parse_questions("not JSON", "A claim")[0]


def test_shortlist_preserves_bm25_leaders_and_question_match() -> None:
    row = {
        "claim_id": "c1", "claim": "Did the museum open in Paris?",
        "retrieved": [
            {"passage_id": "p0", "text": "Unrelated text"},
            {"passage_id": "p1", "text": "Unrelated text too"},
            {"passage_id": "p2", "text": "The museum opened in Paris."},
        ],
    }
    result = shortlist_candidates(row, ["When did the museum open in Paris?"], limit=2, bm25_keep=1)
    assert [item["passage_id"] for item in result] == ["p0", "p2"]


def test_rank_fusion_diversity_and_exact_source_excerpts() -> None:
    fused = reciprocal_rank_scores([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    assert len(fused) == 3
    assert all(score > 0 for score in fused)
    with pytest.raises(ValueError):
        reciprocal_rank_scores([[1.0], [float("nan")]])
    candidates = [
        {"passage_id": "a", "url": "https://a.test/", "text": "A"},
        {"passage_id": "b", "url": "https://a.test", "text": "B"},
        {"passage_id": "c", "url": "https://b.test", "text": "C"},
    ]
    selected = select_diverse(candidates, [3.0, 2.0, 1.0], k=2)
    assert [item["passage_id"] for item in selected] == ["a", "c"]
    source = "A stadium was built in 1984. The museum opened in Paris in 2010. Visitors came from France."
    excerpt = extract_sentences(source, "The museum opened in Paris", ["When did the museum open?"], max_chars=70)
    assert "The museum opened in Paris in 2010." in excerpt
    assert excerpt in source


def test_full_and_excerpt_rankings_keep_identical_ids() -> None:
    row = {
        "claim_id": "c1", "claim": "claim", "retrieved": [
            {"passage_id": "p1", "url": "https://a.test", "score": 1.0,
             "full_text": "long original text", "excerpt_text": "original text", "text": "long original text"}
        ],
    }
    full = prepare_output_row(row, excerpt=False)
    excerpt = prepare_output_row(row, excerpt=True)
    assert full["retrieved"][0]["passage_id"] == excerpt["retrieved"][0]["passage_id"]
    assert full["retrieved"][0]["text"] == "long original text"
    assert excerpt["retrieved"][0]["text"] == "original text"
    assert "full_text" not in json.dumps(full)


def test_script_finalizes_and_resumes_without_reloading_models(tmp_path, monkeypatch) -> None:
    claims = tmp_path / "dev.jsonl"
    rankings = tmp_path / "bm25.jsonl"
    qrels = tmp_path / "qrels.jsonl"
    claim_id = "averitec:dev:00001"
    write_jsonl([{"claim_id": claim_id, "claim": "Museum opened in 2010", "label": "Supported"}], claims)
    write_jsonl([{
        "claim_id": claim_id, "claim": "Museum opened in 2010",
        "retrieved": [
            {"passage_id": "p1", "url": "https://a.test", "text": "A stadium was built. " * 40, "score": 4},
            {"passage_id": "p2", "url": "https://b.test", "text": "The museum opened in 2010. " * 40, "score": 3},
            {"passage_id": "p3", "url": "https://c.test", "text": "This is another source. " * 40, "score": 2},
        ],
    }], rankings)
    write_jsonl([{"claim_id": claim_id, "relevant_passage_ids": ["p2"]}], qrels)
    fake_model = SimpleNamespace(config=SimpleNamespace(_commit_hash="test"))
    fake_tokenizer = SimpleNamespace(init_kwargs={"_commit_hash": "test"})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda *_: None))
    monkeypatch.setattr(runner, "load_qwen", lambda *_: (fake_model, fake_tokenizer, None))
    monkeypatch.setattr(runner, "generate_questions", lambda *_, **__: (["When did the museum open?"], '{}'))
    monkeypatch.setattr(runner, "load_cross_encoder", lambda *_: (fake_model, fake_tokenizer, None))
    monkeypatch.setattr(runner, "score_query", lambda *_, **__: [0.0, 4.0, 1.0])
    arguments = [
        "run_question_guided_retrieval.py", "--input-rankings", str(rankings),
        "--claims", str(claims), "--split", "dev", "--qrels", str(qrels),
        "--output-dir", str(tmp_path), "--run-name", "trial", "--resume",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    runner.main()
    output = tmp_path / "trial"
    report = json.loads((output / "dev_metrics.json").read_text(encoding="utf-8"))
    assert report["claim_count"] == 1
    assert report["evidence_hit_at_3"] == 1.0
    assert report["shortlist_hit_at_32"] == 1.0
    assert report["bm25_candidate_hit_at_100"] == 1.0
    full = read_jsonl(output / "dev_full_rankings.jsonl")
    excerpts = read_jsonl(output / "dev_excerpt_rankings.jsonl")
    assert [x["passage_id"] for x in full[0]["retrieved"]] == [x["passage_id"] for x in excerpts[0]["retrieved"]]
    monkeypatch.setattr(runner, "load_qwen", lambda *_: (_ for _ in ()).throw(AssertionError("model reloaded")))
    runner.main()
