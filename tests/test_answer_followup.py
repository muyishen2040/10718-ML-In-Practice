"""Check provenance, source-grounded packing, resumability and bounded work."""
import argparse
import json

import pytest

from src.data.averitec import CANONICAL_LABELS
from src.retrieval.answer_followup import (QUESTION_PROMPT, cached_questions, candidate_union, pack_answers,
    prepare_answers, prepare_questions, select_answer_evidence)
from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.runs import sha256_file


def fixture_inputs(tmp_path):
    data, parent, questions = tmp_path / "data", tmp_path / "parent", tmp_path / "questions"
    hashes = {}
    for split, count in (("train", 80), ("dev", 8)):
        rows = [{"claim_id": f"{split}:{i}", "claim": f"Event happened in city {i}",
                 "label": CANONICAL_LABELS[i % 4], "evidence": [{"text": "FORBIDDEN GOLD ANSWER"}],
                 "metadata": {"claim_date": "2020-01-01", "question": "FORBIDDEN GOLD QUESTION"}} for i in range(count)]
        path = data / "processed/averitec" / f"{split}.jsonl"
        write_jsonl(rows, path)
        hashes[split + "_claims"] = sha256_file(path)
        for mode in ("raw", "clean_nli"):
            offset = 0 if mode == "raw" else 2
            write_jsonl([{"claim_id": r["claim_id"], "claim": r["claim"], "retrieved": [
                {"passage_id": f"p{j}", "text": f"Event happened in London on Monday. Independent report {j}.",
                 "url": f"https://source{j}.test/article", "original_rank": j + 1,
                 "nli_probabilities": {"entailment": .1, "contradiction": .1, "neutral": .8}, "quality": [0.] * 8}
                for j in range(offset, offset + 3)]} for r in rows], parent / mode / f"{split}_rankings.jsonl")
        write_jsonl([{"claim_id": r["claim_id"], "claim": r["claim"], "questions": ["Where did it happen?", "When did it happen?"],
                     "question_origin": "new_claim_only_generator", "fallback": False} for r in rows], questions / f"{split}_questions.jsonl")
    write_json({"nli_model": "local_nli", "revision": "local", "max_length": 384, "claim_budget": 96,
                "nli_label_order": "contradiction,entailment,neutral", "limit": 0, "input_hashes": hashes}, parent / "preparation.json")
    write_json({"finalized": True}, parent / "preparation_report.json")
    write_json({"provenance": "test"}, questions / "questions.json")
    return argparse.Namespace(data_root=data, parent_prepared=parent, questions=questions, output=tmp_path / "new",
        model="local_qa", revision="local", device="cpu", batch_size=4, max_length=384, limit=0)


class FakeReader:
    calls = 0

    def __init__(self, *args):
        pass

    def __call__(self, questions, candidates):
        FakeReader.calls += 1
        assert questions == ["Where did it happen?", "When did it happen?"]
        assert len(candidates) <= 6
        assert "FORBIDDEN" not in str(candidates)
        return [[{"question_index": 0, "question": questions[0], "answer": "London", "margin": 3.},
                 {"question_index": 1, "question": questions[1], "answer": "Monday", "margin": 2.}] for _ in candidates]


def test_cached_preparation_resumes_torn_row_and_never_uses_gold(tmp_path):
    args = fixture_inputs(tmp_path)
    FakeReader.calls = 0
    prepare_answers(args, reader_factory=FakeReader)
    assert FakeReader.calls == 88
    prepare_answers(args, reader_factory=lambda *a: pytest.fail("Completed run reloaded QA model"))
    report = json.loads((args.output / "preparation_report.json").read_text())
    assert report["totals"]["train"]["candidate_count"] == 80 * 5
    assert report["config"]["distillation_incompatible_modes"] == ["qa_same3", "qa_selected3"]
    for split in ("train", "dev"):
        raw = read_jsonl(args.parent_prepared / "raw" / f"{split}_rankings.jsonl")
        packed = read_jsonl(args.output / "qa_same3" / f"{split}_rankings.jsonl")
        selected = read_jsonl(args.output / "qa_selected3" / f"{split}_rankings.jsonl")
        assert len(packed) == len(raw)
        assert "FORBIDDEN" not in str(packed)
        for a, b in zip(raw, packed):
            assert [p["passage_id"] for p in a["retrieved"]] == [p["passage_id"] for p in b["retrieved"]]
            assert all(len(p["extracted_answers"]) == 2 for p in b["retrieved"])
        assert all(len(r["retrieved"]) <= 3 for r in selected)
    assert (args.output / "train_fit_answer_audit.csv").is_file()
    # A torn final claim is repaired, then only that claim is recomputed.
    journal = args.output / "dev_answers.partial.jsonl"
    rows = journal.read_bytes().splitlines(keepends=True)
    journal.write_bytes(b"".join(rows[:-1]) + b'{"claim_id":')
    prepare_answers(args, reader_factory=FakeReader)
    assert FakeReader.calls == 89
    args.batch_size += 1
    with pytest.raises(ValueError, match="settings changed"):
        prepare_answers(args, reader_factory=FakeReader)


def test_pack_checks_verbatim_and_keeps_both_questions():
    item = {"passage_id": "p", "text": "The event happened in London on Monday."}
    answers = [{"question_index": 0, "question": "Where?", "answer": "London", "margin": 2.},
               {"question_index": 1, "question": "When?", "answer": "Monday", "margin": -1.}]
    packed = pack_answers(item, answers)
    assert "Where?" in packed["text"] and "When?" in packed["text"]
    assert "Candidate source span: London" in packed["text"]
    assert "No confident extractive answer" in packed["text"]
    assert packed["source_text"] == item["text"]
    answers[1]["answer"] = "invented"
    with pytest.raises(ValueError, match="not a substring"):
        pack_answers(item, answers)
    with pytest.raises(ValueError, match="different source text"):
        candidate_union([item], [{**item, "text": "different"}])


def test_selection_keeps_sourced_spans_and_filters_search_pages():
    candidates = [{"passage_id": str(i), "url": f"https://s{i}.test/article", "text": f"London on Monday. Report {i}."} for i in range(5)]
    candidates[0]["url"] = "https://www.google.com/search?q=london"
    answers = [[{"question_index": q, "question": "Where?" if q == 0 else "When?", "answer": "London", "margin": 3.} for q in (0, 1)] for _ in candidates]
    selected = select_answer_evidence("The event was in London", "2020-01-01", candidates, answers)
    assert len(selected) == 3
    assert "0" not in {p["passage_id"] for p in selected}


def test_question_cache_requires_known_provenance_and_exact_claims(tmp_path):
    claims = [{"claim_id": "x", "claim": "A test claim"}]
    path = tmp_path / "claims.jsonl"
    write_jsonl(claims, path)
    cache = tmp_path / "cache/questions.jsonl"
    write_jsonl([{**claims[0], "questions": ["Which event?", "Which date?"]}], cache)
    with pytest.raises(ValueError, match="provenance"):
        cached_questions(cache, path, claims)
    config = {"script": "run_question_guided_retrieval", "claims_sha256": sha256_file(path), "question_prompt": QUESTION_PROMPT}
    write_json(config, cache.parent / "question_guided_config.json")
    assert cached_questions(cache, path, claims)["x"]["questions"] == ["Which event?", "Which date?"]
    config["question_prompt"] = "Uses gold answers"
    write_json(config, cache.parent / "question_guided_config.json")
    with pytest.raises(ValueError, match="provenance"):
        cached_questions(cache, path, claims)


def test_question_generation_resumes_without_reloading_generator(tmp_path):
    prep = fixture_inputs(tmp_path)
    args = argparse.Namespace(data_root=prep.data_root, output=tmp_path / "generated", model="local", revision="local",
        device="cpu", max_new_tokens=80, train_question_cache=None, dev_question_cache=None)
    class Generator:
        def __init__(self, *args):
            pass
        def __call__(self, claim):
            assert "FORBIDDEN" not in claim
            return ["Which event happened?", "Where did the event happen?"], False
    prepare_questions(args, generator_factory=Generator)
    prepare_questions(args, generator_factory=lambda *a: pytest.fail("Reloaded completed generator"))
    assert len(read_jsonl(args.output / "train_questions.jsonl")) == 80


def test_qa_distillation_mismatch_fails_before_model_load(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from src.verification.pair_training import train_reader
    args = fixture_inputs(tmp_path)
    prepare_answers(args, reader_factory=FakeReader)
    train_args = argparse.Namespace(prepared=args.output, mode="qa_same3", distillation_weight=.1, device="cpu")
    with pytest.raises(ValueError, match="no matching NLI teacher"):
        train_reader(train_args)


def test_comparison_reads_prior_results_and_computes_planned_contrasts(tmp_path):
    from src.verification.evidence_experiments import compare_results
    args = fixture_inputs(tmp_path)
    prepare_answers(args, reader_factory=FakeReader)
    rows = read_jsonl(args.data_root / "processed/averitec/dev.jsonl")
    predictions = [{"claim_id": r["claim_id"], "true_label": r["label"], "predicted_label": r["label"],
                    "probabilities": {label: float(label == r["label"]) for label in CANONICAL_LABELS}} for r in rows]
    reference = tmp_path / "previous"
    write_jsonl(predictions, reference / "frozen/raw/predictions.jsonl")
    write_json({}, reference / "frozen/raw/metrics.json")
    for name in ("raw_legacy_d01", "raw_sample_d01", "raw_sample_d0", "qa_same3_d0", "qa_selected3_d0"):
        write_jsonl(predictions, args.output / "pair" / name / "dev/predictions.jsonl")
        write_json({}, args.output / "pair" / name / "dev/metrics.json")
    compare_results(argparse.Namespace(data_root=args.data_root, prepared=args.output, output=args.output,
        reference_root=reference, qrels=None))
    report = json.loads((args.output / "evidence_reader_results.json").read_text())
    assert len(report["planned_paired_contrasts"]) == 4
    assert len(report["verdicts"]) == 6
    assert report["reference_name"] == "previous/frozen/raw"
    assert report["predicted_label_counts"]["pair/qa_same3_d0"]["Not Enough Evidence"] == 2
