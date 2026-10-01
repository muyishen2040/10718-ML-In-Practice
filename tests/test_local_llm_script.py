import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_local_llm_baseline.py"
SPEC = importlib.util.spec_from_file_location("run_local_llm_baseline_script", SCRIPT_PATH)
assert SPEC and SPEC.loader
LOCAL_LLM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOCAL_LLM)


def test_parse_json_verdict_accepts_fenced_object() -> None:
    response = """```json
{"label":"Refuted","selected_passage_ids":["p2"],"reported_confidence":0.8,"brief_rationale":"The passage contradicts the claim."}
```"""

    verdict = LOCAL_LLM.parse_json_verdict(response)

    assert verdict.label == "Refuted"
    assert verdict.selected_passage_ids == ["p2"]


def test_parse_json_verdict_normalizes_common_local_aliases() -> None:
    response = """{
      "verdict": "Supported",
      "evidence": "p1",
      "confidence": 0.7,
      "reason": "The passage directly establishes the claim."
    }"""

    verdict = LOCAL_LLM.parse_json_verdict(response)

    assert verdict.label == "Supported"
    assert verdict.selected_passage_ids == ["p1"]
    assert verdict.reported_confidence == 0.7


def test_success_cache_filters_to_exact_model_and_variant(tmp_path) -> None:
    raw_path = tmp_path / "classify_top3_raw.jsonl"
    LOCAL_LLM.append_jsonl_row(
        raw_path,
        {"claim_id": "c1", "status": "success", "model": "Qwen/Qwen3-4B", "variant": "classify_top3"},
    )
    LOCAL_LLM.append_jsonl_row(
        raw_path,
        {"claim_id": "c2", "status": "success", "model": "other", "variant": "classify_top3"},
    )

    cached = LOCAL_LLM.existing_successes(raw_path, model="Qwen/Qwen3-4B", variant="classify_top3")

    assert set(cached) == {"c1"}


def test_build_cases_supports_claim_only_and_capped_gold_evidence() -> None:
    claims = [
        {
            "claim_id": "c1",
            "claim": "one",
            "evidence": [
                {"passage_id": "c1-g1", "text": "gold one"},
                {"passage_id": "c1-g2", "text": "gold two"},
            ],
        },
        {"claim_id": "c2", "claim": "two", "evidence": []},
    ]

    claim_only_cases, claim_only_excluded = LOCAL_LLM.build_cases(
        claims=claims, rankings_path=None, evidence_mode="claim_only", max_evidence=3
    )
    gold_cases, gold_excluded = LOCAL_LLM.build_cases(
        claims=claims, rankings_path=None, evidence_mode="gold_top3", max_evidence=1
    )

    assert [case["passages"] for case in claim_only_cases] == [[], []]
    assert claim_only_excluded == []
    assert [item["passage_id"] for item in gold_cases[0]["passages"]] == ["c1-g1"]
    assert gold_excluded == ["c2"]


def test_invalid_top_three_citation_does_not_discard_verdict() -> None:
    verdict = LOCAL_LLM.LLMVerdict(
        label="Refuted",
        selected_passage_ids=["made-up-id"],
        reported_confidence=0.6,
        brief_rationale="The supplied passage contradicts the claim.",
    )

    error = LOCAL_LLM.citation_validation_error(verdict, [{"passage_id": "p1", "text": "text"}], "classify_top3")

    assert error == "LLM selected a passage ID outside the supplied candidates"
