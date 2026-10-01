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
