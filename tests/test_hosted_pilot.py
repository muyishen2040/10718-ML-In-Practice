import json
from unittest.mock import Mock

from scripts import run_hosted_pilot as pilot
from src.utils.io import read_jsonl
from src.verification.llm_zero_shot import LLMVerdict


def test_packaged_pilot_preserves_budget_and_reuses_completed_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "PILOT", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.org/untrusted-default")
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(pilot.runner, "openai_client", factory)

    def simulated_provider(**request):
        assert request["model"] == "gpt-5.4-mini-2026-03-17"
        assert request["service_tier"] == "default"
        assert request["reasoning_effort"] == "none"
        assert request["max_completion_tokens"] == 512
        claim_id = "synthetic:1" if "library closes" in request["user_prompt"] else "synthetic:2"
        return LLMVerdict(
            label="Supported" if claim_id == "synthetic:1" else "Refuted",
            selected_passage_ids=[f"{claim_id}:p1"],
            reported_confidence=0.8,
            brief_rationale="Assessment based on the supplied passage.",
        ), {"usage": {"prompt_tokens": 50, "completion_tokens": 30}}

    provider = Mock(side_effect=simulated_provider)
    monkeypatch.setattr(pilot.runner, "call_openai_structured", provider)
    pilot.main([])
    provider.assert_not_called()
    factory.assert_not_called()

    pilot.main(["--live"])
    directory = tmp_path / "runs/synthetic-smoke-v1"
    config = json.loads((directory / "classify_top3_config.json").read_text())
    assert config["completed_claim_count"] == 3
    assert config["max_api_attempts"] == 10
    assert config["reserved_api_attempt_count"] == 2
    assert config["max_retries"] == 0
    assert config["provider_endpoint_sha256"] == pilot.runner.json_sha256(pilot.API_BASE)
    predictions = read_jsonl(directory / "classify_top3_predictions.jsonl")
    assert all(row["predicted_label"] == row["true_label"] for row in predictions)

    factory.reset_mock()
    provider.reset_mock()
    pilot.main(["--live"])
    provider.assert_not_called()
    factory.assert_not_called()
