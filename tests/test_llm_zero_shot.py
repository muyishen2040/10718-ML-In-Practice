import pytest

from src.verification.llm_zero_shot import LLMVerdict, build_user_prompt, validate_llm_verdict


PASSAGES = [
    {"passage_id": "p1", "text": "Evidence one."},
    {"passage_id": "p2", "text": "Evidence two."},
    {"passage_id": "p3", "text": "Evidence three."},
]


def test_rerank_prompt_and_selection_enforce_candidate_ids() -> None:
    prompt = build_user_prompt("A claim", PASSAGES, "rerank_top20_and_classify")
    result = LLMVerdict(
        label="Supported",
        selected_passage_ids=["p1", "p2", "p3"],
        reported_confidence=0.5,
        brief_rationale="The passages support the claim.",
    )

    assert "select exactly 3 passage IDs" in prompt
    validate_llm_verdict(result, PASSAGES, "rerank_top20_and_classify")


def test_rerank_rejects_unknown_or_wrong_count_selection() -> None:
    result = LLMVerdict(
        label="Supported",
        selected_passage_ids=["p1"],
        reported_confidence=0.5,
        brief_rationale="Short rationale.",
    )

    with pytest.raises(ValueError, match="exactly three"):
        validate_llm_verdict(result, PASSAGES, "rerank_top20_and_classify")
