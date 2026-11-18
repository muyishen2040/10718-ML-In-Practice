import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.verification.hosted_verifier import HostedVerifier
from src.verification.llm_zero_shot import LLMVerdict


PASSAGES = [
    {"passage_id": "p1", "text": "First source passage.", "url": "https://example.org/one"},
    {"passage_id": "p2", "text": "Second source passage.", "url": "https://example.org/two"},
    {"passage_id": "p3", "text": "Third source passage.", "url": "https://example.org/three"},
]


@pytest.fixture
def hosted():
    parse = Mock()
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
    return HostedVerifier(model="mock-model", client=client), parse


def completion(*, parsed=None, refusal=None):
    message = SimpleNamespace(parsed=parsed, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], model_dump=Mock(return_value={}))


@pytest.mark.parametrize(
    ("verdict", "count", "citations"),
    [
        ("Supported", 1, ["p1"]),
        ("Refuted", 2, ["p2"]),
        ("Not Enough Evidence", 2, ["p1"]),
        ("Conflicting Evidence", 3, ["p3", "p1"]),
    ],
)
def test_returns_backend_fields_with_original_citations(hosted, verdict, count, citations):
    verifier, parse = hosted
    parse.return_value = completion(parsed=LLMVerdict(
        label=verdict,
        selected_passage_ids=citations,
        reported_confidence=0.65,
        brief_rationale="Assessment based on the supplied sources.",
    ))
    passages = [{**passage, "label": "private-annotation"} for passage in PASSAGES[:count]]
    sources = {passage["passage_id"]: passage for passage in PASSAGES[:count]}

    result = verifier.verify("Example claim", passages)

    assert result == {
        "status": "completed",
        "claim": "Example claim",
        "verdict": verdict,
        "explanation": "Assessment based on the supplied sources.",
        "uncertainty_note": None,
        "evidence": [
            {**sources[passage_id], "rank": rank}
            for rank, passage_id in enumerate(citations, start=1)
        ],
    }
    parse.assert_called_once()
    request = parse.call_args.kwargs
    assert request["model"] == "mock-model"
    assert request["response_format"] is LLMVerdict
    assert request["timeout"] == 30.0
    assert request["max_completion_tokens"] == 512
    assert "temperature" not in request
    assert "tools" not in request
    user_prompt = request["messages"][1]["content"]
    assert "Example claim" in user_prompt
    assert "private-annotation" not in user_prompt
    assert "https://example.org" not in user_prompt
    assert all(passage["text"] in user_prompt for passage in passages)


def test_returned_citation_text_matches_the_excerpt_sent_to_provider(hosted):
    verifier, parse = hosted
    verifier.max_chars_per_passage = 8
    parse.return_value = completion(parsed=LLMVerdict(
        label="Supported",
        selected_passage_ids=["p1"],
        reported_confidence=0.65,
        brief_rationale="The cited excerpt supports the claim.",
    ))
    result = verifier.verify("Example claim", PASSAGES[:1])

    assert result["evidence"][0]["text"] == "First so"
    assert parse.call_args.kwargs["messages"][1]["content"].endswith("First so")


def test_empty_evidence_returns_completed_nei_without_a_call(hosted):
    verifier, parse = hosted
    result = verifier.verify("Example claim", [])
    assert result["status"] == "completed"
    assert result["verdict"] == "Not Enough Evidence"
    assert result["evidence"] == []
    parse.assert_not_called()


@pytest.mark.parametrize(
    ("passages", "message"),
    [
        (PASSAGES + [{"passage_id": "p4", "text": "Fourth passage."}], "at most three"),
        ([PASSAGES[0], PASSAGES[0]], "unique"),
        ([{"text": "Unidentified source."}], "passage ID"),
        ([{"passage_id": "p1", "text": " "}], "source text"),
    ],
)
def test_invalid_sources_are_rejected_before_calling_provider(hosted, passages, message):
    verifier, parse = hosted
    with pytest.raises(ValueError, match=message):
        verifier.verify("Example claim", passages)
    parse.assert_not_called()


@pytest.mark.parametrize(
    ("ids", "explanation"),
    [(["invented"], "Explanation."), (["p1", "p1"], "Explanation."), (["p1"], " ")],
)
def test_invalid_citations_and_blank_explanations_are_errors(hosted, ids, explanation):
    verifier, parse = hosted
    parse.return_value = completion(parsed=LLMVerdict(
        label="Supported",
        selected_passage_ids=ids,
        reported_confidence=0.65,
        brief_rationale=explanation,
    ))
    assert_failed(verifier.verify("Example claim", PASSAGES))


@pytest.mark.parametrize("response", ["timeout", "malformed", "refusal", "unparsed", "no choices"])
def test_provider_failures_have_no_verdict(hosted, response):
    verifier, parse = hosted
    if response == "timeout":
        parse.side_effect = TimeoutError("Private provider details")
    elif response == "malformed":
        parse.side_effect = lambda **request: request["response_format"].model_validate_json(
            json.dumps({"label": "Unknown"})
        )
    elif response == "no choices":
        parse.return_value = SimpleNamespace(choices=[])
    else:
        parse.return_value = completion(refusal="Private provider details" if response == "refusal" else None)
    assert_failed(verifier.verify("Example claim", PASSAGES))


def assert_failed(result):
    assert result == {
        "status": "error",
        "claim": "Example claim",
        "verdict": None,
        "explanation": "The hosted verifier could not complete the check.",
        "uncertainty_note": None,
        "evidence": [],
    }
