import fcntl
import json
import re
import warnings
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import run_llm_baseline as runner
from src.utils.io import read_jsonl, write_jsonl
from src.verification.llm_zero_shot import LLMVerdict, call_openai_structured


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    claims = tmp_path / "claims.jsonl"
    rankings = tmp_path / "rankings.jsonl"
    write_jsonl(
        [{"claim_id": name, "claim": f"Claim {name}", "label": "Supported"} for name in ("a", "b", "c", "d")],
        claims,
    )
    write_jsonl(
        [
            {"claim_id": "c", "retrieved": []},
            {"claim_id": "b", "retrieved": [{"passage_id": "b1", "text": "Evidence b."}]},
            {"claim_id": "a", "retrieved": [
                {"passage_id": "a1", "text": "Evidence a1."},
                {"passage_id": "a2", "text": "Evidence a2."},
            ]},
            {"claim_id": "outside-scope", "retrieved": []},
        ],
        rankings,
    )
    args = [
        "--claims", str(claims), "--rankings", str(rankings),
        "--output-dir", str(tmp_path / "runs"), "--run-name", "mock", "--model", "sdk-mock",
    ]
    factory = Mock(return_value=Mock())
    provider = Mock(side_effect=response)
    monkeypatch.setattr(runner, "openai_client", factory)
    monkeypatch.setattr(runner, "call_openai_structured", provider)
    monkeypatch.setattr(runner, "sleep", Mock())
    return SimpleNamespace(
        args=args, claims=claims, rankings=rankings, directory=tmp_path / "runs" / "mock",
        provider=provider, factory=factory,
    )


def response(**kwargs):
    ids = re.findall(r"\[PASSAGE \d+ \| ID=([^\]]+)\]", kwargs["user_prompt"])
    assert ids
    assert "true_label" not in kwargs["user_prompt"] and "label" not in kwargs
    return LLMVerdict(
        label="Supported", selected_passage_ids=ids, reported_confidence=0.8, brief_rationale="Cited evidence."
    ), {"request_id": f"req-{ids[0]}", "usage": {"total_tokens": 10}}


def test_interrupted_run_repairs_torn_row_and_never_repeats_completed_calls(benchmark):
    run = benchmark

    def interrupt_after_first_call(**kwargs):
        if "Claim a" in kwargs["user_prompt"]:
            return response(**kwargs)
        raise KeyboardInterrupt

    run.provider.side_effect = interrupt_after_first_call
    raw = run.directory / "classify_top3_raw.jsonl"
    with pytest.raises(KeyboardInterrupt):
        runner.main(run.args)
    assert [row["claim_id"] for row in read_jsonl(raw)] == ["a"]
    with raw.open("ab") as handle:
        handle.write(b'{"claim_id": "b",')

    run.provider.reset_mock(side_effect=True)
    run.provider.side_effect = response
    runner.main(run.args)
    run.provider.assert_called_once()
    assert "Claim b" in run.provider.call_args.kwargs["user_prompt"]
    assert [row["claim_id"] for row in read_jsonl(raw)] == ["a", "b", "c"]
    predictions = read_jsonl(run.directory / "classify_top3_predictions.jsonl")
    assert [row["claim_id"] for row in predictions] == ["a", "b", "c"]
    assert predictions[2]["predicted_label"] == "Not Enough Evidence"
    assert predictions[2]["reported_confidence"] is None
    metrics = json.loads((run.directory / "classify_top3_metrics.json").read_text())
    assert metrics["evaluation_status"] == "partial"
    assert metrics["completion_coverage"] == 0.75
    assert metrics["unresolved_claim_ids"] == ["d"]
    assert "macro_f1" not in metrics and metrics["completed_only_metrics"]["claim_count"] == 3
    assert read_jsonl(run.directory / "classify_top3_errors.jsonl")[0]["error"] == "missing_ranking"
    assert read_jsonl(raw)[0]["response"]["request_id"] == "req-a1"

    run.factory.reset_mock()
    run.provider.reset_mock()
    runner.main(run.args)
    run.factory.assert_not_called()
    run.provider.assert_not_called()


@pytest.mark.parametrize("changed", ["passage", "claim", "prompt", "template", "model", "temperature", "endpoint"])
def test_changed_run_rejected_before_provider_call(benchmark, monkeypatch, changed):
    run = benchmark
    runner.main(run.args)
    run.provider.reset_mock()
    raw = run.directory / "classify_top3_raw.jsonl"
    before = raw.read_bytes()
    args = run.args.copy()
    if changed == "passage":
        rows = read_jsonl(run.rankings)
        rows[2]["retrieved"].reverse()
        write_jsonl(rows, run.rankings)
    elif changed == "claim":
        rows = read_jsonl(run.claims)
        rows[0]["claim"] = "Changed claim a"
        write_jsonl(rows, run.claims)
    elif changed == "prompt":
        monkeypatch.setattr(runner, "HOSTED_SYSTEM_PROMPT", "Changed system prompt.")
    elif changed == "template":
        original = runner.build_user_prompt
        monkeypatch.setattr(runner, "build_user_prompt", lambda *a: original(*a) + "\nNew instruction.")
    elif changed == "model":
        args[-1] = "different-model"
    elif changed == "endpoint":
        monkeypatch.setenv("OPENAI_BASE_URL", "https://example.org/v1")
    else:
        args += ["--temperature", "1"]

    with pytest.raises(SystemExit) as exit_info:
        runner.main(args)
    assert exit_info.value.code == 2
    run.provider.assert_not_called()
    assert raw.read_bytes() == before
    runner.main(args + ["--run-name", "changed"])
    assert run.provider.call_count == 2


def test_retries_transient_error_and_resumes_malformed_claim(benchmark):
    run = benchmark
    base = response(model="sdk-mock", user_prompt="[PASSAGE 1 | ID=a1]\nEvidence a1.")
    run.provider.side_effect = [
        TimeoutError("transient"),
        base,
        (base[0].model_copy(update={"selected_passage_ids": ["invented"]}), base[1]),
    ]
    runner.main(run.args)
    assert run.provider.call_count == 3
    runner.sleep.assert_called_once()
    failed = run.directory / "classify_top3_failed_attempts.jsonl"
    assert [(row["claim_id"], row["error"]) for row in read_jsonl(failed)] == [
        ("a", "TimeoutError"), ("b", "ValueError")
    ]
    assert [row["claim_id"] for row in read_jsonl(run.directory / "classify_top3_predictions.jsonl")] == ["a", "c"]

    run.provider.reset_mock(side_effect=True)
    run.provider.side_effect = response
    runner.main(run.args)
    run.provider.assert_called_once()
    assert "Claim b" in run.provider.call_args.kwargs["user_prompt"]
    summary = json.loads((run.directory / "classify_top3_config.json").read_text())
    assert summary["failed_attempt_count"] == 2
    assert summary["usage"]["total_tokens"] == 30


def test_provider_retry_after_pauses_further_calls(benchmark):
    class Throttled(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "60"})

    run = benchmark
    run.provider.side_effect = Throttled
    runner.main(run.args)
    run.provider.assert_called_once()
    runner.sleep.assert_not_called()
    metrics = json.loads((run.directory / "classify_top3_metrics.json").read_text())
    assert metrics["unattempted_claim_ids"] == ["b"]
    assert metrics["completed_claim_count"] == 1


@pytest.mark.parametrize("code", ["credit_balance_exhausted", "project_spend_limit_exceeded"])
def test_billing_failure_pauses_without_spending_retry_allowance(benchmark, code):
    class BillingFailure(Exception):
        status_code = 429

    error = BillingFailure()
    error.code = code
    run = benchmark
    run.provider.side_effect = error

    runner.main(run.args)

    run.provider.assert_called_once()
    runner.sleep.assert_not_called()
    metrics = json.loads((run.directory / "classify_top3_metrics.json").read_text())
    assert metrics["completed_claim_count"] == 1
    assert metrics["unresolved_claim_ids"] == ["a", "b", "d"]


def test_attempt_cap_includes_interrupted_requests_across_restarts(benchmark):
    run = benchmark
    args = run.args + ["--max-api-attempts", "2"]

    def interrupt_after_first_call(**kwargs):
        if "Claim a" in kwargs["user_prompt"]:
            return response(**kwargs)
        raise KeyboardInterrupt

    run.provider.side_effect = interrupt_after_first_call
    with pytest.raises(KeyboardInterrupt):
        runner.main(args)
    reservations = read_jsonl(run.directory / "classify_top3_api_attempts.jsonl")
    assert [row["claim_id"] for row in reservations] == ["a", "b"]
    run.provider.reset_mock(side_effect=True)
    run.factory.reset_mock()
    runner.main(args)

    run.provider.assert_not_called()
    run.factory.assert_not_called()
    summary = json.loads((run.directory / "classify_top3_config.json").read_text())
    assert summary["reserved_api_attempt_count"] == 2
    assert summary["api_attempt_limit_reached"]
    assert summary["unresolved_claim_ids"] == ["b", "d"]

    with pytest.raises(SystemExit):
        runner.main(args + ["--overwrite"])
    assert read_jsonl(run.directory / "classify_top3_api_attempts.jsonl") == reservations


def test_legacy_cache_requires_explicit_reset(benchmark):
    run = benchmark
    raw = run.directory / "classify_top3_raw.jsonl"
    raw.parent.mkdir(parents=True)
    original = {"claim_id": "a", "model": "sdk-mock", "status": "success"}
    write_jsonl([original], raw)
    with pytest.raises(SystemExit) as exit_info:
        runner.main(run.args)
    assert exit_info.value.code == 2
    assert read_jsonl(raw) == [original]
    run.provider.assert_not_called()
    runner.main(run.args + ["--overwrite"])
    assert run.provider.call_count == 2
    assert read_jsonl(raw)[0]["request_sha256"]


def test_concurrent_writer_is_rejected(benchmark):
    run = benchmark
    run.directory.mkdir(parents=True)
    with (run.directory / ".classify_top3.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit) as exit_info:
            runner.main(run.args)
    assert exit_info.value.code == 2
    run.provider.assert_not_called()


def test_rerank_shorter_pool_and_gold_labels_use_shared_scorer(benchmark):
    run = benchmark
    runner.main(run.args + ["--variant", "rerank_top20_and_classify", "--max-claims", "3"])
    metrics = json.loads((run.directory / "rerank_top20_and_classify_metrics.json").read_text())
    assert metrics["evaluation_status"] == "complete"
    assert metrics["claim_count"] == 3
    assert metrics["accuracy"] == pytest.approx(2 / 3)


def test_real_sdk_response_round_trips_through_journal(benchmark, monkeypatch):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    run = benchmark
    monkeypatch.setattr(runner, "call_openai_structured", call_openai_structured)

    def transport(request):
        body = json.loads(request.content)
        assert body["response_format"]["json_schema"]["strict"]
        assert body["reasoning_effort"] == "none"
        assert body["service_tier"] == "default"
        payload = response(user_prompt=body["messages"][1]["content"])[0].model_dump()
        return httpx.Response(200, headers={"x-request-id": "req-simulated"}, json={
            "id": "completion-simulated", "created": 0, "model": "sdk-mock", "object": "chat.completion",
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(payload)},
            }],
        })

    run.factory.return_value = openai.OpenAI(
        api_key="synthetic-test-key", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    args = run.args + ["--max-claims", "1", "--reasoning-effort", "none", "--service-tier", "default"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        runner.main(args)
    row = read_jsonl(run.directory / "classify_top3_raw.jsonl")[0]
    assert row["response"]["request_id"] == "req-simulated"
    assert row["parsed"]["prediction"]["predicted_label"] == "Supported"
    assert row["parsed"]["prediction"]["selected_passage_ids"] == ["a1", "a2"]
    run.factory.reset_mock()
    runner.main(args)
    run.factory.assert_not_called()
