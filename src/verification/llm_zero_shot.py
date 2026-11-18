"""Evidence-only zero-shot LLM baseline with strict structured outputs."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from src.data.averitec import CANONICAL_LABELS

if TYPE_CHECKING:
    from openai import OpenAI

LLMVariant = Literal["classify_top3", "rerank_top20_and_classify"]
NO_EVIDENCE_EXPLANATION = "No evidence passages were supplied to assess the claim."


class LLMVerdict(BaseModel):
    """Required model response; confidence is self-reported, not calibrated."""

    label: Literal["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence"]
    selected_passage_ids: list[str] = Field(min_length=1, max_length=3)
    reported_confidence: float = Field(ge=0.0, le=1.0)
    brief_rationale: str = Field(min_length=1, max_length=600)


class ClaimOnlyLLMVerdict(BaseModel):
    """Required response when the model receives no evidence passages."""

    label: Literal["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence"]
    reported_confidence: float = Field(ge=0.0, le=1.0)
    brief_rationale: str = Field(min_length=1, max_length=600)


SYSTEM_PROMPT = """You are an evidence-grounded claim-verification baseline.
Use only the supplied passages. Do not use outside knowledge, the claim's likely
truth, or information that is absent from the passages. Return one label:
- Supported: evidence establishes the claim.
- Refuted: evidence establishes that the claim is false or materially wrong.
- Not Enough Evidence: supplied evidence does not establish either conclusion.
- Conflicting Evidence: supplied evidence contains material support and refutation.

Select the passage IDs that most directly support your decision. Your reported
confidence is a self-assessment, not a probability. Return the required JSON
object only, with exactly these keys:
{"label": "one of the four labels", "selected_passage_ids": ["passage ID"],
 "reported_confidence": 0.0, "brief_rationale": "one brief explanation"}."""


CLAIM_ONLY_SYSTEM_PROMPT = """You are a claim-only claim-verification baseline.
You receive a claim but no evidence passages. Do not imply that you retrieved,
read, or verified external evidence. Return one label:
- Supported
- Refuted
- Not Enough Evidence
- Conflicting Evidence

Return the required JSON object only, with exactly these keys:
{"label": "one of the four labels", "reported_confidence": 0.0,
 "brief_rationale": "one brief explanation"}."""


def build_user_prompt(
    claim: str,
    passages: list[dict[str, Any]],
    variant: LLMVariant,
    max_chars_per_passage: int = 1200,
    evidence_description: str = "retrieved passages",
) -> str:
    """Build the frozen evidence-only request for one claim."""
    expected = min(3, len(passages))
    if expected < 1:
        raise ValueError("LLM baseline requires at least one candidate passage")
    instructions = (
        f"Rerank the {len(passages)} candidate passages and select exactly {expected} passage IDs before classifying."
        if variant == "rerank_top20_and_classify"
        else f"Classify using these {len(passages)} {evidence_description} and select 1 to {expected} relevant IDs."
    )
    rendered_passages = []
    for index, passage in enumerate(passages, start=1):
        passage_id = str(passage["passage_id"])
        text = str(passage.get("text") or "").strip()[:max_chars_per_passage]
        rendered_passages.append(f"[PASSAGE {index} | ID={passage_id}]\n{text}")
    return f"{instructions}\n\n[CLAIM]\n{claim.strip()}\n\n" + "\n\n".join(rendered_passages)


def build_claim_only_prompt(claim: str) -> str:
    """Build the frozen no-evidence request used by the claim-only diagnostic."""
    return f"Classify this claim without evidence passages.\n\n[CLAIM]\n{claim.strip()}"


def prompt_sha256(system_prompt: str, user_prompt: str) -> str:
    return hashlib.sha256(f"{system_prompt}\n\n{user_prompt}".encode("utf-8")).hexdigest()


def validate_llm_verdict(result: LLMVerdict, passages: list[dict[str, Any]], variant: LLMVariant) -> None:
    """Verify selections against the exact candidate set after schema parsing."""
    candidate_ids = {str(passage["passage_id"]) for passage in passages}
    selected = result.selected_passage_ids
    if len(selected) != len(set(selected)):
        raise ValueError("LLM selected duplicate passage IDs")
    if set(selected) - candidate_ids:
        raise ValueError("LLM selected a passage ID outside the supplied candidates")
    if variant == "rerank_top20_and_classify" and len(selected) != min(3, len(passages)):
        raise ValueError("Reranking variant must select exactly three candidates when available")


def call_openai_structured(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None,
    client: OpenAI | None = None,
    max_completion_tokens: int | None = None,
    timeout: float | None = None,
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
) -> tuple[LLMVerdict, dict[str, Any]]:
    """Call OpenAI's supported Python structured-output interface.

    The OpenAI package is imported only at call time, letting tests run without a
    key and preventing accidental API calls during import.
    """
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("Install openai>=2,<3 before using the hosted LLM.") from error
        client = OpenAI()

    options: dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if max_completion_tokens is not None:
        options["max_completion_tokens"] = max_completion_tokens
    if timeout is not None:
        options["timeout"] = timeout
    if reasoning_effort is not None:
        options["reasoning_effort"] = reasoning_effort
    if service_tier is not None:
        options["service_tier"] = service_tier
    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format=LLMVerdict,
        **options,
    )
    if not completion.choices:
        raise RuntimeError("Model returned no choices")
    message = completion.choices[0].message
    if message.refusal:
        raise RuntimeError(f"Model refusal: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("Model returned no parsed structured response")
    # The parsed result is returned separately. It is SDK metadata, not part
    # of the provider's response, and can trigger generic serialization warnings.
    raw = completion.model_dump(mode="json", exclude={"choices": {"__all__": {"message": {"parsed"}}}})
    raw["request_id"] = getattr(completion, "_request_id", None)
    return message.parsed, raw


def build_prediction(claim_id: str, true_label: str | None, result: LLMVerdict | ClaimOnlyLLMVerdict) -> dict[str, Any]:
    if result.label not in CANONICAL_LABELS:
        raise ValueError(f"Invalid LLM label: {result.label}")
    return {
        "claim_id": claim_id,
        "true_label": true_label,
        "predicted_label": result.label,
        "probabilities": None,
        "selected_passage_ids": getattr(result, "selected_passage_ids", []),
        "reported_confidence": result.reported_confidence,
        "brief_rationale": result.brief_rationale,
    }
