"""Evidence-only zero-shot LLM baseline with strict structured outputs."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.data.averitec import CANONICAL_LABELS

LLMVariant = Literal["classify_top3", "rerank_top20_and_classify"]


class LLMVerdict(BaseModel):
    """Required model response; confidence is self-reported, not calibrated."""

    label: Literal["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence"]
    selected_passage_ids: list[str] = Field(min_length=1, max_length=3)
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
object only."""


def build_user_prompt(claim: str, passages: list[dict[str, Any]], variant: LLMVariant, max_chars_per_passage: int = 1200) -> str:
    """Build the frozen evidence-only request for one claim."""
    expected = min(3, len(passages))
    if expected < 1:
        raise ValueError("LLM baseline requires at least one candidate passage")
    instructions = (
        f"Rerank the {len(passages)} candidate passages and select exactly {expected} passage IDs before classifying."
        if variant == "rerank_top20_and_classify"
        else f"Classify using these top {len(passages)} retrieved passages and select 1 to {expected} relevant IDs."
    )
    rendered_passages = []
    for index, passage in enumerate(passages, start=1):
        passage_id = str(passage["passage_id"])
        text = str(passage.get("text") or "").strip()[:max_chars_per_passage]
        rendered_passages.append(f"[PASSAGE {index} | ID={passage_id}]\n{text}")
    return f"{instructions}\n\n[CLAIM]\n{claim.strip()}\n\n" + "\n\n".join(rendered_passages)


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
    *, model: str, system_prompt: str, user_prompt: str, temperature: float
) -> tuple[LLMVerdict, dict[str, Any]]:
    """Call OpenAI's supported Python structured-output interface.

    The OpenAI package is imported only at call time, letting tests run without a
    key and preventing accidental API calls during import.
    """
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install requirements-colab.txt before using the LLM baseline.") from error
    completion = OpenAI().chat.completions.parse(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format=LLMVerdict,
        temperature=temperature,
    )
    message = completion.choices[0].message
    if message.refusal:
        raise RuntimeError(f"Model refusal: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("Model returned no parsed structured response")
    return message.parsed, completion.model_dump(mode="json")


def build_prediction(claim_id: str, true_label: str | None, result: LLMVerdict) -> dict[str, Any]:
    if result.label not in CANONICAL_LABELS:
        raise ValueError(f"Invalid LLM label: {result.label}")
    return {
        "claim_id": claim_id,
        "true_label": true_label,
        "predicted_label": result.label,
        "probabilities": None,
        "selected_passage_ids": result.selected_passage_ids,
        "reported_confidence": result.reported_confidence,
        "brief_rationale": result.brief_rationale,
    }
