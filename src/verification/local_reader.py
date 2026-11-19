"""Frozen local pair-reader inference and the local adapter for A's backend.

``LocalReaderVerifier.verify`` implements B's side of the proposed internal
adapter boundary in docs/INTEGRATION_CONTRACT.md: it accepts a claim and ordered
source passages (no labels, answers or evaluation metadata) and returns the
contract's verdict/explanation/uncertainty/evidence fields. It does not start an
HTTP service; A's backend owns that.

``infer_rankings`` feeds a saved reader any three-passage rankings file, e.g.
selector output. That is an input intervention on a frozen verifier, not a
verifier trained on those passages.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.data.averitec import CANONICAL_LABELS
from src.evaluation.splits import parse_claim_date
from src.retrieval.evidence_quality import QUALITY_NAMES, quality_features

MAX_DISPLAYED = 3
CONTRACT_VERSION = "proposed-v1"

_EXPLANATIONS = {
    "Supported": "The reader judged that the shown passages, taken together, back the claim.",
    "Refuted": "The reader judged that the shown passages contradict the claim.",
    "Not Enough Evidence": "The shown passages do not establish or contradict the key part of the claim.",
    "Conflicting Evidence": "The shown passages point in different directions or support only part of the claim.",
}


def reader_inputs(claim: str, claim_date: str | None, passages: list[dict]) -> list[dict]:
    """Attach the prediction-time quality features the reader was trained with."""
    date = parse_claim_date(claim_date)
    items = []
    for position, passage in enumerate(passages[:MAX_DISPLAYED], start=1):
        if not str(passage.get("passage_id") or "") or not str(passage.get("text") or "").strip():
            raise ValueError("Each passage needs a passage_id and nonblank text")
        item = dict(passage)
        if len(item.get("quality") or []) != len(QUALITY_NAMES):
            item["quality"] = quality_features(claim, item, int(item.get("original_rank") or position), date)
        items.append(item)
    return items


def uncertainty_note(probabilities: dict[str, float]) -> str:
    """Plain-language uncertainty without probability percentages."""
    ranked = sorted(probabilities.items(), key=lambda kv: -kv[1])
    (first, p1), (second, p2) = ranked[0], ranked[1]
    if p1 - p2 < 0.15:
        note = f"The reader was nearly split between {first} and {second}; read the sources before relying on it."
    elif p1 < 0.5:
        note = "The reader's preference is weak; other verdicts remained plausible."
    else:
        note = "This is an automated preliminary reading of three passages, not a confirmed fact-check."
    return note + " You make the final trust or sharing decision."


class LocalReaderVerifier:
    """Load a saved ``EvidencePairReader`` once and verify claims against passages."""

    def __init__(self, model_dir: Path, *, device: str = "cpu", max_length: int = 384, claim_budget: int = 96,
                 preparation: Path | None = None):
        import torch
        from transformers import AutoTokenizer

        from src.verification.pair_reader import EvidencePairReader, PairCollator

        if preparation is not None:
            settings = json.loads(Path(preparation).read_text(encoding="utf-8"))
            max_length, claim_budget = settings["max_length"], settings["claim_budget"]
        self.model_dir, self.device = Path(model_dir), device
        self.max_length, self.claim_budget = max_length, claim_budget
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self.model = EvidencePairReader.from_pretrained(self.model_dir).to(device).eval()
        labels = [self.model.config.id2label[i] for i in range(4)]
        if labels != list(CANONICAL_LABELS):
            raise ValueError(f"Saved reader label order differs: {labels}")
        self.collator = PairCollator(self.tokenizer)
        weights = sorted(self.model_dir.glob("*.safetensors"))
        digest = hashlib.sha256(b"".join(p.read_bytes() for p in weights)).hexdigest() if weights else None
        self.version = {"model": "evidence_pair_reader", "model_dir": self.model_dir.name, "weights_sha256": digest,
                        "max_length": max_length, "claim_budget": claim_budget, "contract": CONTRACT_VERSION}

    def predict_proba(self, records: list[dict]) -> list[list[float]]:
        """Records need ``claim`` and ``retrieved_evidence``; labels are not read."""
        from src.verification.pair_reader import PairDataset

        rows = [{"claim": r["claim"], "label": CANONICAL_LABELS[0],  # placeholder; PairDataset requires a key
                 "retrieved_evidence": reader_inputs(r["claim"], (r.get("metadata") or {}).get("claim_date"),
                                                     r.get("retrieved_evidence") or [])} for r in records]
        dataset = PairDataset(rows, self.tokenizer, max_length=self.max_length, claim_budget=self.claim_budget)
        batch = {k: v.to(self.device) for k, v in self.collator([dataset[i] for i in range(len(dataset))]).items() if k != "labels"}
        with self._torch.inference_mode(), self._torch.autocast(device_type="cuda", dtype=self._torch.float16,
                                                                  enabled=self.device.startswith("cuda")):
            return self.model(**batch).logits.float().softmax(-1).cpu().tolist()

    def verify(self, claim: str, passages: list[dict], *, claim_date: str | None = None) -> dict[str, Any]:
        """Contract-shaped response for one claim and its ranked passages."""
        claim = str(claim or "").strip()
        shown = [p for p in passages if str(p.get("text") or "").strip()][:MAX_DISPLAYED]
        evidence = [{"passage_id": str(p["passage_id"]), "text": str(p["text"]), "url": p.get("url"), "rank": i}
                    for i, p in enumerate(shown, start=1)]
        base = {"status": "completed", "claim": claim, "evidence": evidence, "model": self.version}
        if not claim:
            return {**base, "status": "error", "verdict": None, "explanation": "No claim text was supplied.", "uncertainty_note": None}
        if not shown:
            return {**base, "status": "insufficient_sources", "verdict": None,
                    "explanation": "No source passages were available to check this claim.", "uncertainty_note": None}
        try:
            vector = self.predict_proba([{"claim": claim, "metadata": {"claim_date": claim_date}, "retrieved_evidence": shown}])[0]
        except Exception as error:  # reported as a service failure, never as an NEI verdict
            return {**base, "status": "error", "verdict": None, "explanation": f"Local reader failed: {type(error).__name__}",
                    "uncertainty_note": None}
        probabilities = dict(zip(CANONICAL_LABELS, vector, strict=True))
        verdict = max(probabilities, key=probabilities.get)
        return {**base, "verdict": verdict, "explanation": _EXPLANATIONS[verdict],
                "uncertainty_note": uncertainty_note(probabilities)}


def infer_rankings(args) -> dict[str, Any]:
    """Score a saved reader on a claims file plus any rankings file (resumable)."""
    from src.data.evidence import attach_evidence
    from src.utils.checkpoints import bind_run
    from src.utils.io import read_jsonl, write_json
    from src.utils.runs import sha256_file
    from src.verification.inference import predict_cached

    verifier = LocalReaderVerifier(args.model, device=args.device, preparation=args.preparation)
    records = attach_evidence(read_jsonl(args.claims), args.rankings, exact=True)
    args.output.mkdir(parents=True, exist_ok=True)
    bind_run(args.output / "infer_run.json", {"model": verifier.version, "claims_sha256": sha256_file(args.claims),
             "rankings_sha256": sha256_file(args.rankings), "claims": str(args.claims), "rankings": str(args.rankings),
             "intervention": "frozen reader; only the supplied passages change"})
    metrics = predict_cached(records, verifier.predict_proba, args.output, batch_size=args.batch_size,
                             description=f"Frozen pair reader / {Path(args.rankings).parent.name}")
    write_json(metrics, args.output / "metrics.json")
    return metrics
