"""Adapter that runs the saved BM25-top-3 DistilBERT baseline on benchmark claims.

Real inference, restricted to known AVeriTeC dev claims: the saved dev BM25
rankings supply each claim's three passages. Inputs are built exactly like
scripts/train_transformer_verifier.py::compose_text, and the model, label order
and token budgets come from the saved baseline run. Gold labels are never read.
"""
import json
from pathlib import Path

LABELS = ("Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence")
MAX_EVIDENCE = 3

# Standing caveat. 0.52 accuracy / 0.347 macro-F1 are the saved baseline's
# results on the 500 official dev claims (final_dev_metrics.json).
BASELINE_NOTE = ("This is a simple baseline model. On the benchmark it agreed with the "
                 "reference label about half the time, so treat it as a rough lead only.")


def compose_text(claim, passages, tokenizer, claim_budget, evidence_budget):
    """Mirror of scripts/train_transformer_verifier.py::compose_text. Keep in sync."""
    claim_tokens = tokenizer.encode(str(claim), add_special_tokens=False)[:claim_budget]
    sections = [f"Claim: {tokenizer.decode(claim_tokens, skip_special_tokens=True)}"]
    usable = [p for p in passages[:MAX_EVIDENCE] if str(p.get("text") or "").strip()]
    if not usable:
        sections.append("Evidence: No candidate passage was retrieved.")
    for i, p in enumerate(usable, start=1):
        tokens = tokenizer.encode(str(p["text"]), add_special_tokens=False)[:evidence_budget]
        sections.append(f"Evidence {i}: {tokenizer.decode(tokens, skip_special_tokens=True)}")
    return "\n".join(sections)


def load_rankings(path):
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["claim_id"])] = row
    return rows


def lean_note(probabilities):
    """Plain-language strength of the model's lean. No percentages (not calibrated)."""
    top = max(probabilities)
    if top >= 0.7:
        return "The model leaned clearly toward this label."
    if top >= 0.45:
        return "The model leaned toward this label, but not strongly."
    return "The model had no strong lean; the labels were close."


class BaselineAdapter:
    name = "baseline_distilbert_bm25_top3"

    def __init__(self, rankings, predict_fn):
        """rankings: {claim_id: row with 'claim' and 'retrieved'}.
        predict_fn(claim, passages) -> four probabilities in LABELS order."""
        self.rankings = rankings
        self.predict_fn = predict_fn
        self._cache = {}

    @classmethod
    def from_paths(cls, baseline_dir, rankings_path, device="cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        baseline_dir = Path(baseline_dir)
        settings = json.loads((baseline_dir / "transformer_run_config.json").read_text(encoding="utf-8"))
        model_dir = baseline_dir / "final_model"
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(device).eval()
        if [model.config.id2label.get(i) for i in range(4)] != list(LABELS):
            raise ValueError("Saved model's label order differs from the canonical labels")

        def predict(claim, passages):
            text = compose_text(claim, passages, tokenizer, settings["claim_token_budget"],
                                settings["evidence_token_budget_per_passage"])
            enc = tokenizer(text, truncation=True, max_length=settings["max_length"], return_tensors="pt")
            with torch.inference_mode():
                logits = model(**{k: v.to(device) for k, v in enc.items()}).logits.float()
            return logits.softmax(-1)[0].cpu().tolist()

        return cls(load_rankings(rankings_path), predict)

    def list_claims(self, query="", limit=50):
        q = (query or "").strip().lower()
        out = []
        for cid, row in self.rankings.items():
            if q in cid.lower() or q in str(row.get("claim", "")).lower():
                out.append({"claim_id": cid, "claim": row.get("claim", "")})
                if len(out) >= limit:
                    break
        return out

    def get(self, claim_id):
        row = self.rankings.get(claim_id)
        if row is None:
            return None
        if claim_id not in self._cache:
            self._cache[claim_id] = self._verify(row)
        return self._cache[claim_id]

    def _verify(self, row):
        claim = row["claim"]
        passages = row.get("retrieved") or []
        evidence = [{"passage_id": str(p["passage_id"]), "text": p["text"],
                     "url": p.get("url") or "", "rank": i}
                    for i, p in enumerate(passages[:MAX_EVIDENCE], start=1)
                    if str(p.get("text") or "").strip()]
        if not evidence:
            return {"claim_id": row["claim_id"], "claim": claim, "status": "insufficient_sources",
                    "explanation": "No source passages were found for this claim.", "evidence": []}
        probs = self.predict_fn(claim, passages)
        verdict = LABELS[max(range(4), key=lambda i: probs[i])]
        return {"claim_id": row["claim_id"], "claim": claim, "status": "completed", "verdict": verdict,
                "explanation": "A baseline classifier read the claim and the three passages shown "
                               "and chose this label. It does not give reasons.",
                "uncertainty_note": lean_note(probs) + " " + BASELINE_NOTE,
                "evidence": evidence}
