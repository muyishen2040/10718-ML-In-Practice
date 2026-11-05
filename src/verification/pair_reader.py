"""NLI-initialized, permutation-invariant three-passage verdict reader.

Four-way verdict supervision applies to the aggregate, never to every passage.
A small distillation term retains the pretrained three-way NLI behavior using
cached teacher probabilities computed from prediction-time text alone.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.data.averitec import CANONICAL_LABELS
from src.retrieval.evidence_quality import QUALITY_NAMES, nli_label_order, pair_premise


def weighted_verdict_loss(logits, labels, weights, normalization="sample_mean"):
    """Keep class weights effective even for singleton/single-class microbatches.

    Weights have dataset mean one. Averaging weighted per-example losses gives
    the same gradient under equal-size microbatch accumulation as a full batch.
    The legacy option is retained for old checkpoints and controlled ablations.
    """
    if normalization not in ("sample_mean", "batch_weight_mean"):
        raise ValueError(f"Unknown loss normalization: {normalization}")
    reduction = "none" if normalization == "sample_mean" else "mean"
    loss = nn.functional.cross_entropy(logits.float(), labels, weight=weights.float(), reduction=reduction)
    return loss.mean()


class PairReaderConfig(PretrainedConfig):
    model_type = "evidence_pair_reader"

    def __init__(self, encoder_config=None, projection_size=192, dropout=0.15,
                 nli_labels=None, class_weights=None, distillation_weight=0.1, freeze_word_embeddings=True,
                 loss_normalization="batch_weight_mean", **kwargs):
        kwargs.pop("num_labels", None)
        super().__init__(num_labels=4, **kwargs)
        self.encoder_config = encoder_config or {}
        self.projection_size = projection_size
        self.dropout = dropout
        self.nli_labels = nli_labels or ["contradiction", "entailment", "neutral"]
        self.class_weights = class_weights or [1.0] * 4
        self.distillation_weight = distillation_weight
        self.freeze_word_embeddings = freeze_word_embeddings
        if loss_normalization not in ("sample_mean", "batch_weight_mean"):
            raise ValueError("Unknown verdict-loss normalization")
        self.loss_normalization = loss_normalization
        self.id2label = dict(enumerate(CANONICAL_LABELS))
        self.label2id = {v: k for k, v in self.id2label.items()}


class EvidencePairReader(PreTrainedModel):
    config_class = PairReaderConfig
    base_model_prefix = "reader"
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        encoder = dict(config.encoder_config)
        kind = encoder.pop("model_type")
        self.reader = AutoModelForSequenceClassification.from_config(AutoConfig.for_model(kind, **encoder))
        hidden = int(self.reader.config.hidden_size)
        size, q = config.projection_size, len(QUALITY_NAMES)
        self.project = nn.Sequential(nn.Linear(hidden + 3 + q, size), nn.GELU(), nn.LayerNorm(size))
        self.attention = nn.Linear(size, 1)
        self.verdict = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(2 * size + q + 8, size),
                                     nn.GELU(), nn.Dropout(config.dropout), nn.Linear(size, 4))
        self.register_buffer("verdict_weights", torch.tensor(config.class_weights, dtype=torch.float32))
        self.post_init()
        if config.freeze_word_embeddings:
            self.reader.get_input_embeddings().requires_grad_(False)

    @classmethod
    def from_nli(cls, name, revision=None, *, class_weights=None, label_order=None, distillation_weight=0.1,
                 freeze_word_embeddings=True, loss_normalization="sample_mean"):
        reader = AutoModelForSequenceClassification.from_pretrained(name, revision=revision)
        if int(reader.config.num_labels) != 3:
            raise ValueError("Reader initialization requires a three-label NLI model")
        order = nli_label_order(reader.config.id2label, label_order)
        config = PairReaderConfig(encoder_config=reader.config.to_dict(), nli_labels=order,
                                  class_weights=class_weights, distillation_weight=distillation_weight,
                                  freeze_word_embeddings=freeze_word_embeddings, loss_normalization=loss_normalization)
        model = cls(config)
        model.reader = reader
        if freeze_word_embeddings:
            model.reader.get_input_embeddings().requires_grad_(False)
        return model

    def forward(self, input_ids, attention_mask, evidence_mask, quality, teacher_probabilities,
                token_type_ids=None, labels=None):
        batch, passages, length = input_ids.shape
        encoded = {"input_ids": input_ids.reshape(-1, length), "attention_mask": attention_mask.reshape(-1, length)}
        if token_type_ids is not None:
            encoded["token_type_ids"] = token_type_ids.reshape(-1, length)
        output = self.reader(**encoded, output_hidden_states=True, return_dict=True)
        hidden = output.hidden_states[-1][:, 0].reshape(batch, passages, -1)
        nli = output.logits.float().reshape(batch, passages, 3)
        indices = [self.config.nli_labels.index(k) for k in ("entailment", "contradiction", "neutral")]
        nli = nli[..., indices]
        probabilities = nli.softmax(-1)
        valid = evidence_mask.to(hidden.dtype)
        projected = self.project(torch.cat([hidden, probabilities.to(hidden.dtype), quality.to(hidden.dtype)], dim=-1))
        attention_logits = self.attention(projected).squeeze(-1).masked_fill(valid == 0, -1e4)
        weights = attention_logits.softmax(-1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        pooled = (projected * weights.unsqueeze(-1)).sum(1)
        any_evidence = (valid.sum(1, keepdim=True) > 0).to(hidden.dtype)
        maximum = projected.masked_fill(valid.unsqueeze(-1) == 0, -1e4).max(1).values * any_evidence
        mean_nli = (probabilities * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        max_nli = probabilities.masked_fill(valid.unsqueeze(-1) == 0, 0).max(1).values
        mean_quality = (quality * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        conflict = max_nli[:, :1] * max_nli[:, 1:2]
        features = torch.cat([pooled, maximum, mean_nli, max_nli, mean_quality, any_evidence, conflict], dim=-1)
        logits = self.verdict(features.to(hidden.dtype))
        loss = None
        if labels is not None:
            loss = weighted_verdict_loss(logits, labels, self.verdict_weights, self.config.loss_normalization)
            if self.config.distillation_weight:
                teacher = teacher_probabilities.float().clamp_min(1e-7)
                kl = nn.functional.kl_div(nli.log_softmax(-1), teacher, reduction="none").sum(-1)
                loss = loss + self.config.distillation_weight * (kl * valid).sum() / valid.sum().clamp_min(1)
        return SequenceClassifierOutput(loss=loss, logits=logits)


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, records, tokenizer, *, max_length=384, claim_budget=96):
        from src.utils.progress import progress

        self.records, self.features = records, []
        with progress(total=len(records), description="Encode evidence pairs", unit="claim", leave=False) as bar:
            for record in records:
                hypothesis = tokenizer.decode(tokenizer.encode(record["claim"], add_special_tokens=False)[:claim_budget], skip_special_tokens=True)
                evidence = (record.get("retrieved_evidence") or [])[:3]
                pairs, mask, quality, teacher = [], [], [], []
                for i in range(3):
                    item = evidence[i] if i < len(evidence) else {"text": "No evidence available."}
                    pairs.append(tokenizer(pair_premise(item), hypothesis, truncation="only_first", max_length=max_length))
                    mask.append(float(i < len(evidence)))
                    quality.append(item.get("quality", [0.0] * len(QUALITY_NAMES)))
                    p = item.get("nli_probabilities", {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0})
                    teacher.append([p[k] for k in ("entailment", "contradiction", "neutral")])
                self.features.append({"pairs": pairs, "evidence_mask": mask, "quality": quality,
                                      "teacher_probabilities": teacher, "labels": CANONICAL_LABELS.index(record["label"])})
                bar.update(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


class PairCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        padded = self.tokenizer.pad([p for r in rows for p in r["pairs"]], return_tensors="pt", pad_to_multiple_of=8)
        result = {k: v.reshape(len(rows), 3, -1) for k, v in padded.items()}
        for key in ("evidence_mask", "quality", "teacher_probabilities"):
            result[key] = torch.tensor([r[key] for r in rows], dtype=torch.float32)
        result["labels"] = torch.tensor([r["labels"] for r in rows], dtype=torch.long)
        return result
