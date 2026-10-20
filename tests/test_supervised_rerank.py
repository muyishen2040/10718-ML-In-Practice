import numpy as np
import pytest

from scripts.supervised_rerank_common import normalized_bm25, proxy_hit, rerank, weak_labels
from scripts.prepare_answer_aligned_labels import align_claim
from scripts.train_cross_encoder_weak import training_pairs
from scripts.train_projection_reranker import cache_embeddings, fit_projection, projected_scores


def example_rows():
    rows = [
        {"claim_id": "train:1", "claim": "red moon", "retrieved": [
            {"passage_id": "bad1", "text": "red sky", "url": "https://bad.example", "score": 3.0},
            {"passage_id": "good1", "text": "red moon", "url": "https://good.example/1/", "score": 2.0},
            {"passage_id": "bad2", "text": "blue sky", "url": "https://other.example", "score": 1.0},
        ]},
        {"claim_id": "train:2", "claim": "blue sun", "retrieved": [
            {"passage_id": "bad3", "text": "blue stars", "url": "https://bad.example", "score": 3.0},
            {"passage_id": "good2", "text": "blue sun", "url": "https://good.example/2", "score": 2.0},
            {"passage_id": "bad4", "text": "yellow stars", "url": "https://other.example", "score": 1.0},
        ]},
    ]
    claims = {
        "train:1": {"source_urls": ["https://good.example/1"]},
        "train:2": {"source_urls": ["https://good.example/2/"]},
    }
    return rows, claims


def test_proxy_training_labels_and_rerank_are_separate():
    rows, claims = example_rows()
    assert weak_labels(rows[0], claims["train:1"], 3) == [False, True, False]
    assert normalized_bm25(rows[0], 3) == [1.0, 0.5, 0.0]
    assert proxy_hit(rows, claims, 3) == 1.0
    ranked = rerank(rows[0], [0.0, 1.0, 0.0], 1.0, 3, "learned")
    assert [item["passage_id"] for item in ranked["retrieved"]] == ["good1", "bad1", "bad2"]
    assert "source_urls" not in str(ranked)


def test_cross_encoder_hard_negative_budget_is_per_claim():
    rows, claims = example_rows()
    pairs = training_pairs(rows, claims, [0, 1], top_n=3, negatives=1)
    assert len(pairs) == 4
    assert [pair[2] for pair in pairs] == [1.0, 0.0, 1.0, 0.0]
    assert pairs[1][1] == "red sky"
    assert pairs[3][1] == "blue stars"


def test_answer_alignment_uses_gold_only_for_train_labels():
    ranking = {"claim_id": "train:1", "claim": "The red moon was seen", "retrieved": [
        {"passage_id": "wrong", "text": "The moon was blue in the photo", "url": "https://cited.example", "score": 4.0},
        {"passage_id": "right", "text": "The red moon photograph was taken in 2014", "url": "https://cited.example/", "score": 3.0},
        {"passage_id": "negative", "text": "The moon was red", "url": "https://other.example", "score": 2.0},
    ]}
    claim = {"claim_id": "train:1", "source_urls": ["https://cited.example"], "evidence": [
        {"url": "https://cited.example", "text": "The photograph was taken in 2014", "metadata": {"question": "When was the photograph taken?"}},
    ]}
    aligned = align_claim(claim, ranking, 3, 0.15, 2)
    assert aligned["positive_passage_id"] == "right"
    pairs = training_pairs([ranking], {"train:1": claim}, [0], top_n=3, negatives=1,
                           positive_ids={"train:1": aligned["positive_passage_id"]})
    assert pairs == [(ranking["claim"], ranking["retrieved"][1]["text"], 1.0),
                     (ranking["claim"], ranking["retrieved"][2]["text"], 0.0)]


class FakeEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **_):
        self.calls += 1
        return np.asarray([[float("red" in text), float("blue" in text)] for text in texts], dtype=np.float32)


def test_projection_cache_resume_and_score(tmp_path):
    rows, _ = example_rows()
    encoder = FakeEncoder()
    q, p, mask = cache_embeddings(encoder, rows, tmp_path / "cache", 3, 4, 1, "")
    assert q.shape == (2, 2)
    assert p.shape == (2, 3, 2)
    assert mask.sum() == 6
    assert encoder.calls == 4
    q2, p2, mask2 = cache_embeddings(encoder, rows, tmp_path / "cache", 3, 4, 1, "")
    assert encoder.calls == 4  # Reused both complete checkpoint chunks.
    assert np.array_equal(q, q2) and np.array_equal(p, p2) and np.array_equal(mask, mask2)
    pytest.importorskip("torch")
    labels = np.asarray([[False, True, False], [False, True, False]])
    layer = fit_projection(q, p, mask, labels, [0, 1], lr=0.001, epochs=1, seed=718, device="cpu")
    scores = projected_scores(layer, q, p, mask)
    assert len(scores) == 2 and all(len(item) == 3 for item in scores)
