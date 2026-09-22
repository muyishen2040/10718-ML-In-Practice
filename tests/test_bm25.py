from src.data.schema import EvidenceItem
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.metrics import evaluate_rankings


def test_bm25_retrieves_expected_passage() -> None:
    retriever = BM25Retriever(
        [
            EvidenceItem(passage_id="a", text="Cats sleep many hours per day."),
            EvidenceItem(passage_id="b", text="Mars has two moons."),
            EvidenceItem(passage_id="c", text="A river flows through the city."),
        ]
    )

    results = retriever.retrieve("How many moons does Mars have?", k=1)

    assert results[0].passage_id == "b"


def test_retrieval_metrics() -> None:
    metrics = evaluate_rankings({"c1": ["x", "gold"]}, {"c1": ["gold"]}, k=3)

    assert metrics["recall_at_3"] == 1.0
    assert metrics["mrr"] == 0.5
