from src.evaluation.end_to_end import annotate_retrieval_coverage, evaluate_by_retrieval_coverage


def test_retrieval_conditioning_uses_verifier_evidence_ids_when_present() -> None:
    predictions = [
        {
            "claim_id": "c1",
            "true_label": "Refuted",
            "predicted_label": "Refuted",
            "evidence_passage_ids": ["gold"],
        },
        {
            "claim_id": "c2",
            "true_label": "Supported",
            "predicted_label": "Supported",
            "evidence_passage_ids": ["not-gold"],
        },
    ]
    rankings = [
        {"claim_id": "c1", "retrieved": [{"passage_id": "not-gold"}]},
        {"claim_id": "c2", "retrieved": [{"passage_id": "gold"}]},
    ]
    qrels = {"c1": ["gold"], "c2": ["gold"]}

    annotations = annotate_retrieval_coverage(predictions, rankings, qrels)
    report = evaluate_by_retrieval_coverage(predictions, rankings, qrels)

    assert [row["retrieval_coverage"] for row in annotations] == ["evidence_covered", "evidence_missed"]
    assert report["retrieval_covered_count"] == 1
    assert report["retrieval_missed_count"] == 1
