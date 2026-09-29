from src.data.corpus import build_url_qrels, chunk_text, documents_to_passages


def test_chunking_and_url_qrels_preserve_provenance() -> None:
    passages, audit = documents_to_passages(
        [{"document_id": "d1", "url": "https://example.org/a/", "text": "one two three four five"}],
        max_words=3,
        overlap_words=1,
    )
    qrels, qrels_audit = build_url_qrels(
        [{"claim_id": "c1", "source_urls": ["https://example.org/a"]}], passages
    )

    assert chunk_text("one two three four", max_words=3, overlap_words=1) == ["one two three", "three four"]
    assert audit["passage_count"] == 2
    assert len(qrels[0]["relevant_passage_ids"]) == 2
    assert qrels_audit["judged_claim_count"] == 1


def test_url_qrels_respect_per_claim_candidate_pools() -> None:
    qrels, _ = build_url_qrels(
        [
            {"claim_id": "c1", "source_urls": ["https://example.org/shared"]},
            {"claim_id": "c2", "source_urls": ["https://example.org/shared"]},
        ],
        [
            {"passage_id": "c1-p", "url": "https://example.org/shared", "metadata": {"candidate_claim_id": "c1"}},
            {"passage_id": "c2-p", "url": "https://example.org/shared", "metadata": {"candidate_claim_id": "c2"}},
        ],
    )

    assert qrels[0]["relevant_passage_ids"] == ["c1-p"]
    assert qrels[1]["relevant_passage_ids"] == ["c2-p"]
