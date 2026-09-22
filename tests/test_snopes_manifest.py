from src.external.snopes_manifest import build_source_manifest, source_screening_status


def test_manifest_deduplicates_urls_and_preserves_claim_provenance() -> None:
    manifest, audit = build_source_manifest(
        [
            {"claim_id": "snopes:1", "source_urls": ["https://example.org/a", "https://snopes.com/x"]},
            {"claim_id": "snopes:2", "source_urls": ["https://example.org/a"]},
        ]
    )

    by_url = {row["url"]: row for row in manifest}
    assert by_url["https://example.org/a"]["claim_ids"] == ["snopes:1", "snopes:2"]
    assert by_url["https://snopes.com/x"]["screening_status"] == "excluded_fact_check_source"
    assert audit["unique_source_url_count"] == 2


def test_known_fact_check_domains_require_review() -> None:
    assert source_screening_status("www.politifact.com") == "requires_fact_check_review"
