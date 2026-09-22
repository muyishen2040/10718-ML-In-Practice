import csv

from src.data.snopes import load_snopes_records, normalize_snopes_label, parse_source_urls


def test_normalize_snopes_labels_handles_archival_variants() -> None:
    assert normalize_snopes_label("True") == "Supported"
    assert normalize_snopes_label("Moslty true, exagerated or missing details") == "Supported"
    assert normalize_snopes_label("Moslty false or misleading") == "Refuted"
    assert normalize_snopes_label("Incomplete") == "Not Enough Evidence"
    assert normalize_snopes_label("Unknown") is None


def test_parse_source_urls_deduplicates_and_discards_non_urls() -> None:
    urls = parse_source_urls("https://example.org/a., not a URL, https://example.org/a., https://example.org/b")

    assert urls == ("https://example.org/a", "https://example.org/b")


def test_loader_filters_to_mappable_snopes_records(tmp_path) -> None:
    source = tmp_path / "Dataset.csv"
    rows = [
        {"Num": "1", "Website": "snopes", "Claim": "A", "Evaluation": "True", "Article sources": "https://source.test/a"},
        {"Num": "2", "Website": "poligrafo", "Claim": "B", "Evaluation": "False", "Article sources": "https://source.test/b"},
        {"Num": "3", "Website": "snopes", "Claim": "", "Evaluation": "False", "Article sources": ""},
        {"Num": "4", "Website": "snopes", "Claim": "C", "Evaluation": "Unknown", "Article sources": ""},
    ]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    records, audit, rejections = load_snopes_records(source)

    assert [record.claim_id for record in records] == ["snopes:1"]
    assert audit["snopes_row_count"] == 3
    assert audit["accepted_record_count"] == 1
    assert {row["reason"] for row in rejections} == {"empty_claim", "unmappable_evaluation"}
