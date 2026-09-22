import json

from src.data.averitec import CANONICAL_LABELS, load_split_with_rejections, raw_to_record


def test_raw_to_record_normalizes_label_and_extracts_gold_answers() -> None:
    raw = {
        "claim": "A claim.",
        "label": "Conflicting Evidence/Cherrypicking",
        "questions": [
            {
                "question": "What happened?",
                "answers": [
                    {"answer": "An answer.", "source_url": "https://example.org/source"}
                ],
            }
        ],
    }

    record = raw_to_record(raw, "train", 7)

    assert record.claim_id == "averitec:train:00007"
    assert record.label == "Conflicting Evidence"
    assert record.label in CANONICAL_LABELS
    assert record.evidence[0].text == "An answer."
    assert record.source_urls == ("https://example.org/source",)


def test_loader_audits_empty_claims(tmp_path) -> None:
    source = tmp_path / "claims.json"
    source.write_text(
        json.dumps(
            [
                {"claim": "Valid", "label": "Supported", "questions": []},
                {"claim": "", "label": "Refuted", "questions": []},
            ]
        ),
        encoding="utf-8",
    )

    records, rejections = load_split_with_rejections(source, "train")

    assert len(records) == 1
    assert rejections == [{"source_index": 1, "reason": "empty_claim", "source_label": "Refuted", "fact_checking_article": None}]
