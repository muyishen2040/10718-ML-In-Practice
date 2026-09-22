from src.verification.tfidf_logreg import compose_input


def test_gold_evidence_input_contains_claim_and_evidence() -> None:
    text = compose_input(
        {
            "claim": "Mars has two moons.",
            "evidence": [{"text": "Phobos and Deimos orbit Mars."}],
        },
        "gold",
    )

    assert "[CLAIM] Mars has two moons." in text
    assert "[EVIDENCE 1] Phobos and Deimos orbit Mars." in text


def test_claim_only_input_excludes_evidence() -> None:
    text = compose_input(
        {"claim": "Mars has two moons.", "evidence": [{"text": "This must not appear."}]},
        "claim_only",
    )

    assert text == "[CLAIM] Mars has two moons."
