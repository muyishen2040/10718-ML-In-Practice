from src.verification.tfidf_logreg import compose_input
from src.evaluation.classification import evaluate_predictions


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


def test_three_class_evaluation_does_not_require_conflicting_label() -> None:
    metrics = evaluate_predictions(
        [{"true_label": "Supported", "predicted_label": "Supported"}],
        labels=("Supported", "Refuted", "Not Enough Evidence"),
    )

    assert metrics["macro_f1"] == 1 / 3
    assert metrics["confusion_matrix"]["labels"] == ["Supported", "Refuted", "Not Enough Evidence"]
