from src.external.sampling import stratified_sample


def test_stratified_sample_is_deterministic_and_balanced_when_possible() -> None:
    records = [
        {"claim_id": "s1", "label": "Supported"},
        {"claim_id": "s2", "label": "Supported"},
        {"claim_id": "r1", "label": "Refuted"},
        {"claim_id": "r2", "label": "Refuted"},
        {"claim_id": "n1", "label": "Not Enough Evidence"},
        {"claim_id": "n2", "label": "Not Enough Evidence"},
    ]

    sampled_once = stratified_sample(records, ("Supported", "Refuted", "Not Enough Evidence"), 3, 718)
    sampled_twice = stratified_sample(records, ("Supported", "Refuted", "Not Enough Evidence"), 3, 718)

    assert sampled_once == sampled_twice
    assert {record["label"] for record in sampled_once} == {"Supported", "Refuted", "Not Enough Evidence"}
