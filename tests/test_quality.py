from src.data.quality import split_overlap_audit
from src.data.schema import ClaimRecord


def test_overlap_audit_detects_normalized_claims() -> None:
    report = split_overlap_audit(
        {
            "train": [ClaimRecord(claim_id="train:1", claim="Mars has two moons!", label="Supported")],
            "dev": [ClaimRecord(claim_id="dev:1", claim="mars has two moons", label="Supported")],
        }
    )

    assert report["dev__train"]["normalized_claim"]["overlap_count"] == 1
