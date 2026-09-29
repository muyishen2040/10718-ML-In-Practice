# Human-evaluation protocol

Offline retrieval and verdict metrics are proxies. The intended user outcome is
whether people become less likely to trust or share refuted or unsupported
claims without becoming excessively skeptical of supported claims.

## Minimum session record

Store one consented, pseudonymous record per participant/claim condition:

```json
{
  "participant_id": "random-study-id",
  "claim_id": "averitec:dev:00001",
  "condition": "unassisted | evidence_only | evidence_and_verdict",
  "displayed_passage_ids": ["p1", "p2", "p3"],
  "displayed_system_label": "Supported | Refuted | Not Enough Evidence | Conflicting Evidence | null",
  "displayed_system_confidence": 0.0,
  "participant_assessment": "Supported | Refuted | Not Enough Evidence | Conflicting Evidence",
  "trust_or_share_intent": "trust | share | neither | investigate",
  "participant_confidence_1_to_5": 1,
  "decision_time_seconds": 0.0,
  "perceived_usefulness_1_to_5": 1
}
```

Do not collect names, email addresses, social-media accounts, political
identity, or unneeded sensitive data.

## Design

- Recruit approximately 20–30 accessible adult participants for a pilot; a
  larger, more diverse longitudinal study is the desired deployment evaluation.
- Counterbalance claims and conditions so no participant sees the same claim in
  more than one condition.
- Use unfamiliar claims and preserve the held-out claim labels until analysis.
- Compare unassisted, evidence-only, and evidence-plus-verdict conditions.

## Outcomes

Report claim-assessment accuracy, time, confidence calibration, usefulness, and
two decision harms separately:

- **False reassurance:** a participant trusts/shares a Refuted, Not Enough
  Evidence, or Conflicting Evidence claim.
- **False alarm:** a participant rejects a Supported claim.

Also audit whether users follow a wrong system output more often in the
evidence-plus-verdict condition than in the evidence-only condition.
