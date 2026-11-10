# Proposed interface and model integration contract

This is the shared design for the [development backend](../apps/SETUP.md).
The prototype supports fixture replay and the saved benchmark model. A owns the extension and
backend boundary, B provides the local-model adapter, C provides the hosted-model
adapter. Agree on shared contract and evaluation changes before implementation.

The assistant accepts a factual claim, presents up to three ranked sources and
a four-way preliminary verdict, and leaves the trust/share decision to the user.
Implementation/model settings belong in server configuration and experiment
logs rather than the ordinary user flow.

## Request and response

Suggested route: `POST /verify`. A benchmark request resolves a known claim and
its saved candidate evidence on the backend:

```json
{"mode": "benchmark", "claim_id": "averitec:dev:00000"}
```

The local-model backend also supports `GET /models` to discover selectable
pipelines. A benchmark request may include `model_id` (`distilbert_bm25`,
`distilbert_bge`, `distilbert_fusion`, or `nli_reader`). Omitting it keeps the
BM25 default. Unknown IDs return 400; unavailable choices return 503. Responses
identify the selected pipeline with `model_id`, `model_label` and `provenance`.
The first three share a BM25-trained classifier; only their evidence changes.
This selector is for comparing experimental pipelines during supervised tests.

A live request needs a separately implemented retrieval path for new claims:

```json
{"mode": "live", "claim": "User-selected factual claim", "page_url": null}
```

The current saved rankings are claim-specific benchmark artifacts. They do not
provide arbitrary web search. Until live retrieval is implemented, the service
must identify live mode as unavailable rather than attach another benchmark
claim's evidence. C may prototype live search as a separately evaluated feature
in coordination with A; that is not part of the initial fixed-evidence benchmark.

Example completed response, using faithful source text in the actual service:

```json
{
  "mode": "benchmark",
  "status": "completed",
  "claim": "The claim being checked",
  "verdict": "Not Enough Evidence",
  "explanation": "The supplied passages do not establish the key claim.",
  "uncertainty_note": "Additional sources would be needed.",
  "evidence": []
}
```

Each evidence item contains `passage_id`, `text`, `url` and `rank`. Return at most
three, ordered by estimated relevance. Use the canonical labels Supported,
Refuted, Not Enough Evidence and Conflicting Evidence. Explain uncertainty in
plain language; do not invent calibrated probability percentages.

Use `status: "insufficient_sources"` or `"error"` with a null verdict when the
service cannot complete verification. A completed NEI prediction and an API or
backend failure are distinct outcomes. Benchmark scoring must not discard failed
requests or silently label them as NEI.

## Internal adapter boundary

Both local and hosted verifiers should accept the same claim and ordered source
passages, and return the same verdict/explanation/citation fields. Gold labels,
human answers and evaluation metadata stay in the scorer, outside prediction
inputs. For the first API comparison, freeze the same BM25 top-three evidence
as the local baseline; compare another selector by changing both adapters' inputs
in a separately recorded experiment.

The backend records model/version, evidence IDs, prompt/config hash, request IDs,
timing, token use and failures where applicable. Credentials remain on the
backend. The extension submits the selected text only when the user requests
verification. A mounted personal Drive is a development data location, not an
assumed deployment filesystem.

## Ownership and fixtures

- A implements mock, benchmark, loading, missing-evidence and error flows, then
  connects real adapters. Put UI/backend code under `apps/` as described there.
- B and C keep model/retrieval implementations reusable under `src/` and supply
  adapters to the shared backend. Avoid independent incompatible HTTP services.
- Agree on small mock fixtures before implementation. Tests should cover valid
  evidence links, the three-item limit, unavailable live mode and service errors.
- A frozen benchmark demo does not establish live claim verification or improved
  user decisions. Label demo mode and report its scope accurately.

Version any breaking contract change and update the task briefs and fixtures
before downstream integration. No backend or extension is deployed by this file.
