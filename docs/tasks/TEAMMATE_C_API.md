# Teammate C: hosted API models

## Project and your outcome

Test whether a hosted language model improves our misinformation verifier, and
provide an adapter for the browser-extension backend. Given a claim and the same
three source passages supplied to local models, return Supported, Refuted,
Not Enough Evidence or Conflicting Evidence, with a short grounded explanation
and valid passage citations.

- Repository: <https://github.com/muyishen2040/10718-ML-In-Practice>
- Drive/Colab setup, report and artifacts: [TEAM_HANDOFF.md](../TEAM_HANDOFF.md).
- Baseline results: [TEAM_PROGRESS.md](../TEAM_PROGRESS.md).
- Coordinator-owned evaluation release: [EVALUATION.md](../EVALUATION.md).
- Common backend boundary: [INTEGRATION_CONTRACT.md](../INTEGRATION_CONTRACT.md).

Explore OpenAI or another provider. No particular provider/model has been chosen
by this brief. Read current official provider documentation when implementing
calls; record exact model/version, prompt, decoding and output-schema settings.

## Ownership and starting files

Suggested branch: `team/teammate-c-api`, from the published shared base listed in
the handoff. Own provider integration and API evaluation scripts/notebooks. Keep
reusable provider code under `src/verification/`; a `providers/` subdirectory may
be added when implemented. Coordinate local-reader changes with B and the HTTP
boundary with A. Do not build an independent incompatible frontend.

Starting code is `scripts/run_llm_baseline.py` and
`src/verification/llm_zero_shot.py`, not a completed hosted-model benchmark. The
current runner skips claims without exactly the expected passage count, binds
cached calls too loosely for changed prompts, and writes raw calls at the end
of a run. Audit and fix per-call persistence, input/prompt/config hash binding,
resume behavior and incomplete-claim handling before a full comparison.

## Inputs and split readiness

Mount your own Drive after adding the shared `10718_averitec` shortcut. The
active root is `/content/drive/MyDrive/10718_averitec/v2_source_documents`.
Reuse the saved split-specific BM25 rankings under `outputs/bm25/`; model calls
need their retrieved text, not a rebuild of the full passage corpus. Write new
outputs to a unique personal Drive run directory.

The coordinator will publish the time-based fit/validation manifest path and
hash through [team_evaluation.json](../../configs/team_evaluation.json). While
pending, implement mocks, parsing, caching and integration. Start paid prompt
pilots on designated training-fit cases after release. Do not choose a private
replacement split or tune on official dev to speed up comparison.

## First bounded experiment sequence

1. Choose one hosted model initially and record a spending cap and token/output
   limits. Build a provider adapter returning the common response fields. Keep
   credentials in the backend environment, outside Git and the extension.
2. Run a small training-fit pilot to verify formatting, passage-ID citations,
   error/retry handling, interruption recovery, token use and approximate cost.
   A repeated completed request should reuse an exactly matched cache entry.
3. Compare prompts/models using the released training-validation population,
   freeze the selected configuration, then run the agreed official-dev evaluation.
   First hold the claim and BM25 top-three passages fixed, with external search
   disabled, to isolate verifier quality.
4. Keep all 500 dev claim IDs, including empty-evidence cases. Permit empty source
   lists as prediction inputs, but distinguish a completed NEI prediction from
   a failed call. Report unresolved failures and incomplete coverage; do not
   silently drop claims or turn errors into NEI predictions.
5. If B supplies a validated selector, compare the same API verifier on its
   three full passages. This isolates evidence changes from model changes.
   Any assisted-labeling work for B is a different training-only task, with its
   own prompt, cache and outputs, not a deployed input protocol.
6. Optional live web retrieval can support A's arbitrary-browser-claim flow.
   Record it as a separate experiment with source URLs, retrieval/publication
   dates, evidence budget and costs. It is not a fixed-evidence baseline or proof
   that the static benchmark source constraints were met.

## Results and completion

Use the common scorer and split IDs. Report macro-F1, accuracy, per-class scores,
FR/FA counts and denominators, completion coverage, malformed responses,
retries/errors, token counts, total cost and latency. Self-reported confidence
is not a calibrated probability. An API model may have encountered old claims
during pretraining; a project-level temporal split does not establish otherwise.

Deliver a resumable script/notebook, provider adapter, frozen prompt/configuration,
prediction and error journals, selection record and metric/cost report in Drive.
Provide A with a working adapter or identify what integration remains missing.
Use the existing local scoring code; a provider-hosted evaluation dashboard is
not required. OpenAI's [evaluation guidance](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
provides background on task-specific tests and run logging.

Return a PR and a self-contained update using [the template](../team_updates/TEMPLATE.md).
Include model/prompt/code versions, protocol ID/hash, exact artifact paths and
stable Drive links, runtime/cost, completed and interrupted work, observed errors
and the next action. Do not report a hosted-model score before calls and scoring
have actually completed.
