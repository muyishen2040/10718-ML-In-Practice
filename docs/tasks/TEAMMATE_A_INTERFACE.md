# Teammate A: browser extension and interface

## Project and your outcome

Build the product interface for the CMU 10-718 misinformation screening
assistant. A user selects or pastes a factual claim, explicitly requests a check,
and sees a preliminary verdict plus up to three ranked evidence passages. The
four labels are Supported, Refuted, Not Enough Evidence and Conflicting Evidence.
The user reviews the sources and makes the final trust/share decision.

Your deliverable is a runnable browser extension and common backend boundary.
You do not need to train the research models or create evaluation splits.

- Repository: <https://github.com/muyishen2040/10718-ML-In-Practice>
- Shared setup, report link and artifacts: [TEAM_HANDOFF.md](../TEAM_HANDOFF.md).
- Current method results: [TEAM_PROGRESS.md](../TEAM_PROGRESS.md).
- Proposed requests/responses: [INTEGRATION_CONTRACT.md](../INTEGRATION_CONTRACT.md).
- Shared evaluation protocol: [EVALUATION.md](../EVALUATION.md).

## Ownership and source setup

To run the existing extension with the private Colab CPU backend, use
`feat/colab-extension-backend` and follow
[the complete serving guide](../../apps/COLAB_BACKEND.md). It includes the model
selector. The general task-branch instructions below apply to new development,
not to reproducing that unmerged user-test setup.

Suggested branch: `team/teammate-a-interface`, created from the published shared
base commit listed in the handoff. Fetch `origin/main`,
then create your task branch from it. The handoff includes the verified code
snapshot and checkout commands.

Put extension code under `apps/browser-extension/` and the HTTP/integration layer
under `apps/backend/`. A prototype with mock responses and a saved-model adapter
is available; see [setup](../../apps/SETUP.md) and the
[Colab CPU workflow](../../apps/COLAB_BACKEND.md). B and C supply reusable local
and hosted verifier adapters. Coordinate contract changes rather than duplicating
model logic inside the frontend.

## First implementation sequence

1. Build against small mock responses using the common contract. Implement claim
   editing, submission, loading, results, missing-source and service-error states.
   This can proceed while the date-based split release is pending.
2. Show the verdict, short explanation, faithful source excerpts and clickable
   URLs. Display the three items in relevance order. Use plain-language uncertainty
   rather than an uncalibrated percentage. Keep model/configuration details out
   of the normal user flow.
3. Implement the backend boundary and connect a verifier adapter from B or C.
   Keep provider credentials on the backend, not in the extension package.
   Submit selected text when the user requests a check.
4. Add a clearly labeled benchmark-demo mode. Given an AVeriTeC claim ID, the
   backend can read its saved claim/evidence and call the chosen adapter, or
   replay explicitly labeled saved predictions for a demo fixture. Do not describe
   replayed results as a new inference experiment.
5. Connect arbitrary live claims only after a real retrieval path is implemented.
   The cached rankings belong to known benchmark claims and cannot act as a
   general web index. Coordinate live search with C; until available, show an
   explicit unsupported/insufficient-source state for live mode.

## Data and deployment boundary

For development access, add the shared `10718_averitec` shortcut to your own
My Drive and mount it in Colab if needed. The active data root is
`/content/drive/MyDrive/10718_averitec/v2_source_documents`. Existing rankings and
checkpoints are described in the handoff; do not download the full corpus to
build mock UI. Save new outputs outside shared reference directories.

The browser cannot directly read another user's mounted Colab Drive path.
Model loading, retrieval and inference run behind the backend. Local development
and any persistent deployment are separate deliverables; document what actually
runs and do not assume a Colab session is an always-on service.

## Done when

- Another teammate can load the extension and start the development backend
  using your instructions, including dependencies and environment settings.
- Selected/pasted claim submission and all loading/result/error states work.
- Evidence links, ranking order and the three-item limit are checked.
- Benchmark replay, actual inference and unsupported/live mode are labeled clearly.
- The UI works with the common response schema and at least one integrated adapter,
  or the missing integration is identified explicitly.
- A short demo and self-contained update are available. Do not claim a user-study
  benefit or an ML improvement from a working interface alone.

## Return to the team

Submit a PR, a runnable setup guide and a short demo. Add a new update based on
[the template](../team_updates/TEMPLATE.md), naming the code commit, contract
version, tested flows, mock/real integration status, known limitations and next
dependency. Keep service secrets and generated datasets out of Git. Add reviewed
progress to the shared result record.
