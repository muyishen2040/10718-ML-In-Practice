# Team work plan

Updated November 5, 2026. Work covers the interface, local ML improvements and
hosted API experiments, using shared evaluation and integration rules.
These are work assignments,
not claims that the implementations or new experiments are complete.

Read [the shared handoff](TEAM_HANDOFF.md) for access and artifact locations,
then [current results](TEAM_PROGRESS.md), then your task brief.

## Responsibilities and source locations

| Owner | Task brief | Code area | First deliverable |
| --- | --- | --- | --- |
| Teammate A | [Interface / extension](tasks/TEAMMATE_A_INTERFACE.md) | `apps/browser-extension/`, `apps/backend/` (planned) | Mock user flow and backend boundary, then one real adapter |
| Teammate B | [Main ML method](tasks/TEAMMATE_B_METHOD.md) | `src/retrieval/`, local-reader modules in `src/verification/`, method scripts/notebooks | Reproduce the raw-reader reference, then one bounded selection/reader improvement |
| Teammate C | [Hosted API models](tasks/TEAMMATE_C_API.md) | Provider modules in `src/verification/`, API scripts/notebooks | Reliable same-evidence pilot, then metric/cost/latency comparison |

Suggested task branches are `team/teammate-a-interface`, `team/teammate-b-method`
and `team/teammate-c-api`, each from the same remotely verified shared base.
The reader code and team documents are merged into the shared `main` branch.
The handoff records the tested code snapshot and checkout instructions. Record
the exact commit used for each task.

Script paths are preserved; historical notebook locations are indexed in
[`notebooks/archive`](../notebooks/archive/README.md). New product code follows
[apps/README.md](../apps/README.md); experimental model/evaluation code stays
reusable under `src/` with thin script entry points. Coordinate changes to shared
interfaces instead of creating incompatible private copies.

## Shared evaluation release

The planned release splits official train by time, stores the manifest in
shared Drive and exposes its path, hash and fold IDs through
[configs/team_evaluation.json](../configs/team_evaluation.json). The current
status is **pending**. [EVALUATION.md](EVALUATION.md) defines the release and
the instructions B/C use when it becomes ready.

Initially plan two earlier-to-later training-validation windows. Date boundaries,
missing dates, duplicate groups and rare-class coverage must be checked before
publication. Keep original official dev and test separate.
Existing full-train checkpoints are not independent baselines on an internal
validation subset they already trained on.

Teammates should not create their own replacement temporal splits. Existing
portable notebooks do not automatically read the new release pointer. The
manifest and evaluator support must be agreed across model runners before new
scores are described as the agreed temporal comparison.

## Parallel work and integration milestones

1. **Prepare independently:** A builds mock UI, B inspects and reproduces existing
   model mechanics, and C builds provider mocks/parsing/resume support. Confirm
   shared data access and check evaluation readiness before comparative runs.
2. **Use the released evaluation:** B trains from the released fit IDs and C
   runs bounded prompt/model experiments on the same protocol. A connects a
   benchmark backend using the agreed [integration contract](INTEGRATION_CONTRACT.md).
3. **Compare before combining:** keep BM25's three supplied passages fixed to
   compare local and hosted verifiers. If B's selector helps, compare both
   verifiers on its passages as a separately recorded input change.
4. **Integrate the product:** A connects the selected adapters. Arbitrary live
   claims require an implemented retrieval path, distinct from cached benchmark
   claim pools. Coordinate optional live retrieval with C and label it separately.
5. **Consolidate:** review results, errors, cost/latency and the
   demo for the report. A working interface does not establish a user-study
   benefit, and a stronger dev ablation is not retroactive train-only selection.

## Return and review

Each teammate returns a PR and one update file per experiment or interface
milestone using [the template](team_updates/TEMPLATE.md). Include code/source
version, release protocol/hash, population, run settings, stable Drive links,
metrics or tested flows, runtime/cost, failures and remaining work. Keep datasets,
weights, secrets and large raw outputs outside Git.

After review, add accepted updates to [TEAM_PROGRESS.md](TEAM_PROGRESS.md)
and change the handoff when artifact locations or setup change. Keep result-table
changes in small PRs to avoid conflicting edits. New run outputs use
personal owner-specific directories; shared completed artifacts remain inputs.

The cleanup, reader follow-up and handoff are published on `main`.
The split release remains pending; no extension deployment or new paid
API experiment is established by this publication.
