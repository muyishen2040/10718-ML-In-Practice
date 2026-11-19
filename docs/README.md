# Documentation map

Start with the shared handoff, then your task brief. These documents are written
for teammates and agents who have no access to earlier conversations.

| Document | Purpose |
| --- | --- |
| [TEAM_HANDOFF.md](TEAM_HANDOFF.md) | Project introduction, Drive/Colab/Git setup and artifact map |
| [TEAM_PROGRESS.md](TEAM_PROGRESS.md) | Historical report results, later experiments and current limitations |
| [TEAM_WORKPLAN.md](TEAM_WORKPLAN.md) | Current responsibilities, dependencies and integration order |
| [EVALUATION.md](EVALUATION.md) | Chronological split release and common scoring |
| [INTEGRATION_CONTRACT.md](INTEGRATION_CONTRACT.md) | Proposed common interface/backend/model boundary |
| [Colab CPU backend](../apps/COLAB_BACKEND.md) | Shared-Drive model loading and private extension testing on one computer |
| [Interface task](tasks/TEAMMATE_A_INTERFACE.md) | Browser extension and backend integration |
| [Local-method task](tasks/TEAMMATE_B_METHOD.md) | Retrieval/selection and verifier improvements |
| [API-model task](tasks/TEAMMATE_C_API.md) | Hosted verifier comparison and provider adapter |
| [Update template](team_updates/TEMPLATE.md) | One self-contained update per experiment or interface milestone |
| [Historical baseline runbook](archive/BASELINE_RUNBOOK.md) | Earlier README command recipes |
| [Archive index](archive/README.md) | Superseded specifications and runbooks |
| [Script map](../scripts/README.md) | Current, reproduction, historical and deferred entry points |
| [Dependency profiles](../requirements/README.md) | Which environment to install, without root-level profile clutter |
| [Human evaluation plan](HUMAN_EVALUATION.md) | Earlier user-study planning; not proof of a completed study |

Keep task-specific instructions in the task briefs, common evaluation
settings in the evaluation release, and accepted results in the progress record.
An experiment update links to its Drive outputs rather than committing datasets,
weights or large prediction dumps. Review updates before merging them into the
canonical result table.
