# Documentation map

Start with the shared handoff, then your task brief. These documents are written
for teammates and agents who have no access to earlier conversations.

| Document | Purpose | Owner |
| --- | --- | --- |
| [TEAM_HANDOFF.md](TEAM_HANDOFF.md) | Project introduction, Drive/Colab/Git setup and artifact map | Coordinator |
| [TEAM_PROGRESS.md](TEAM_PROGRESS.md) | Historical report results, later experiments and current limitations | Coordinator, incorporating teammate updates |
| [TEAM_WORKPLAN.md](TEAM_WORKPLAN.md) | Current responsibilities, dependencies and integration order | Coordinator |
| [EVALUATION.md](EVALUATION.md) | How the coordinator releases chronological splits and common scoring | Coordinator |
| [INTEGRATION_CONTRACT.md](INTEGRATION_CONTRACT.md) | Proposed common interface/backend/model boundary | Coordinator with A/B/C |
| [Interface task](tasks/TEAMMATE_A_INTERFACE.md) | Browser extension and backend integration | Teammate A |
| [Local-method task](tasks/TEAMMATE_B_METHOD.md) | Retrieval/selection and verifier improvements | Teammate B |
| [API-model task](tasks/TEAMMATE_C_API.md) | Hosted verifier comparison and provider adapter | Teammate C |
| [Update template](team_updates/TEMPLATE.md) | One self-contained update per experiment or interface milestone | Each teammate |
| [Historical baseline runbook](archive/BASELINE_RUNBOOK.md) | Earlier README command recipes | Reference only |
| [Archive index](archive/README.md) | Superseded specifications and runbooks | Reference only |
| [Script map](../scripts/README.md) | Current, reproduction, historical and deferred entry points | Reference by task |
| [Dependency profiles](../requirements/README.md) | Which environment to install, without root-level profile clutter | Reference by workflow |
| [Human evaluation plan](HUMAN_EVALUATION.md) | Earlier user-study planning; not proof of a completed study | Reference only |

Keep task-specific instructions in the three task briefs, common evaluation
settings in the evaluation release, and accepted results in the progress record.
An experiment update links to its Drive outputs rather than committing datasets,
weights or large prediction dumps. The coordinator consolidates updates so every
teammate does not edit the canonical result table at once.
