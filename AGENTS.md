# Instructions for project agents

This repository builds an evidence-grounded misinformation screening assistant
for CMU 10-718. Given a claim, it shows up to three evidence passages and one of
four verdicts: Supported, Refuted, Not Enough Evidence, or Conflicting Evidence.
The user makes the final trust or sharing decision.

## Read before starting

1. [Team handoff](docs/TEAM_HANDOFF.md): access, Colab setup, Git workflow,
   artifact locations, and the current bottlenecks.
2. [Progress and results](docs/TEAM_PROGRESS.md): distinguish the submitted
   report from later experiments and distinguish findings from hypotheses.
3. [Team work plan](docs/TEAM_WORKPLAN.md): identify the assigned
   workstream and its dependencies before changing shared interfaces.
4. Open the assigned brief: [A: interface](docs/tasks/TEAMMATE_A_INTERFACE.md),
   [B: main method](docs/tasks/TEAMMATE_B_METHOD.md), or
   [C: API models](docs/tasks/TEAMMATE_C_API.md).

The work split is A: browser extension/interface, B: local ML method,
and C: hosted API models. Use the shared evaluation release for comparisons;
do not create independent replacement splits.

The task prompt should name your role: A, B or C. If it does not,
complete the handoff's first-session checklist and ask which role to own before
changing shared code. Use that checklist to report your source version, data
access, first deliverable and any missing dependency; no earlier chat is needed.

Read [the evaluation release guide](docs/EVALUATION.md) and
`configs/team_evaluation.json` before new comparative training or prompt selection.
While its status is pending, continue setup, mocks and recorded-mechanics checks.
Historical notebooks do not automatically consume this new pointer. Use
[the common integration contract](docs/INTEGRATION_CONTRACT.md) for new product
adapters. Script entry-point paths are stable; older experiment notebooks are
under `notebooks/archive/`, indexed by its README. Put new product code under
`apps/` and shared ML/evaluation logic under `src/`.

These files are the current onboarding entry point. The root contains only
`README.md`, `AGENTS.md` and the base `requirements.txt` entry point as visible
files. Historical guides are in `docs/archive/`; dependency profiles are under
`requirements/`. Private notes/report drafts are preserved under ignored `local/`.
Resolve status against current code, saved run configurations, and result JSONs.
Do not delete a script merely because its method failed: tests and other runners
still import helpers from historical entry points. Consult `scripts/README.md`.

Portable reference notebooks are frozen self-contained snapshots. Their old
virtual dependency filenames are restored inside their runtime, not at the repo
root. Current builders use the new dependency paths. Do not regenerate a frozen
experiment merely to reorganize files; validate current generator output separately.

## Keep knowledge portable

- Write for a teammate and agent with no access to previous conversations.
  Define the dataset split, model, evidence input, metric denominator, and
  artifact location whenever reporting a result. Explain what an experiment
  tested and what its results do and do not establish.
- Record each completed or interrupted experiment using
  [the update template](docs/team_updates/TEMPLATE.md). Use a new update file
  per experiment. After review, add accepted findings to `docs/TEAM_PROGRESS.md`
  and update the handoff when setup or locations change.
- Include the Git commit or portable-notebook source hash, run configuration,
  owner, stable Drive link, relative artifact paths, and next action. Do not
  describe an experiment as completed without its outputs.
- Confirm a branch exists remotely before telling teammates to check it out.
  A local commit or an embedded notebook does not imply publication on GitHub.

## Preserve comparable experiments

- Read shared inputs from `10718_averitec/v2_source_documents`. Write new outputs
  to a separate, owner-specific run directory. Existing datasets, rankings,
  checkpoints and completed experiments are reference artifacts.
- Use official train labels for fitting. Record selection on training-only
  validation before official-dev evaluation. Official dev has been inspected
  repeatedly and is exploratory; official test remains untouched.
- Keep the same 500 official dev claims for primary verdict comparisons and
  the original 386 URL-judged claims for retrieval comparisons. Report the
  evaluation population when using any additional subset.
- URL overlap and extractive answerability do not establish decisive passage
  relevance. Gold-answer controls are unavailable at prediction time. Feeding
  a frozen BM25-trained verifier different dev passages is an input intervention,
  not a verifier trained on those retrieval outputs.
- Preserve progress bars, recoverable checkpoints, input/configuration checks,
  and small smoke tests in new Colab experiments. Record runtime and memory
  limits instead of implying that an unrun experiment succeeded.
- Keep fixed-evidence hosted-model comparisons separate from web-assisted runs.
  A live browser claim needs an implemented retrieval path; benchmark rankings
  cannot be treated as a general search index for arbitrary claims. Label mock,
  benchmark-demo and live results clearly in the interface and reports.
