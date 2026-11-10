# Misinformation Screening Assistant

CMU 10-718 project: given a factual claim, retrieve up to three evidence passages
and predict Supported, Refuted, Not Enough Evidence, or Conflicting Evidence.
The user reviews the sources and makes the final trust or sharing decision.

## Start here

Use the shared [`main` branch](https://github.com/muyishen2040/10718-ML-In-Practice/tree/main).
It includes the reader follow-up, repository cleanup and team handoff.

1. Read [the shared handoff](docs/TEAM_HANDOFF.md) for Drive access, Colab setup
   and verified artifact locations.
2. Read [current results and limitations](docs/TEAM_PROGRESS.md). The submitted
   report and later exploratory experiments are recorded separately.
3. Open your task brief:

| Owner | Task brief | Main responsibility |
| --- | --- | --- |
| Teammate A | [Interface / browser extension](docs/tasks/TEAMMATE_A_INTERFACE.md) | User flow, results panel and shared backend integration |
| Teammate B | [Main ML method](docs/tasks/TEAMMATE_B_METHOD.md) | Evidence selection and local verifier improvements |
| Teammate C | [Hosted API models](docs/tasks/TEAMMATE_C_API.md) | Same-evidence provider comparisons, cost and latency |

Agents should also read [AGENTS.md](AGENTS.md). The
[work plan](docs/TEAM_WORKPLAN.md) explains dependencies and ownership.

## Data and evaluation

Code lives in Git. The dataset, saved retrieval rankings and model checkpoints
live in [the shared Drive folder](https://drive.google.com/drive/folders/19tuNxxB1y9arSmLnrQcxRuPvDQBf6YeK).
After adding its shortcut to your own My Drive, the active Colab data root is:

```text
/content/drive/MyDrive/10718_averitec/v2_source_documents
```

Use saved rankings when possible; the development passage corpus alone is about
22.85 GB. Save each new run to a separate personal output directory. See the
handoff before running dataset preparation or full dense retrieval.

Chronological splits inside official train are planned.
[configs/team_evaluation.json](configs/team_evaluation.json) is the single
release pointer for their path, hash and fold IDs. Its current status is
**pending**; it does not invent a completed split. Existing notebooks do not
automatically consume this new pointer. Follow [the evaluation guide](docs/EVALUATION.md)
once a ready release is published.

All primary historical verdict results use the same 500 official dev claims.
Dev has been inspected repeatedly; official test stays untouched. Compare hosted
and local verifiers on identical supplied evidence before adding live web search.

## Repository map

```text
AGENTS.md                  Agent entry point and collaboration rules
apps/                      Prototype extension, development backend and Colab setup
configs/                   Baseline protocol and pending team-evaluation pointer
docs/                      Shared setup, results, contracts, task briefs and updates
notebooks/                 Current reader references; historical workflows in archive/
scripts/                   Stable entry points; purpose/status map in scripts/README.md
src/data/                  Normalization, schemas and validated evidence joins
src/retrieval/             BM25, evidence quality and selection helpers
src/verification/          Verifiers, training and cached inference
src/evaluation/            Splits, classification metrics and paired comparisons
tests/                     Existing automated checks
requirements/              Dependency profiles and installation guide
local/                     Private draft notes/reports; ignored by Git
outputs/                   Local generated artifacts; ignored by Git
```

Place new extension/backend code according to [apps/README.md](apps/README.md).
Use [the proposed integration contract](docs/INTEGRATION_CONTRACT.md) so local
and hosted models can share one interface. Cached benchmark claims and arbitrary
live browser claims require different retrieval paths.

## Dependencies

The root `requirements.txt` installs the lightweight base environment:

```bash
python -m pip install -r requirements.txt
```

Neural-reader and older Colab/local-LLM profiles live under
[`requirements/`](requirements/README.md). Use the profile for the chosen workflow
instead of installing all environments together. Portable notebooks retain their
bundled runtime setup independently of root dependency filenames.

## Run and return work

- Check the published branch/commit in the handoff before creating a task branch.
  Local code or an embedded notebook is not proof that a remote branch exists.
- Use [the notebook catalog](notebooks/README.md) and the chosen workflow's
  dependency file. A portable notebook restores its own source; use a fresh
  runtime rather than mixing it with an older checkout's imports.
- Return a PR and an update based on [the template](docs/team_updates/TEMPLATE.md),
  including code version, split/evidence settings, results, runtime and stable
  Drive links. Keep datasets, weights, credentials and large raw outputs out of Git.

## Earlier commands and specifications

[Baseline runbook](docs/archive/BASELINE_RUNBOOK.md), [COLAB.md](docs/archive/BASELINE_COLAB.md),
[implementation plan](docs/archive/INITIAL_IMPLEMENTATION_PLAN.md), and
[evidence-reader design](docs/archive/EVIDENCE_READER_DESIGN.md) preserve earlier instructions.
The old root guide files have been removed; their full contents remain in
`docs/archive/`. The current handoff and result record define progress and ownership.
Script paths remain stable. Eleven older notebooks are in `notebooks/archive/`;
the notebook catalog links to their new locations. Tests and historical source
snapshots are retained for reproducibility.

Local automated checks, after installing the relevant workflow dependencies:

```bash
python -m pytest
```
