# Colab notebook catalog

Read [the shared handoff](../docs/TEAM_HANDOFF.md) first for Drive shortcuts,
input/output paths and Git publication status. Many expensive runs already have
saved outputs. A notebook's presence does not establish that every cell was run
or that the file is already available on the remote branch.

| Notebook | When to use it | Existing artifact root under `DATA_ROOT/outputs` |
| --- | --- | --- |
| [evidence_reader_followup_colab.ipynb](evidence_reader_followup_colab.ipynb) | Latest completed weighting/distillation/QA study; inspect or reproduce deliberately | `evidence_reader_followup_v2` |
| [evidence_aware_reader_colab.ipynb](evidence_aware_reader_colab.ipynb) | Original NLI scoring and pair-reader reference, before the later weighting correction; reuse cached inputs | `evidence_aware_reader_v1` |

The other 11 workflows are in [the notebook archive](archive/README.md), with
their purpose and reason for archiving. They include original dense/fusion runs,
projection/weak-supervision studies, QA selection, controls and limitation audits.
Read the archive index and saved run configuration before reproducing one.
Their original source snapshots are preserved. Only external dependency-install
paths in older clone-based setup cells were migrated during the root cleanup;
the archive records that change and its original file hashes.

## New team evaluation

The chronological training split release is pending. Existing portable
notebooks contain historical source/configuration snapshots and do not
automatically consume [the team evaluation pointer](../configs/team_evaluation.json).
Adapt the relevant runner to the published manifest before reporting a result
as the new temporal comparison. Do not silently use a notebook's old random or
grouped holdout because the new release is still pending.

Use a fresh session for a different bundled source/dependency snapshot, change
new output paths to your own Drive folder, and keep shared reference inputs
unchanged. New notebooks should retain progress bars, resumable journals and
input/configuration binding. New hosted-model runs need their own runner audit;
the legacy API script is not yet a turnkey full-population, interruption-safe
benchmark. See [C's brief](../docs/tasks/TEAMMATE_C_API.md).

The two reader reference notebooks retain their frozen self-contained source
and dependency files. Their legacy virtual dependency filename is restored
inside the temporary runtime and does not require a root file in this checkout.
Current builders now bundle `requirements/reader-colab.txt`. A rebuilt notebook
gets a new source hash and needs its own run directory.

The [script map](../scripts/README.md) distinguishes active building blocks from
reproduction, historical experiments and deferred external-data commands.
