# Historical experiment notebooks

These notebooks are retained for reproducibility, failed hypotheses and report
provenance. They were first moved here byte-for-byte on November 4, 2026.
The second cleanup updated only external dependency-install paths in ten setup
cells after profiles moved into `requirements/`. Original/new file hashes and
changed cell IDs are recorded in [dependency_path_migration.json](dependency_path_migration.json).
Model/prompt code, metadata and portable embedded snapshots were not changed.
All 11 had empty saved output cells at the time of archiving. Completed results
remain in Drive and [the progress record](../../docs/TEAM_PROGRESS.md).

They are not the default workflow for the next team iteration. Start with
[the main catalog](../README.md), [task briefs](../../docs/README.md) and the
coordinator's [evaluation release](../../docs/EVALUATION.md).

| Notebook | Why it is historical |
| --- | --- |
| [initial_ml_solution_colab.ipynb](initial_ml_solution_colab.ipynb) | Original four-experiment workflow, superseded by later rerun/runtime fixes |
| [initial_ml_solution_colab_rerun.ipynb](initial_ml_solution_colab_rerun.ipynb) | Completed initial dense/fusion/verifier workflow; full dense retrieval is expensive |
| [initial_ml_solution_followup_colab.ipynb](initial_ml_solution_followup_colab.ipynb) | Earlier reranking and verifier controls, including reported claim-only/gold results |
| [initial_ml_solution_next_colab.ipynb](initial_ml_solution_next_colab.ipynb) | Earlier ablations and source-diversity/weak-supervision exploration |
| [initial_ml_solution_supervised_rerank_colab.ipynb](initial_ml_solution_supervised_rerank_colab.ipynb) | Earlier projection, weak-supervision and answer-aware candidate methods |
| [initial_ml_solution_answer_aligned_top100_colab.ipynb](initial_ml_solution_answer_aligned_top100_colab.ipynb) | Completed larger-pool BM25 and silver answer-alignment study; cached rankings are reusable |
| [initial_ml_solution_question_guided_colab.ipynb](initial_ml_solution_question_guided_colab.ipynb) | Earlier question-guided evidence selection; generated-question caches are reusable |
| [initial_ml_solution_answer_aware_colab.ipynb](initial_ml_solution_answer_aware_colab.ipynb) | Earlier extractive-QA/selector proposal; presence is not proof that every experiment ran |
| [initial_ml_solution_dense_fusion_comparison_colab.ipynb](initial_ml_solution_dense_fusion_comparison_colab.ipynb) | Earlier attempt to complete dense/fusion verifier comparisons; do not infer matched full-train runs from its title |
| [initial_ml_solution_fast_bge_verifier_swap_colab.ipynb](initial_ml_solution_fast_bge_verifier_swap_colab.ipynb) | Completed frozen-verifier evidence-input comparison used in the report |
| [initial_ml_solution_limitation_study_colab.ipynb](initial_ml_solution_limitation_study_colab.ipynb) | Earlier bounded limitation audit/inference/training study; follow its saved artifacts for completed cells |

Older code may assume old dependency versions or hard-coded output directories.
Use a fresh runtime and new personal output path for an intentional reproduction.
Do not overwrite a finished shared run. Historical source snapshots and holdouts
do not automatically adopt the coordinator's new time-based splits.

Relevant script helpers remain at stable paths because tests and other runners
import them. Archiving the notebooks does not make those helpers safe to delete.
