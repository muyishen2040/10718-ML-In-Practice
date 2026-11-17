# Part 1 revision experiments: interrupted run

Stopped on November 17, 2026 at the user's request to avoid leaving a paid
Colab runtime running near the Codex usage limit. The CLI confirmed that
`misinfo-part1-revision` was terminated. No replacement runtime was started.

Code: branch `feat/part1-temporal-evaluation`, release commit `f453171`.
Training source snapshot SHA-256:
`06a8d8edaabebe2de940d56855664248eba4c2838f13af05aea7bf84436c675b`.
The project test directory passed: 110 tests. An earlier unrestricted pytest
invocation collected third-party packages under ignored local dependencies;
use `pytest tests`, not unrestricted discovery.

## Saved work

Shared inputs are under the mounted
`/content/drive/MyDrive/10718_averitec/v2_source_documents`, accessible from the
[shared Drive folder](https://drive.google.com/drive/folders/19tuNxxB1y9arSmLnrQcxRuPvDQBf6YeK).
Original claims, rankings and models were preserved.

New outputs are in the personal directory
`/content/drive/MyDrive/10718_runs/muyi/report_part1_20261117_v1`.
This directory is separate from shared inputs; teammates need access before
attempting to read it.

- Released manifest: shared-input relative path
  `processed/averitec/team_splits/trust_temporal_20261117_v1/split_manifest.json`.
  SHA-256: `af2d5436167ed39f658255c099b41acab16cdd1ecc89e25e42a95bfd0af20d5e`.
- `historical_metrics.json` and `.csv`: completed recalculation of all 11
  report rows on the same 500 official dev claims, using binary balanced
  accuracy for Supported versus all other labels. No model was retrained for
  this historical table. Official test remains untouched.
- `temporal/spring_2020/logistic/`: three completed regularization trials,
  saved models, validation predictions, the frozen selection record,
  evaluation predictions and `evaluation_complete.json`.
- Transformer training had started but no completed transformer trial was
  observed before shutdown. Treat it as unfinished until checking Drive.
  The summer window was not reached. The aggregate summarize stage was not run.

## Completed spring logistic-regression result

Training uses BM25's top three passages for each claim. Fit/validation/future
evaluation populations are 1,411 / 355 / 575 claims. The future window is
March 1 through June 3, 2020, with 97 Supported and 478 other claims.
All vectorizer fitting and classifier training use only fit claims.

Validation selected regularization `C=10` and a Supported-score threshold of
0.25 from `C={0.1,1,10}` and argmax plus 17 thresholds from 0.10 to 0.90.
The choice was saved before future-window scoring; no refit was performed.

| Decision policy on the selected model | Binary balanced accuracy | Binary accuracy | False reassurance / 478 | False alarm / 97 |
| --- | ---: | ---: | ---: | ---: |
| Four-way argmax | 0.5841 | 0.8209 | 28 | 75 |
| Validation-selected threshold 0.25 | 0.6229 | 0.7009 | 124 | 48 |

Balanced accuracy averages the Supported acceptance recall and other-claim
withholding recall. Thresholding improved this metric by 3.88 percentage points,
but increased false reassurance substantially. Equal weighting of these two
class recalls does not encode a strict false-reassurance safety constraint.
This is one completed window for one model, not the final model comparison.

## Resume

Create a fresh GPU runtime and authorize Drive again. Check out the release
branch. Install the recorded dependencies from `experiment_config.json`, restore
the same source files, and cache the same normalized train claims and BM25 train
rankings locally. The runner rejects changes to bound inputs/configuration.

With `DATA_ROOT` pointing to that local cache (same relative input paths),
`SHARED_ROOT` to the mounted shared inputs, and `OUT` to the personal output
directory above, run:

```bash
python scripts/run_part1_revision.py train --data-root "$DATA_ROOT" --shared-root "$SHARED_ROOT" --output "$OUT"
python scripts/run_part1_revision.py summarize --data-root "$DATA_ROOT" --shared-root "$SHARED_ROOT" --output "$OUT"
```

Completed trials/evaluations are reused from Drive. Current-trial epoch
checkpoints were runtime-local, so an unfinished transformer trial restarts
after VM termination. Do not overwrite the manifest or saved historical outputs.
Inspect coverage and selection records before reporting any additional result.
