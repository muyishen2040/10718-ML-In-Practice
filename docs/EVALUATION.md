# Shared evaluation release

The released evaluation uses time-based splits inside the official AVeriTeC
training set. Local and hosted models must use the same published split and
evaluation entry point. Do not invent independent replacement splits for the
team's comparable result table.

The release pointer is [configs/team_evaluation.json](../configs/team_evaluation.json).
Protocol `trust_temporal_20261117_v1` was released on November 17, 2026 on
`feat/part1-temporal-evaluation`. Its manifest is saved under
`processed/averitec/team_splits/trust_temporal_20261117_v1/split_manifest.json`
in shared Drive. The pointer records its SHA-256. Existing notebooks require
explicit adaptation to consume it.

## Released windows and runners

- `spring_2020` uses 1,411 fit claims through November 8, 2019, 355 validation
  claims from November 9 through February 29, 2020, and 575 evaluation claims
  from March 1 through June 3, 2020.
- `summer_2020` uses 1,870 fit claims through March 15, 2020, 468 validation
  claims from March 16 through June 3, and 611 evaluation claims from June 4
  through August 25, 2020.
- The manifest quarantines 105 missing dates and nine pre-2000 date values.
  It preserves their IDs and raw values. Earlier copies of normalized claim
  text are purged when they cross partition boundaries. Date ties stay together.
- `scripts/run_part1_revision.py` prepares the release, recalculates historical
  dev metrics, trains logistic regression and DistilBERT, and summarizes results.
  `scripts/score_trust_predictions.py` validates coverage and scores a released
  partition. Historical reader notebooks still need manifest integration.

Select hyperparameters and Supported-score thresholds on each fold's validation
partition before future-window prediction. The first revision evaluates the
selected fit-trained checkpoint directly, retaining validation as held out.
Rolling windows share historical training data. Report each window separately.

## Checklist for subsequent releases

1. Audit training dates, missing dates, duplicate groups and per-class counts.
   Define two earlier-to-later windows initially, adjusting date boundaries to
   provide useful coverage of the rare labels. Describe ties, missing dates and
   any excluded claims. Keep a duplicate group from crossing fit/validation.
2. Save an immutable manifest in a new versioned Drive directory. A suggested
   location is `processed/averitec/team_splits/PROTOCOL_ID/split_manifest.json`
   relative to `DATA_ROOT`; this directory is proposed and not yet populated.
3. Include a protocol ID, hash of the normalized source claims, per-fold fit and
   validation claim IDs, date boundaries, class counts and exclusion policies.
   Validate no duplicate/unknown IDs, no overlap and the claimed time order.
   A later window may legitimately have a larger training prefix; do not pool
   overlapping validation periods as independent examples without explaining it.
4. Provide an evaluation entry point that joins predictions to the released IDs,
   validates claim coverage and labels, and reports the common metrics. Document
   its exact command and prediction schema. Make each relevant runner consume
   this manifest rather than recreating a random/grouped split internally.
5. Update the pointer in Git with `status: "ready"`, a concrete `protocol_id`,
   `split_manifest` object containing `relative_path` and `sha256`, concrete
   `fold_ids`, and the executable `evaluation_command`. Publish the commit and
   notify teammates of the file path and version.
6. If a release changes, issue a new protocol ID/path/hash and preserve the old
   results. Do not overwrite the same manifest while teammates are running it.

Before release, agree on the manifest schema and loader across model runners.
The revision runner consumes the published manifest. The pair-reader CLI still
needs manifest support before it can train a comparable temporal model.

## Loading the published path in Colab

After mounting the shared Drive and checking out the published repository, this
cell resolves the common manifest. It does not train a model or implement the
models. Keep `DATA_ROOT` as a shared input and `OUT` personal.

```python
import hashlib
import json
from pathlib import Path

DATA_ROOT = Path('/content/drive/MyDrive/10718_averitec/v2_source_documents')
release = json.loads(Path('configs/team_evaluation.json').read_text())
if release['status'] != 'ready':
    raise RuntimeError('The shared evaluation release is not ready yet.')
reference = release['split_manifest']
manifest_path = DATA_ROOT / reference['relative_path']
if not manifest_path.is_file():
    raise FileNotFoundError(manifest_path)
digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if digest != reference['sha256']:
    raise ValueError('Split manifest hash differs from the published release.')
print('Protocol:', release['protocol_id'])
print('Folds:', release['fold_ids'])
print('Manifest:', manifest_path)
print('Evaluation command:', release['evaluation_command'])
```

For a portable notebook restored outside the checkout, load the pointer from
the actual repository checkout or a versioned copy at the same Git
revision. Do not assume bundled historical notebooks contain this new file.

## Scoring and leakage rules

- Per temporal fold, fit models, vocabularies, class weights and learned selectors
  using only the fit IDs. An existing checkpoint trained on all 3,067 claims has
  already seen the proposed internal validation claims; retrain for an honest
  fold comparison. A pretrained foundation model is a separate initialization.
- Tune prompts/hyperparameters on the designated training-validation protocol,
  freeze them, and record all trials. Apply the same protocol to local and API
  comparisons. Time splits control project training data, not what an external
  foundation model may have seen during pretraining.
- Retain the fixed 500 official dev claims for the historical end-to-end
  comparison. Official test remains untouched. Date-window results and original
  dev results are separate tables with explicit populations.
- Primary metric is Supported-versus-other binary balanced accuracy, the mean
  of Supported acceptance recall and other-claim withholding recall. Report
  four-class macro-F1 and accuracy as secondary diagnostics. Conditional FR
  divides by non-Supported claims and conditional FA divides by Supported
  claims. Retain historical all-claim rates with their original labels.
  Report missing predictions and API errors explicitly.
- The display budget is three passages. URL-proxy retrieval uses the original
  386 judged dev claims; MRR@3 and MRR over top 20 must remain separately named.
  Gold-answer controls and web-assisted runs are separate evidence settings.

Every update records the protocol ID, manifest hash, fold ID, input rankings,
code/model/prompt version and exact artifact links. Review teammate updates
before adding accepted results to the canonical score table.

## Existing prediction format and scorer

Historical prediction files are JSONL, one completed prediction per claim.
The required scoring fields are `claim_id`, `true_label` and `predicted_label`.
`evidence_passage_ids` records the supplied evidence. Neural models may also
provide `probabilities`, a normalized map over all four canonical labels;
an LLM's self-reported confidence is not this probability map.

The product response calls the prediction `verdict`; its evaluation adapter
maps that to `predicted_label`. Join `true_label` from the evaluation claims
after inference. Do not send true labels or annotated answers to a verifier.
Keep failed requests in a separate error journal until resolved.

[evaluate_predictions](../src/evaluation/classification.py) computes the existing
metrics from completed rows. It does not check the expected claim-ID population
or detect duplicate IDs on its own. The caller must validate exact coverage,
unique IDs and correct gold-label joins before reporting a full-population
score. `scripts/score_trust_predictions.py` enforces these checks and joins
gold labels outside inference.
