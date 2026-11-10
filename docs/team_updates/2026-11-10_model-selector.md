# Benchmark model selector

- Workstream: A, extension and local-model integration.
- Completed November 10, 2026, on `feat/colab-extension-backend`.
- Portable notebook source hash:
  `b46efdb1896b46c1ba817de39a57ca3da6a4a085017365c6dcaa18dd1c51611b`.
- This changes serving and interface behavior; it does not train models or
  select a new winner on official dev. The shared temporal evaluation release
  remains pending.

## What is available

The Benchmark popup now offers DistilBERT + BM25 (default), DistilBERT + BGE,
DistilBERT + BM25/BGE fusion, and Corrected NLI reader. The first three share
the original BM25-trained classifier and use different saved dev evidence.
The fourth loads `raw_sample_d01` from the reader follow-up study. It is a
safety-oriented experimental alternative, not that study's train-only winner.
These options cover the same 500 official dev claims and display up to three
passages. They do not verify arbitrary new browser claims.

Mode, selected pipeline and backend URL are stored locally by Chrome. Switching
models preserves the claim and clears its old result. Delayed responses from
an earlier selection are ignored. The response badge identifies the pipeline.
Unavailable options are disabled, and explicit unavailable/unknown requests
return errors rather than predictions from another model.

## Inputs and reproduction

Follow [the Colab backend guide](../../apps/COLAB_BACKEND.md), which lists all
relative artifact paths. Inputs are read from the shared
[`10718_averitec/v2_source_documents` folder](https://drive.google.com/drive/folders/19tuNxxB1y9arSmLnrQcxRuPvDQBf6YeK).
Only model/tokenizer/config files and saved dev evidence are staged to the CPU
runtime cache. Shared inputs remain unchanged. The NLI adapter checks its
preparation and evidence hashes against `reader_run.json`, uses the original
token budgets and prediction-time quality features, and supplies no gold label.
Gold-answer and claim-only controls are excluded from the menu.

Use `notebooks/extension_backend_cpu_colab.ipynb` and
`scripts/colab_extension.ps1 -Action Prepare`, then `-Action Relay`.
After a source update, restart the kernel before Prepare to avoid mixing Python
module versions. Cached files and the mounted Drive can be reused. Models run
on CPU with two threads; requests share an inference lock, while prediction
caches are separate by pipeline. Stop the named runtime when testing ends.

The API adds `GET /models` and optional benchmark `model_id` in `POST /verify`.
Existing requests retain the BM25 default. See
[the integration contract](../INTEGRATION_CONTRACT.md).

## Checks and limits

Live CPU requests exercised all four choices on dev claims `00000` and `00015`.
Returned evidence IDs and provenance changed with the selected pipeline. NLI
verdicts and evidence IDs matched the saved predictions for both checked claims.
This is a serving reproduction check, not a new accuracy estimate.
All four choices also returned a null verdict and `insufficient_sources` for
the empty-pool claim `00417` through the local relay. The full project suite
passed 130 tests with one optional saved-model test skipped; real saved-model
inference was checked separately in Colab as described above.

Browser automation used the actual popup files, a local same-origin test proxy
to the running relay, and a Chrome-storage shim. It checked dropdown choices,
request model IDs, matching result badges, persistent preferences, preserving
the claim, clearing old results, disabled unavailable options and ignoring late
responses. This does not replace reloading and testing the installed extension
in Chrome. No participant study, new training or model-performance improvement
is claimed.
