# Interface update: private Colab CPU backend

- Workstream: extension/backend integration.
- Date: November 10, 2026.
- Source branch: `feat/colab-extension-backend`, based on interface commit
  `99eadf5c59b80b5ef298281b171b973930689a6f`. The commit introducing this update
  records the implementation. Notebook metadata contains its bundled source hash.
- Scope: serving the existing BM25-trained DistilBERT on known dev claims.
  No fitting, model selection or new retrieval was performed.

## Inputs and availability

The mounted shared input root is
`/content/drive/MyDrive/10718_averitec/v2_source_documents`.
Only the saved baseline's `final_model` and `transformer_run_config.json` under
`outputs/initial_ml_solution_rerun_bs16/transformer/bm25_top3/`, and
`outputs/bm25/dev_bm25_top20_v2/dev_rankings.jsonl`, are needed. No gold labels,
training answers, qrels or full evidence corpus are read for inference.

The roughly 283-MB bundle is copied to an atomic, hash-checked runtime cache.
Interrupted copies can be resumed. Existing shared artifacts are unchanged.
The model runs on CPU with two Torch threads; the exercised runtime used
PyTorch 2.14.1+cpu and Transformers 4.57.1.

## Implementation and reproduction

- [Guide](../../apps/COLAB_BACKEND.md) and
  [portable notebook](../../notebooks/extension_backend_cpu_colab.ipynb).
- `apps/backend/colab_runtime.py`: loopback-only server, model loading,
  input staging and repeatable start/stop.
- `apps/backend/colab_relay.py`: bounded requests through authenticated
  `colab exec`, local claim-catalog search, timeouts and managed shutdown.
- `scripts/colab_extension.ps1`: Windows/Ubuntu-WSL preparation, relay, status
  and stop commands. Colab CLI 0.7.5 is installed in an isolated environment.
- Neither Colab credentials nor Drive files are included in the extension.
  The existing extension localhost permissions are sufficient.
- No artificial keepalive, public tunnel or participant event logger is added.

## Verification

The project suite passed 125 tests, with one optional real-model test skipped
because local model paths were not configured. After the shutdown helper was
added, the focused backend suite passed 20 tests with the same skip.
Tests cover copy interruption/corruption, input preservation, model reuse,
private routes, request serialization, catalog caching, portable source hashes
and local shutdown-token validation.

Separately, the actual shared-Drive model was loaded in a paid Colab CPU runtime.
Requests from Windows through the local relay produced:

| Request | Result |
| --- | --- |
| Claim catalog | 500 known dev claims; cached search works locally |
| `averitec:dev:00000` | Completed, Refuted, three evidence passages |
| `averitec:dev:00001` | Completed, Refuted, three evidence passages |
| `averitec:dev:00417` | `insufficient_sources`, no verdict/evidence |
| Arbitrary live claim | Unavailable, no attached evidence or verdict |

The tested verification requests took about 1.4–2.6 seconds including CLI
transport. The initial notebook smoke inference took about 0.7 seconds inside
Colab. These are a few engineering checks, not a representative latency or
classification evaluation. Model-correctness claims require the existing
benchmark evaluation rather than interpreting these outputs as new scores.

## Remaining user actions and limitations

Load/reload the unpacked extension in Chrome and select Benchmark mode. Confirm
the baseline-model provenance badge and evidence links in the actual popup.
The installed-Chrome flow for this private relay has not been automated here.
For a controlled study, select claims and record assisted/unassisted judgments,
confidence and timing separately; this setup does not collect those responses.

Colab may disconnect; its normal paid-runtime policies and limits still apply.
The relay is private to the testing computer. Additional participant computers
need a separately planned connection; do not share Google credentials.
After testing, use the Stop action to release the runtime and managed relay.
