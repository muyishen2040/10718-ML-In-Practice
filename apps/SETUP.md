# Run the prototype (development only)

For the shared-Drive CPU session and private local relay, use
[the Colab backend guide](COLAB_BACKEND.md). It includes a portable notebook,
Windows/WSL commands and extension setup without a public tunnel.

Status: mock interface, a fixture-replay backend, and an optional adapter that runs
the saved BM25-top-3 DistilBERT baseline on AVeriTeC dev claims. No retrieval for
new claims and no hosted API is connected. Contract: [INTEGRATION_CONTRACT.md](../docs/INTEGRATION_CONTRACT.md).

## Requirements

Chrome or another Chromium browser; Python 3.9+. The backend uses only the
standard library, and the extension has no build step. Set no environment
variables and no secrets are needed.

## Start the backend

```bash
python3 apps/backend/server.py            # http://127.0.0.1:8000
python3 -m unittest discover -s apps/backend/tests
```

Routes: `GET /health`, `GET /benchmark/claims`, `POST /verify`.

## Load the extension

1. Open `chrome://extensions`, enable Developer mode.
2. Choose **Load unpacked** and select `apps/browser-extension/`.
3. Select text on any normal web page, click the extension icon, press
   **Check this claim**. Chrome's own pages block selection; paste instead.

## Run the real baseline on benchmark claims (optional)

Download these shared-Drive files (path relative to `10718_averitec/v2_source_documents/outputs`)
into the git-ignored `local/baseline/`:

| File | Drive path | Size |
| --- | --- | --- |
| `final_model/{model.safetensors,config.json,tokenizer.json,tokenizer_config.json,vocab.txt,special_tokens_map.json}` | `initial_ml_solution_rerun_bs16/transformer/bm25_top3/final_model/` | ~269 MB |
| `transformer_run_config.json` | `initial_ml_solution_rerun_bs16/transformer/bm25_top3/` | 1 KB |
| `dev_rankings.jsonl` | `bm25/dev_bm25_top20_v2/` | 14 MB |

Skip the `checkpoint-*` subfolders. Then:

```bash
python3 -m venv --system-site-packages local/venv-backend
local/venv-backend/bin/pip install -r apps/backend/requirements-model.txt
local/venv-backend/bin/python apps/backend/server.py \
  --baseline-dir local/baseline --dev-rankings local/baseline/dev_rankings.jsonl
```

Needs PyTorch (already in many environments; otherwise `pip install torch`). Inference
runs on CPU. The popup's Benchmark mode then searches the 500 dev claims. To run the
real-model test: `BASELINE_DIR=local/baseline DEV_RANKINGS=local/baseline/dev_rankings.jsonl`
before `python -m unittest discover -s apps/backend/tests`.

What this is: the saved baseline (macro-F1 0.347, accuracy 0.52 on the 500 official
dev claims) predicting from each claim's saved BM25 top-three passages. It is not a
check of arbitrary text, and gold labels are never read. Claims with no saved passage
return "not enough sources" rather than a guess.

## Three result sources (always labeled in the popup)

| Mode | Needs backend | What it does |
| --- | --- | --- |
| Mock | no | Built-in synthetic samples for every UI state. Not a check. |
| Benchmark demo | yes | Default: replays synthetic fixtures. With `--baseline-dir`: runs the baseline model on a chosen dev claim. Labeled either way. |
| Live | yes | Returns "not enough sources": no retrieval path exists for new claims. |

## Known limits

- Fixtures in `apps/backend/fixtures/demo_claims.json` are synthetic (example.com
  URLs), not AVeriTeC records. Replace with saved benchmark predictions once B or C
  provides them, via an `Adapter` in `apps/backend/server.py`.
- The backend adds an optional `provenance` field (`fixture_replay` or
  `baseline_distilbert_bm25_top3`) to responses. This is an additive proposal to the contract, not yet agreed.
- Nothing here shows that the interface improves user decisions.
