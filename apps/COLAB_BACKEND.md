# Test the extension with a private Colab CPU backend

The saved models run in your Colab session and read the shared Drive
artifacts. A small authenticated relay runs on your own computer. Chrome calls
`http://127.0.0.1:8000`; the relay forwards verification requests with `colab exec`.
Neither the Drive mount nor the runtime is published to the web. No ngrok,
Cloudflare tunnel, SSH daemon, GPU, training or full-corpus scan is needed.

Use a paid Colab account with a positive compute balance. Normal Colab resource
and usage limits still apply. This setup is for supervised testing on your own
computer, using known AVeriTeC dev claims. It does not add retrieval for arbitrary
claims, provide an always-on service, or establish a completed user study.

## Windows setup

Ubuntu WSL with Python 3 and `python3-venv` is required. Work from the repository
root on the branch containing this guide. The extension code is based on
`team/teammate-a-interface` at `99eadf5`.

In an interactive PowerShell terminal:

```powershell
.\scripts\colab_extension.ps1 -Action Prepare
.\scripts\colab_extension.ps1 -Action Relay
```

`Prepare` installs Colab CLI 0.7.5 into an isolated WSL environment, creates or
reuses the named **CPU** session `claim-check-extension`, requests Drive access,
and executes the portable notebook. Open Google's authorization link when
prompted, complete consent, then press Enter in that terminal. No API keys are
needed. CLI authentication may also require a Google sign-in on a new computer.

The source notebook is copied under ignored `local/` before execution. CLI output
notebooks stay there instead of being committed to Git. Only the last `Relay`
command needs to remain open while testing; Ctrl+C stops it.

## Drive files and startup

Add the shared `10718_averitec` folder as a My Drive shortcut if necessary.
The data root in Colab is:

```text
/content/drive/MyDrive/10718_averitec/v2_source_documents
```

The notebook copies only these serving inputs to `/content/claim_check_serving`:

- `outputs/initial_ml_solution_rerun_bs16/transformer/bm25_top3/final_model/`
- The same run's `transformer_run_config.json`
- `outputs/bm25/dev_bm25_top20_v2/dev_rankings.jsonl`
- BGE and fusion dev rankings under `outputs/initial_ml_solution_rerun_bs16/`:
  `dense/dev_bge_small_v15/dev_rankings.jsonl` and `hybrid/dev_rrf60/hybrid_rankings.jsonl`
- Corrected reader: `outputs/evidence_reader_followup_v2/pair/raw_sample_d01/`
  (`final_model/` and `reader_run.json`), with `raw/dev_rankings.jsonl` and
  `preparation.json` from `outputs/evidence_aware_reader_v1/prepared/`

The original baseline bundle is roughly 283 MB. Loading all four options adds
about 600 MB of reader weights, tokenizer files and evidence. Shared inputs are unchanged.
Interrupted copies can be rerun; complete files are validated against the serving
manifest. The model and all 500 claims' saved rankings then stay in memory.
Training data, gold labels, qrels, full passage corpora and vector indexes are not
needed by this backend. An empty-evidence claim returns `insufficient_sources`.

The notebook starts a loopback-only HTTP thread and returns control to the kernel.
The relay caches the claim catalog locally so popup searches do not repeatedly
invoke remote Python. Verification requests are serialized and have timeouts.
CLI startup adds transport latency; record it separately from model inference
when analyzing user decision time. Do not use artificial keepalive loops.

## Load and test the extension

1. In Chrome, open `chrome://extensions` and enable Developer mode.
2. Choose **Load unpacked**, then select `apps/browser-extension` in this checkout.
   If it was already loaded from this folder, click **Reload**.
3. Open the popup, select **Benchmark**, and leave Backend URL at
   `http://127.0.0.1:8000`. Choose **Model / evidence pipeline**, search for a
   benchmark claim and check it.
4. Confirm that the result badge names your selected pipeline and up to three
   source passages appear. The mode, model and backend URL are remembered.
   Mock mode never calls Colab; Live is unavailable.

Model choices are DistilBERT + BM25 (the default), DistilBERT + BGE, DistilBERT +
BM25/BGE fusion, and Corrected NLI reader. The first three reuse the BM25-trained
classifier with different saved evidence; the NLI reader has separate weights
and uses its original prepared BM25 passages. Switching models keeps the claim
and clears the old result. Missing or unloadable models appear disabled; the
backend never silently substitutes another model for a verification request.
These are recorded experimental pipelines, not newly trained models or proof
of improved user decisions. Gold-answer and claim-only controls are excluded.

`GET /models` lists model IDs, labels, descriptions and availability.
`POST /verify` accepts optional `model_id` with the benchmark claim ID. Omitting
it preserves the original BM25 default. Responses include `model_id`,
`model_label` and `provenance`; prediction caches are separate for each pipeline.

The portable notebook also creates `/content/claim_check_extension.zip` containing
only the matching extension code. It is useful when another computer needs the UI;
the model, Drive files and credentials are not included.

Before inviting participants, test one completed response, the empty-evidence
case and a stopped-backend error. Select a fixed claim set for a controlled study;
record assisted/unassisted assignments, judgments and timing separately. This
workflow does not yet collect participant responses or randomize conditions.

## Resume and stop

For a running session, `-Action Relay` reconnects without retraining or recopying.
If its kernel restarted, rerun `-Action Prepare`. A replacement VM needs another
Drive mount and serving-bundle copy. Changed notebook source requires stopping
the previous backend and restarting its kernel, or creating a fresh session,
rather than mixing imported versions. Reuse the mounted Drive and cached files
when restarting the kernel.

After testing, stop both the managed local relay and the named runtime:

```powershell
.\scripts\colab_extension.ps1 -Action Stop
```

The wrapper keeps a private stop token under ignored `local/`, not in the
extension. Ctrl+C also stops a foreground relay. Closing Chrome does not stop
Colab. `-Action Status` checks the named session.
If local port 8000 is occupied, do not terminate an unknown process. Use
`-Action Relay -Port 8001` and change the extension URL and host permissions to
match, or stop the earlier relay you started.

## Linux/macOS commands

The same workflow can be run without the PowerShell wrapper:

```bash
python -m pip install google-colab-cli==0.7.5
colab new -s claim-check-extension
colab drivemount -s claim-check-extension
mkdir -p local
cp notebooks/extension_backend_cpu_colab.ipynb local/colab-extension.ipynb
colab exec -s claim-check-extension --timeout 600 -f local/colab-extension.ipynb
python apps/backend/colab_relay.py --session claim-check-extension
# After stopping the relay with Ctrl+C:
colab stop -s claim-check-extension
```

For browser-run Colab, open `notebooks/extension_backend_cpu_colab.ipynb` on CPU,
run it in order, then use `colab sessions` to select that same runtime in the relay.
Do not accidentally create a second runtime containing no loaded backend.

References: [Colab CLI](https://github.com/googlecolab/google-colab-cli),
[Colab usage and Drive guidance](https://research.google.com/colaboratory/faq.html).
