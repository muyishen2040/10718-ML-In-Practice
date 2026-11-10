"""Build the portable CPU notebook for the extension's private Colab backend."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.notebooks import NotebookBuilder, emit_notebook

TARGET = ROOT / "notebooks/extension_backend_cpu_colab.ipynb"


def build():
    paths = ["apps/backend/server.py", "apps/backend/baseline_adapter.py",
             "apps/backend/model_registry.py", "apps/backend/nli_adapter.py",
             "apps/backend/colab_runtime.py",
             "apps/backend/requirements-model.txt"]
    paths += ['src/__init__.py', 'src/verification/__init__.py', 'src/verification/pair_reader.py',
              'src/data/__init__.py', 'src/data/averitec.py', 'src/data/schema.py', 'src/data/corpus.py',
              'src/retrieval/__init__.py', 'src/retrieval/evidence_quality.py', 'src/retrieval/question_guided.py',
              'src/evaluation/__init__.py', 'src/evaluation/splits.py', 'src/utils/__init__.py', 'src/utils/progress.py']
    paths += [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "apps/browser-extension").iterdir()) if path.is_file()]
    sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in paths}
    payload = json.dumps(sources, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    nb = NotebookBuilder(TARGET.name)
    md, code = nb.markdown, nb.code
    md("""# Extension backend: private Colab CPU session

Load saved models and evidence from shared Drive. The extension can select
DistilBERT with BM25, BGE or fused evidence, or the corrected NLI reader.
The three DistilBERT options share classifier weights; their evidence differs.
No training, full-corpus retrieval, GPU, public tunnel or third-party web proxy is used.
This notebook contains its source and runs directly in the kernel. Requests arrive
from an authenticated CLI relay on your own computer, where Chrome still uses
`http://127.0.0.1:8000`. Backend and relay both bind only to their own loopback.

Use your paid Colab account with positive compute credits. This is a supervised
prototype on known benchmark claims, not arbitrary live claim verification or
an always-on service. Colab's normal resource and usage policies still apply.

**CLI route:** create a named CPU session, authorize Drive with `colab drivemount`,
then execute this notebook with `colab exec --timeout 600 -f ...`. See
`apps/COLAB_BACKEND.md` or `scripts/colab_extension.ps1` in the repository.
**Browser route:** select CPU/None as accelerator and run in order; use
`colab sessions` on your computer to identify this same runtime before relaying.
""")
    md("## 1. Restore the bundled source")
    code(f"""
from pathlib import Path
import hashlib, importlib.util, json, os, subprocess, sys, time
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
SOURCES = json.loads({payload!r})
SOURCE_SHA256 = {digest!r}
assert hashlib.sha256(json.dumps(SOURCES, sort_keys=True, ensure_ascii=True).encode()).hexdigest() == SOURCE_SHA256
RUNTIME = Path('/content') / ('claim_check_source_' + SOURCE_SHA256[:12])
for name, module in list(sys.modules.items()):
    if name.startswith('claim_check_colab_runtime_') and name != 'claim_check_colab_runtime_' + SOURCE_SHA256[:12] and getattr(module, '_ACTIVE', None):
        raise RuntimeError('A different backend source is still loaded. Use a fresh CPU session or stop the previous backend first.')
    if name == 'src' and getattr(module, '__file__', None) and not Path(module.__file__).is_relative_to(RUNTIME):
        raise RuntimeError('Different project code is already imported. Restart the kernel before running this notebook.')
for name, content in SOURCES.items():
    path = RUNTIME / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(RUNTIME / 'apps/backend'))
print('Portable source:', SOURCE_SHA256)
""", hidden=True)
    md("## 2. Install the text-only model dependencies")
    code("""
if 'transformers' in sys.modules and sys.modules['transformers'].__version__ != '4.57.1':
    raise RuntimeError('Use a fresh session before loading this notebook.')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                str(RUNTIME / 'apps/backend/requirements-model.txt')], check=True)
import torch
print('PyTorch:', torch.__version__, '| inference device: CPU')
""")
    md("## 3. Mount Drive and locate the serving inputs\n\nKeep the shared shortcut named `10718_averitec`. A CLI-mounted Drive is reused. Only the model, tokenizer, run settings and saved dev rankings are read; no training labels or full passage corpus are needed.")
    code("""
if not Path('/content/drive/MyDrive').is_dir():
    from google.colab import drive
    drive.mount('/content/drive')
DATA_ROOT = Path('/content/drive/MyDrive/10718_averitec/v2_source_documents')
CACHE_ROOT = Path('/content/claim_check_serving')
PORT = 8000
CPU_THREADS = 2
if not DATA_ROOT.is_dir():
    raise FileNotFoundError('Check shared-folder access and the My Drive shortcut: ' + str(DATA_ROOT))
print('Shared input root:', DATA_ROOT)
""")
    md("## 4. Copy the small serving bundle and start the private backend\n\nComplete cached files are reused after interruption. Shared Drive files are never changed. This returns to the kernel after starting a background HTTP thread; there is no blocking server loop or artificial keepalive. Rerunning with the same source/settings reuses the loaded backend.")
    code("""
module_name = 'claim_check_colab_runtime_' + SOURCE_SHA256[:12]
if module_name not in sys.modules:
    spec = importlib.util.spec_from_file_location(module_name, RUNTIME / 'apps/backend/colab_runtime.py')
    backend = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = backend
    spec.loader.exec_module(backend)
else:
    backend = sys.modules[module_name]
BACKEND_INFO = backend.start_backend(DATA_ROOT, CACHE_ROOT, port=PORT, cpu_threads=CPU_THREADS)
print(json.dumps(BACKEND_INFO, indent=2))
""")
    md("## 5. Check one real inference request")
    code("""
import urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
base = BACKEND_INFO['url']
with opener.open(base + '/health', timeout=10) as reply:
    health = json.load(reply)
assert health['adapter'] == 'model_registry', health
with opener.open(base + '/models', timeout=10) as reply:
    models = json.load(reply)
print('Available models:', [m['label'] for m in models['models'] if m['available']])
with opener.open(base + '/benchmark/claims?limit=1', timeout=10) as reply:
    claim_id = json.load(reply)['claims'][0]['claim_id']
request = urllib.request.Request(base + '/verify',
    data=json.dumps({'mode': 'benchmark', 'claim_id': claim_id, 'model_id': models['default_model_id']}).encode(),
    headers={'Content-Type': 'application/json'})
started = time.monotonic()
with opener.open(request, timeout=90) as reply:
    result = json.load(reply)
assert result['status'] in ('completed', 'insufficient_sources'), result
print('Claim:', claim_id, '| status:', result['status'], '| verdict:', result.get('verdict'))
print('Evidence items:', len(result['evidence']), '| runtime inference seconds:', round(time.monotonic() - started, 2))
print('Backend is ready for the local relay. No gold verdict labels were read.')
""")
    md("## 6. Optional: package this matching extension version\n\nThe zip contains extension code only, not Drive files, weights, credentials or predictions. Download with the Colab file browser, extract locally, then use Chrome's Load unpacked on the extracted folder.")
    code("""
import zipfile
extension_zip = Path('/content/claim_check_extension.zip')
with zipfile.ZipFile(extension_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((RUNTIME / 'apps/browser-extension').iterdir()):
        if path.is_file():
            archive.write(path, path.name)
print('Extension zip:', extension_zip)
print('Source snapshot:', SOURCE_SHA256)
""")
    md("""## Connect and stop

On your computer, run the repository's `scripts/colab_extension.ps1 -Action Relay`
or the Python relay command in `apps/COLAB_BACKEND.md`. Keep that terminal open.
Load the unpacked extension, use **Benchmark**, select **Model / evidence pipeline**, and keep Backend URL at
`http://127.0.0.1:8000`. Mock mode does not call the trained model; Live is unavailable.

Run `colab sessions` if you opened this notebook in the browser; select that same
session in the relay, rather than creating another one. CLI command startup adds
transport latency to the first verification of each claim; repeated inference is
cached separately for each pipeline in the remote adapter. Record this overhead if measuring decision time.

If the kernel restarts, rerun the notebook in order. A replacement VM needs a new
Drive mount and cache copy. Changing settings while the backend is loaded requires
`backend.stop_backend()` first. A new notebook source needs a fresh runtime.

When finished, stop the relay with Ctrl+C and stop the named Colab runtime:
`colab stop -s claim-check-extension`. Closing the extension does not stop the VM.
""")
    document = nb.document(source_sha256=digest)
    document["metadata"].pop("accelerator", None)
    for i, cell in enumerate(document["cells"]):
        cell["id"] = f"extension-backend-{i:03d}"
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", action="store_true")
    args = parser.parse_args()
    emit_notebook(build(), TARGET, patch=args.patch)
