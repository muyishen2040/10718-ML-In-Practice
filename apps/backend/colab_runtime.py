"""Start the private benchmark backend inside a Colab CPU kernel."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import threading
import time
from http.server import ThreadingHTTPServer

RUN_PATH = Path("outputs/initial_ml_solution_rerun_bs16/transformer/bm25_top3")
RANKINGS_PATH = Path("outputs/bm25/dev_bm25_top20_v2/dev_rankings.jsonl")
_ACTIVE = None


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_artifacts(data_root, cache_root):
    """Copy a small serving bundle atomically; never write to shared inputs."""
    data_root, cache_root = Path(data_root).resolve(), Path(cache_root).resolve()
    run = data_root / RUN_PATH
    model = run / "final_model"
    required = [run / "transformer_run_config.json", model / "config.json", data_root / RANKINGS_PATH]
    missing = [str(path) for path in required if not path.is_file()]
    if not model.is_dir() or not any((model / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        missing.append(str(model / "model.safetensors"))
    if missing:
        raise FileNotFoundError("Check the mounted shared Drive and DATA_ROOT:\n" + "\n".join(missing))
    if cache_root == data_root or cache_root.is_relative_to(data_root):
        raise ValueError("Use a runtime cache outside the shared input folder")
    # final_model contains the saved tokenizer and weights. Training checkpoint
    # subdirectories and the full evidence corpus are deliberately not copied.
    sources = [path for path in sorted(model.iterdir()) if path.is_file()]
    sources += [run / "transformer_run_config.json", data_root / RANKINGS_PATH]
    manifest_path = cache_root / "serving_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    previous = previous if previous.get("data_root") == str(data_root) else {}
    manifest = {"data_root": str(data_root), "files": {}}
    for source in sources:
        relative = (Path("baseline/final_model") / source.name if source.parent == model
                    else Path("baseline/transformer_run_config.json") if source.parent == run
                    else Path("dev_rankings.jsonl"))
        target = cache_root / relative
        stat = source.stat()
        fingerprint = {"source": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        old = previous.get("files", {}).get(relative.as_posix(), {})
        unchanged = all(old.get(key) == value for key, value in fingerprint.items())
        if unchanged and target.is_file() and file_hash(target) == old.get("sha256"):
            print("Reusing", relative, flush=True)
            manifest["files"][relative.as_posix()] = old
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".copying")
        print(f"Copying {relative} ({stat.st_size / 1e6:.1f} MB)", flush=True)
        shutil.copyfile(source, temporary)
        after = source.stat()
        if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError("Source changed during copy; rerun after it is stable: " + str(source))
        digest = file_hash(temporary)
        os.replace(temporary, target)
        manifest["files"][relative.as_posix()] = {**fingerprint, "sha256": digest}
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return cache_root / "baseline", cache_root / "dev_rankings.jsonl", manifest


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SerializedAdapter:
    """Keep concurrent requests from oversubscribing a small CPU runtime."""
    def __init__(self, adapter):
        self.adapter, self.name = adapter, adapter.name
        self.lock = threading.Lock()

    def get(self, claim_id):
        with self.lock:
            return self.adapter.get(claim_id)

    def list_claims(self, query="", limit=50):
        return self.adapter.list_claims(query, limit)


def start_backend(data_root, cache_root="/content/claim_check_serving", *, port=8000,
                  cpu_threads=2, adapter_factory=None):
    """Load once, serve on runtime loopback, and return to the notebook kernel.

    The local authenticated CLI relay is the only intended client. No public
    tunnel, retraining, new retrieval, or artificial runtime keepalive is used.
    """
    global _ACTIVE
    settings = (str(Path(data_root).resolve()), str(Path(cache_root).resolve()), port, cpu_threads)
    if _ACTIVE:
        if _ACTIVE["settings"] != settings:
            raise RuntimeError("Stop the existing backend before changing its settings")
        if _ACTIVE["thread"].is_alive():
            return {**_ACTIVE["info"], "reused": True}
        raise RuntimeError("Previous backend stopped unexpectedly; call stop_backend() and restart")
    if not 0 <= port <= 65535 or not 1 <= cpu_threads <= 64:
        raise ValueError("Invalid port or CPU thread count")
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    baseline, rankings, manifest = stage_artifacts(data_root, cache_root)
    if adapter_factory is None:
        import torch
        torch.set_num_threads(cpu_threads)
        adapter_factory = _module("_claim_check_baseline", "baseline_adapter.py").BaselineAdapter.from_paths
    started = time.monotonic()
    print("Loading the saved model and rankings on CPU...", flush=True)
    adapter = SerializedAdapter(adapter_factory(baseline, rankings, "cpu"))
    server_module = _module("_claim_check_http", "server.py")
    server = ThreadingHTTPServer(("127.0.0.1", port), server_module.make_handler(adapter))
    thread = threading.Thread(target=server.serve_forever, name="claim-check-backend", daemon=True)
    thread.start()
    info = {"url": f"http://127.0.0.1:{server.server_port}", "adapter": adapter.name,
            "device": "cpu", "loaded_in_seconds": round(time.monotonic() - started, 2),
            "claim_count": len(adapter.list_claims("", 10000)), "cache_root": str(Path(cache_root)),
            "manifest": str(Path(cache_root) / "serving_manifest.json"), "reused": False}
    _ACTIVE = {"server": server, "thread": thread, "adapter": adapter,
               "settings": settings, "info": info}
    print("Private backend ready:", info["url"], flush=True)
    return info


def stop_backend():
    global _ACTIVE
    if _ACTIVE:
        _ACTIVE["server"].shutdown()
        _ACTIVE["server"].server_close()
        _ACTIVE["thread"].join(timeout=5)
        _ACTIVE = None
        print("Backend stopped. Stop the Colab session separately when finished.", flush=True)
