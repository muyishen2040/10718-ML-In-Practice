# Dependency profiles

Run installation commands from the repository root. Select the profile for the
workflow instead of installing every profile into one Colab runtime. Package
constraints were preserved when these files moved from the root on November 4,
2026; nested `-r` includes are relative to this directory.

| Profile | Purpose | Command |
| --- | --- | --- |
| [base.txt](base.txt) | Core data/metrics, BM25, TF-IDF and lightweight tests | `python -m pip install -r requirements/base.txt` |
| [reader-colab.txt](reader-colab.txt) | NLI reader and text-only transformer work; retains Colab's CUDA PyTorch | `python -m pip install -r requirements/reader-colab.txt` |
| [colab.txt](colab.txt) | Earlier broad dense-retrieval/hosted-baseline environment | `python -m pip install -r requirements/colab.txt` |
| [local-llm-colab.txt](local-llm-colab.txt) | Earlier quantized local Qwen workflow | `python -m pip install -r requirements/local-llm-colab.txt` |

The root `requirements.txt` is a convenience entry point for `base.txt`.
The reader profile pins Transformers 4.57.1 and avoids the broad
sentence-transformers/TensorFlow import path that caused earlier runtime issues.
Use the chosen notebook's setup and a fresh session when changing environments.

Portable reader notebooks bundle their own dependency file inside their restored
runtime. An older virtual filename inside that snapshot is intentional and does
not require a matching file at the repository root. New notebook builders use
the current `requirements/` paths. Regenerating a portable notebook changes its
source snapshot; use a new run directory rather than mixing saved run versions.

Core tests run without neural dependencies and skip optional model checks. The
full model tests need an appropriate PyTorch/Transformers environment. Additional
API providers or frontend/backend implementations should declare their own
dependencies when implemented; those components are not created by this move.
