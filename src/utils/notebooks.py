"""Minimal reproducible Colab artifact helpers (standard library only)."""
import hashlib
import json
from pathlib import Path


class NotebookBuilder:
    def __init__(self, name):
        self.name, self.cells = name, []

    def markdown(self, text):
        self.cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)})

    def code(self, text, *, hidden=False):
        self.cells.append({"cell_type": "code", "execution_count": None, "outputs": [],
                           "metadata": {"jupyter": {"source_hidden": True}} if hidden else {},
                           "source": (text.strip() + "\n").splitlines(keepends=True)})

    def document(self, **metadata):
        for i, cell in enumerate(self.cells):
            cell["id"] = f"evidence-reader-{i:03d}"
        return {"cells": self.cells, "metadata": {"colab": {"name": self.name}, "accelerator": "GPU",
                "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                "language_info": {"name": "python"}, **metadata}, "nbformat": 4, "nbformat_minor": 5}


def source_snapshot(root, extra_files):
    root = Path(root)
    paths = sorted((root / "src").rglob("*.py")) + [root / p for p in extra_files]
    sources = {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in paths}
    sources["scripts/__init__.py"] = ""
    payload = json.dumps(sources, sort_keys=True, ensure_ascii=True)
    return payload, hashlib.sha256(payload.encode()).hexdigest()


def emit_notebook(document, target, *, patch=False):
    text = json.dumps(document, ensure_ascii=True, indent=1) + "\n"
    target = Path(target)
    if not patch:
        print(text, end="")
        return
    print("*** Begin Patch")
    if target.exists():
        print("*** Update File: " + target.as_posix())
        print("@@")
        print("\n".join("-" + line for line in target.read_text(encoding="utf-8").splitlines()))
    else:
        print("*** Add File: " + target.as_posix())
    print("\n".join("+" + line for line in text.splitlines()))
    print("*** End Patch")
