import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def snapshot_sources(notebook):
    sources = None
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        parsed = ast.parse(text, filename=f"cell_{i}")
        assert cell["execution_count"] is None and not cell["outputs"]
        for node in parsed.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SOURCES" for target in node.targets):
                sources = json.loads(ast.literal_eval(node.value.args[0]))
    assert sources is not None
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    assert digest == notebook["metadata"]["source_sha256"]
    assert "scripts/run_answer_followup.py" in sources
    assert "src/verification/pair_reader.py" in sources
    for name, text in sources.items():
        if name.endswith(".py"):
            ast.parse(text, filename=name)
    return sources


def test_frozen_followup_snapshot_is_self_contained_after_repository_reorganization(tmp_path):
    # An executed portable snapshot is intentionally independent of later source
    # refactors. Validate its own hash/code and the actual install-file boundary.
    notebook = json.loads((ROOT / "notebooks/evidence_reader_followup_colab.ipynb").read_text(encoding="utf-8"))
    sources = snapshot_sources(notebook)
    installed = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        parsed = ast.parse("".join(cell["source"]))
        for node in ast.walk(parsed):
            if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                    and isinstance(node.left, ast.Name) and node.left.id == "RUNTIME"
                    and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)
                    and node.right.value.endswith(".txt")):
                installed.append(node.right.value)
    assert installed, "Notebook must install a bundled profile"
    for relative in installed:
        assert relative in sources, f"Install profile missing from snapshot: {relative}"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sources[relative], encoding="utf-8")
        assert path.read_text(encoding="utf-8") == sources[relative]


def test_current_generator_embeds_relocated_profile_and_current_sources():
    from scripts.build_answer_followup_notebook import build

    notebook = build()
    sources = snapshot_sources(notebook)
    assert "requirements/reader-colab.txt" in sources
    assert "RUNTIME / 'requirements/reader-colab.txt'" in "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    for name, text in sources.items():
        if name == "scripts/__init__.py":
            continue
        assert text == (ROOT / name).read_text(encoding="utf-8"), f"Stale notebook source: {name}"
