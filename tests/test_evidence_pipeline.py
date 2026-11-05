import ast
import hashlib
import json

import numpy as np
import pytest

from src.data.evidence import attach_evidence
from src.evaluation.splits import selection_split
from src.retrieval.evidence_quality import (EvidencePolicy, aggregate_features, clean_candidates, is_search_page,
    nli_label_order, original_url, quality_features, select_evidence)
from src.utils.checkpoints import append_journal, bind_run, read_journal
from src.utils.io import write_jsonl


def test_cleaning_removes_search_and_copies_but_not_archive_capture_dates():
    text = "An independent report with actual evidence about a particular event."
    candidates = [
        {"passage_id": "search", "text": "results", "url": "https://www.google.com/search?q=thing"},
        {"passage_id": "one", "text": text, "url": "https://a.test/report"},
        {"passage_id": "copy", "text": text.upper(), "url": "https://staging.a.test/report"},
        {"passage_id": "later", "text": "later report", "url": "https://b.test", "metadata": {"published_at": "2024-01-18"}},
        {"passage_id": "archive", "text": "This article projects population growth by 2050.", "url": "https://web.archive.org/web/20240101000000/https://c.test/2020/05/01/report"},
    ]
    selected, rejected = clean_candidates("An event happened", "2020-10-02", candidates, EvidencePolicy())
    assert [p["passage_id"] for p in selected] == ["one", "archive"]
    assert {p["reason"].split(":")[0] for p in rejected} == {"search_results_page", "duplicate_of", "published_after_claim"}
    assert selected[1]["publication_date"] is None
    assert selected[1]["quality"][6] == 1  # Future mention is a flag, not exclusion.


def test_cleaning_never_reads_gold_urls_or_labels():
    passage = {"passage_id": "p", "text": "some raw evidence", "url": "https://a.test", "label": "Supported", "gold_answer": "SECRET"}
    selected, _ = clean_candidates("a claim", None, [passage], EvidencePolicy())
    assert "SECRET" not in json.dumps(selected)
    assert "label" not in json.dumps(selected)
    assert original_url("https://web.archive.org/web/20240101000000id_/https://a.test/x") == "https://a.test/x"
    assert is_search_page("https://web.archive.org/web/20240101000000/https://www.google.com/search?q=x")


def test_selection_treats_contradictions_as_useful_and_uses_at_most_three():
    rows = []
    for i, p in enumerate((dict(entailment=.01, contradiction=.98, neutral=.01), dict(entailment=.1, contradiction=.1, neutral=.8))):
        row = {"passage_id": str(i), "text": "a claim", "url": "https://a.test/" + str(i), "original_rank": i+1,
               "quality": quality_features("a claim", {"text": "a claim"}, i+1, None), "nli_probabilities": p}
        rows.append(row)
    assert select_evidence(rows, rerank=True)[0]["passage_id"] == "0"
    assert np.isfinite(aggregate_features({"retrieved": []})).all()
    assert aggregate_features({"retrieved": rows}).shape == aggregate_features({"retrieved": []}).shape
    with pytest.raises(ValueError, match="NLI labels"):
        nli_label_order({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_grouped_holdout_keeps_duplicate_claims_together():
    rows = [{"claim_id": str(i), "claim": f"claim number {i//2}!", "label": str((i//2) % 4)} for i in range(80)]
    train, val = selection_split(rows, .1, 718, "grouped")
    assert {rows[i]["claim"] for i in train}.isdisjoint({rows[i]["claim"] for i in val})


def test_journal_repairs_only_torn_final_row_and_rejects_config_changes(tmp_path):
    path = tmp_path / "journal.jsonl"
    append_journal(path, [{"claim_id": "one"}])
    with path.open("ab") as handle:
        handle.write(b'{"claim_id":')
    assert set(read_journal(path, {"one", "two"})) == {"one"}
    append_journal(path, [{"claim_id": "two"}])
    assert set(read_journal(path, {"one", "two"})) == {"one", "two"}
    bind_run(tmp_path / "run.json", {"x": 1})
    with pytest.raises(ValueError, match="changed"):
        bind_run(tmp_path / "run.json", {"x": 2})


def test_evidence_join_detects_duplicate_rankings(tmp_path):
    path = tmp_path / "ranks.jsonl"
    write_jsonl([{"claim_id": "a", "retrieved": []}] * 2, path)
    with pytest.raises(ValueError, match="duplicate"):
        attach_evidence([{"claim_id": "a", "claim": "claim"}], path)


def test_main_method_notebook_compiles_and_embeds_its_dependencies():
    from scripts.build_evidence_reader_notebook import build
    notebook = build()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    bundled = next(c for c in notebook["cells"] if "SOURCES = json.loads" in "".join(c["source"]))
    assignment = next(n for n in ast.parse("".join(bundled["source"])).body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SOURCES" for t in n.targets))
    payload = ast.literal_eval(assignment.value.args[0])
    assert hashlib.sha256(payload.encode()).hexdigest() == notebook["metadata"]["source_sha256"]
    sources = json.loads(payload)
    assert "scripts/run_evidence_reader.py" in sources
    assert "src/verification/pair_reader.py" in sources
    assert "requirements/reader-colab.txt" in sources
    for name, text in sources.items():
        if name.endswith(".py"):
            ast.parse(text, filename=name)
