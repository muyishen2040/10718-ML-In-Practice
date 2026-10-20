import importlib.util
import json
import sys
import zipfile
from pathlib import Path

from src.utils.io import read_jsonl, write_jsonl


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_averitec_archive_bm25.py"
SPEC = importlib.util.spec_from_file_location("run_averitec_archive_bm25_script", SCRIPT_PATH)
assert SPEC and SPEC.loader
ARCHIVE_BM25 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARCHIVE_BM25)


def test_archive_member_becomes_source_document_passages(tmp_path) -> None:
    archive_path = tmp_path / "train.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "output_train/12.json",
            json.dumps({"url": "https://example.org", "url2text": ["one two", "three four"]}) + "\n",
        )
    with zipfile.ZipFile(archive_path) as archive:
        documents = list(ARCHIVE_BM25.documents_from_official_member(archive, archive.infolist()[0]))

    passages = ARCHIVE_BM25.passages_from_documents(documents, "averitec:train:00012", max_words=3, overlap_words=1)

    assert len(documents) == 1
    assert [item.text for item in passages] == ["one two three", "three four"]
    assert all(item.metadata["candidate_claim_id"] == "averitec:train:00012" for item in passages)


def test_separate_archive_invocations_do_not_duplicate_checkpoint_rows(tmp_path, monkeypatch) -> None:
    processed = tmp_path / "data" / "processed" / "averitec"
    claims = processed / "train.jsonl"
    write_jsonl([
        {"claim_id": "averitec:train:00000", "claim": "first", "metadata": {"source_index": 0}},
        {"claim_id": "averitec:train:00001", "claim": "second", "metadata": {"source_index": 1}},
    ], claims)
    output_base = tmp_path / "bm25"
    for index in (0, 1):
        archive_path = tmp_path / f"train_{index}.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(f"output_train/{index}.json", json.dumps({
                "url": f"https://example.org/{index}", "url2text": ["first second evidence"],
            }) + "\n")
        monkeypatch.setattr(sys, "argv", [
            "archive-bm25", "--archive", str(archive_path), "--data-root", str(tmp_path / "data"),
            "--output-dir", str(output_base), "--run-name", "test", "--resume",
        ])
        ARCHIVE_BM25.main()
    partial = output_base / "test" / "train_rankings.partial.jsonl"
    assert len(read_jsonl(partial)) == 2
    monkeypatch.setattr(sys, "argv", [
        "archive-bm25", "--data-root", str(tmp_path / "data"), "--output-dir", str(output_base),
        "--run-name", "test", "--resume", "--finalize",
    ])
    ARCHIVE_BM25.main()
    assert len(read_jsonl(output_base / "test" / "train_rankings.jsonl")) == 2
