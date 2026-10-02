import importlib.util
import json
import zipfile
from pathlib import Path


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
