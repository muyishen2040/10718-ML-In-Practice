import json
import zipfile

from src.data.knowledge_store import inspect_zip_archive, normalize_zip_to_documents, normalize_zip_to_jsonl
from src.utils.io import read_jsonl


def test_archive_inspection_reports_member_shapes(tmp_path) -> None:
    archive_path = tmp_path / "knowledge.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("documents/example.json", json.dumps({"url": "https://example.org", "text": "evidence"}))
        archive.writestr("documents/example.tsv", "url\ttext\nhttps://example.org\tevidence\n")

    report = inspect_zip_archive(archive_path)

    assert report["member_count"] == 2
    assert report["suffix_counts"] == {".json": 1, ".tsv": 1}
    assert report["sample_members"][0]["json_keys"] == ["text", "url"]


def test_archive_normalizer_handles_json_and_tsv_url_text_records(tmp_path) -> None:
    archive_path = tmp_path / "knowledge.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("documents/one.json", json.dumps({"url": "https://example.org/one", "text": "first evidence"}))
        archive.writestr("documents/two.tsv", "url\ttext\nhttps://example.org/two\tsecond evidence\n")

    documents, audit = normalize_zip_to_documents(archive_path)

    assert {document["url"] for document in documents} == {"https://example.org/one", "https://example.org/two"}
    assert audit["normalized_document_count"] == 2


def test_streaming_archive_normalizer_writes_jsonl_and_audits_limit(tmp_path) -> None:
    archive_path = tmp_path / "knowledge.zip"
    output_path = tmp_path / "documents.jsonl"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "documents/rows.jsonl",
            '\n'.join(
                json.dumps({"url": f"https://example.org/{index}", "text": "evidence"}) for index in range(3)
            ),
        )

    audit = normalize_zip_to_jsonl(archive_path, output_path, max_documents=2)

    assert len(read_jsonl(output_path)) == 2
    assert audit["stopped_early_at_document_limit"] is True
    assert audit["warning"]


def test_streaming_normalizer_handles_official_averitec_per_claim_jsonl(tmp_path) -> None:
    archive_path = tmp_path / "knowledge.zip"
    output_path = tmp_path / "documents.jsonl"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "output_dev/7.json",
            "\n".join(
                [
                    json.dumps({"url": "https://example.org", "url2text": ["first sentence"]}),
                    json.dumps({"url": "https://example.org", "url2text": ["second sentence"]}),
                ]
            )
            + "\n",
        )

    audit = normalize_zip_to_jsonl(archive_path, output_path)
    documents = read_jsonl(output_path)

    assert audit["normalized_document_count"] == 2
    assert {document["metadata"]["candidate_claim_index"] for document in documents} == {7}
    assert {document["metadata"]["source_format"] for document in documents} == {"averitec_url2text"}
    inspection = inspect_zip_archive(archive_path)
    assert inspection["sample_members"][0]["jsonl_with_json_suffix"] is True
    assert inspection["sample_members"][0]["json_keys"] == ["url", "url2text"]
