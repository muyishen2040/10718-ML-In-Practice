"""Safe AVeriTeC knowledge-store archive inspection for Colab."""

from __future__ import annotations

import json
import csv
import io
import zipfile
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from src.utils.progress import progress


URL_KEYS = ("url", "source_url", "document_url")
TEXT_KEYS = ("text", "scraped_text", "content", "lines", "document", "body")
_CLAIM_FILE_PATTERN = re.compile(r"(?:^|/)(\d+)\.json(?:#\d+)?$")


def inspect_zip_archive(path: str | Path, sample_members: int = 10) -> dict[str, Any]:
    """Inspect archive structure without extracting large files or assuming schema."""
    archive_path = Path(path)
    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        suffix_counts = Counter(PurePosixPath(member.filename).suffix.casefold() or "[no extension]" for member in members)
        samples = []
        inspected_members = members[:sample_members]
        with progress(total=len(inspected_members), description="Inspecting archive", unit="file") as bar:
            for member in inspected_members:
                sample: dict[str, Any] = {"name": member.filename, "compressed_size": member.compress_size, "file_size": member.file_size}
                suffix = PurePosixPath(member.filename).suffix.casefold()
                if suffix in {".json", ".jsonl"} and member.file_size <= 2_000_000:
                    with archive.open(member) as handle:
                        text = handle.read().decode("utf-8", errors="replace")
                    try:
                        parsed = json.loads(text) if suffix == ".json" else json.loads(next(line for line in text.splitlines() if line.strip()))
                    except json.JSONDecodeError:
                        # Official AVeriTeC stores JSONL rows in files named
                        # `N.json`; show the first row shape instead of calling it
                        # an opaque parse failure in the inspection report.
                        try:
                            parsed = json.loads(next(line for line in text.splitlines() if line.strip()))
                            sample["jsonl_with_json_suffix"] = suffix == ".json"
                        except (json.JSONDecodeError, StopIteration):
                            sample["json_parse_error"] = True
                            parsed = None
                    except StopIteration:
                        sample["json_parse_error"] = True
                        parsed = None
                    if parsed is not None:
                        sample["json_type"] = type(parsed).__name__
                        if isinstance(parsed, dict):
                            sample["json_keys"] = sorted(parsed)[:30]
                        elif isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                            sample["first_row_keys"] = sorted(parsed[0])[:30]
                elif suffix in {".tsv", ".csv"} and member.file_size <= 2_000_000:
                    with archive.open(member) as handle:
                        sample["header"] = handle.readline().decode("utf-8", errors="replace").strip()
                samples.append(sample)
                bar.update(1)
    return {
        "archive": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "member_count": len(members),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "sample_members": samples,
    }


def _text_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        text = "\n".join(item.strip() for item in value if item.strip())
        return text or None
    return None


def _candidate_claim_index(source_member: str) -> int | None:
    """Read the zero-based claim index from official `output_<split>/N.json` names."""
    match = _CLAIM_FILE_PATTERN.search(source_member)
    return int(match.group(1)) if match else None


def _documents_from_value(
    value: Any, source_member: str, *, url2text_mode: str = "sentence"
) -> list[dict[str, Any]]:
    """Interpret common URL/text JSON shapes without fabricating content."""
    if url2text_mode not in {"sentence", "source_document"}:
        raise ValueError("url2text_mode must be 'sentence' or 'source_document'")
    if isinstance(value, list):
        documents = []
        for index, item in enumerate(value):
            documents.extend(
                _documents_from_value(item, f"{source_member}#{index}", url2text_mode=url2text_mode)
            )
        return documents
    if not isinstance(value, dict):
        return []
    url = next((str(value[key]).strip() for key in URL_KEYS if value.get(key)), None)
    # The official AVeriTeC knowledge store contains one JSONL file per claim.
    # Each row associates a URL with `url2text`, a list of extracted sentences.
    # Preserve the candidate-pool boundary so retrieval never crosses claims.
    sentence_values = value.get("url2text")
    if url and isinstance(sentence_values, list) and all(isinstance(item, str) for item in sentence_values):
        candidate_claim_index = _candidate_claim_index(source_member)
        cleaned_sentences = [sentence.strip() for sentence in sentence_values if sentence.strip()]
        if url2text_mode == "source_document" and cleaned_sentences:
            # Each official JSONL row is a URL and its extracted sentences.
            # Keep that natural source-document unit, then let the later
            # passage builder produce overlapping 160-word retrieval chunks.
            # This is substantially smaller than making every sentence an
            # independent candidate while preserving the same evidence text.
            return [
                {
                    "document_id": f"{source_member}:source_document",
                    "url": url,
                    "text": "\n".join(cleaned_sentences),
                    "metadata": {
                        "archive_member": source_member.split("#", 1)[0],
                        "candidate_claim_index": candidate_claim_index,
                        "source_format": "averitec_url2text_source_document",
                    },
                }
            ]
        return [
            {
                "document_id": f"{source_member}:sentence:{sentence_index}",
                "url": url,
                "text": sentence.strip(),
                "metadata": {
                    "archive_member": source_member.split("#", 1)[0],
                    "candidate_claim_index": candidate_claim_index,
                    "source_format": "averitec_url2text",
                },
            }
            for sentence_index, sentence in enumerate(sentence_values)
            if sentence.strip()
        ]
    text = next((_text_value(value[key]) for key in TEXT_KEYS if _text_value(value.get(key))), None)
    if url and text:
        return [{"document_id": f"{source_member}:{url}", "url": url, "text": text, "metadata": {"archive_member": source_member}}]
    # Some archives use a dictionary mapping source URL to extracted text/lines.
    documents = []
    for key, nested_value in value.items():
        if key.startswith(("http://", "https://")):
            nested_text = _text_value(nested_value)
            if nested_text:
                documents.append(
                    {"document_id": f"{source_member}:{key}", "url": key, "text": nested_text, "metadata": {"archive_member": source_member}}
                )
    return documents


def normalize_zip_to_documents(
    archive_path: str | Path, *, url2text_mode: str = "sentence"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract recognized JSON/JSONL/TSV/CSV URL-text records directly from a zip.

    Unknown members are counted and omitted rather than guessed. Review the
    resulting audit before treating the documents as a valid evidence corpus.
    """
    source = Path(archive_path)
    documents: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"archive": str(source), "member_counts": Counter(), "recognized_member_counts": Counter(), "skipped_member_samples": []}
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            suffix = PurePosixPath(member.filename).suffix.casefold()
            audit["member_counts"][suffix or "[no extension]"] += 1
            try:
                with archive.open(member) as binary_handle:
                    if suffix == ".json":
                        value = json.load(io.TextIOWrapper(binary_handle, encoding="utf-8", errors="replace"))
                        parsed = _documents_from_value(value, member.filename, url2text_mode=url2text_mode)
                    elif suffix == ".jsonl":
                        parsed = []
                        for index, line in enumerate(io.TextIOWrapper(binary_handle, encoding="utf-8", errors="replace")):
                            if line.strip():
                                parsed.extend(
                                    _documents_from_value(
                                        json.loads(line), f"{member.filename}#{index}", url2text_mode=url2text_mode
                                    )
                                )
                    elif suffix in {".tsv", ".csv"}:
                        delimiter = "\t" if suffix == ".tsv" else ","
                        reader = csv.DictReader(io.TextIOWrapper(binary_handle, encoding="utf-8", errors="replace"), delimiter=delimiter)
                        parsed = _documents_from_value(list(reader), member.filename, url2text_mode=url2text_mode)
                    else:
                        parsed = []
                if parsed:
                    audit["recognized_member_counts"][suffix] += 1
                    documents.extend(parsed)
                elif len(audit["skipped_member_samples"]) < 20:
                    audit["skipped_member_samples"].append(member.filename)
            except (json.JSONDecodeError, UnicodeError, csv.Error) as error:
                if len(audit["skipped_member_samples"]) < 20:
                    audit["skipped_member_samples"].append(f"{member.filename} ({type(error).__name__})")
    audit["member_counts"] = dict(sorted(audit["member_counts"].items()))
    audit["recognized_member_counts"] = dict(sorted(audit["recognized_member_counts"].items()))
    audit["normalized_document_count"] = len(documents)
    audit["documents_with_url"] = sum(bool(document.get("url")) for document in documents)
    audit["url2text_mode"] = url2text_mode
    return documents, audit


def normalize_zip_to_jsonl(
    archive_path: str | Path,
    output_path: str | Path,
    *,
    max_documents: int | None = None,
    url2text_mode: str = "sentence",
) -> dict[str, Any]:
    """Write recognized archive records to JSONL without retaining the corpus in RAM.

    This is the Colab-facing path.  It streams JSONL/CSV/TSV records and
    writes each normalized document immediately.  JSON members still need to
    be valid standalone JSON values, but no list of documents from the whole
    archive is accumulated in memory.
    """
    if max_documents is not None and max_documents < 1:
        raise ValueError("max_documents must be positive when supplied")
    if url2text_mode not in {"sentence", "source_document"}:
        raise ValueError("url2text_mode must be 'sentence' or 'source_document'")
    source = Path(archive_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "archive": str(source),
        "member_counts": Counter(),
        "recognized_member_counts": Counter(),
        "skipped_member_samples": [],
        "max_documents": max_documents,
        "url2text_mode": url2text_mode,
    }
    normalized_document_count = 0
    stopped_early = False

    with zipfile.ZipFile(source) as archive, destination.open("w", encoding="utf-8", newline="\n") as output:
        members = [member for member in archive.infolist() if not member.is_dir()]
        with progress(total=len(members), description="Normalizing archive", unit="file") as bar:
            for member in members:
                suffix = PurePosixPath(member.filename).suffix.casefold()
                audit["member_counts"][suffix or "[no extension]"] += 1
                recognized_member = False
                try:
                    with archive.open(member) as binary_handle:
                        text_handle = io.TextIOWrapper(binary_handle, encoding="utf-8", errors="replace")
                        if suffix == ".json":
                            # AVeriTeC calls its per-claim JSONL files `.json`.
                            # Try ordinary JSON first, then safely fall back to one
                            # JSON object per line without treating an invalid file
                            # as a generic format.
                            content = text_handle.read()
                            try:
                                parsed_rows = _documents_from_value(
                                    json.loads(content), member.filename, url2text_mode=url2text_mode
                                )
                                row_iterator = iter(parsed_rows)
                            except json.JSONDecodeError:
                                def json_rows_with_json_suffix() -> Any:
                                    for index, line in enumerate(content.splitlines()):
                                        if line.strip():
                                            yield from _documents_from_value(
                                                json.loads(line),
                                                f"{member.filename}#{index}",
                                                url2text_mode=url2text_mode,
                                            )
                                row_iterator = json_rows_with_json_suffix()
                        elif suffix == ".jsonl":
                            def jsonl_rows() -> Any:
                                for index, line in enumerate(text_handle):
                                    if line.strip():
                                        yield from _documents_from_value(
                                            json.loads(line),
                                            f"{member.filename}#{index}",
                                            url2text_mode=url2text_mode,
                                        )
                            row_iterator = jsonl_rows()
                        elif suffix in {".tsv", ".csv"}:
                            delimiter = "\t" if suffix == ".tsv" else ","
                            reader = csv.DictReader(text_handle, delimiter=delimiter)
                            def tabular_rows() -> Any:
                                for index, row in enumerate(reader):
                                    yield from _documents_from_value(
                                        row, f"{member.filename}#{index}", url2text_mode=url2text_mode
                                    )
                            row_iterator = tabular_rows()
                        else:
                            row_iterator = iter(())
                        for document in row_iterator:
                            output.write(json.dumps(document, ensure_ascii=False) + "\n")
                            normalized_document_count += 1
                            recognized_member = True
                            if max_documents is not None and normalized_document_count >= max_documents:
                                stopped_early = True
                                break
                    if recognized_member:
                        audit["recognized_member_counts"][suffix] += 1
                    elif len(audit["skipped_member_samples"]) < 20:
                        audit["skipped_member_samples"].append(member.filename)
                except (json.JSONDecodeError, UnicodeError, csv.Error) as error:
                    if len(audit["skipped_member_samples"]) < 20:
                        audit["skipped_member_samples"].append(f"{member.filename} ({type(error).__name__})")
                bar.update(1)
                bar.set_postfix(documents=normalized_document_count)
                if stopped_early:
                    break

    audit["member_counts"] = dict(sorted(audit["member_counts"].items()))
    audit["recognized_member_counts"] = dict(sorted(audit["recognized_member_counts"].items()))
    audit["normalized_document_count"] = normalized_document_count
    audit["stopped_early_at_document_limit"] = stopped_early
    audit["warning"] = (
        "This normalized corpus is intentionally incomplete because max_documents was used. "
        "Do not report retrieval metrics from it as full-corpus AVeriTeC results."
        if stopped_early
        else None
    )
    return audit
