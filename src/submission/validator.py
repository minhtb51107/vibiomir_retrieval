from __future__ import annotations

import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq


class SubmissionValidationError(ValueError):
    """Raised when an organizer submission fails strict validation."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _assert_finite_numbers(value: Any, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SubmissionValidationError(f"non-finite number at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_numbers(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_numbers(child, f"{location}[{index}]")


def load_submission_payload(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            if len(names) != 1 or "/" in names[0] or not names[0].endswith(".json"):
                raise SubmissionValidationError(
                    "ZIP must contain exactly one root-level .json file"
                )
            payload = json.loads(archive.read(names[0]).decode("utf-8"))
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SubmissionValidationError("submission root must be a JSON array")
    _assert_finite_numbers(payload, "submission")
    return payload


def canonical_provenance(
    chunks_path: str | Path,
) -> tuple[set[int], set[tuple[int, str]]]:
    table = pq.read_table(chunks_path, columns=["doc_id", "raw_text"])
    pairs = {
        (int(row["doc_id"]), str(row["raw_text"]))
        for row in table.to_pylist()
    }
    return {doc_id for doc_id, _ in pairs}, pairs


def expected_query_ids(queries_path: str | Path) -> list[int]:
    table = pq.read_table(queries_path, columns=["id"])
    return [int(value) for value in table.column("id").to_pylist()]


def load_source_documents(
    documents_path: str | Path, doc_ids: set[int] | None = None
) -> dict[int, str]:
    """Load processed `normalized_text` per doc_id for source-span validation."""
    table = pq.read_table(documents_path, columns=["doc_id", "normalized_text"])
    output: dict[int, str] = {}
    for row in table.to_pylist():
        doc_id = int(row["doc_id"])
        if doc_ids is None or doc_id in doc_ids:
            output[doc_id] = str(row["normalized_text"])
    return output


def validate_submission(
    path: str | Path,
    *,
    expected_queries: list[int],
    valid_document_ids: set[int],
    valid_chunks: set[tuple[int, str]],
    require_exact_order: bool = True,
    source_documents: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Strictly validate a submission.

    By default every chunk object must equal a canonical chunk. When
    `source_documents` is supplied, a chunk that is not canonical is still
    accepted if and only if its text is a verbatim contiguous substring of the
    processed source document for its doc_id (expanded-window experiments).
    """
    payload = load_submission_payload(path)
    if len(payload) != len(expected_queries):
        raise SubmissionValidationError(
            f"query count {len(payload)} != expected {len(expected_queries)}"
        )
    query_ids: list[int] = []
    document_total = 0
    chunk_total = 0
    unique_documents: set[int] = set()
    document_counts: list[int] = []
    chunk_counts: list[int] = []
    source_span_chunks = 0
    for index, record in enumerate(payload):
        location = f"query[{index}]"
        if not isinstance(record, dict):
            raise SubmissionValidationError(f"{location} must be an object")
        if set(record) != {"id", "relevant_docs", "relevant_chunks"}:
            raise SubmissionValidationError(f"{location} has incorrect fields")
        query_id = record["id"]
        if not _is_int(query_id):
            raise SubmissionValidationError(f"{location}.id must be an integer")
        query_ids.append(query_id)
        documents = record["relevant_docs"]
        chunks = record["relevant_chunks"]
        if not isinstance(documents, list) or not isinstance(chunks, list):
            raise SubmissionValidationError(
                f"{location} result fields must be arrays"
            )
        seen_documents: set[int] = set()
        for result_index, doc_id in enumerate(documents):
            if not _is_int(doc_id):
                raise SubmissionValidationError(
                    f"{location}.relevant_docs[{result_index}] must be an integer"
                )
            if doc_id in seen_documents:
                raise SubmissionValidationError(
                    f"duplicate document {doc_id} at {location}"
                )
            if doc_id not in valid_document_ids:
                raise SubmissionValidationError(
                    f"unknown local document {doc_id} at {location}"
                )
            seen_documents.add(doc_id)
            unique_documents.add(doc_id)
        seen_chunks: set[tuple[int, str]] = set()
        for result_index, chunk in enumerate(chunks):
            chunk_location = f"{location}.relevant_chunks[{result_index}]"
            if not isinstance(chunk, dict):
                raise SubmissionValidationError(f"{chunk_location} must be an object")
            required = {"doc_id", "chunk_text"}
            allowed = required | {"chunk_order"}
            if not required.issubset(chunk) or not set(chunk).issubset(allowed):
                raise SubmissionValidationError(
                    f"{chunk_location} has incorrect fields"
                )
            doc_id = chunk["doc_id"]
            text = chunk["chunk_text"]
            if not _is_int(doc_id):
                raise SubmissionValidationError(f"{chunk_location}.doc_id must be integer")
            if not isinstance(text, str):
                raise SubmissionValidationError(f"{chunk_location}.chunk_text must be string")
            if "chunk_order" in chunk and (
                not _is_int(chunk["chunk_order"]) or chunk["chunk_order"] < 0
            ):
                raise SubmissionValidationError(
                    f"{chunk_location}.chunk_order must be a non-negative integer"
                )
            key = (doc_id, text)
            if key in seen_chunks:
                raise SubmissionValidationError(
                    f"duplicate chunk object at {chunk_location}"
                )
            if doc_id not in valid_document_ids:
                raise SubmissionValidationError(
                    f"unknown local document {doc_id} at {chunk_location}"
                )
            if key not in valid_chunks:
                document_text = (
                    source_documents.get(doc_id) if source_documents is not None else None
                )
                if document_text is None or not text or text not in document_text:
                    raise SubmissionValidationError(
                        f"chunk provenance mismatch at {chunk_location}"
                    )
                source_span_chunks += 1
            seen_chunks.add(key)
            unique_documents.add(doc_id)
        document_total += len(documents)
        chunk_total += len(chunks)
        document_counts.append(len(documents))
        chunk_counts.append(len(chunks))
    duplicates = [item for item, count in Counter(query_ids).items() if count > 1]
    if duplicates:
        raise SubmissionValidationError(f"duplicate query IDs: {duplicates[:10]}")
    expected_set = set(expected_queries)
    actual_set = set(query_ids)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise SubmissionValidationError(
            f"query IDs differ; missing={missing[:10]} extra={extra[:10]}"
        )
    if require_exact_order and query_ids != expected_queries:
        raise SubmissionValidationError("query order differs from official query input")
    return {
        "path": str(path),
        "valid": True,
        "query_count": len(payload),
        "document_result_count": document_total,
        "chunk_result_count": chunk_total,
        "unique_doc_ids": len(unique_documents),
        "document_minimum_per_query": min(document_counts, default=0),
        "document_maximum_per_query": max(document_counts, default=0),
        "chunk_minimum_per_query": min(chunk_counts, default=0),
        "chunk_maximum_per_query": max(chunk_counts, default=0),
        "invalid_doc_ids": 0,
        "duplicate_doc_ids": 0,
        "duplicate_chunk_objects": 0,
        "chunk_provenance_mismatches": 0,
        "source_span_chunks_verified": source_span_chunks,
    }
