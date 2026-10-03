from __future__ import annotations

import json
import sqlite3
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.storage.body_archive import BodyArchive

from .chunking import ChunkConfig, LightweightUnicodeTokenizer, chunk_document
from .extraction import ExtractionConfig, config_from_dict, extract_document
from .models import CleanDocument, ExtractionStatus


def _percentile(values: list[int], fraction: float) -> int | float:
    if not values:
        return 0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    result = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return round(result, 2)


def _distribution(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "p95": 0, "max": 0}
    return {
        "min": min(values),
        "p25": _percentile(values, 0.25),
        "median": _percentile(values, 0.5),
        "p75": _percentile(values, 0.75),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _empty_document(row: sqlite3.Row, status: ExtractionStatus) -> CleanDocument:
    return CleanDocument(
        doc_id=int(row["doc_id"]),
        original_url=str(row["original_url"]),
        final_url=row["final_url"],
        title=None,
        language_signal="unknown",
        selected_encoding=None,
        normalized_text="",
        sections=[],
        extraction_method=None,
        extraction_status=status,
        raw_byte_length=0,
        raw_char_count=0,
        clean_char_count=0,
        paragraph_count=0,
        replacement_character_count=0,
        semantic_char_count=0,
        density_char_count=0,
        boilerplate_signal_count=0,
        raw_sha256=None,
        archive_shard=None,
        archive_member_offset=None,
        archive_checksum_sha256=None,
    )


def _missing_body_status(row: sqlite3.Row) -> ExtractionStatus:
    status = str(row["status"])
    if status == "ROBOTS_BLOCKED":
        return ExtractionStatus.ROBOTS_BLOCKED
    if status == "ACCESS_RESTRICTED" or (
        status == "HTTP_ERROR" and row["http_status"] in {401, 403}
    ):
        return ExtractionStatus.ACCESS_RESTRICTED
    if bool(row["js_shell_candidate"]):
        return ExtractionStatus.JS_SHELL
    return ExtractionStatus.EXTRACTION_FAILED


def _content_type_header(metadata: sqlite3.Row) -> str | None:
    content_type = metadata["content_type"]
    declared = metadata["declared_http_encoding"]
    if content_type and declared:
        return f"{content_type}; charset={declared}"
    return str(content_type) if content_type else None


def process_archive(
    *,
    crawl_db: str | Path,
    archive_dir: str | Path,
    extraction_config_path: str | Path,
    chunking_config_path: str | Path,
    documents_out: str | Path,
    chunks_dir: str | Path,
    summary_out: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    extraction_raw = yaml.safe_load(
        Path(extraction_config_path).read_text(encoding="utf-8")
    )
    extraction_config: ExtractionConfig = config_from_dict(extraction_raw)
    chunking_raw = yaml.safe_load(Path(chunking_config_path).read_text(encoding="utf-8"))
    chunk_configs = [
        ChunkConfig(
            name=str(item["name"]),
            target_tokens=int(item["target_tokens"]),
            overlap_tokens=int(item["overlap_tokens"]),
        )
        for item in chunking_raw["experiments"]
    ]
    short_threshold = int(chunking_raw["very_short_tokens"])
    tokenizer = LightweightUnicodeTokenizer()

    connection = sqlite3.connect(crawl_db)
    connection.row_factory = sqlite3.Row
    sql = "SELECT * FROM crawl_results ORDER BY doc_id"
    parameters: tuple[int, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        parameters = (limit,)
    rows = list(connection.execute(sql, parameters))
    connection.close()

    documents: list[CleanDocument] = []
    with BodyArchive(archive_dir) as archive:
        for row in rows:
            doc_id = int(row["doc_id"])
            if not archive.contains(doc_id):
                documents.append(_empty_document(row, _missing_body_status(row)))
                continue
            metadata = archive.metadata(doc_id)
            try:
                body = archive.get_body(doc_id)
                documents.append(
                    extract_document(
                        doc_id=doc_id,
                        original_url=str(row["original_url"]),
                        final_url=row["final_url"],
                        body=body,
                        content_type_header=_content_type_header(metadata),
                        archive_shard=str(metadata["shard_name"]),
                        archive_member_offset=int(metadata["member_offset"]),
                        archive_checksum_sha256=str(metadata["checksum_sha256"]),
                        config=extraction_config,
                    )
                )
            except Exception as exc:
                failed = _empty_document(row, ExtractionStatus.EXTRACTION_FAILED)
                failed.error_type = type(exc).__name__
                failed.error_message = str(exc)[:1000]
                documents.append(failed)

        archive_rows = archive.count
        compressed_bytes = int(
            archive.connection.execute(
                "SELECT COALESCE(SUM(compressed_length), 0) FROM bodies"
            ).fetchone()[0]
        )
        original_bytes = int(
            archive.connection.execute(
                "SELECT COALESCE(SUM(original_length), 0) FROM bodies"
            ).fetchone()[0]
        )

    documents_path = Path(documents_out)
    documents_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([doc.as_dict() for doc in documents]), documents_path)

    status_counts = Counter(doc.extraction_status.value for doc in documents)
    encoding_counts = Counter(
        doc.selected_encoding or "missing" for doc in documents if doc.raw_byte_length
    )
    language_counts = Counter(
        doc.language_signal for doc in documents if doc.extraction_status == ExtractionStatus.SUCCESS
    )
    method_counts = Counter(
        (doc.extraction_method or "missing").split("+")[0]
        for doc in documents
        if doc.extraction_status == ExtractionStatus.SUCCESS
    )
    full_method_counts = Counter(
        doc.extraction_method or "missing"
        for doc in documents
        if doc.extraction_status == ExtractionStatus.SUCCESS
    )
    successful = [doc for doc in documents if doc.extraction_status == ExtractionStatus.SUCCESS]
    ratios = [doc.clean_char_count / max(doc.raw_char_count, 1) for doc in successful]
    paragraphs = [doc.paragraph_count for doc in successful]

    chunk_output_dir = Path(chunks_dir)
    chunk_output_dir.mkdir(parents=True, exist_ok=True)
    chunk_summaries: dict[str, dict[str, Any]] = {}
    docs_by_id = {doc.doc_id: doc for doc in documents}
    for chunk_config in chunk_configs:
        chunks = [
            chunk
            for document in successful
            for chunk in chunk_document(document, chunk_config, tokenizer)
        ]
        pq.write_table(
            pa.Table.from_pylist([chunk.as_dict() for chunk in chunks]),
            chunk_output_dir / f"{chunk_config.name}.parquet",
        )
        token_counts = [chunk.token_count for chunk in chunks]
        crossing = 0
        qa_crossing = 0
        for chunk in chunks:
            document = docs_by_id[chunk.doc_id]
            overlapping = [
                section
                for section in document.sections
                if section.start_offset < chunk.end_offset
                and section.end_offset > chunk.start_offset
            ]
            if len(overlapping) > 1:
                crossing += 1
            kinds = {section.section_type for section in overlapping}
            if {"question", "answer"} <= kinds:
                qa_crossing += 1
        chunk_summaries[chunk_config.name] = {
            "target_tokens": chunk_config.target_tokens,
            "overlap_tokens": chunk_config.overlap_tokens,
            "chunk_count": len(chunks),
            "documents_with_chunks": len({chunk.doc_id for chunk in chunks}),
            "chunks_per_document_mean": round(len(chunks) / max(len(successful), 1), 4),
            "token_count_mean": round(statistics.fmean(token_counts), 2) if token_counts else 0,
            "token_count_median": statistics.median(token_counts) if token_counts else 0,
            "token_count_distribution": _distribution(token_counts),
            "very_short_chunk_count": sum(value < short_threshold for value in token_counts),
            "very_short_chunk_rate": round(
                sum(value < short_threshold for value in token_counts) / max(len(chunks), 1), 6
            ),
            "very_long_chunk_count": sum(
                value > chunk_config.target_tokens for value in token_counts
            ),
            "section_crossing_count": crossing,
            "section_crossing_rate": round(crossing / max(len(chunks), 1), 6),
            "qa_boundary_crossing_count": qa_crossing,
        }

    summary = {
        "phase": "3",
        "documents_processed": len(documents),
        "archive": {
            "stored_bodies": archive_rows,
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": round(compressed_bytes / max(original_bytes, 1), 6),
        },
        "extraction": {
            "status_counts": dict(sorted(status_counts.items())),
            "success_rate_all_metadata_rows": round(
                len(successful) / max(len(documents), 1), 6
            ),
            "success_rate_archived_bodies": round(
                len(successful) / max(archive_rows, 1), 6
            ),
            "encoding_counts": dict(sorted(encoding_counts.items())),
            "language_signal_counts": dict(sorted(language_counts.items())),
            "method_counts": dict(sorted(method_counts.items())),
            "full_method_counts": dict(sorted(full_method_counts.items())),
            "structured_qa_documents": sum(
                "+qa_structured" in (doc.extraction_method or "")
                for doc in successful
            ),
            "clean_raw_character_ratio_mean": round(statistics.fmean(ratios), 6)
            if ratios
            else 0,
            "clean_raw_character_ratio_median": round(statistics.median(ratios), 6)
            if ratios
            else 0,
            "paragraph_count_mean": round(statistics.fmean(paragraphs), 2)
            if paragraphs
            else 0,
            "paragraph_count_median": statistics.median(paragraphs) if paragraphs else 0,
            "paragraph_count_distribution": _distribution(paragraphs),
            "replacement_character_total": sum(
                doc.replacement_character_count for doc in documents
            ),
        },
        "chunking": chunk_summaries,
        "outputs": {
            "documents_parquet": str(documents_path),
            "chunks_directory": str(chunk_output_dir),
        },
    }
    summary_path = Path(summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
