from __future__ import annotations

import os
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.indexing.embedder import file_sha256
from src.processing.extraction import config_from_dict, extract_document
from src.processing.pipeline import _content_type_header, _empty_document, _missing_body_status
from src.processing.models import ExtractionStatus
from src.storage.body_archive import BodyArchive

from .state import PartitionCheckpoint


DOCUMENT_SCHEMA = pa.schema(
    [
        ("doc_id", pa.int64()),
        ("original_url", pa.string()),
        ("final_url", pa.string()),
        ("title", pa.string()),
        ("language_signal", pa.string()),
        ("selected_encoding", pa.string()),
        ("normalized_text", pa.string()),
        (
            "sections",
            pa.list_(
                pa.struct(
                    [
                        ("section_type", pa.string()),
                        ("text", pa.string()),
                        ("start_offset", pa.int64()),
                        ("end_offset", pa.int64()),
                        ("heading_path", pa.list_(pa.string())),
                    ]
                )
            ),
        ),
        ("extraction_method", pa.string()),
        ("extraction_status", pa.string()),
        ("raw_byte_length", pa.int64()),
        ("raw_char_count", pa.int64()),
        ("clean_char_count", pa.int64()),
        ("paragraph_count", pa.int64()),
        ("replacement_character_count", pa.int64()),
        ("semantic_char_count", pa.int64()),
        ("density_char_count", pa.int64()),
        ("boilerplate_signal_count", pa.int64()),
        ("raw_sha256", pa.string()),
        ("archive_shard", pa.string()),
        ("archive_member_offset", pa.int64()),
        ("archive_checksum_sha256", pa.string()),
        ("error_type", pa.string()),
        ("error_message", pa.string()),
    ]
)


def _extract_row(row: sqlite3.Row, archive: BodyArchive, config: Any) -> dict[str, Any]:
    doc_id = int(row["doc_id"])
    if not archive.contains(doc_id):
        return _empty_document(row, _missing_body_status(row)).as_dict()
    metadata = archive.metadata(doc_id)
    try:
        body = archive.get_body(doc_id)
        return extract_document(
            doc_id=doc_id,
            original_url=str(row["original_url"]),
            final_url=row["final_url"],
            body=body,
            content_type_header=_content_type_header(metadata),
            archive_shard=str(metadata["shard_name"]),
            archive_member_offset=int(metadata["member_offset"]),
            archive_checksum_sha256=str(metadata["checksum_sha256"]),
            config=config,
        ).as_dict()
    except Exception as error:
        failed = _empty_document(row, ExtractionStatus.EXTRACTION_FAILED)
        failed.error_type = type(error).__name__
        failed.error_message = str(error)[:1000]
        return failed.as_dict()


def build_document_partitions(
    *,
    crawl_database: str | Path,
    archive_directory: str | Path,
    extraction_config_path: str | Path,
    output_directory: str | Path,
    checkpoint_path: str | Path,
    rows_per_partition: int = 1000,
) -> dict[str, Any]:
    if rows_per_partition <= 0:
        raise ValueError("rows_per_partition must be positive")
    crawl = Path(crawl_database)
    extraction_path = Path(extraction_config_path)
    extraction_config = config_from_dict(
        yaml.safe_load(extraction_path.read_text(encoding="utf-8"))
    )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        "crawl_sha256": file_sha256(crawl),
        "extraction_config_sha256": file_sha256(extraction_path),
        "rows_per_partition": rows_per_partition,
        "format_version": 1,
    }
    started = time.perf_counter()
    connection = sqlite3.connect(f"file:{crawl}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    cursor = connection.execute("SELECT * FROM crawl_results ORDER BY doc_id")
    partitions_written = 0
    rows_written = 0
    initial_completed = 0
    try:
        with BodyArchive(archive_directory) as archive, PartitionCheckpoint(
            checkpoint_path, signature=signature
        ) as checkpoint:
            initial_completed = checkpoint.count
            partition_index = 0
            while rows := cursor.fetchmany(rows_per_partition):
                item_ids = [str(int(row["doc_id"])) for row in rows]
                if all(checkpoint.is_completed(item_id) for item_id in item_ids):
                    partition_index += 1
                    continue
                if any(checkpoint.is_completed(item_id) for item_id in item_ids):
                    raise RuntimeError("partially committed deterministic extraction partition")
                records = [_extract_row(row, archive, extraction_config) for row in rows]
                partition_id = f"part-{partition_index:06d}"
                destination = output / f"{partition_id}.parquet"
                temporary = destination.with_suffix(".parquet.tmp")
                pq.write_table(pa.Table.from_pylist(records, schema=DOCUMENT_SCHEMA), temporary)
                os.replace(temporary, destination)
                checkpoint.commit_partition(
                    partition_id=partition_id,
                    item_ids=item_ids,
                    output_sha256=file_sha256(destination),
                )
                partitions_written += 1
                rows_written += len(rows)
                partition_index += 1
            if checkpoint.integrity_check() != "ok":
                raise RuntimeError("extraction checkpoint integrity failure")
            final_completed = checkpoint.count
    finally:
        connection.close()

    status_counts: Counter[str] = Counter()
    successful_chars = 0
    for partition in sorted(output.glob("part-*.parquet")):
        for batch in pq.ParquetFile(partition).iter_batches(
            columns=["extraction_status", "clean_char_count"]
        ):
            for row in batch.to_pylist():
                status_counts[str(row["extraction_status"])] += 1
                if row["extraction_status"] == ExtractionStatus.SUCCESS.value:
                    successful_chars += int(row["clean_char_count"])
    return {
        "crawl_database": str(crawl),
        "crawl_sha256": signature["crawl_sha256"],
        "extraction_config_sha256": signature["extraction_config_sha256"],
        "rows_per_partition": rows_per_partition,
        "initial_completed_rows": initial_completed,
        "final_completed_rows": final_completed,
        "rows_written_this_run": rows_written,
        "partitions_written_this_run": partitions_written,
        "status_counts": dict(sorted(status_counts.items())),
        "successful_clean_characters": successful_chars,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "output_directory": str(output),
        "checkpoint": str(checkpoint_path),
    }
