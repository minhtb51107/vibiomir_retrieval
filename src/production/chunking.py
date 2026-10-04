from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.indexing.embedder import file_sha256
from src.indexing.tokenizer_validation import HuggingFaceOffsetTokenizer
from src.processing.chunking import ChunkConfig, chunk_document
from src.processing.models import CleanDocument, DocumentSection, ExtractionStatus

from .state import PartitionCheckpoint


CHUNK_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("doc_id", pa.int64()),
        ("chunk_index", pa.int64()),
        ("raw_text", pa.string()),
        ("normalized_text", pa.string()),
        ("token_count", pa.int64()),
        ("start_offset", pa.int64()),
        ("end_offset", pa.int64()),
        ("section_type", pa.string()),
        ("heading_path", pa.list_(pa.string())),
        ("source_url", pa.string()),
        ("extraction_method", pa.string()),
    ]
)


def _document(row: dict[str, Any]) -> CleanDocument:
    return CleanDocument(
        **{
            **row,
            "sections": [DocumentSection(**value) for value in row["sections"]],
            "extraction_status": ExtractionStatus(row["extraction_status"]),
        }
    )


def build_chunk_partitions(
    *,
    tokenizer: Any,
    source_documents: str | Path,
    output_directory: str | Path,
    checkpoint_path: str | Path,
    tokenizer_revision: str,
    content_token_budget: int = 508,
    overlap_tokens: int = 64,
    documents_per_partition: int = 1000,
) -> dict[str, Any]:
    if content_token_budget != 508 or overlap_tokens != 64:
        raise ValueError("production chunking requires the validated 508/64 policy")
    if documents_per_partition <= 0:
        raise ValueError("documents_per_partition must be positive")
    source = Path(source_documents)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    source_hash = file_sha256(source)
    signature = {
        "chunk_algorithm_version": 2,
        "source_sha256": source_hash,
        "tokenizer_revision": tokenizer_revision,
        "content_token_budget": content_token_budget,
        "overlap_tokens": overlap_tokens,
        "documents_per_partition": documents_per_partition,
    }
    adapter = HuggingFaceOffsetTokenizer(tokenizer)
    chunk_config = ChunkConfig(
        "bge_m3_512_overlap_64", content_token_budget, overlap_tokens
    )
    started = time.perf_counter()
    initial_completed = 0
    written_documents = 0
    written_chunks = 0
    partitions_written = 0
    maximum_tokens = 0
    parquet = pq.ParquetFile(source)
    with PartitionCheckpoint(checkpoint_path, signature=signature) as checkpoint:
        initial_completed = checkpoint.count
        for partition_index, batch in enumerate(
            parquet.iter_batches(batch_size=documents_per_partition)
        ):
            rows = batch.to_pylist()
            item_ids = [str(int(row["doc_id"])) for row in rows]
            if all(checkpoint.is_completed(item_id) for item_id in item_ids):
                continue
            if any(checkpoint.is_completed(item_id) for item_id in item_ids):
                raise RuntimeError("partially committed deterministic chunk partition")
            chunk_rows = []
            for row in rows:
                document = _document(row)
                if document.extraction_status != ExtractionStatus.SUCCESS:
                    continue
                for chunk in chunk_document(document, chunk_config, adapter):
                    chunk_rows.append(chunk.as_dict())
                    maximum_tokens = max(maximum_tokens, chunk.token_count)
            partition_id = f"part-{partition_index:06d}"
            destination = output / f"{partition_id}.parquet"
            temporary = destination.with_suffix(".parquet.tmp")
            table = pa.Table.from_pylist(chunk_rows, schema=CHUNK_SCHEMA)
            pq.write_table(table, temporary)
            os.replace(temporary, destination)
            output_hash = file_sha256(destination)
            checkpoint.commit_partition(
                partition_id=partition_id,
                item_ids=item_ids,
                output_sha256=output_hash,
            )
            written_documents += len(rows)
            written_chunks += len(chunk_rows)
            partitions_written += 1
        if checkpoint.integrity_check() != "ok":
            raise RuntimeError("chunk partition checkpoint integrity failure")
        completed = checkpoint.count
    return {
        "source_documents": str(source),
        "source_sha256": source_hash,
        "tokenizer_revision": tokenizer_revision,
        "content_token_budget": content_token_budget,
        "maximum_tokens_with_specials": 512,
        "overlap_tokens": overlap_tokens,
        "documents_per_partition": documents_per_partition,
        "initial_completed_documents": initial_completed,
        "final_completed_documents": completed,
        "documents_written_this_run": written_documents,
        "chunks_written_this_run": written_chunks,
        "partitions_written_this_run": partitions_written,
        "maximum_content_tokens": maximum_tokens,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "output_directory": str(output),
        "checkpoint": str(checkpoint_path),
    }
