from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from .embedder import file_sha256


SCHEMA = """
CREATE TABLE metadata (
    faiss_row INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    doc_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    section_type TEXT NOT NULL,
    heading_path_json TEXT NOT NULL,
    source_url TEXT NOT NULL,
    extraction_method TEXT NOT NULL
);
CREATE INDEX idx_metadata_doc_id ON metadata(doc_id);
CREATE TABLE manifest (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class ChunkMetadataStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ChunkMetadataStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM metadata").fetchone()[0])

    def manifest(self) -> dict[str, str]:
        return {
            str(row["key"]): str(row["value"])
            for row in self.connection.execute("SELECT key, value FROM manifest")
        }

    def get_rows(self, row_ids: Iterable[int]) -> list[dict[str, Any]]:
        requested = [int(value) for value in row_ids]
        if not requested:
            return []
        found: dict[int, sqlite3.Row] = {}
        for offset in range(0, len(requested), 900):
            group = requested[offset : offset + 900]
            placeholders = ",".join("?" for _ in group)
            for row in self.connection.execute(
                f"SELECT * FROM metadata WHERE faiss_row IN ({placeholders})", group
            ):
                found[int(row["faiss_row"])] = row
        result = []
        for row_id in requested:
            row = found[row_id]
            item = dict(row)
            item["heading_path"] = json.loads(item.pop("heading_path_json"))
            result.append(item)
        return result


def build_metadata_store(chunks_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    source = Path(chunks_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    connection.executescript(SCHEMA)
    digest = hashlib.sha256()
    row_number = 0
    parquet = pq.ParquetFile(source)
    columns = [
        "chunk_id", "doc_id", "chunk_index", "raw_text", "normalized_text",
        "token_count", "start_offset", "end_offset", "section_type",
        "heading_path", "source_url", "extraction_method",
    ]
    for batch in parquet.iter_batches(batch_size=2048, columns=columns):
        records = batch.to_pylist()
        values = []
        for record in records:
            chunk_id = str(record["chunk_id"])
            digest.update(chunk_id.encode("utf-8"))
            digest.update(b"\0")
            values.append(
                (
                    row_number,
                    chunk_id,
                    int(record["doc_id"]),
                    int(record["chunk_index"]),
                    str(record["raw_text"]),
                    str(record["normalized_text"]),
                    int(record["token_count"]),
                    int(record["start_offset"]),
                    int(record["end_offset"]),
                    str(record["section_type"]),
                    json.dumps(record["heading_path"], ensure_ascii=False),
                    str(record["source_url"]),
                    str(record["extraction_method"]),
                )
            )
            row_number += 1
        with connection:
            connection.executemany(
                "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
    manifest = {
        "chunk_source": str(source),
        "chunk_source_sha256": file_sha256(source),
        "chunk_id_order_sha256": digest.hexdigest(),
        "row_count": str(row_number),
    }
    with connection:
        connection.executemany(
            "INSERT INTO manifest(key, value) VALUES (?, ?)", manifest.items()
        )
    connection.execute("PRAGMA integrity_check").fetchone()
    connection.close()
    os.replace(temporary, destination)
    return {**manifest, "path": str(destination), "size_bytes": destination.stat().st_size}


def verify_metadata_store(
    chunks_path: str | Path, metadata_path: str | Path
) -> dict[str, int]:
    checked = 0
    mismatches = 0
    with ChunkMetadataStore(metadata_path) as store:
        for batch in pq.ParquetFile(chunks_path).iter_batches(
            batch_size=2048, columns=["chunk_id", "doc_id"]
        ):
            source_rows = batch.to_pylist()
            mapped_rows = store.get_rows(range(checked, checked + len(source_rows)))
            for source, mapped in zip(source_rows, mapped_rows, strict=True):
                if (
                    str(source["chunk_id"]) != str(mapped["chunk_id"])
                    or int(source["doc_id"]) != int(mapped["doc_id"])
                ):
                    mismatches += 1
            checked += len(source_rows)
    return {"rows_checked": checked, "mismatches": mismatches}
