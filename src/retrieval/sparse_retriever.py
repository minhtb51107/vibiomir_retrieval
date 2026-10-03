from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import statistics
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq

from src.indexing.embedder import file_sha256, normalize_text
from src.indexing.metadata_store import ChunkMetadataStore

from .lexical_tokenizer import TOKENIZER_NAME, tokenize_lexical


SCHEMA = """
CREATE TABLE documents (
    row_id INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    doc_id INTEGER NOT NULL,
    token_count INTEGER NOT NULL
);
CREATE TABLE terms (
    term_id INTEGER PRIMARY KEY,
    term TEXT NOT NULL UNIQUE,
    document_frequency INTEGER NOT NULL
);
CREATE TABLE postings (
    term_id INTEGER NOT NULL,
    row_id INTEGER NOT NULL,
    term_frequency INTEGER NOT NULL,
    PRIMARY KEY(term_id, row_id)
);
CREATE TABLE manifest (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _chunk_order_hash(chunks_path: Path) -> str:
    digest = hashlib.sha256()
    for batch in pq.ParquetFile(chunks_path).iter_batches(
        batch_size=2048, columns=["chunk_id"]
    ):
        for chunk_id in batch.column(0).to_pylist():
            digest.update(str(chunk_id).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def build_sparse_index(
    *,
    chunks_path: str | Path,
    output_path: str | Path,
    k1: float,
    b: float,
    dense_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
    source = Path(chunks_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    source_hash = file_sha256(source)
    order_hash = _chunk_order_hash(source)
    if dense_metadata_path is not None:
        with ChunkMetadataStore(dense_metadata_path) as dense_metadata:
            manifest = dense_metadata.manifest()
            if manifest.get("chunk_source_sha256") != source_hash:
                raise ValueError("validated chunks do not match dense metadata source")
            if manifest.get("chunk_id_order_sha256") != order_hash:
                raise ValueError("validated chunk order does not match dense metadata")

    started = time.perf_counter()
    connection = sqlite3.connect(temporary)
    connection.executescript(SCHEMA)
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA journal_mode=OFF")
    document_frequencies: Counter[str] = Counter()
    total_tokens = 0
    document_lengths: list[int] = []
    document_count = 0
    duplicate_chunk_ids = 0
    columns = ["chunk_id", "doc_id", "normalized_text"]
    parquet = pq.ParquetFile(source)
    try:
        with connection:
            for batch in parquet.iter_batches(batch_size=512, columns=columns):
                document_rows = []
                for record in batch.to_pylist():
                    tokens = tokenize_lexical(str(record["normalized_text"]))
                    document_frequencies.update(set(tokens))
                    document_rows.append(
                        (
                            document_count,
                            str(record["chunk_id"]),
                            int(record["doc_id"]),
                            len(tokens),
                        )
                    )
                    total_tokens += len(tokens)
                    document_lengths.append(len(tokens))
                    document_count += 1
                try:
                    connection.executemany(
                        "INSERT INTO documents VALUES (?, ?, ?, ?)", document_rows
                    )
                except sqlite3.IntegrityError as error:
                    duplicate_chunk_ids += 1
                    raise ValueError("duplicate chunk_id in validated chunks") from error

            ordered_terms = sorted(document_frequencies)
            term_ids = {term: index for index, term in enumerate(ordered_terms)}
            connection.executemany(
                "INSERT INTO terms VALUES (?, ?, ?)",
                (
                    (term_ids[term], term, int(document_frequencies[term]))
                    for term in ordered_terms
                ),
            )

            row_id = 0
            for batch in parquet.iter_batches(batch_size=128, columns=["normalized_text"]):
                postings = []
                for text in batch.column(0).to_pylist():
                    counts = Counter(tokenize_lexical(str(text)))
                    postings.extend(
                        (term_ids[term], row_id, int(frequency))
                        for term, frequency in sorted(counts.items())
                    )
                    row_id += 1
                connection.executemany(
                    "INSERT INTO postings VALUES (?, ?, ?)", postings
                )
            connection.execute("CREATE INDEX idx_postings_term ON postings(term_id)")
            connection.execute("CREATE INDEX idx_documents_doc ON documents(doc_id)")
            average_length = total_tokens / max(document_count, 1)
            manifest = {
                "format_version": "1",
                "tokenizer": TOKENIZER_NAME,
                "chunk_source": str(source),
                "chunk_source_sha256": source_hash,
                "chunk_id_order_sha256": order_hash,
                "document_count": str(document_count),
                "term_count": str(len(ordered_terms)),
                "total_tokens": str(total_tokens),
                "average_document_length": repr(average_length),
                "bm25_k1": repr(float(k1)),
                "bm25_b": repr(float(b)),
            }
            connection.executemany(
                "INSERT INTO manifest(key, value) VALUES (?, ?)", manifest.items()
            )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"sparse index integrity check failed: {integrity}")
    finally:
        connection.close()
    os.replace(temporary, destination)
    elapsed = time.perf_counter() - started
    ordered_lengths = sorted(document_lengths)
    p95_index = round(0.95 * (len(ordered_lengths) - 1)) if ordered_lengths else 0
    return {
        **manifest,
        "document_count": document_count,
        "term_count": len(document_frequencies),
        "total_tokens": total_tokens,
        "average_document_length": average_length,
        "document_length_tokens": {
            "min": min(ordered_lengths, default=0),
            "median": statistics.median(ordered_lengths) if ordered_lengths else 0,
            "p95": ordered_lengths[p95_index] if ordered_lengths else 0,
            "max": max(ordered_lengths, default=0),
        },
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "build_seconds": round(elapsed, 6),
        "index_path": str(destination),
        "index_size_bytes": destination.stat().st_size,
        "index_sha256": file_sha256(destination),
        "integrity_check": integrity,
    }


class SparseRetriever:
    def __init__(
        self,
        *,
        index_path: str | Path,
        metadata: ChunkMetadataStore,
    ):
        self.path = Path(index_path)
        self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        self.metadata = metadata
        manifest = {
            str(row["key"]): str(row["value"])
            for row in self.connection.execute("SELECT key, value FROM manifest")
        }
        self.document_count = int(manifest["document_count"])
        self.average_document_length = float(manifest["average_document_length"])
        self.k1 = float(manifest["bm25_k1"])
        self.b = float(manifest["bm25_b"])
        self.tokenizer_name = manifest["tokenizer"]
        dense_manifest = metadata.manifest()
        if manifest["chunk_source_sha256"] != dense_manifest.get("chunk_source_sha256"):
            raise ValueError("sparse index and dense metadata sources differ")
        if manifest["chunk_id_order_sha256"] != dense_manifest.get("chunk_id_order_sha256"):
            raise ValueError("sparse index and dense metadata row order differs")
        if self.document_count != metadata.count:
            raise ValueError("sparse index and metadata row counts differ")
        self._document_lengths = {
            int(row["row_id"]): int(row["token_count"])
            for row in self.connection.execute("SELECT row_id, token_count FROM documents")
        }

    def close(self) -> None:
        self._postings.cache_clear()
        self.connection.close()

    def __enter__(self) -> SparseRetriever:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @lru_cache(maxsize=20000)
    def _postings(self, term: str) -> tuple[int, tuple[tuple[int, int], ...]]:
        term_row = self.connection.execute(
            "SELECT term_id, document_frequency FROM terms WHERE term = ?", (term,)
        ).fetchone()
        if term_row is None:
            return 0, ()
        rows = self.connection.execute(
            "SELECT row_id, term_frequency FROM postings WHERE term_id = ? ORDER BY row_id",
            (int(term_row["term_id"]),),
        )
        return int(term_row["document_frequency"]), tuple(
            (int(row["row_id"]), int(row["term_frequency"])) for row in rows
        )

    def retrieve(
        self,
        query_ids: Sequence[int],
        query_texts: Sequence[str],
        *,
        top_k: int,
    ) -> list[list[dict[str, Any]]]:
        if len(query_ids) != len(query_texts):
            raise ValueError("query IDs and texts must have equal length")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        return [
            self._retrieve_one(int(query_id), str(query_text), top_k)
            for query_id, query_text in zip(query_ids, query_texts, strict=True)
        ]

    def _retrieve_one(
        self, query_id: int, query_text: str, top_k: int
    ) -> list[dict[str, Any]]:
        normalized_query = normalize_text(query_text)
        query_terms = Counter(tokenize_lexical(normalized_query))
        scores: dict[int, float] = {}
        for term, query_frequency in sorted(query_terms.items()):
            document_frequency, postings = self._postings(term)
            if not postings:
                continue
            inverse_document_frequency = math.log(
                1.0
                + (self.document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for row_id, term_frequency in postings:
                document_length = self._document_lengths[row_id]
                denominator = term_frequency + self.k1 * (
                    1.0
                    - self.b
                    + self.b * document_length / max(self.average_document_length, 1e-12)
                )
                contribution = inverse_document_frequency * (
                    term_frequency * (self.k1 + 1.0) / denominator
                )
                scores[row_id] = scores.get(row_id, 0.0) + query_frequency * contribution
        ranked_rows = sorted(scores, key=lambda row_id: (-scores[row_id], row_id))[:top_k]
        metadata_rows = self.metadata.get_rows(ranked_rows)
        results = []
        for rank, (row_id, row) in enumerate(
            zip(ranked_rows, metadata_rows, strict=True), start=1
        ):
            results.append(
                {
                    "query_id": query_id,
                    "query_text": normalized_query,
                    "rank": rank,
                    "score": float(scores[row_id]),
                    "chunk_id": row["chunk_id"],
                    "doc_id": int(row["doc_id"]),
                    "chunk_text": row["raw_text"],
                    "source_url": row["source_url"],
                    "sparse_row": row_id,
                    "chunk_index": int(row["chunk_index"]),
                    "start_offset": int(row["start_offset"]),
                    "end_offset": int(row["end_offset"]),
                    "section_type": row["section_type"],
                    "heading_path": row["heading_path"],
                    "extraction_method": row["extraction_method"],
                }
            )
        return results


def verify_sparse_mapping(
    index_path: str | Path, metadata_path: str | Path
) -> dict[str, int]:
    connection = sqlite3.connect(index_path)
    connection.row_factory = sqlite3.Row
    checked = 0
    mismatches = 0
    try:
        with ChunkMetadataStore(metadata_path) as metadata:
            for offset in range(0, metadata.count, 900):
                expected = metadata.get_rows(range(offset, min(offset + 900, metadata.count)))
                actual = list(
                    connection.execute(
                        "SELECT row_id, chunk_id, doc_id FROM documents "
                        "WHERE row_id >= ? AND row_id < ? ORDER BY row_id",
                        (offset, min(offset + 900, metadata.count)),
                    )
                )
                for dense, sparse in zip(expected, actual, strict=True):
                    if (
                        int(dense["faiss_row"]) != int(sparse["row_id"])
                        or str(dense["chunk_id"]) != str(sparse["chunk_id"])
                        or int(dense["doc_id"]) != int(sparse["doc_id"])
                    ):
                        mismatches += 1
                    checked += 1
    finally:
        connection.close()
    return {"rows_checked": checked, "mismatches": mismatches}
