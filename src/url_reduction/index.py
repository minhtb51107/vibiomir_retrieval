from __future__ import annotations

import hashlib
import heapq
import json
import os
import sqlite3
import struct
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from src.indexing.embedder import file_sha256

from .url_text import UrlTextFeatures, extract_url_features, lexical_tokens


SCHEMA = """
CREATE TABLE queries (
    query_id INTEGER PRIMARY KEY,
    query_text TEXT NOT NULL,
    tokens_json TEXT NOT NULL,
    token_count INTEGER NOT NULL
);
CREATE TABLE terms (
    term_id INTEGER PRIMARY KEY,
    term TEXT NOT NULL UNIQUE,
    document_frequency INTEGER NOT NULL
);
CREATE TABLE documents (
    doc_id INTEGER PRIMARY KEY,
    original_url TEXT NOT NULL,
    url_text TEXT NOT NULL,
    domain TEXT NOT NULL,
    scheme TEXT NOT NULL,
    document_length INTEGER NOT NULL,
    path_token_count INTEGER NOT NULL,
    numeric_ratio REAL NOT NULL,
    has_han INTEGER NOT NULL,
    vietnamese_like INTEGER NOT NULL,
    opaque INTEGER NOT NULL,
    text_hash TEXT NOT NULL,
    lexical_match INTEGER NOT NULL
);
CREATE VIRTUAL TABLE url_fts USING fts5(tokens, content='');
CREATE TABLE fallback (
    domain TEXT NOT NULL,
    slot INTEGER NOT NULL,
    doc_id INTEGER NOT NULL UNIQUE,
    PRIMARY KEY(domain, slot)
);
CREATE TABLE domain_stats (
    domain TEXT PRIMARY KEY,
    corpus_count INTEGER NOT NULL
);
CREATE TABLE manifest (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _document_row(feature: UrlTextFeatures, doc_id: int, lexical_match: bool) -> tuple[Any, ...]:
    return (
        doc_id,
        feature.original_url,
        feature.normalized_text,
        feature.domain,
        feature.scheme,
        max(len(feature.tokens), 1),
        feature.path_token_count,
        feature.numeric_ratio,
        int(feature.has_han),
        int(feature.vietnamese_like),
        int(feature.opaque),
        f"{feature.text_hash_u64:016x}",
        int(lexical_match),
    )


def _duplicate_hash_stats(path: Path, rows: int) -> dict[str, int]:
    values = np.memmap(path, dtype="<u8", mode="r+", shape=(rows,))
    values.sort()
    duplicate_rows = 0
    duplicate_groups = 0
    previous_equal = False
    block_size = 1_000_000
    for start in range(1, rows, block_size):
        stop = min(rows, start + block_size)
        equal = np.asarray(values[start:stop] == values[start - 1 : stop - 1])
        duplicate_rows += int(equal.sum())
        if len(equal):
            starts = equal & np.concatenate(([not previous_equal], ~equal[:-1]))
            duplicate_groups += int(starts.sum())
            previous_equal = bool(equal[-1])
    del values
    return {
        "duplicate_text_groups": duplicate_groups,
        "repeated_rows_beyond_first": duplicate_rows,
        "unique_text_hashes": rows - duplicate_rows,
    }


def build_query_conditioned_index(
    *,
    corpus_path: str | Path,
    query_path: str | Path,
    output_path: str | Path,
    hash_path: str | Path,
    stopwords: frozenset[str],
    batch_rows: int,
    fallback_rows_per_domain: int,
    fallback_seed: str,
    sqlite_cache_mib: int,
    maximum_url_characters: int,
) -> dict[str, Any]:
    if batch_rows <= 0 or fallback_rows_per_domain <= 0:
        raise ValueError("batch and fallback sizes must be positive")
    queries_table = pq.read_table(query_path, columns=["id", "query"])
    query_rows = queries_table.to_pylist()
    query_tokens: dict[int, list[str]] = {
        int(row["id"]): lexical_tokens(
            str(row["query"]), stopwords=stopwords, accent_folded=True
        )
        for row in query_rows
    }
    vocabulary = sorted({term for values in query_tokens.values() for term in values})
    term_ids = {term: index for index, term in enumerate(vocabulary)}
    df = np.zeros(len(vocabulary), dtype=np.int64)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    hashes = Path(hash_path)
    hashes.parent.mkdir(parents=True, exist_ok=True)
    hash_temporary = hashes.with_suffix(hashes.suffix + ".tmp")
    if hash_temporary.exists():
        hash_temporary.unlink()

    connection = sqlite3.connect(temporary)
    connection.executescript(SCHEMA)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(f"PRAGMA cache_size=-{max(1024, sqlite_cache_mib * 1024)}")
    connection.executemany(
        "INSERT INTO queries VALUES (?, ?, ?, ?)",
        (
            (
                int(row["id"]),
                str(row["query"]),
                json.dumps(query_tokens[int(row["id"])], ensure_ascii=False),
                len(query_tokens[int(row["id"])]),
            )
            for row in query_rows
        ),
    )

    corpus = pq.ParquetFile(corpus_path)
    domain_counts: Counter[str] = Counter()
    scheme_counts: Counter[str] = Counter()
    fallback_heaps: dict[str, list[tuple[int, int, tuple[Any, ...]]]] = {}
    total_length = 0
    matched_documents = 0
    han_count = 0
    vietnamese_count = 0
    opaque_count = 0
    path_signal_count = 0
    peak_rss = _rss_bytes() or 0
    scan_started = time.perf_counter()
    processed = 0
    with hash_temporary.open("wb") as hash_handle:
        for batch in corpus.iter_batches(batch_size=batch_rows, columns=["id", "url"]):
            document_rows: list[tuple[Any, ...]] = []
            fts_rows: list[tuple[int, str]] = []
            hash_buffer = bytearray()
            for raw_id, raw_url in zip(
                batch.column("id").to_pylist(), batch.column("url").to_pylist(), strict=True
            ):
                doc_id = int(raw_id)
                feature = extract_url_features(
                    str(raw_url),
                    stopwords=stopwords,
                    maximum_characters=maximum_url_characters,
                )
                total_length += max(len(feature.tokens), 1)
                domain_counts[feature.domain] += 1
                scheme_counts[feature.scheme or "missing"] += 1
                han_count += int(feature.has_han)
                vietnamese_count += int(feature.vietnamese_like)
                opaque_count += int(feature.opaque)
                path_signal_count += int(feature.path_token_count > 0)
                hash_buffer.extend(struct.pack("<Q", feature.text_hash_u64))

                matching = [
                    (term_ids[term], frequency)
                    for term, frequency in feature.token_counts
                    if term in term_ids
                ]
                if matching:
                    matched_documents += 1
                    document_rows.append(_document_row(feature, doc_id, True))
                    indexed_frequency = 0
                    encoded_terms: list[str] = []
                    for term_id, frequency in matching:
                        df[term_id] += 1
                        indexed_frequency += frequency
                        encoded_terms.extend([f"t{term_id}"] * frequency)
                    # Preserve URL-token length in FTS5's BM25 normalization
                    # without indexing the unbounded non-query vocabulary.
                    encoded_terms.extend(
                        ["urltoken"] * max(len(feature.tokens) - indexed_frequency, 0)
                    )
                    fts_rows.append((doc_id, " ".join(encoded_terms)))

                rank = int.from_bytes(
                    hashlib.blake2b(
                        f"{fallback_seed}:{doc_id}".encode(), digest_size=8
                    ).digest(),
                    "big",
                )
                heap = fallback_heaps.setdefault(feature.domain, [])
                candidate = (-rank, doc_id, _document_row(feature, doc_id, False))
                if len(heap) < fallback_rows_per_domain:
                    heapq.heappush(heap, candidate)
                elif candidate > heap[0]:
                    heapq.heapreplace(heap, candidate)
            with connection:
                connection.executemany(
                    "INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    document_rows,
                )
                connection.executemany(
                    "INSERT INTO url_fts(rowid, tokens) VALUES (?, ?)", fts_rows
                )
            hash_handle.write(hash_buffer)
            processed += len(batch)
            peak_rss = max(peak_rss, _rss_bytes() or 0)
            if processed % (batch_rows * 20) < len(batch):
                elapsed = time.perf_counter() - scan_started
                print(f"processed {processed}/{corpus.metadata.num_rows} URLs ({processed / max(elapsed, 1e-9):.1f}/s)")
    scan_seconds = time.perf_counter() - scan_started

    finalize_started = time.perf_counter()
    fallback_rows: list[tuple[str, int, int]] = []
    fallback_documents: list[tuple[Any, ...]] = []
    for domain, heap in sorted(fallback_heaps.items()):
        ordered = sorted(((-negative, doc_id, row) for negative, doc_id, row in heap))
        for slot, (_, doc_id, row) in enumerate(ordered):
            fallback_rows.append((domain, slot, doc_id))
            fallback_documents.append(row)
    with connection:
        connection.executemany(
            "INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            fallback_documents,
        )
        connection.executemany("INSERT INTO fallback VALUES (?, ?, ?)", fallback_rows)
        connection.executemany(
            "INSERT INTO terms VALUES (?, ?, ?)",
            ((index, term, int(df[index])) for index, term in enumerate(vocabulary)),
        )
        connection.executemany(
            "INSERT INTO domain_stats VALUES (?, ?)", sorted(domain_counts.items())
        )
        manifest = {
            "format_version": "1",
            "corpus_sha256": file_sha256(corpus_path),
            "query_sha256": file_sha256(query_path),
            "corpus_rows": str(processed),
            "query_count": str(len(query_rows)),
            "query_vocabulary_terms": str(len(vocabulary)),
            "average_document_length": repr(total_length / max(processed, 1)),
            "fallback_seed": fallback_seed,
            "fallback_rows_per_domain": str(fallback_rows_per_domain),
        }
        connection.executemany("INSERT INTO manifest VALUES (?, ?)", manifest.items())
        connection.execute("CREATE INDEX idx_documents_domain ON documents(domain)")
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    stored_documents = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    posting_count = int(df.sum())
    connection.close()
    os.replace(temporary, destination)
    os.replace(hash_temporary, hashes)
    duplicate_stats = _duplicate_hash_stats(hashes, processed)
    finalize_seconds = time.perf_counter() - finalize_started
    return {
        "corpus_rows": processed,
        "query_count": len(query_rows),
        "query_vocabulary_terms": len(vocabulary),
        "stored_candidate_documents": stored_documents,
        "lexically_matched_documents": matched_documents,
        "posting_count": posting_count,
        "average_url_token_length": round(total_length / max(processed, 1), 6),
        "full_corpus": {
            "unique_domains": len(domain_counts),
            "domain_counts": dict(domain_counts.most_common()),
            "scheme_counts": dict(sorted(scheme_counts.items())),
            "han_url_count": han_count,
            "vietnamese_like_url_count": vietnamese_count,
            "opaque_url_count": opaque_count,
            "path_signal_count": path_signal_count,
            **duplicate_stats,
        },
        "runtime": {
            "preprocessing_scan_seconds": round(scan_seconds, 6),
            "index_finalize_seconds": round(finalize_seconds, 6),
            "total_seconds": round(scan_seconds + finalize_seconds, 6),
            "urls_per_second": round(processed / max(scan_seconds, 1e-9), 3),
            "peak_process_rss_bytes": peak_rss or None,
        },
        "storage": {
            "index_path": str(destination),
            "index_size_bytes": destination.stat().st_size,
            "index_sha256": file_sha256(destination),
            "hash_path": str(hashes),
            "hash_size_bytes": hashes.stat().st_size,
        },
        "integrity_check": integrity,
        "manifest": manifest,
    }
