from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from src.indexing.embedder import Embedder, normalize_text
from src.indexing.faiss_index import ExactFaissIndex
from src.indexing.metadata_store import ChunkMetadataStore

from .dense_retriever import DenseRetriever
from .ranking import aggregate_documents, compare_aggregations


VIETNAMESE_CHARS = frozenset(
    "ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    "ÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
    "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)


def language_signal(text: str) -> str:
    han = sum("\u3400" <= character <= "\u9fff" for character in text)
    vi = sum(character in VIETNAMESE_CHARS for character in text)
    if han >= 20:
        return "zh-Han-script"
    if vi >= 5:
        return "vi-signal"
    return "unknown"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def run_retrieval(
    *,
    query_path: str | Path,
    index_path: str | Path,
    metadata_path: str | Path,
    output_directory: str | Path,
    embedder: Embedder,
    query_batch_size: int,
    top_k_chunks: int,
    top_k_docs: int,
    top_n_mean: int,
    sanity_query_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    query_table = pq.read_table(query_path, columns=["id", "query"])
    queries = query_table.to_pylist()
    model_metadata = embedder.metadata()
    query_token_stats = None
    tokenizer = getattr(embedder, "tokenizer", None)
    if tokenizer is not None:
        query_token_counts = [
            len(
                tokenizer(
                    normalize_text(str(row["query"])),
                    add_special_tokens=True,
                    truncation=False,
                    verbose=False,
                )["input_ids"]
            )
            for row in queries
        ]
        limit = int(model_metadata["max_sequence_length"])
        query_token_stats = {
            "median": _percentile([float(value) for value in query_token_counts], 0.5),
            "p95": round(
                _percentile([float(value) for value in query_token_counts], 0.95), 2
            ),
            "max": max(query_token_counts),
            "exceeding_limit": sum(value > limit for value in query_token_counts),
        }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    chunk_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    mean_rows: list[dict[str, Any]] = []
    embedding_per_query_ms: list[float] = []
    query_vectors = []
    normalized_query_texts = [normalize_text(str(row["query"])) for row in queries]
    for offset in range(0, len(queries), query_batch_size):
        texts = normalized_query_texts[offset : offset + query_batch_size]
        started = time.perf_counter()
        query_vectors.append(embedder.encode(texts))
        elapsed_ms = (time.perf_counter() - started) * 1000
        embedding_per_query_ms.extend([elapsed_ms / len(texts)] * len(texts))
        print(f"embedded {min(offset + query_batch_size, len(queries))}/{len(queries)} queries")
    vectors = np.vstack(query_vectors).astype(np.float32, copy=False)

    index = ExactFaissIndex.load(index_path)
    search_per_query_ms: list[float] = []
    with ChunkMetadataStore(metadata_path) as metadata:
        retriever = DenseRetriever(embedder=embedder, index=index, metadata=metadata)
        for offset in range(0, len(queries), query_batch_size):
            batch = queries[offset : offset + query_batch_size]
            started = time.perf_counter()
            retrieved = retriever.retrieve_vectors(
                [int(row["id"]) for row in batch],
                normalized_query_texts[offset : offset + len(batch)],
                vectors[offset : offset + len(batch)],
                top_k=top_k_chunks,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            search_per_query_ms.extend([elapsed_ms / len(batch)] * len(batch))
            for chunks in retrieved:
                chunk_rows.extend(chunks)
                best_rows.extend(
                    aggregate_documents(
                        chunks, method="best_chunk", top_k_docs=top_k_docs
                    )
                )
                mean_rows.extend(
                    aggregate_documents(
                        chunks,
                        method="top_n_mean",
                        top_k_docs=top_k_docs,
                        top_n=top_n_mean,
                    )
                )
    per_query_ms = [
        embedding + search
        for embedding, search in zip(
            embedding_per_query_ms, search_per_query_ms, strict=True
        )
    ]

    pq.write_table(pa.Table.from_pylist(chunk_rows), output / "chunk_rankings.parquet")
    pq.write_table(pa.Table.from_pylist(best_rows), output / "documents_best_chunk.parquet")
    pq.write_table(pa.Table.from_pylist(mean_rows), output / "documents_top_n_mean.parquet")
    benchmark = {
        "model": model_metadata,
        "query_count": len(queries),
        "query_token_stats": query_token_stats,
        "top_k_chunks": top_k_chunks,
        "top_k_docs": top_k_docs,
        "query_batch_size": query_batch_size,
        "mean_latency_ms_per_query": round(statistics.fmean(per_query_ms), 3),
        "p50_latency_ms_per_query": round(_percentile(per_query_ms, 0.5), 3),
        "p95_latency_ms_per_query": round(_percentile(per_query_ms, 0.95), 3),
        "mean_embedding_latency_ms_per_query": round(
            statistics.fmean(embedding_per_query_ms), 3
        ),
        "mean_faiss_and_mapping_latency_ms_per_query": round(
            statistics.fmean(search_per_query_ms), 3
        ),
        "chunk_ranking_rows": len(chunk_rows),
        "document_ranking_rows_per_method": len(best_rows),
        "aggregation_comparison": compare_aggregations(best_rows, mean_rows),
        "outputs": {
            "chunk_rankings": str(output / "chunk_rankings.parquet"),
            "best_chunk_documents": str(output / "documents_best_chunk.parquet"),
            "top_n_mean_documents": str(output / "documents_top_n_mean.parquet"),
        },
    }

    by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in chunk_rows:
        by_query[int(row["query_id"])].append(row)
    ordered_queries = sorted(queries, key=lambda row: (len(str(row["query"])), int(row["id"])))
    if sanity_query_count >= len(ordered_queries):
        selected = ordered_queries
    else:
        selected = [
            ordered_queries[round(index * (len(ordered_queries) - 1) / (sanity_query_count - 1))]
            for index in range(sanity_query_count)
        ]
    sanity_rows = []
    overall_languages: Counter[str] = Counter()
    for query in selected:
        rows = by_query[int(query["id"])]
        top_ten = rows[:10]
        signals = Counter(language_signal(row["chunk_text"]) for row in top_ten)
        overall_languages.update(signals)
        docs = Counter(int(row["doc_id"]) for row in rows)
        sanity_rows.append(
            {
                "query_id": int(query["id"]),
                "query_characters": len(str(query["query"])),
                "query_preview": str(query["query"])[:120],
                "top_10_language_signals": dict(sorted(signals.items())),
                "top_doc_id": int(rows[0]["doc_id"]),
                "top_chunk_id": rows[0]["chunk_id"],
                "top_score": round(float(rows[0]["score"]), 6),
                "unique_docs_in_top_k_chunks": len(docs),
                "maximum_chunks_from_one_doc": max(docs.values()),
            }
        )
    sanity = {
        "label": "SANITY CHECK — not an official evaluation",
        "selection": "queries evenly spaced across query character-length order",
        "query_count": len(selected),
        "top_10_language_signal_totals": dict(sorted(overall_languages.items())),
        "queries_with_vietnamese_result": sum(
            row["top_10_language_signals"].get("vi-signal", 0) > 0 for row in sanity_rows
        ),
        "queries_with_chinese_result": sum(
            row["top_10_language_signals"].get("zh-Han-script", 0) > 0 for row in sanity_rows
        ),
        "rows": sanity_rows,
    }
    return benchmark, sanity
