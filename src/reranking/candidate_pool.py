from __future__ import annotations

import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


SOURCES = ("dense", "sparse", "hybrid")
PROVENANCE_FIELDS = (
    "chunk_index",
    "start_offset",
    "end_offset",
    "section_type",
    "heading_path",
    "extraction_method",
)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _source_rows(path: str | Path, depth: int) -> Iterable[dict[str, Any]]:
    for row in pq.read_table(path).to_pylist():
        if int(row["rank"]) <= depth:
            yield row


def build_unified_candidate_pool(
    *,
    dense_rows: Iterable[dict[str, Any]],
    sparse_rows: Iterable[dict[str, Any]],
    hybrid_rows: Iterable[dict[str, Any]],
    max_depth: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_depth <= 0:
        raise ValueError("max_depth must be positive")
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    preparation_ms: dict[int, float] = defaultdict(float)
    source_counts = {source: 0 for source in SOURCES}
    duplicate_source_rows = 0

    for source, rows in zip(SOURCES, (dense_rows, sparse_rows, hybrid_rows), strict=True):
        for original in rows:
            started = time.perf_counter()
            query_id = int(original["query_id"])
            chunk_id = str(original["chunk_id"])
            existing = grouped[query_id].get(chunk_id)
            if existing is None:
                existing = {
                    "query_id": query_id,
                    "query_text": str(original["query_text"]),
                    "chunk_id": chunk_id,
                    "doc_id": int(original["doc_id"]),
                    "chunk_text": str(original["chunk_text"]),
                    "source_url": str(original["source_url"]),
                    "dense_rank": None,
                    "dense_score": None,
                    "sparse_rank": None,
                    "sparse_score": None,
                    "hybrid_rank": None,
                    "hybrid_score": None,
                }
                for field in PROVENANCE_FIELDS:
                    existing[field] = original.get(field)
                grouped[query_id][chunk_id] = existing
            else:
                invariant = (
                    str(original["query_text"]),
                    int(original["doc_id"]),
                    str(original["chunk_text"]),
                    str(original["source_url"]),
                )
                expected = (
                    existing["query_text"],
                    existing["doc_id"],
                    existing["chunk_text"],
                    existing["source_url"],
                )
                if invariant != expected:
                    raise ValueError(
                        f"candidate provenance mismatch for query={query_id}, chunk={chunk_id}"
                    )
            rank_key = f"{source}_rank"
            score_key = f"{source}_score"
            if existing[rank_key] is not None:
                duplicate_source_rows += 1
                if (int(original["rank"]), -float(original["score"])) >= (
                    int(existing[rank_key]),
                    -float(existing[score_key]),
                ):
                    preparation_ms[query_id] += (time.perf_counter() - started) * 1000
                    continue
            existing[rank_key] = int(original["rank"])
            existing[score_key] = float(original["score"])
            source_counts[source] += 1
            preparation_ms[query_id] += (time.perf_counter() - started) * 1000

    output: list[dict[str, Any]] = []
    union_sizes: list[int] = []
    selected_sizes: list[int] = []
    for query_id in sorted(grouped):
        started = time.perf_counter()
        rows = list(grouped[query_id].values())
        union_sizes.append(len(rows))
        infinity = 10**9
        rows.sort(
            key=lambda row: (
                row["hybrid_rank"] is None,
                row["hybrid_rank"] if row["hybrid_rank"] is not None else infinity,
                min(
                    row["dense_rank"] if row["dense_rank"] is not None else infinity,
                    row["sparse_rank"] if row["sparse_rank"] is not None else infinity,
                ),
                row["dense_rank"] if row["dense_rank"] is not None else infinity,
                row["sparse_rank"] if row["sparse_rank"] is not None else infinity,
                row["chunk_id"],
            )
        )
        selected = rows[:max_depth]
        selected_sizes.append(len(selected))
        for pool_rank, row in enumerate(selected, start=1):
            membership = [source for source in SOURCES if row[f"{source}_rank"] is not None]
            row["pool_rank"] = pool_rank
            row["source_membership"] = membership
            row["source_membership_key"] = "+".join(membership)
            output.append(row)
        preparation_ms[query_id] += (time.perf_counter() - started) * 1000

    timings = list(preparation_ms.values())
    summary = {
        "query_count": len(grouped),
        "candidate_rows": len(output),
        "max_depth": max_depth,
        "source_rows_read": source_counts,
        "duplicate_source_rows": duplicate_source_rows,
        "union_size": {
            "min": min(union_sizes, default=0),
            "median": statistics.median(union_sizes) if union_sizes else 0,
            "max": max(union_sizes, default=0),
            "mean": statistics.fmean(union_sizes) if union_sizes else 0.0,
        },
        "selected_size": {
            "min": min(selected_sizes, default=0),
            "max": max(selected_sizes, default=0),
        },
        "preparation_latency_ms_per_query": {
            "mean": statistics.fmean(timings) if timings else 0.0,
            "p50": _percentile(timings, 0.5),
            "p95": _percentile(timings, 0.95),
        },
    }
    return output, summary


def build_pool_from_paths(
    *,
    dense_path: str | Path,
    sparse_path: str | Path,
    hybrid_path: str | Path,
    dense_depth: int,
    sparse_depth: int,
    hybrid_depth: int,
    max_depth: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return build_unified_candidate_pool(
        dense_rows=_source_rows(dense_path, dense_depth),
        sparse_rows=_source_rows(sparse_path, sparse_depth),
        hybrid_rows=_source_rows(hybrid_path, hybrid_depth),
        max_depth=max_depth,
    )
