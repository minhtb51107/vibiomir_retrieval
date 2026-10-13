"""Bounded, disk-backed registry and candidate-ledger helpers.

The functions in this module deliberately operate on metadata and frozen scores.
They never invoke a model or mutate source artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq


def canonical_domain(url: str) -> str:
    return (urlsplit(str(url)).hostname or "unknown").lower().removeprefix("www.")


def content_digest(text: str) -> str:
    normalized = " ".join(str(text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def atomic_parquet_writer(path: Path, schema: pa.Schema) -> tuple[pq.ParquetWriter, Path]:
    """Return a writer backed by a process-unique file in the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    return pq.ParquetWriter(temporary, schema, compression="zstd"), temporary


def commit_parquet(writer: pq.ParquetWriter, temporary: Path, destination: Path) -> None:
    writer.close()
    os.replace(temporary, destination)


def query_groups(path: Path, columns: list[str] | None = None) -> Iterator[list[dict]]:
    """Yield query-contiguous Parquet rows while keeping memory bounded."""
    current: int | None = None
    rows: list[dict] = []
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=4096):
        for row in batch.to_pylist():
            query_id = int(row["query_id"])
            if current is not None and query_id != current:
                yield rows
                rows = []
            current = query_id
            rows.append(row)
    if rows:
        yield rows


def aggregate_candidate_rows(
    query_id: int,
    arm_rows: Iterable[tuple[str, list[dict]]],
    scores: dict[str, float],
    final11: set[int],
    final12: set[int],
    canonical_domains: dict[int, str],
) -> list[dict]:
    """Create one diagnostic row per canonical query/document pair.

    Repeated discovery of an exact chunk through multiple arms is retained as
    arm provenance but counted once for score aggregation.
    """
    documents: dict[int, dict] = {}
    for arm, rows in arm_rows:
        for row in rows:
            doc_id = int(row["doc_id"])
            chunk_id = str(row["chunk_id"])
            rank = int(row.get("source_rank") or row.get("pool_rank") or 0)
            item = documents.setdefault(doc_id, {
                "arms": defaultdict(lambda: math.inf), "chunks": {}, "candidate_rows": 0,
            })
            item["candidate_rows"] += 1
            item["arms"][arm] = min(item["arms"][arm], rank)
            score = scores.get(chunk_id)
            if score is None or not math.isfinite(score):
                raise ValueError(f"missing/nonfinite frozen score: query_id={query_id} chunk_id={chunk_id}")
            previous = item["chunks"].setdefault(chunk_id, (doc_id, float(score)))
            if previous != (doc_id, float(score)):
                raise ValueError(f"conflicting exact candidate identity: query_id={query_id} chunk_id={chunk_id}")
    output = []
    for doc_id, item in documents.items():
        values = sorted((value[1] for value in item["chunks"].values()), reverse=True)
        arm_ranks = {arm: int(rank) for arm, rank in sorted(item["arms"].items())}
        output.append({
            "query_id": query_id,
            "doc_id": doc_id,
            "canonical_domain": canonical_domains[doc_id],
            "candidate_arms": sorted(arm_ranks),
            "original_rank_by_arm_json": json.dumps(arm_ranks, ensure_ascii=False, sort_keys=True),
            "candidate_chunk_count": len(values),
            "candidate_row_count": int(item["candidate_rows"]),
            "duplicate_chunk_competition": int(item["candidate_rows"] - len(values)),
            "best_reranker_score": values[0],
            "mean_top2_score": sum(values[:2]) / min(2, len(values)),
            "mean_top3_score": sum(values[:3]) / min(3, len(values)),
            "reached_phase11c_docs20": doc_id in final11,
            "reached_phase12_docs20": doc_id in final12,
        })
    return sorted(output, key=lambda row: (row["query_id"], row["doc_id"]))


def summarize_ledger(rows: Iterable[dict]) -> dict:
    """Small deterministic summary used by tests and the M1.5 audit."""
    rows = list(rows)
    by_arm = Counter()
    for row in rows:
        by_arm.update(row["candidate_arms"])
    return {
        "query_document_pairs": len(rows),
        "multi_arm_query_document_pairs": sum(len(row["candidate_arms"]) > 1 for row in rows),
        "duplicate_chunk_competition": sum(int(row["duplicate_chunk_competition"]) for row in rows),
        "candidate_documents_by_arm": dict(sorted(by_arm.items())),
        "phase11c_docs20_pairs": sum(bool(row["reached_phase11c_docs20"]) for row in rows),
        "phase12_docs20_pairs": sum(bool(row["reached_phase12_docs20"]) for row in rows),
    }
