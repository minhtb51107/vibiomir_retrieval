from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any


def aggregate_documents(
    chunks: list[dict[str, Any]],
    *,
    method: str,
    top_k_docs: int,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    if method not in {"best_chunk", "top_n_mean"}:
        raise ValueError(f"unsupported aggregation method: {method}")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        grouped[int(chunk["doc_id"])].append(chunk)
    ranked: list[dict[str, Any]] = []
    for doc_id, candidates in grouped.items():
        ordered = sorted(candidates, key=lambda row: (-float(row["score"]), row["chunk_id"]))
        selected = ordered[: max(1, top_n)]
        score = (
            float(ordered[0]["score"])
            if method == "best_chunk"
            else statistics.fmean(float(row["score"]) for row in selected)
        )
        best = ordered[0]
        ranked.append(
            {
                "query_id": best["query_id"],
                "query_text": best["query_text"],
                "score": score,
                "doc_id": doc_id,
                "best_chunk_id": best["chunk_id"],
                "chunk_text": best["chunk_text"],
                "source_url": best["source_url"],
                "aggregation_method": method,
                "contributing_chunks": len(selected),
            }
        )
    ranked.sort(key=lambda row: (-float(row["score"]), int(row["doc_id"])))
    result = ranked[:top_k_docs]
    for rank, row in enumerate(result, start=1):
        row["rank"] = rank
    return result


def compare_aggregations(
    best_rows: list[dict[str, Any]], mean_rows: list[dict[str, Any]]
) -> dict[str, float | int]:
    best_by_query: dict[int, list[int]] = defaultdict(list)
    mean_by_query: dict[int, list[int]] = defaultdict(list)
    for row in best_rows:
        best_by_query[int(row["query_id"])].append(int(row["doc_id"]))
    for row in mean_rows:
        mean_by_query[int(row["query_id"])].append(int(row["doc_id"]))
    query_ids = sorted(set(best_by_query) & set(mean_by_query))
    top1_agreement = sum(
        best_by_query[qid][:1] == mean_by_query[qid][:1] for qid in query_ids
    )
    overlaps = [
        len(set(best_by_query[qid]) & set(mean_by_query[qid]))
        / max(len(set(best_by_query[qid]) | set(mean_by_query[qid])), 1)
        for qid in query_ids
    ]
    return {
        "query_count": len(query_ids),
        "top1_agreement_count": top1_agreement,
        "top1_agreement_rate": round(top1_agreement / max(len(query_ids), 1), 6),
        "mean_top_k_jaccard": round(statistics.fmean(overlaps), 6) if overlaps else 0.0,
    }
