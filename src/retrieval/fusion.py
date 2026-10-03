from __future__ import annotations

from typing import Any, Sequence


def deduplicate_chunks(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        chunk_id = str(row["chunk_id"])
        current = best.get(chunk_id)
        if current is None or (
            -float(row["score"]), int(row.get("rank", 0)), chunk_id
        ) < (
            -float(current["score"]), int(current.get("rank", 0)), chunk_id
        ):
            best[chunk_id] = dict(row)
    return sorted(
        best.values(),
        key=lambda row: (-float(row["score"]), int(row.get("rank", 0)), str(row["chunk_id"])),
    )


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[dict[str, Any]]],
    *,
    rrf_constant: int = 60,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    if rrf_constant < 0:
        raise ValueError("rrf_constant must be non-negative")
    fused: dict[str, dict[str, Any]] = {}
    for source_index, ranking in enumerate(rankings):
        seen: set[str] = set()
        for fallback_rank, original in enumerate(ranking, start=1):
            chunk_id = str(original["chunk_id"])
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank = int(original.get("rank", fallback_rank))
            if chunk_id not in fused:
                row = dict(original)
                row["score"] = 0.0
                row["source_ranks"] = [None] * len(rankings)
                fused[chunk_id] = row
            fused[chunk_id]["score"] += 1.0 / (rrf_constant + rank)
            fused[chunk_id]["source_ranks"][source_index] = rank
    result = list(fused.values())
    result.sort(
        key=lambda row: (
            -float(row["score"]),
            min(rank for rank in row["source_ranks"] if rank is not None),
            str(row["chunk_id"]),
        )
    )
    if top_k is not None:
        result = result[:top_k]
    for rank, row in enumerate(result, start=1):
        row["rank"] = rank
        row["fusion_method"] = "rrf"
        row["rrf_constant"] = rrf_constant
    return result


def reciprocal_rank_fusion_documents(
    rankings: Sequence[Sequence[dict[str, Any]]],
    *,
    rrf_constant: int = 60,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    converted = []
    for ranking in rankings:
        converted.append(
            [
                {
                    **row,
                    "chunk_id": f"doc:{int(row['doc_id'])}",
                    "document_source_rank": int(row["rank"]),
                }
                for row in ranking
            ]
        )
    fused = reciprocal_rank_fusion(
        converted, rrf_constant=rrf_constant, top_k=top_k
    )
    for row in fused:
        row.pop("chunk_id", None)
        row["aggregation_method"] = "rrf_after_document_aggregation"
    return fused
