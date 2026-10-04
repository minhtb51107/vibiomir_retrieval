from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any, Sequence

from src.retrieval.ranking import aggregate_documents


def is_boilerplate_like(
    row: dict[str, Any], *, max_characters: int, section_types: set[str]
) -> bool:
    text = str(row["chunk_text"]).strip()
    return len(text) <= max_characters and str(row.get("section_type", "")) in section_types


def select_candidates(
    rows: Sequence[dict[str, Any]],
    *,
    top_k: int,
    exact_text_suppression: bool = False,
    max_chunks_per_doc: int | None = None,
    boilerplate_penalty_enabled: bool = False,
    boilerplate_max_characters: int = 80,
    boilerplate_section_types: set[str] | None = None,
    boilerplate_penalty: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if max_chunks_per_doc is not None and max_chunks_per_doc <= 0:
        raise ValueError("max_chunks_per_doc must be positive")
    section_types = boilerplate_section_types or {"title", "heading"}
    prepared = []
    eligible = 0
    for original in rows:
        row = dict(original)
        penalized = is_boilerplate_like(
            row,
            max_characters=boilerplate_max_characters,
            section_types=section_types,
        )
        eligible += int(penalized)
        row["boilerplate_like"] = penalized
        row["selection_score"] = float(row["rerank_score"]) - (
            boilerplate_penalty
            if boilerplate_penalty_enabled and penalized
            else 0.0
        )
        prepared.append(row)
    prepared.sort(
        key=lambda row: (
            -float(row["selection_score"]),
            -float(row["rerank_score"]),
            str(row["chunk_id"]),
        )
    )
    selected = []
    seen_text: set[str] = set()
    document_counts: Counter[int] = Counter()
    exact_suppressed = 0
    document_suppressed = 0
    for row in prepared:
        text = str(row["chunk_text"])
        doc_id = int(row["doc_id"])
        if exact_text_suppression and text in seen_text:
            exact_suppressed += 1
            continue
        if max_chunks_per_doc is not None and document_counts[doc_id] >= max_chunks_per_doc:
            document_suppressed += 1
            continue
        selected.append(row)
        seen_text.add(text)
        document_counts[doc_id] += 1
        if len(selected) == top_k:
            break
    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank
    return selected, {
        "input_candidates": len(rows),
        "selected_candidates": len(selected),
        "exact_text_suppressed": exact_suppressed,
        "max_chunks_per_doc_suppressed": document_suppressed,
        "boilerplate_like_candidates": eligible,
        "boilerplate_penalized": eligible if boilerplate_penalty_enabled else 0,
    }


def aggregate_reranked_documents(
    rows: Sequence[dict[str, Any]],
    *,
    method: str,
    top_k_docs: int,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    compatible = [
        {**row, "score": float(row["rerank_score"])}
        for row in rows
    ]
    aggregated = aggregate_documents(
        compatible, method=method, top_k_docs=top_k_docs, top_n=top_n
    )
    for row in aggregated:
        row["rerank_score"] = float(row["score"])
        row["aggregation_method"] = f"rerank_{method}"
    return aggregated


def selection_summary(
    selected_by_query: dict[int, list[dict[str, Any]]]
) -> dict[str, float | int]:
    unique_docs = [
        len({int(row["doc_id"]) for row in rows})
        for rows in selected_by_query.values()
    ]
    concentration = [
        max(Counter(int(row["doc_id"]) for row in rows).values(), default=0)
        for rows in selected_by_query.values()
    ]
    return {
        "query_count": len(selected_by_query),
        "mean_unique_docs": statistics.fmean(unique_docs) if unique_docs else 0.0,
        "median_unique_docs": statistics.median(unique_docs) if unique_docs else 0,
        "minimum_unique_docs": min(unique_docs, default=0),
        "maximum_chunks_from_one_doc": max(concentration, default=0),
        "median_maximum_chunks_from_one_doc": statistics.median(concentration)
        if concentration
        else 0,
    }
