from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.indexing.embedder import file_sha256
from src.retrieval.pipeline import language_signal

from .candidate_pool import SOURCES, build_pool_from_paths
from .checkpoint import RerankCheckpoint
from .reranker import Reranker
from .selection import (
    aggregate_reranked_documents,
    select_candidates,
    selection_summary,
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


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms_per_query": round(statistics.fmean(values), 6) if values else 0.0,
        "p50_ms_per_query": round(_percentile(values, 0.5), 6),
        "p95_ms_per_query": round(_percentile(values, 0.95), 6),
        "max_ms_per_query": round(max(values, default=0.0), 6),
    }


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def prepare_candidate_pool(
    config: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    inputs = config["inputs"]
    pool_config = config["candidate_pool"]
    max_depth = max(int(value) for value in pool_config["rerank_depths"])
    rows, summary = build_pool_from_paths(
        dense_path=inputs["dense_candidates"],
        sparse_path=inputs["sparse_candidates"],
        hybrid_path=inputs["hybrid_candidates"],
        dense_depth=int(pool_config["dense_depth"]),
        sparse_depth=int(pool_config["sparse_depth"]),
        hybrid_depth=int(pool_config["hybrid_depth"]),
        max_depth=max_depth,
    )
    output = Path(config["outputs"]["directory"])
    path = output / f"unified_candidate_pool_depth{max_depth}.parquet"
    _write_parquet(rows, path)
    summary["path"] = str(path)
    summary["sha256"] = file_sha256(path)
    summary["input_hashes"] = {
        "dense_candidates_sha256": file_sha256(inputs["dense_candidates"]),
        "sparse_candidates_sha256": file_sha256(inputs["sparse_candidates"]),
        "hybrid_candidates_sha256": file_sha256(inputs["hybrid_candidates"]),
        "candidate_pool_sha256": summary["sha256"],
    }
    return path, summary


def _language_signal_extended(text: str) -> str:
    signal = language_signal(text)
    if signal != "unknown":
        return signal
    latin = sum(character.isascii() and character.isalpha() for character in text)
    return "latin-other-signal" if latin >= 20 else "unknown"


def _diagnostic_result(row: dict[str, Any], source: str) -> dict[str, Any]:
    score_key = "rerank_score" if source == "reranked" else f"{source}_score"
    rank_key = "rerank_rank" if source == "reranked" else f"{source}_rank"
    text = str(row["chunk_text"])
    return {
        "doc_id": int(row["doc_id"]),
        "chunk_id": str(row["chunk_id"]),
        "rank": int(row[rank_key]),
        "score": round(float(row[score_key]), 6),
        "language_signal": _language_signal_extended(text),
        "text_preview": text[:180],
    }


def build_multilingual_sanity(
    *,
    pool_path: str | Path,
    reranked_path: str | Path,
    sample_count: int,
) -> dict[str, Any]:
    """Build deterministic, descriptive side-by-side diagnostics.

    The sample is evenly spaced across query character-length order. Language
    labels are lightweight script/diacritic signals, not language detection,
    and the output is explicitly not a relevance evaluation.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    pool_by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(pool_path).to_pylist():
        pool_by_query[int(row["query_id"])].append(row)
    reranked_by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(reranked_path).to_pylist():
        reranked_by_query[int(row["query_id"])].append(row)

    ordered_queries = sorted(
        (
            (len(str(rows[0]["query_text"])), query_id, str(rows[0]["query_text"]))
            for query_id, rows in pool_by_query.items()
        ),
        key=lambda item: (item[0], item[1]),
    )
    count = min(sample_count, len(ordered_queries))
    if count == 1:
        selected_indices = [0]
    else:
        selected_indices = [
            round(index * (len(ordered_queries) - 1) / (count - 1))
            for index in range(count)
        ]

    signal_totals: Counter[str] = Counter()
    queries_with_signal: Counter[str] = Counter()
    for rows in reranked_by_query.values():
        signals = [
            _language_signal_extended(str(row["chunk_text"]))
            for row in sorted(rows, key=lambda row: int(row["rerank_rank"]))[:10]
        ]
        signal_totals.update(signals)
        queries_with_signal.update(set(signals))

    sample_rows = []
    for selected_index in selected_indices:
        query_characters, query_id, query_text = ordered_queries[selected_index]
        pool_rows = pool_by_query[query_id]
        reranked_rows = reranked_by_query[query_id]
        comparison: dict[str, Any] = {}
        for source in SOURCES:
            candidates = [row for row in pool_rows if row[f"{source}_rank"] is not None]
            top = min(
                candidates,
                key=lambda row: (int(row[f"{source}_rank"]), str(row["chunk_id"])),
            )
            comparison[source] = _diagnostic_result(top, source)
        reranked_top = min(
            reranked_rows,
            key=lambda row: (int(row["rerank_rank"]), str(row["chunk_id"])),
        )
        comparison["reranked"] = _diagnostic_result(reranked_top, "reranked")
        sample_rows.append(
            {
                "query_id": query_id,
                "query_characters": query_characters,
                "query_preview": query_text[:180],
                "hybrid_to_reranked_top1_changed": (
                    comparison["hybrid"]["chunk_id"]
                    != comparison["reranked"]["chunk_id"]
                ),
                **comparison,
            }
        )

    return {
        "label": "SANITY CHECK — descriptive signals, not relevance evaluation",
        "selection": "queries evenly spaced across query character-length order",
        "query_count": len(sample_rows),
        "reranked_top_10_all_queries": {
            "result_signal_totals": dict(sorted(signal_totals.items())),
            "queries_with_signal": dict(sorted(queries_with_signal.items())),
        },
        "sample_top1_changes": sum(
            int(row["hybrid_to_reranked_top1_changed"]) for row in sample_rows
        ),
        "rows": sample_rows,
    }


def _rerank_depth(
    rows_by_query: dict[int, list[dict[str, Any]]], depth: int
) -> tuple[dict[int, list[dict[str, Any]]], list[float]]:
    output: dict[int, list[dict[str, Any]]] = {}
    timings = []
    for query_id in sorted(rows_by_query):
        started = time.perf_counter()
        rows = [dict(row) for row in rows_by_query[query_id] if int(row["pool_rank"]) <= depth]
        rows.sort(key=lambda row: (-float(row["rerank_score"]), str(row["chunk_id"])))
        for rank, row in enumerate(rows, start=1):
            row["rerank_rank"] = rank
            row["score"] = float(row["rerank_score"])
        output[query_id] = rows
        timings.append((time.perf_counter() - started) * 1000)
    return output, timings


def _movement_summary(
    reranked: dict[int, list[dict[str, Any]]], *, depth: int
) -> dict[str, Any]:
    absolute_movements = []
    top1_changed = 0
    promoted_membership: Counter[str] = Counter()
    before_membership: Counter[str] = Counter()
    after_membership: Counter[str] = Counter()
    before_languages: Counter[str] = Counter()
    after_languages: Counter[str] = Counter()
    promoted_languages: Counter[str] = Counter()
    unique_docs_before = []
    unique_docs_after = []
    for rows in reranked.values():
        by_pool = sorted(rows, key=lambda row: int(row["pool_rank"]))
        top1_changed += rows[0]["chunk_id"] != by_pool[0]["chunk_id"]
        rank_by_chunk = {str(row["chunk_id"]): rank for rank, row in enumerate(rows, start=1)}
        absolute_movements.extend(
            abs(int(row["pool_rank"]) - rank_by_chunk[str(row["chunk_id"])])
            for row in rows
        )
        for row in by_pool[:10]:
            before_membership[str(row["source_membership_key"])] += 1
            before_languages[_language_signal_extended(str(row["chunk_text"]))] += 1
        for row in rows[:10]:
            membership = str(row["source_membership_key"])
            signal = _language_signal_extended(str(row["chunk_text"]))
            after_membership[membership] += 1
            after_languages[signal] += 1
            if int(row["pool_rank"]) > 10:
                promoted_membership[membership] += 1
                promoted_languages[signal] += 1
        unique_docs_before.append(len({int(row["doc_id"]) for row in by_pool[:20]}))
        unique_docs_after.append(len({int(row["doc_id"]) for row in rows[:20]}))
    return {
        "depth": depth,
        "query_count": len(reranked),
        "top1_changed_count": top1_changed,
        "top1_changed_rate": round(top1_changed / max(len(reranked), 1), 6),
        "absolute_rank_movement": {
            "mean": round(statistics.fmean(absolute_movements), 6),
            "median": _percentile([float(value) for value in absolute_movements], 0.5),
            "p95": _percentile([float(value) for value in absolute_movements], 0.95),
            "max": max(absolute_movements, default=0),
        },
        "top10_source_membership_before": dict(sorted(before_membership.items())),
        "top10_source_membership_after": dict(sorted(after_membership.items())),
        "promoted_into_top10_by_membership": dict(sorted(promoted_membership.items())),
        "top10_language_signals_before": dict(sorted(before_languages.items())),
        "top10_language_signals_after": dict(sorted(after_languages.items())),
        "promoted_into_top10_by_language_signal": dict(sorted(promoted_languages.items())),
        "top20_document_diversity": {
            "mean_unique_docs_before": round(statistics.fmean(unique_docs_before), 6),
            "mean_unique_docs_after": round(statistics.fmean(unique_docs_after), 6),
        },
    }


def _checkpoint_signature(
    pool_summary: dict[str, Any], model_metadata: dict[str, Any]
) -> dict[str, Any]:
    return {
        "input_hashes": dict(pool_summary["input_hashes"]),
        "model_name": model_metadata["model_name"],
        "resolved_revision": model_metadata["resolved_revision"],
        "max_sequence_length": model_metadata["max_sequence_length"],
        "inference_precision": model_metadata["inference_precision"],
        "batch_size": model_metadata["batch_size"],
    }


def score_pool(
    *,
    config: dict[str, Any],
    reranker: Reranker,
    pool_path: str | Path,
    pool_summary: dict[str, Any],
    progress_every: int = 1000,
) -> dict[str, Any]:
    """Stream-score the pool into the transactional checkpoint (GPU stage).

    Only already-persisted scores are skipped; nothing is rescored. Rows are
    streamed from Parquet in small batches so host memory stays bounded.
    """
    pool_path = Path(pool_path)
    pool_row_count = pq.ParquetFile(pool_path).metadata.num_rows
    model_metadata = reranker.metadata()
    signature = _checkpoint_signature(pool_summary, model_metadata)
    inference_started = time.perf_counter()
    with RerankCheckpoint(config["outputs"]["checkpoint"], signature=signature) as checkpoint:
        initial_count = checkpoint.count
        completed = checkpoint.completed_keys()
        write_batch_size = max(32, int(model_metadata["batch_size"]) * 32)
        scored_this_run = 0
        columns = ["query_id", "query_text", "chunk_id", "chunk_text"]
        for parquet_batch in pq.ParquetFile(pool_path).iter_batches(
            batch_size=1024, columns=columns
        ):
            pending = [
                row
                for row in parquet_batch.to_pylist()
                if (int(row["query_id"]), str(row["chunk_id"])) not in completed
            ]
            for offset in range(0, len(pending), write_batch_size):
                batch = pending[offset : offset + write_batch_size]
                pairs = [
                    (str(row["query_text"]), str(row["chunk_text"])) for row in batch
                ]
                started = time.perf_counter()
                scores = reranker.score_pairs(pairs)
                elapsed_ms = (time.perf_counter() - started) * 1000
                per_pair_ms = elapsed_ms / max(len(batch), 1)
                checkpoint.write_scores(
                    (
                        int(row["query_id"]),
                        str(row["chunk_id"]),
                        float(score),
                        per_pair_ms,
                    )
                    for row, score in zip(batch, scores, strict=True)
                )
                scored_this_run += len(batch)
                done = initial_count + scored_this_run
                if done % progress_every < len(batch) or done == pool_row_count:
                    elapsed = time.perf_counter() - inference_started
                    print(
                        f"reranked {done}/{pool_row_count} pairs "
                        f"({scored_this_run / max(elapsed, 1e-9):.2f}/s)",
                        flush=True,
                    )
        if checkpoint.integrity_check() != "ok":
            raise RuntimeError("rerank checkpoint integrity check failed")
        final_count = checkpoint.count
    return {
        "initial_rows": initial_count,
        "final_rows": final_count,
        "rows_scored_this_run": final_count - initial_count,
        "pool_rows": pool_row_count,
        "runtime_seconds": round(time.perf_counter() - inference_started, 6),
        "complete": final_count == pool_row_count,
    }


def run_reranking(
    *,
    config: dict[str, Any],
    reranker: Reranker,
    pool_path: str | Path,
    pool_summary: dict[str, Any],
    progress_every: int = 1000,
) -> dict[str, Any]:
    """Score and finalize in one process (kept for small/mock runs)."""
    scoring = score_pool(
        config=config,
        reranker=reranker,
        pool_path=pool_path,
        pool_summary=pool_summary,
        progress_every=progress_every,
    )
    model_metadata = reranker.metadata()
    reranker.release()
    return finalize_reranking(
        config=config,
        pool_path=pool_path,
        pool_summary=pool_summary,
        model_metadata=model_metadata,
        scoring_runs=[scoring],
    )


def finalize_reranking(
    *,
    config: dict[str, Any],
    pool_path: str | Path,
    pool_summary: dict[str, Any],
    model_metadata: dict[str, Any],
    scoring_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """CPU-only post-processing from a complete checkpoint (never loads a model)."""
    wall_started = time.perf_counter()
    pool_config = config["candidate_pool"]
    output = Path(config["outputs"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    pool_path = Path(pool_path)
    pool_row_count = pq.ParquetFile(pool_path).metadata.num_rows
    candidate_count = pool_row_count
    input_hashes = dict(pool_summary["input_hashes"])
    signature = _checkpoint_signature(pool_summary, model_metadata)
    with RerankCheckpoint(config["outputs"]["checkpoint"], signature=signature) as checkpoint:
        if checkpoint.integrity_check() != "ok":
            raise RuntimeError("rerank checkpoint integrity check failed")
        score_records = checkpoint.read_scores()
        final_count = checkpoint.count
    if final_count != pool_row_count:
        raise RuntimeError(
            f"rerank checkpoint is incomplete: {final_count}/{pool_row_count}"
        )
    total_inference_ms = sum(ms for _, ms in score_records.values())
    initial_count = int(scoring_runs[0]["initial_rows"]) if scoring_runs else 0
    inference_seconds = total_inference_ms / 1000
    pool_rows = pq.read_table(pool_path).to_pylist()
    inference_ms_by_query: dict[int, float] = defaultdict(float)
    for row in pool_rows:
        key = (int(row["query_id"]), str(row["chunk_id"]))
        score, inference_ms = score_records[key]
        row["rerank_score"] = score
        inference_ms_by_query[int(row["query_id"])] += inference_ms
    del score_records

    pool_by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        pool_by_query[int(row["query_id"])].append(row)
    depth_summaries = []
    default_depth = int(pool_config["default_depth"])
    base: dict[int, list[dict[str, Any]]] | None = None
    post_ms_by_query: dict[int, float] = defaultdict(float)
    for depth_value in pool_config["rerank_depths"]:
        depth = int(depth_value)
        reranked, timings = _rerank_depth(pool_by_query, depth)
        if depth == default_depth:
            base = reranked
        for query_id, elapsed in zip(sorted(reranked), timings, strict=True):
            post_ms_by_query[query_id] += elapsed
        rows = [row for query_id in sorted(reranked) for row in reranked[query_id]]
        _write_parquet(rows, output / f"reranked_chunks_depth{depth}.parquet")
        depth_summaries.append(_movement_summary(reranked, depth=depth))
        if depth != default_depth:
            del reranked

    if base is None:
        raise RuntimeError("default rerank depth was not produced")
    del pool_rows
    query_ids = sorted(pool_by_query)
    del pool_by_query
    selection_config = config["selection"]
    boilerplate = selection_config["boilerplate_penalty"]
    strategies = {
        "pure_rerank": {
            "exact_text_suppression": False,
            "max_chunks_per_doc": None,
            "boilerplate_penalty_enabled": False,
        },
        "max_chunks_per_doc": {
            "exact_text_suppression": False,
            "max_chunks_per_doc": int(selection_config["max_chunks_per_doc"]),
            "boilerplate_penalty_enabled": False,
        },
        "controlled": {
            "exact_text_suppression": True,
            "max_chunks_per_doc": int(selection_config["max_chunks_per_doc"]),
            "boilerplate_penalty_enabled": True,
        },
    }
    control_summary: dict[str, Any] = {}
    selection_ms_by_query: dict[int, float] = defaultdict(float)
    aggregation_ms_by_query: dict[int, float] = defaultdict(float)
    output_paths: dict[str, dict[str, str]] = {}
    for name, options in strategies.items():
        selected_by_query = {}
        totals: Counter[str] = Counter()
        document_best = []
        document_mean = []
        for query_id in sorted(base):
            selection_started = time.perf_counter()
            selected, stats = select_candidates(
                base[query_id],
                top_k=int(selection_config["final_top_k_chunks"]),
                exact_text_suppression=options["exact_text_suppression"],
                max_chunks_per_doc=options["max_chunks_per_doc"],
                boilerplate_penalty_enabled=options["boilerplate_penalty_enabled"],
                boilerplate_max_characters=int(boilerplate["max_characters"]),
                boilerplate_section_types=set(boilerplate["section_types"]),
                boilerplate_penalty=float(boilerplate["penalty"]),
            )
            selection_ms_by_query[query_id] += (
                time.perf_counter() - selection_started
            ) * 1000
            selected_by_query[query_id] = selected
            totals.update(stats)
            aggregation_started = time.perf_counter()
            document_best.extend(
                aggregate_reranked_documents(
                    selected,
                    method="best_chunk",
                    top_k_docs=int(selection_config["final_top_k_docs"]),
                )
            )
            document_mean.extend(
                aggregate_reranked_documents(
                    selected,
                    method="top_n_mean",
                    top_k_docs=int(selection_config["final_top_k_docs"]),
                    top_n=int(config["documents"]["top_n_mean"]),
                )
            )
            aggregation_ms_by_query[query_id] += (
                time.perf_counter() - aggregation_started
            ) * 1000
        selected_rows = [
            row for query_id in sorted(selected_by_query) for row in selected_by_query[query_id]
        ]
        chunks_path = output / f"selected_{name}_chunks.parquet"
        best_path = output / f"selected_{name}_documents_best_chunk.parquet"
        mean_path = output / f"selected_{name}_documents_top_n_mean.parquet"
        _write_parquet(selected_rows, chunks_path)
        _write_parquet(document_best, best_path)
        _write_parquet(document_mean, mean_path)
        output_paths[name] = {
            "chunks": str(chunks_path),
            "documents_best_chunk": str(best_path),
            "documents_top_n_mean": str(mean_path),
        }
        control_summary[name] = {
            "controls": options,
            "activations": dict(totals),
            "selection": selection_summary(selected_by_query),
        }

    preparation_mean = float(pool_summary["preparation_latency_ms_per_query"]["mean"])
    inference_values = [inference_ms_by_query.get(query_id, 0.0) for query_id in query_ids]
    post_values = [
        post_ms_by_query[query_id] + selection_ms_by_query[query_id]
        for query_id in query_ids
    ]
    aggregation_values = [aggregation_ms_by_query[query_id] for query_id in query_ids]
    total_values = [
        preparation_mean + inference + post + aggregation
        for inference, post, aggregation in zip(
            inference_values, post_values, aggregation_values, strict=True
        )
    ]
    return {
        "model": model_metadata,
        "input_hashes": input_hashes,
        "candidate_pool": pool_summary,
        "checkpoint": {
            "path": config["outputs"]["checkpoint"],
            "first_run_initial_rows": initial_count,
            "final_rows": final_count,
            "scoring_runs": scoring_runs,
            "integrity_check": "ok",
        },
        "inference": {
            "pairs": candidate_count,
            "accounting": (
                "sum of per-pair GPU inference_ms stored in the checkpoint "
                "across all resumed runs"
            ),
            "total_inference_seconds": round(inference_seconds, 6),
            "pairs_per_second": round(
                candidate_count / max(inference_seconds, 1e-9), 6
            ),
            "latency": latency_summary(inference_values),
        },
        "post_processing_latency": latency_summary(post_values),
        "document_aggregation_latency": latency_summary(aggregation_values),
        "total_latency": latency_summary(total_values),
        "total_wall_seconds": round(time.perf_counter() - wall_started, 6),
        "depth_experiments": depth_summaries,
        "control_summary": control_summary,
        "outputs": output_paths,
    }


def compact_benchmark(result: dict[str, Any]) -> dict[str, Any]:
    return result


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
