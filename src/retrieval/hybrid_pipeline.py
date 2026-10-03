from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.indexing.embedder import file_sha256, normalize_text
from src.indexing.metadata_store import ChunkMetadataStore

from .fusion import reciprocal_rank_fusion_documents
from .hybrid_retriever import HybridRetriever
from .pipeline import language_signal
from .ranking import aggregate_documents
from .sparse_retriever import SparseRetriever


REQUIRED_CHUNK_RESULT_FIELDS = {
    "query_id",
    "query_text",
    "rank",
    "score",
    "chunk_id",
    "doc_id",
    "chunk_text",
    "source_url",
}


def load_hybrid_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["sparse"]["tokenizer"] != "unicode_words_han_unigrams_bigrams_v1":
        raise ValueError("unsupported sparse tokenizer")
    if config["fusion"]["method"] != "rrf":
        raise ValueError("Phase 6 baseline requires RRF")
    if config["documents"]["default_aggregation"] not in {
        "best_chunk",
        "top_n_mean",
    }:
        raise ValueError("unsupported default document aggregation")
    return config


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms_per_query": round(statistics.fmean(values), 6) if values else 0.0,
        "p50_ms_per_query": round(_percentile(values, 0.5), 6),
        "p95_ms_per_query": round(_percentile(values, 0.95), 6),
        "max_ms_per_query": round(max(values, default=0.0), 6),
    }


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty retrieval output: {path}")
    pq.write_table(pa.Table.from_pylist(rows), path)


def _validate_schema(rows: list[dict[str, Any]], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} returned no results")
    missing = REQUIRED_CHUNK_RESULT_FIELDS - set(rows[0])
    if missing:
        raise ValueError(f"{label} result schema missing: {sorted(missing)}")


def run_sparse_retrieval(
    *,
    query_path: str | Path,
    index_path: str | Path,
    metadata_path: str | Path,
    output_directory: str | Path,
    top_k_chunks: int,
    top_k_docs: int,
    top_n_mean: int,
) -> dict[str, Any]:
    queries = pq.read_table(query_path, columns=["id", "query"]).to_pylist()
    output = Path(output_directory)
    all_chunks: list[dict[str, Any]] = []
    best_documents: list[dict[str, Any]] = []
    mean_documents: list[dict[str, Any]] = []
    per_query_ms: list[float] = []
    empty_result_queries = 0
    with ChunkMetadataStore(metadata_path) as metadata:
        with SparseRetriever(index_path=index_path, metadata=metadata) as retriever:
            for query in queries:
                started = time.perf_counter()
                chunks = retriever.retrieve(
                    [int(query["id"])], [str(query["query"])], top_k=top_k_chunks
                )[0]
                per_query_ms.append((time.perf_counter() - started) * 1000)
                if not chunks:
                    empty_result_queries += 1
                    continue
                all_chunks.extend(chunks)
                best_documents.extend(
                    aggregate_documents(
                        chunks, method="best_chunk", top_k_docs=top_k_docs
                    )
                )
                mean_documents.extend(
                    aggregate_documents(
                        chunks,
                        method="top_n_mean",
                        top_k_docs=top_k_docs,
                        top_n=top_n_mean,
                    )
                )
    _validate_schema(all_chunks, "sparse")
    paths = {
        "chunk_rankings": output / "sparse_chunk_rankings.parquet",
        "documents_best_chunk": output / "sparse_documents_best_chunk.parquet",
        "documents_top_n_mean": output / "sparse_documents_top_n_mean.parquet",
    }
    _write_parquet(all_chunks, paths["chunk_rankings"])
    _write_parquet(best_documents, paths["documents_best_chunk"])
    _write_parquet(mean_documents, paths["documents_top_n_mean"])
    return {
        "query_count": len(queries),
        "empty_result_queries": empty_result_queries,
        "top_k_chunks": top_k_chunks,
        "top_k_docs": top_k_docs,
        "chunk_result_rows": len(all_chunks),
        "document_result_rows_per_method": len(best_documents),
        "latency": _latency_summary(per_query_ms),
        "outputs": {key: str(path) for key, path in paths.items()},
    }


def _group_results(path: str | Path, top_k: int | None = None) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(path).to_pylist():
        if top_k is None or int(row["rank"]) <= top_k:
            grouped[int(row["query_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["rank"]))
    return grouped


def _jaccard(left: set[Any], right: set[Any]) -> float:
    return len(left & right) / max(len(left | right), 1)


def _method_diagnostics(
    by_query: dict[int, list[dict[str, Any]]]
) -> dict[str, Any]:
    unique_docs = []
    concentration = []
    duplicate_chunk_rows = 0
    top_scores = []
    ranks: list[float] = []
    for rows in by_query.values():
        chunk_ids = [str(row["chunk_id"]) for row in rows]
        duplicate_chunk_rows += len(chunk_ids) - len(set(chunk_ids))
        counts = Counter(int(row["doc_id"]) for row in rows)
        unique_docs.append(len(counts))
        concentration.append(max(counts.values(), default=0))
        if rows:
            top_scores.append(float(rows[0]["score"]))
            ranks.extend(float(row["rank"]) for row in rows)
    return {
        "query_count": len(by_query),
        "duplicate_chunk_rows": duplicate_chunk_rows,
        "duplicate_chunk_rate": round(
            duplicate_chunk_rows / max(sum(len(rows) for rows in by_query.values()), 1),
            6,
        ),
        "unique_docs_per_candidate_pool": {
            "mean": round(statistics.fmean(unique_docs), 6),
            "median": _percentile([float(value) for value in unique_docs], 0.5),
            "min": min(unique_docs, default=0),
            "max": max(unique_docs, default=0),
        },
        "maximum_chunks_from_one_doc": {
            "mean": round(statistics.fmean(concentration), 6),
            "median": _percentile([float(value) for value in concentration], 0.5),
            "max": max(concentration, default=0),
        },
        "top1_score_distribution": {
            "min": round(min(top_scores, default=0.0), 6),
            "median": round(_percentile(top_scores, 0.5), 6),
            "p95": round(_percentile(top_scores, 0.95), 6),
            "max": round(max(top_scores, default=0.0), 6),
        },
        "rank_distribution": {
            "result_rows": len(ranks),
            "min": int(min(ranks, default=0)),
            "median": _percentile(ranks, 0.5),
            "p95": _percentile(ranks, 0.95),
            "max": int(max(ranks, default=0)),
        },
    }


def _language_signal_extended(text: str) -> str:
    signal = language_signal(text)
    if signal != "unknown":
        return signal
    latin_letters = sum(character.isascii() and character.isalpha() for character in text)
    return "latin-other-signal" if latin_letters >= 20 else "unknown"


def _multilingual_summary(
    methods: dict[str, dict[int, list[dict[str, Any]]]], *, top_k: int = 10
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "label": "SANITY CHECK — script-based signals, not language identification or evaluation",
        "top_k": top_k,
        "methods": {},
    }
    for method, by_query in methods.items():
        totals: Counter[str] = Counter()
        queries_with: Counter[str] = Counter()
        for rows in by_query.values():
            signals = Counter(
                _language_signal_extended(str(row["chunk_text"]))
                for row in rows[:top_k]
            )
            totals.update(signals)
            for signal in signals:
                queries_with[signal] += 1
        result["methods"][method] = {
            "result_signal_totals": dict(sorted(totals.items())),
            "queries_with_signal": dict(sorted(queries_with.items())),
        }
    return result


def _corpus_duplicate_stats(chunks_path: str | Path) -> dict[str, int | float]:
    groups: dict[str, list[int]] = defaultdict(list)
    chunk_ids: set[str] = set()
    duplicate_chunk_ids = 0
    row_count = 0
    for batch in pq.ParquetFile(chunks_path).iter_batches(
        batch_size=1024, columns=["chunk_id", "doc_id", "normalized_text"]
    ):
        for row in batch.to_pylist():
            row_count += 1
            chunk_id = str(row["chunk_id"])
            if chunk_id in chunk_ids:
                duplicate_chunk_ids += 1
            chunk_ids.add(chunk_id)
            groups[normalize_text(str(row["normalized_text"]))].append(int(row["doc_id"]))
    duplicates = [doc_ids for doc_ids in groups.values() if len(doc_ids) > 1]
    return {
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "exact_duplicate_text_groups": len(duplicates),
        "rows_beyond_first_in_duplicate_groups": sum(len(group) - 1 for group in duplicates),
        "rows_beyond_first_rate": round(
            sum(len(group) - 1 for group in duplicates) / max(row_count, 1), 6
        ),
        "groups_repeated_within_same_document": sum(
            len(group) > len(set(group)) for group in duplicates
        ),
        "groups_shared_across_documents": sum(len(set(group)) > 1 for group in duplicates),
    }


def run_hybrid_comparison(
    *,
    query_path: str | Path,
    chunks_path: str | Path,
    dense_results_path: str | Path,
    sparse_results_path: str | Path,
    output_directory: str | Path,
    dense_top_k: int,
    sparse_top_k: int,
    hybrid_top_k: int,
    top_k_docs: int,
    top_n_mean: int,
    rrf_constant: int,
    sanity_query_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    queries = pq.read_table(query_path, columns=["id", "query"]).to_pylist()
    query_ids = [int(row["id"]) for row in queries]
    dense = _group_results(dense_results_path, dense_top_k)
    sparse = _group_results(sparse_results_path, sparse_top_k)
    if set(dense) != set(query_ids) or set(sparse) != set(query_ids):
        raise ValueError("dense/sparse results do not cover all official queries")
    _validate_schema(dense[query_ids[0]], "dense")
    _validate_schema(sparse[query_ids[0]], "sparse")

    hybrid_engine = HybridRetriever(rrf_constant=rrf_constant)
    hybrid: dict[int, list[dict[str, Any]]] = {}
    fusion_per_query_ms: list[float] = []
    document_outputs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query_id in query_ids:
        started = time.perf_counter()
        fused = hybrid_engine.fuse(
            [dense[query_id]], [sparse[query_id]], top_k=hybrid_top_k
        )[0]
        fusion_per_query_ms.append((time.perf_counter() - started) * 1000)
        hybrid[query_id] = fused
        per_method_documents: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for method_name, chunks in (
            ("dense", dense[query_id]),
            ("sparse", sparse[query_id]),
            ("hybrid", fused),
        ):
            best = aggregate_documents(
                chunks, method="best_chunk", top_k_docs=top_k_docs
            )
            mean = aggregate_documents(
                chunks,
                method="top_n_mean",
                top_k_docs=top_k_docs,
                top_n=top_n_mean,
            )
            document_outputs[f"{method_name}_best_chunk"].extend(best)
            document_outputs[f"{method_name}_top_n_mean"].extend(mean)
            per_method_documents[method_name] = {
                "best_chunk": best,
                "top_n_mean": mean,
            }
        document_outputs["document_rrf_best_chunk"].extend(
            reciprocal_rank_fusion_documents(
                [
                    per_method_documents["dense"]["best_chunk"],
                    per_method_documents["sparse"]["best_chunk"],
                ],
                rrf_constant=rrf_constant,
                top_k=top_k_docs,
            )
        )
        document_outputs["document_rrf_top_n_mean"].extend(
            reciprocal_rank_fusion_documents(
                [
                    per_method_documents["dense"]["top_n_mean"],
                    per_method_documents["sparse"]["top_n_mean"],
                ],
                rrf_constant=rrf_constant,
                top_k=top_k_docs,
            )
        )

    output = Path(output_directory)
    hybrid_rows = [row for query_id in query_ids for row in hybrid[query_id]]
    _validate_schema(hybrid_rows, "hybrid")
    _write_parquet(hybrid_rows, output / "hybrid_chunk_rankings_rrf.parquet")
    for name, rows in document_outputs.items():
        _write_parquet(rows, output / f"{name}_documents.parquet")

    dense_best = defaultdict(list)
    sparse_best = defaultdict(list)
    hybrid_best = defaultdict(list)
    for label, target in (
        ("dense_best_chunk", dense_best),
        ("sparse_best_chunk", sparse_best),
        ("hybrid_best_chunk", hybrid_best),
    ):
        for row in document_outputs[label]:
            target[int(row["query_id"])].append(row)

    def comparison(
        left_documents: dict[int, list[dict[str, Any]]],
        right_documents: dict[int, list[dict[str, Any]]],
        left_chunks: dict[int, list[dict[str, Any]]],
        right_chunks: dict[int, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        chunk_top1 = sum(
            left_chunks[qid][0]["chunk_id"] == right_chunks[qid][0]["chunk_id"]
            for qid in query_ids
        )
        document_top1 = sum(
            int(left_documents[qid][0]["doc_id"])
            == int(right_documents[qid][0]["doc_id"])
            for qid in query_ids
        )
        return {
            "query_count": len(query_ids),
            "chunk_rank1_agreement_count": chunk_top1,
            "chunk_rank1_agreement_rate": round(chunk_top1 / len(query_ids), 6),
            "document_rank1_agreement_count": document_top1,
            "document_rank1_agreement_rate": round(document_top1 / len(query_ids), 6),
            "mean_top10_chunk_jaccard": round(
                statistics.fmean(
                    _jaccard(
                        {row["chunk_id"] for row in left_chunks[qid][:10]},
                        {row["chunk_id"] for row in right_chunks[qid][:10]},
                    )
                    for qid in query_ids
                ),
                6,
            ),
            "mean_top10_document_jaccard": round(
                statistics.fmean(
                    _jaccard(
                        {int(row["doc_id"]) for row in left_documents[qid][:10]},
                        {int(row["doc_id"]) for row in right_documents[qid][:10]},
                    )
                    for qid in query_ids
                ),
                6,
            ),
            "mean_candidate_chunk_jaccard": round(
                statistics.fmean(
                    _jaccard(
                        {row["chunk_id"] for row in left_chunks[qid]},
                        {row["chunk_id"] for row in right_chunks[qid]},
                    )
                    for qid in query_ids
                ),
                6,
            ),
        }

    dense_sparse = comparison(dense_best, sparse_best, dense, sparse)
    dense_hybrid = comparison(dense_best, hybrid_best, dense, hybrid)
    overlaps = {
        "label": "DESCRIPTIVE DIAGNOSTICS — not relevance evaluation",
        "candidate_depths": {
            "dense": dense_top_k,
            "sparse": sparse_top_k,
            "hybrid": hybrid_top_k,
            "documents": top_k_docs,
        },
        "dense_vs_sparse": dense_sparse,
        "dense_vs_hybrid": dense_hybrid,
        "methods": {
            "dense": _method_diagnostics(dense),
            "sparse": _method_diagnostics(sparse),
            "hybrid": _method_diagnostics(hybrid),
        },
        "corpus_exact_duplicates": _corpus_duplicate_stats(chunks_path),
    }

    multilingual = _multilingual_summary(
        {"dense": dense, "sparse": sparse, "hybrid": hybrid}
    )
    ordered_queries = sorted(
        queries, key=lambda row: (len(str(row["query"])), int(row["id"]))
    )
    if sanity_query_count >= len(ordered_queries):
        selected = ordered_queries
    else:
        selected = [
            ordered_queries[
                round(index * (len(ordered_queries) - 1) / (sanity_query_count - 1))
            ]
            for index in range(sanity_query_count)
        ]
    manual_rows = []
    for query in selected:
        query_id = int(query["id"])
        row = {
            "query_id": query_id,
            "query_characters": len(str(query["query"])),
            "query_preview": str(query["query"])[:140],
        }
        for name, values in (
            ("dense", dense[query_id]),
            ("sparse", sparse[query_id]),
            ("hybrid", hybrid[query_id]),
        ):
            first = values[0]
            row[name] = {
                "doc_id": int(first["doc_id"]),
                "chunk_id": str(first["chunk_id"]),
                "score": round(float(first["score"]), 6),
                "language_signal": _language_signal_extended(str(first["chunk_text"])),
                "text_preview": str(first["chunk_text"])[:180],
            }
        manual_rows.append(row)
    multilingual["manual_side_by_side"] = {
        "label": "SANITY CHECK — not evaluation",
        "selection": "queries evenly spaced across query character-length order",
        "query_count": len(manual_rows),
        "rows": manual_rows,
    }
    multilingual["observations"] = [
        "Dense results contained both Vietnamese- and Han-script signals, confirming cross-script candidates occur without translation.",
        "Sparse top-10 results were almost entirely Vietnamese-signal because the official queries are Vietnamese and BM25 requires lexical overlap.",
        "RRF retained some Han-script candidates while shifting the top-10 distribution toward same-script lexical matches.",
        "The bounded side-by-side sample contains both topic-related and non-specific or boilerplate matches across methods; without ground truth this is not evidence that one method is better.",
        "The latin-other signal is not a reliable English-language classifier, so English behavior remains inconclusive in this pilot.",
    ]
    fusion_summary = {
        "label": "DESCRIPTIVE COMPARISON — no ground truth and no quality claim",
        "query_count": len(query_ids),
        "method": "reciprocal_rank_fusion",
        "rrf_constant": rrf_constant,
        "candidate_depths": {
            "dense": dense_top_k,
            "sparse": sparse_top_k,
            "hybrid": hybrid_top_k,
            "documents": top_k_docs,
        },
        "fusion_latency": _latency_summary(fusion_per_query_ms),
        "result_rows": len(hybrid_rows),
        "source_hashes": {
            "validated_chunks_sha256": file_sha256(chunks_path),
            "dense_results_sha256": file_sha256(dense_results_path),
            "sparse_results_sha256": file_sha256(sparse_results_path),
        },
    }
    return fusion_summary, overlaps, multilingual


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
