from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.indexing.embedder import file_sha256

from .safety import GIB, DiskAudit, evaluate_disk_guard


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _scenario(low: float, expected: float, high: float) -> dict[str, int]:
    return {"low": round(low), "expected": round(expected), "high": round(high)}


def _scale(per_row: float, rows: dict[str, int]) -> dict[str, int]:
    return {name: round(per_row * count) for name, count in rows.items()}


def build_capacity_plan(
    config: dict[str, Any],
    *,
    c_drive: DiskAudit,
    d_drive: DiskAudit,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    corpus_rows = int(config["source"]["rows"])
    crawl = _read_json("artifacts/crawl_benchmark/benchmark_summary.json")
    processing = _read_json("artifacts/phase3_pilot/summary.json")
    dense = _read_json("artifacts/phase4_dense/index_benchmark.json")
    sparse = _read_json("artifacts/phase6_hybrid/sparse_index_benchmark.json")
    reranking = _read_json("artifacts/phase7_reranking/benchmark.json")

    bounded_path = Path("artifacts/phase8_scaling/bounded_benchmark.json")
    bounded = _read_json(bounded_path) if bounded_path.exists() else None

    pilot_rows = int(processing["documents_processed"])
    successful = int(processing["extraction"]["status_counts"]["SUCCESS"])
    validated_chunks = int(dense["chunk_count"])
    if bounded is not None:
        pilot_rows = int(bounded["tiers"]["bounded_live_urls"])
        successful = int(bounded["extraction"]["status_counts"]["SUCCESS"])
        validated_chunks = int(bounded["tiers"]["real_bge_chunks"])
        sparse = bounded["indexes"]["sparse"]
        crawl = bounded["crawl"]
    expected_success_rate = successful / pilot_rows
    success_rates = {
        "low": min(0.75, expected_success_rate * 0.9),
        "expected": expected_success_rate,
        "high": min(0.98, max(0.95, expected_success_rate * 1.05)),
    }
    chunks_per_success = {
        "low": 3.0,
        "expected": validated_chunks / successful,
        "high": 8.0,
    }
    successful_docs = {
        name: round(corpus_rows * rate) for name, rate in success_rates.items()
    }
    chunk_counts = {
        name: round(successful_docs[name] * chunks_per_success[name])
        for name in success_rates
    }

    expected_transfer = 287.7e9
    archive_ratio = 0.32
    if bounded is not None:
        expected_transfer = (
            float(bounded["crawl"]["database"]["downloaded_bytes"])
            / pilot_rows
            * corpus_rows
        )
        archive = bounded["archive_integrity"]
        archive_ratio = float(archive["compressed_bytes"]) / max(
            float(archive["original_bytes"]), 1.0
        )
    raw_transfer = _scenario(
        min(220e9, expected_transfer * 0.8),
        expected_transfer,
        max(360e9, expected_transfer * 1.25),
    )
    compressed_archive = {
        "low": round(raw_transfer["low"] * 0.25),
        "expected": round(raw_transfer["expected"] * archive_ratio),
        "high": round(raw_transfer["high"] * 0.40),
    }
    metadata_bytes_per_row = float(
        crawl["database"].get(
            "estimated_main_database_bytes_per_result_row",
            crawl["database"].get("estimated_bytes_per_result_row"),
        )
    )
    metadata = _scale(
        metadata_bytes_per_row,
        {name: corpus_rows for name in chunk_counts},
    )
    documents_path = Path("data/processed/phase3_pilot_documents.parquet")
    chunks_path = Path("data/chunks/phase4_bge_m3/tokens_512_overlap_64.parquet")
    dense_metadata_path = Path("data/indexes/phase4_dense/chunk_metadata.sqlite")
    document_bytes = documents_path.stat().st_size
    chunk_bytes = chunks_path.stat().st_size
    dense_metadata_bytes = dense_metadata_path.stat().st_size
    if bounded is not None:
        document_bytes = int(bounded["storage"]["documents_bytes"])
        chunk_bytes = int(bounded["storage"]["chunks_bytes"])
        dense_metadata_bytes = int(bounded["storage"]["dense_metadata_bytes"])
    processed_documents = _scale(
        document_bytes / pilot_rows,
        {name: corpus_rows for name in chunk_counts},
    )
    chunk_parquet = _scale(chunk_bytes / validated_chunks, chunk_counts)
    dense_vectors = _scale(1024 * 4, chunk_counts)
    dense_index = dict(dense_vectors)
    dense_metadata = _scale(
        dense_metadata_bytes / validated_chunks, chunk_counts
    )
    sparse_index = _scale(
        int(sparse["index_size_bytes"]) / validated_chunks, chunk_counts
    )
    retrieval_outputs = _scenario(0.5e9, 1.0e9, 2.0e9)
    checkpoints = _scenario(2e9, 3e9, 5e9)
    components = {
        "crawl_metadata": metadata,
        "compressed_body_archive": compressed_archive,
        "processed_documents": processed_documents,
        "validated_chunk_parquet": chunk_parquet,
        "dense_embeddings_float32": dense_vectors,
        "faiss_index_flat_ip": dense_index,
        "dense_metadata_sqlite": dense_metadata,
        "bm25_sqlite": sparse_index,
        "retrieval_outputs": retrieval_outputs,
        "checkpoints_and_manifests": checkpoints,
    }
    retained_total = {
        name: sum(component[name] for component in components.values())
        for name in ("low", "expected", "high")
    }
    transient_peak = {
        name: retained_total[name] + sparse_index[name]
        for name in retained_total
    }
    storage = {
        "units": "bytes unless a field says otherwise",
        "measured_sources": {
            "crawl_rows": pilot_rows,
            "crawl_bytes_per_metadata_row": metadata_bytes_per_row,
            "pilot_successful_documents": successful,
            "validated_chunks": validated_chunks,
            "chunk_parquet_bytes": chunk_bytes,
            "dense_metadata_bytes": dense_metadata_bytes,
            "sparse_index_bytes": sparse["index_size_bytes"],
            "phase8_bounded_evidence_used": bounded is not None,
        },
        "network_transfer": raw_transfer,
        "components": components,
        "retained_total": retained_total,
        "transient_peak_during_atomic_sparse_build": transient_peak,
        "note": (
            "Ranges extrapolate bounded pilot density and are capacity estimates, "
            "not exact production sizes. Network transfer is not added to disk totals."
        ),
    }

    embedding_throughput = float(dense["embedding_throughput_chunks_per_second"])
    sparse_per_chunk = float(sparse["build_seconds"]) / validated_chunks
    exact_search_per_query = 0.003302 * (chunk_counts["expected"] / 5643)
    exact_search_range = None
    if bounded is not None:
        ann_exact_ms = float(
            bounded["ann"]["indexes"]["IndexFlatIP"]["mean_ms_per_query"]
        )
        bounded_linear = ann_exact_ms / 1000 * chunk_counts["expected"] / 50_000
        exact_search_range = [
            round(min(bounded_linear, exact_search_per_query), 3),
            round(max(bounded_linear, exact_search_per_query), 3),
        ]
    rerank_seconds = float(reranking["inference"]["total_inference_seconds"])
    timing = {
        "crawl_planning_range_days": [25, 35],
        "crawl_linear_estimate_days": round(
            corpus_rows / (float(crawl["run"]["urls_completed_per_minute"]) * 60 * 24),
            3,
        ),
        "dense_embedding_seconds": {
            name: round(count / embedding_throughput, 3)
            for name, count in chunk_counts.items()
        },
        "sparse_build_linear_seconds": {
            name: round(count * sparse_per_chunk, 3)
            for name, count in chunk_counts.items()
        },
        "index_flat_ip_estimated_seconds_per_query": round(exact_search_per_query, 3),
        "index_flat_ip_bounded_linear_range_seconds_per_query": exact_search_range,
        "rerank_all_1200_queries_seconds": rerank_seconds,
        "rerank_scope": "fixed candidate depth 100; never the whole corpus",
    }
    capacity = {
        "source": {
            "path": config["source"]["corpus"],
            "rows": corpus_rows,
            "sha256": file_sha256(config["source"]["corpus"]),
        },
        "disk_audit": {"C": c_drive.as_dict(), "D": d_drive.as_dict()},
        "pilot": {
            "rows": pilot_rows,
            "success_rate": round(successful / pilot_rows, 6),
            "validated_chunks": validated_chunks,
            "validated_chunks_per_successful_document": round(
                validated_chunks / successful, 6
            ),
            "crawl_urls_per_minute": crawl["run"]["urls_completed_per_minute"],
            "embedding_chunks_per_second": embedding_throughput,
            "sparse_build_seconds": sparse["build_seconds"],
            "rerank_pairs_per_second": reranking["inference"]["pairs_per_second"],
            "phase8_bounded_evidence_used": bounded is not None,
        },
        "projection_assumptions": {
            "success_rates": success_rates,
            "chunks_per_successful_document": chunks_per_success,
        },
        "projected_successful_documents": successful_docs,
        "projected_validated_chunks": chunk_counts,
        "timing": timing,
        "storage_artifact": "artifacts/phase8_scaling/storage_projection.json",
    }

    safety = config["safety"]
    common = {
        "headroom_multiplier": float(safety["projected_growth_headroom_multiplier"]),
        "maximum_projected_disk_fraction": float(safety["maximum_projected_disk_fraction"]),
    }
    bounded_5k_growth = round(compressed_archive["expected"] * 5000 / corpus_rows) + 2 * GIB
    gates = {
        "8A": evaluate_disk_guard(
            d_drive,
            projected_growth_bytes=bounded_5k_growth,
            minimum_free_bytes=int(safety["minimum_free_before_stage_gib"]["acquisition"] * GIB),
            **common,
        ),
        "8B": evaluate_disk_guard(
            d_drive,
            projected_growth_bytes=4 * GIB,
            minimum_free_bytes=int(safety["minimum_free_before_stage_gib"]["indexing"] * GIB),
            **common,
        ),
        "8C": evaluate_disk_guard(
            d_drive,
            projected_growth_bytes=compressed_archive["expected"] + metadata["expected"],
            minimum_free_bytes=int(safety["stop_free_space_gib"] * GIB),
            **common,
        ),
        "8D": evaluate_disk_guard(
            d_drive,
            projected_growth_bytes=processed_documents["expected"] + chunk_parquet["expected"],
            minimum_free_bytes=int(safety["stop_free_space_gib"] * GIB),
            **common,
        ),
        "8E": evaluate_disk_guard(
            d_drive,
            projected_growth_bytes=(
                dense_vectors["expected"]
                + dense_index["expected"]
                + dense_metadata["expected"]
                + 2 * sparse_index["expected"]
            ),
            minimum_free_bytes=int(safety["stop_free_space_gib"] * GIB),
            **common,
        ),
    }
    gates["8F"] = {
        "allowed": False,
        "reason": "blocked until 8C, 8D, and 8E complete and validate",
    }
    production_gate = {
        "unrestricted_enabled": bool(safety["unrestricted_enabled"]),
        "current_decision": "BLOCKED",
        "reason": (
            "expected full retained and transient storage exceed both local free "
            "space and D: total capacity"
        ),
        "stage_gates": gates,
        "conditions_before_unrestricted": [
            "provision external storage exceeding the high transient projection plus reserve",
            "complete and validate bounded 8A and 8B manifests",
            "choose and benchmark a scalable dense ANN index instead of silently changing IndexFlatIP",
            "choose and benchmark a sparse backend or sharding plan if SQLite projection remains unsafe",
            "set unrestricted_enabled only through an explicit reviewed config change",
        ],
    }
    return capacity, storage, production_gate
