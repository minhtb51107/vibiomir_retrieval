#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import faiss
import numpy as np


def _timed_search(index: object, queries: np.ndarray, top_k: int) -> tuple[float, np.ndarray]:
    started = time.perf_counter()
    _, rows = index.search(queries, top_k)
    return time.perf_counter() - started, rows


def _overlap(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(
        np.mean(
            [len(set(left).intersection(right)) / len(left) for left, right in zip(reference, candidate, strict=True)]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded exact/ANN operational benchmark.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--vectors", type=int, default=50_000)
    args = parser.parse_args()
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    count = int(metadata["chunk_count"])
    dimension = int(metadata["model"]["embedding_dimension"])
    source = np.memmap(args.embeddings, dtype=np.float32, mode="r", shape=(count, dimension))
    selected = min(args.vectors, 50_000)
    repetitions = (selected + count - 1) // count
    values = np.tile(np.asarray(source), (repetitions, 1))[:selected].copy()
    if repetitions > 1:
        rng = np.random.default_rng(2026)
        values += rng.normal(0.0, 1e-5, values.shape).astype(np.float32)
        faiss.normalize_L2(values)
    queries = np.ascontiguousarray(values[:100])
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "vector_count": selected,
        "source_vector_count": count,
        "dimension": dimension,
        "synthetic_expansion": repetitions > 1,
        "note": "Expanded rows only measure operational scaling; no relevance claim is made.",
        "indexes": {},
    }
    started = time.perf_counter()
    exact = faiss.IndexFlatIP(dimension)
    exact.add(values)
    build = time.perf_counter() - started
    search, exact_rows = _timed_search(exact, queries, 10)
    exact_path = output / "flat.index"
    faiss.write_index(exact, str(exact_path))
    results["indexes"]["IndexFlatIP"] = {
        "build_seconds": round(build, 6),
        "search_100_queries_seconds": round(search, 6),
        "mean_ms_per_query": round(search * 10, 6),
        "size_bytes": exact_path.stat().st_size,
        "top10_row_overlap_with_exact": 1.0,
    }
    started = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 80
    hnsw.hnsw.efSearch = 64
    hnsw.add(values)
    build = time.perf_counter() - started
    search, rows = _timed_search(hnsw, queries, 10)
    hnsw_path = output / "hnsw.index"
    faiss.write_index(hnsw, str(hnsw_path))
    results["indexes"]["IndexHNSWFlat_M32"] = {
        "build_seconds": round(build, 6),
        "search_100_queries_seconds": round(search, 6),
        "mean_ms_per_query": round(search * 10, 6),
        "size_bytes": hnsw_path.stat().st_size,
        "top10_row_overlap_with_exact": round(_overlap(exact_rows, rows), 6),
    }
    started = time.perf_counter()
    quantizer = faiss.IndexFlatIP(dimension)
    ivfpq = faiss.IndexIVFPQ(quantizer, dimension, 128, 32, 8, faiss.METRIC_INNER_PRODUCT)
    ivfpq.nprobe = 16
    ivfpq.train(values[: min(30_000, selected)])
    ivfpq.add(values)
    build = time.perf_counter() - started
    search, rows = _timed_search(ivfpq, queries, 10)
    pq_path = output / "ivfpq.index"
    faiss.write_index(ivfpq, str(pq_path))
    results["indexes"]["IndexIVFPQ_128x32x8"] = {
        "build_seconds": round(build, 6),
        "search_100_queries_seconds": round(search, 6),
        "mean_ms_per_query": round(search * 10, 6),
        "size_bytes": pq_path.stat().st_size,
        "top10_row_overlap_with_exact": round(_overlap(exact_rows, rows), 6),
    }
    destination = Path(args.summary_out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
