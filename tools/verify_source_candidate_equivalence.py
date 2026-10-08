#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indexing.embedder import normalize_text
from src.retrieval.fusion import reciprocal_rank_fusion
from src.source_census.pipeline import (
    SourceBM25Index,
    _bm25_rank,
    load_config,
    stable_top_k_indices,
)


def host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def main() -> int:
    config = load_config("configs/source_census.yaml")
    chunks = pq.read_table(config["outputs"]["chunks"]).to_pylist()
    triage = json.loads((Path(config["outputs"]["artifacts"]) / "technical_triage.json").read_text(encoding="utf-8"))
    healthy = set(triage["healthy_sources"])
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in chunks:
        source = host(str(row.get("source_url") or ""))
        if source in healthy:
            by_source[source].append(row)
    ordered = sorted(by_source, key=lambda source: (len(by_source[source]), source))
    selected = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    queries = pq.read_table(config["inputs"]["queries"], columns=["id", "query"]).to_pylist()[:20]
    query_texts = [normalize_text(str(row["query"])) for row in queries]
    dimension = int(config["models"]["embedder"]["dimension"])
    all_vectors = np.memmap(
        Path(config["outputs"]["root"]) / "round1/chunk_embeddings.f32",
        dtype=np.float32, mode="r", shape=(len(chunks), dimension),
    )
    query_vectors = np.memmap(
        "data/source_relevance_probe/query_embeddings.f32",
        dtype=np.float32, mode="r", shape=(1200, dimension),
    )
    chunk_position = {str(row["chunk_id"]): index for index, row in enumerate(chunks)}
    old_seconds = 0.0
    new_seconds = 0.0
    comparisons = 0
    for source in selected:
        rows = by_source[source]
        texts = [str(row["normalized_text"]) for row in rows]
        matrix = np.asarray(all_vectors[[chunk_position[str(row["chunk_id"])] for row in rows]], dtype=np.float32)
        dense_scores = np.asarray(query_vectors[:20]) @ matrix.T
        start = time.perf_counter()
        old_sparse = [_bm25_rank(texts, query, k1=1.2, b=.75, top_k=8) for query in query_texts]
        old_dense = [np.argsort(-dense_scores[index], kind="stable")[:8] for index in range(20)]
        old_seconds += time.perf_counter() - start
        start = time.perf_counter()
        index = SourceBM25Index(texts, k1=1.2, b=.75)
        new_sparse = [index.rank(query, top_k=8) for query in query_texts]
        new_dense = [stable_top_k_indices(dense_scores[q], 8) for q in range(20)]
        new_seconds += time.perf_counter() - start
        for q in range(20):
            if [row for row, _ in old_sparse[q]] != [row for row, _ in new_sparse[q]]:
                raise AssertionError(f"BM25 IDs differ for {source} query {q}")
            if not np.allclose([score for _, score in old_sparse[q]], [score for _, score in new_sparse[q]], rtol=0.0, atol=1e-12):
                raise AssertionError(f"BM25 scores differ for {source} query {q}")
            if old_dense[q].tolist() != new_dense[q].tolist():
                raise AssertionError(f"dense IDs differ for {source} query {q}")
            def ranking(order, sparse):
                dense = [{"chunk_id": rows[int(i)]["chunk_id"], "score": float(dense_scores[q, i]), "rank": rank} for rank, i in enumerate(order, 1)]
                sparse_rows = [{"chunk_id": rows[i]["chunk_id"], "score": score, "rank": rank} for rank, (i, score) in enumerate(sparse, 1)]
                return reciprocal_rank_fusion([dense, sparse_rows], rrf_constant=60, top_k=8)
            old_fused = ranking(old_dense[q], old_sparse[q]); new_fused = ranking(new_dense[q], new_sparse[q])
            if [row["chunk_id"] for row in old_fused] != [row["chunk_id"] for row in new_fused]:
                raise AssertionError(f"fused IDs differ for {source} query {q}")
            if not np.allclose([row["score"] for row in old_fused], [row["score"] for row in new_fused], rtol=0.0, atol=1e-15):
                raise AssertionError(f"fused scores differ for {source} query {q}")
            comparisons += 1
    result = {
        "equivalent": True,
        "sources": [{"source": source, "chunks": len(by_source[source])} for source in selected],
        "queries": len(queries), "comparisons": comparisons,
        "old_seconds": round(old_seconds, 6), "optimized_seconds": round(new_seconds, 6),
        "speedup": round(old_seconds / max(new_seconds, 1e-12), 3),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
