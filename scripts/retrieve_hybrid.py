#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.indexing.embedder import file_sha256
from src.retrieval.hybrid_pipeline import (
    load_hybrid_config,
    run_hybrid_comparison,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fuse existing Phase 4 dense and Phase 6 sparse rankings."
    )
    parser.add_argument("--config", default="configs/hybrid_retrieval.yaml")
    args = parser.parse_args()
    config = load_hybrid_config(args.config)
    inputs = config["inputs"]
    sparse_results = (
        Path(config["outputs"]["directory"]) / "sparse_chunk_rankings.parquet"
    )
    fusion, overlaps, multilingual = run_hybrid_comparison(
        query_path=inputs["queries"],
        chunks_path=inputs["chunks"],
        dense_results_path=inputs["dense_chunk_results"],
        sparse_results_path=sparse_results,
        output_directory=config["outputs"]["directory"],
        dense_top_k=int(config["fusion"]["dense_top_k"]),
        sparse_top_k=int(config["fusion"]["sparse_top_k"]),
        hybrid_top_k=int(config["fusion"]["hybrid_top_k"]),
        top_k_docs=int(config["documents"]["top_k"]),
        top_n_mean=int(config["documents"]["top_n_mean"]),
        rrf_constant=int(config["fusion"]["rrf_constant"]),
        sanity_query_count=int(config["diagnostics"]["sanity_query_count"]),
    )
    artifact_directory = Path(config["outputs"]["artifacts"])
    sparse_index = json.loads(
        (artifact_directory / "sparse_index_benchmark.json").read_text(encoding="utf-8")
    )
    sparse_retrieval = json.loads(
        (artifact_directory / "sparse_retrieval_benchmark.json").read_text(encoding="utf-8")
    )
    dense_benchmark = json.loads(Path(inputs["dense_benchmark"]).read_text(encoding="utf-8"))
    fusion_latency = fusion["fusion_latency"]
    dense_latency = float(dense_benchmark["mean_latency_ms_per_query"])
    sparse_latency = float(sparse_retrieval["latency"]["mean_ms_per_query"])
    benchmark = {
        "label": "PERFORMANCE BENCHMARK — no relevance evaluation",
        "query_count": sparse_retrieval["query_count"],
        "sparse_index": sparse_index,
        "dense_reuse": {
            "status": "reused existing Phase 4 rankings and index without modification",
            "index_path": inputs["dense_index"],
            "index_sha256": file_sha256(inputs["dense_index"]),
            "measured_phase4_mean_ms_per_query": dense_latency,
        },
        "sparse_retrieval": sparse_retrieval,
        "fusion": fusion,
        "hybrid_latency": {
            "mean_ms_per_query": round(
                dense_latency + sparse_latency + float(fusion_latency["mean_ms_per_query"]),
                6,
            ),
            "p50_component_sum_ms_per_query": round(
                float(dense_benchmark["p50_latency_ms_per_query"])
                + float(sparse_retrieval["latency"]["p50_ms_per_query"])
                + float(fusion_latency["p50_ms_per_query"]),
                6,
            ),
            "p95_component_sum_ms_per_query": round(
                float(dense_benchmark["p95_latency_ms_per_query"])
                + float(sparse_retrieval["latency"]["p95_ms_per_query"])
                + float(fusion_latency["p95_ms_per_query"]),
                6,
            ),
            "components": "sums of separately measured Phase 4 dense, Phase 6 sparse, and fusion statistics; excludes one-time model/index loading and is not a jointly timed latency distribution",
        },
    }
    write_json(artifact_directory / "benchmark.json", benchmark)
    write_json(artifact_directory / "overlap_stats.json", overlaps)
    write_json(artifact_directory / "multilingual_sanity.json", multilingual)
    write_json(artifact_directory / "fusion_comparison.json", fusion)
    print(json.dumps(benchmark, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
