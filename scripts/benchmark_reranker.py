#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.reranking.candidate_pool import build_pool_from_paths
from src.reranking.reranker import (
    TransformerCrossEncoderReranker,
    load_reranking_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark GPU-safe reranker batches.")
    parser.add_argument("--config", default="configs/reranking.yaml")
    parser.add_argument(
        "--output", default="artifacts/phase7_reranking/gpu_benchmark.json"
    )
    args = parser.parse_args()
    config = load_reranking_config(args.config)
    inputs = config["inputs"]
    pool_config = config["candidate_pool"]
    rows, pool_summary = build_pool_from_paths(
        dense_path=inputs["dense_candidates"],
        sparse_path=inputs["sparse_candidates"],
        hybrid_path=inputs["hybrid_candidates"],
        dense_depth=int(pool_config["dense_depth"]),
        sparse_depth=int(pool_config["sparse_depth"]),
        hybrid_depth=int(pool_config["hybrid_depth"]),
        max_depth=max(int(value) for value in pool_config["rerank_depths"]),
    )
    sample = sorted(
        rows,
        key=lambda row: (
            -(len(str(row["query_text"])) + len(str(row["chunk_text"]))),
            int(row["query_id"]),
            str(row["chunk_id"]),
        ),
    )[:64]
    pairs = [(str(row["query_text"]), str(row["chunk_text"])) for row in sample]
    model_config = config["model"]
    reranker = TransformerCrossEncoderReranker(
        model_name=model_config["name"],
        revision=model_config["revision"],
        cache_dir=model_config["cache_dir"],
        device=model_config["device"],
        batch_size=1,
        max_length=int(model_config["max_sequence_length"]),
        use_half_on_cuda=model_config["precision"] == "float16",
        seed=int(model_config["seed"]),
    )
    torch = reranker._torch
    reranker.score_pairs(pairs[:1])
    original_lengths = reranker.token_lengths(pairs, truncation=False)
    effective_lengths = reranker.token_lengths(pairs, truncation=True)
    results = []
    allow_next = True
    benchmark_config = config.get("hardware_benchmark", {})
    batch_sizes = [int(value) for value in benchmark_config.get("batch_sizes", [1, 2])]
    known_unsafe = {
        int(value)
        for value in benchmark_config.get("known_unsafe_batch_sizes", [])
    }
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("hardware benchmark batch sizes must be positive")
    if set(batch_sizes) & known_unsafe:
        raise ValueError("refusing to benchmark a known-unsafe batch size")
    for batch_size in batch_sizes:
        if not allow_next:
            break
        reranker.batch_size = batch_size
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            scores = reranker.score_pairs(pairs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            reserved = int(torch.cuda.max_memory_reserved())
            allocated = int(torch.cuda.max_memory_allocated())
            total = int(torch.cuda.get_device_properties(0).total_memory)
            result = {
                "batch_size": batch_size,
                "status": "SUCCESS",
                "sample_pairs": len(pairs),
                "elapsed_seconds": round(elapsed, 6),
                "pairs_per_second": round(len(pairs) / elapsed, 6),
                "mean_latency_ms_per_pair": round(elapsed * 1000 / len(pairs), 6),
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved,
                "peak_reserved_vram_fraction": round(reserved / total, 6),
                "safe_headroom_bytes": total - reserved,
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
            }
            results.append(result)
            allow_next = reserved / total <= 0.75 and total - reserved >= 1_000_000_000
        except torch.cuda.OutOfMemoryError as error:
            results.append(
                {
                    "batch_size": batch_size,
                    "status": "OOM",
                    "error": str(error),
                }
            )
            torch.cuda.empty_cache()
            allow_next = False
    successful = [row for row in results if row["status"] == "SUCCESS"]
    safe = [
        row
        for row in successful
        if row["peak_reserved_vram_fraction"] <= 0.75
        and row["safe_headroom_bytes"] >= 1_000_000_000
    ]
    chosen = int((safe or successful)[-1]["batch_size"])
    reranker.batch_size = chosen
    summary = {
        "model": reranker.metadata(),
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "sample_strategy": "64 largest query+chunk character pairs from unified top-100 pools",
        "sample_original_token_lengths": {
            "min": min(original_lengths),
            "max": max(original_lengths),
        },
        "sample_effective_token_lengths": {
            "min": min(effective_lengths),
            "max": max(effective_lengths),
        },
        "pool_summary": pool_summary,
        "results": results,
        "chosen_batch_size": chosen,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
