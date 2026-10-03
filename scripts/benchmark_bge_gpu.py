#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.indexing.config import load_dense_config
from src.indexing.embedder import SentenceTransformerEmbedder


def main() -> int:
    parser = argparse.ArgumentParser(description="Conservative BGE-M3 CUDA batch benchmark.")
    parser.add_argument("--config", default="configs/dense_retrieval.yaml")
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--summary-out", default="artifacts/phase4_dense/gpu_benchmark.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    config = load_dense_config(args.config)
    model_config = config["model"]
    chunk_path = config["index"]["chunk_source"]
    rows = pq.read_table(chunk_path, columns=["normalized_text", "token_count"]).to_pylist()
    rows.sort(key=lambda row: int(row["token_count"]), reverse=True)
    sample = rows[: args.sample_size]
    texts = [str(row["normalized_text"]) for row in sample]
    embedder = SentenceTransformerEmbedder(
        model_name=model_config["name"],
        revision=model_config["revision"],
        device="cuda",
        batch_size=1,
        max_length=int(model_config["max_length"]),
        normalize_embeddings=bool(model_config["normalize_embeddings"]),
        use_half_on_cuda=True,
        seed=int(model_config["seed"]),
    )
    embedder.encode(texts[:1])
    torch.cuda.synchronize()
    total_vram = torch.cuda.get_device_properties(0).total_memory
    results = []
    previous_peak = None
    for batch_size in (1, 2, 4):
        if batch_size == 4 and previous_peak is not None:
            batch_one_peak = results[0]["peak_reserved_bytes"]
            projected = previous_peak + 2 * max(previous_peak - batch_one_peak, 0)
            if previous_peak / total_vram > 0.70 or projected / total_vram > 0.85:
                results.append(
                    {
                        "batch_size": 4,
                        "status": "SKIPPED_SAFETY_HEADROOM",
                        "reason": "batch-2 peak or conservative projection exceeded threshold",
                    }
                )
                break
        embedder.batch_size = batch_size
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            vectors = embedder.encode(texts)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            results.append(
                {
                    "batch_size": batch_size,
                    "status": "SUCCESS",
                    "sample_chunks": len(texts),
                    "sample_token_count_min": min(int(row["token_count"]) for row in sample),
                    "sample_token_count_max": max(int(row["token_count"]) for row in sample),
                    "elapsed_seconds": round(elapsed, 4),
                    "chunks_per_second": round(len(texts) / elapsed, 4),
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "peak_reserved_vram_fraction": round(peak_reserved / total_vram, 6),
                    "safe_headroom_bytes": total_vram - peak_reserved,
                    "embedding_shape": list(vectors.shape),
                }
            )
            previous_peak = peak_reserved
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            results.append(
                {"batch_size": batch_size, "status": "OOM", "error": str(exc)[:500]}
            )
            break
    successful = [row for row in results if row["status"] == "SUCCESS"]
    chosen = max(int(row["batch_size"]) for row in successful)
    summary = {
        "model_name": model_config["name"],
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "total_vram_bytes": total_vram,
        "precision": "float16 model / float32 output embeddings",
        "sample_strategy": "64 longest validated chunks by BGE-M3 content token count",
        "results": results,
        "chosen_batch_size": chosen,
    }
    output = Path(args.summary_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
