#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.indexing.config import load_dense_config
from src.indexing.embedder import SentenceTransformerEmbedder
from src.retrieval.pipeline import run_retrieval


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve with exact BGE-M3 dense index.")
    parser.add_argument("--config", default="configs/dense_retrieval.yaml")
    parser.add_argument(
        "--benchmark-out", default="artifacts/phase4_dense/retrieval_benchmark.json"
    )
    parser.add_argument(
        "--sanity-out", default="artifacts/phase4_dense/sanity_summary.json"
    )
    args = parser.parse_args()
    config = load_dense_config(args.config)
    model = config["model"]
    index = config["index"]
    retrieval = config["retrieval"]
    embedder = SentenceTransformerEmbedder(
        model_name=model["name"],
        revision=model["revision"],
        device=model["device"],
        batch_size=int(model["query_batch_size"]),
        max_length=int(model["max_length"]),
        normalize_embeddings=bool(model["normalize_embeddings"]),
        use_half_on_cuda=bool(model["use_half_on_cuda"]),
        seed=int(model["seed"]),
    )
    benchmark, sanity = run_retrieval(
        query_path=retrieval["query_source"],
        index_path=Path(index["directory"]) / "chunks.index",
        metadata_path=Path(index["directory"]) / "chunk_metadata.sqlite",
        output_directory=retrieval["output_directory"],
        embedder=embedder,
        query_batch_size=int(model["query_batch_size"]),
        top_k_chunks=int(retrieval["top_k_chunks"]),
        top_k_docs=int(retrieval["top_k_docs"]),
        top_n_mean=int(retrieval["top_n_mean"]),
        sanity_query_count=int(retrieval["sanity_query_count"]),
    )
    benchmark_path = Path(args.benchmark_out)
    sanity_path = Path(args.sanity_out)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    sanity_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sanity_path.write_text(
        json.dumps(sanity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(benchmark, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
