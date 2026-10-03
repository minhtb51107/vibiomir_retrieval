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
from src.indexing.pipeline import build_dense_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build exact BGE-M3 dense index.")
    parser.add_argument("--config", default="configs/dense_retrieval.yaml")
    parser.add_argument("--summary-out", default="artifacts/phase4_dense/index_benchmark.json")
    args = parser.parse_args()
    config = load_dense_config(args.config)
    model = config["model"]
    index = config["index"]
    embedder = SentenceTransformerEmbedder(
        model_name=model["name"],
        revision=model["revision"],
        device=model["device"],
        batch_size=int(model["batch_size"]),
        max_length=int(model["max_length"]),
        normalize_embeddings=bool(model["normalize_embeddings"]),
        use_half_on_cuda=bool(model["use_half_on_cuda"]),
        seed=int(model["seed"]),
    )
    summary = build_dense_index(
        chunks_path=index["chunk_source"],
        embedding_directory=index["embedding_directory"],
        index_directory=index["directory"],
        embedder=embedder,
        add_batch_size=int(index["add_batch_size"]),
    )
    output = Path(args.summary_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
