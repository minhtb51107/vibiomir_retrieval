#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.retrieval.hybrid_pipeline import load_hybrid_config, write_json
from src.retrieval.sparse_retriever import build_sparse_index, verify_sparse_mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic BM25 sparse index.")
    parser.add_argument("--config", default="configs/hybrid_retrieval.yaml")
    parser.add_argument(
        "--summary-out", default="artifacts/phase6_hybrid/sparse_index_benchmark.json"
    )
    args = parser.parse_args()
    config = load_hybrid_config(args.config)
    inputs = config["inputs"]
    sparse = config["sparse"]
    summary = build_sparse_index(
        chunks_path=inputs["chunks"],
        output_path=sparse["index_path"],
        k1=float(sparse["k1"]),
        b=float(sparse["b"]),
        dense_metadata_path=inputs["dense_metadata"],
    )
    verification = verify_sparse_mapping(
        sparse["index_path"], inputs["dense_metadata"]
    )
    summary["mapping_verification"] = verification
    summary["mapping_verified"] = verification["mismatches"] == 0
    if not summary["mapping_verified"]:
        raise RuntimeError("sparse row/chunk/document mapping verification failed")
    write_json(args.summary_out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
