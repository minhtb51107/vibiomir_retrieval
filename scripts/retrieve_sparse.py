#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.retrieval.hybrid_pipeline import (
    load_hybrid_config,
    run_sparse_retrieval,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve all queries with BM25.")
    parser.add_argument("--config", default="configs/hybrid_retrieval.yaml")
    parser.add_argument(
        "--summary-out", default="artifacts/phase6_hybrid/sparse_retrieval_benchmark.json"
    )
    args = parser.parse_args()
    config = load_hybrid_config(args.config)
    summary = run_sparse_retrieval(
        query_path=config["inputs"]["queries"],
        index_path=config["sparse"]["index_path"],
        metadata_path=config["inputs"]["dense_metadata"],
        output_directory=config["outputs"]["directory"],
        top_k_chunks=int(config["sparse"]["top_k_chunks"]),
        top_k_docs=int(config["documents"]["top_k"]),
        top_n_mean=int(config["documents"]["top_n_mean"]),
    )
    write_json(args.summary_out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
