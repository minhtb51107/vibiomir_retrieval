#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.indexing.metadata_store import build_metadata_store, verify_metadata_store
from src.production.config import load_production_config
from src.retrieval.sparse_retriever import build_sparse_index, verify_sparse_mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Build bounded production sparse index.")
    parser.add_argument("--config", default="configs/production_pipeline.yaml")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--work-directory", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    config = load_production_config(args.config)
    work = Path(args.work_directory)
    metadata_path = work / "chunk_metadata.sqlite"
    metadata = build_metadata_store(args.chunks, metadata_path)
    metadata["mapping_verification"] = verify_metadata_store(args.chunks, metadata_path)
    if metadata["mapping_verification"]["mismatches"]:
        raise RuntimeError("bounded chunk metadata mapping mismatch")
    sparse_path = work / "bm25.sqlite"
    sparse = build_sparse_index(
        chunks_path=args.chunks,
        output_path=sparse_path,
        k1=float(config["sparse"]["k1"]),
        b=float(config["sparse"]["b"]),
        dense_metadata_path=metadata_path,
    )
    sparse["mapping_verification"] = verify_sparse_mapping(sparse_path, metadata_path)
    if sparse["mapping_verification"]["mismatches"]:
        raise RuntimeError("bounded sparse/chunk mapping mismatch")
    summary = {
        "metadata": metadata,
        "sparse": sparse,
        "dense_embedding": {
            "executed": False,
            "reason": "host available RAM was 632 MiB; cached BGE-M3 native load exited before checkpoint creation",
            "projection_source": "Phase 4 measured 49.0083 chunks/second",
        },
    }
    destination = Path(args.summary_out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
