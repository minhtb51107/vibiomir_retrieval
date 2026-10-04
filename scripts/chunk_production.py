#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.production.chunking import build_chunk_partitions
from src.production.config import load_production_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build resumable BGE-M3-tokenized chunk partitions."
    )
    parser.add_argument("--config", default="configs/production_pipeline.yaml")
    parser.add_argument("--documents", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    config = load_production_config(args.config)
    chunking = config["chunking"]
    tokenizer = AutoTokenizer.from_pretrained(
        chunking["tokenizer_model"],
        revision=chunking["tokenizer_revision"],
        use_fast=True,
        local_files_only=True,
    )
    summary = build_chunk_partitions(
        tokenizer=tokenizer,
        source_documents=args.documents,
        output_directory=args.output_directory,
        checkpoint_path=args.checkpoint,
        tokenizer_revision=chunking["tokenizer_revision"],
        content_token_budget=int(chunking["content_token_budget"]),
        overlap_tokens=int(chunking["overlap_tokens"]),
        documents_per_partition=int(chunking["partition_documents"]),
    )
    destination = Path(args.summary_out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
