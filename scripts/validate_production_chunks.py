#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.indexing.tokenizer_validation import percentile
from src.production.config import load_production_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BGE tokens, offsets, and provenance.")
    parser.add_argument("--config", default="configs/production_pipeline.yaml")
    parser.add_argument("--documents", required=True)
    parser.add_argument("--chunks", required=True)
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
    documents: dict[int, str] = {}
    for batch in pq.ParquetFile(args.documents).iter_batches(
        columns=["doc_id", "normalized_text"]
    ):
        for row in batch.to_pylist():
            doc_id = int(row["doc_id"])
            if doc_id in documents:
                raise ValueError(f"duplicate document ID: {doc_id}")
            documents[doc_id] = str(row["normalized_text"])
    lengths: list[int] = []
    chunk_ids: set[str] = set()
    invalid_offsets = 0
    wrong_documents = 0
    for batch in pq.ParquetFile(args.chunks).iter_batches(
        batch_size=256,
        columns=["chunk_id", "doc_id", "raw_text", "start_offset", "end_offset"],
    ):
        rows = batch.to_pylist()
        texts = [str(row["raw_text"]) for row in rows]
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            return_length=True,
            verbose=False,
        )
        lengths.extend(int(value) for value in encoded["length"])
        for row in rows:
            chunk_id = str(row["chunk_id"])
            if chunk_id in chunk_ids:
                raise ValueError(f"duplicate chunk ID: {chunk_id}")
            chunk_ids.add(chunk_id)
            doc_id = int(row["doc_id"])
            source = documents.get(doc_id)
            if source is None:
                wrong_documents += 1
                continue
            start = int(row["start_offset"])
            end = int(row["end_offset"])
            if start < 0 or end < start or end > len(source) or source[start:end] != row["raw_text"]:
                invalid_offsets += 1
    maximum = int(chunking["maximum_tokens_with_specials"])
    summary = {
        "model": chunking["tokenizer_model"],
        "revision": chunking["tokenizer_revision"],
        "chunk_count": len(lengths),
        "unique_chunk_ids": len(chunk_ids),
        "token_counts_with_specials": {
            "median": statistics.median(lengths) if lengths else 0,
            "p95": percentile(lengths, 0.95),
            "max": max(lengths, default=0),
            "over_limit": sum(value > maximum for value in lengths),
        },
        "offset_mismatches": invalid_offsets,
        "wrong_document_references": wrong_documents,
        "result": "PASS"
        if not invalid_offsets
        and not wrong_documents
        and not any(value > maximum for value in lengths)
        else "FAIL",
    }
    if summary["result"] != "PASS":
        raise RuntimeError("production chunk validation failed")
    destination = Path(args.summary_out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
