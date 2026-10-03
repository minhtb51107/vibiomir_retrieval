#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.indexing.config import load_dense_config
from src.indexing.tokenizer_validation import validate_and_maybe_regenerate


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 3 chunks with BGE-M3.")
    parser.add_argument("--config", default="configs/dense_retrieval.yaml")
    parser.add_argument(
        "--summary-out", default="artifacts/phase4_dense/tokenizer_stats.json"
    )
    args = parser.parse_args()
    config = load_dense_config(args.config)
    model = config["model"]
    validation = config["tokenizer_validation"]
    tokenizer = AutoTokenizer.from_pretrained(
        model["name"], revision=model["revision"], use_fast=True
    )
    summary = validate_and_maybe_regenerate(
        tokenizer=tokenizer,
        source_chunks=validation["source_chunks"],
        source_documents=validation["source_documents"],
        validated_chunks=validation["validated_chunks"],
        max_tokens=int(validation["intended_max_tokens"]),
        overlap_tokens=int(validation["overlap_tokens"]),
        material_exceedance_rate=float(validation["material_exceedance_rate"]),
    )
    output = Path(args.summary_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
