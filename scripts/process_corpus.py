#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.processing.pipeline import process_archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean and chunk an archived crawl subset.")
    parser.add_argument("--crawl-db", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--extraction-config", default="configs/extraction.yaml")
    parser.add_argument("--chunking-config", default="configs/chunking.yaml")
    parser.add_argument("--documents-out", default="data/processed/clean_documents.parquet")
    parser.add_argument("--chunks-dir", default="data/chunks")
    parser.add_argument("--summary-out", default="artifacts/phase3_pilot/summary.json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    summary = process_archive(
        crawl_db=args.crawl_db,
        archive_dir=args.archive_dir,
        extraction_config_path=args.extraction_config,
        chunking_config_path=args.chunking_config,
        documents_out=args.documents_out,
        chunks_dir=args.chunks_dir,
        summary_out=args.summary_out,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
