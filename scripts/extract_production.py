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

from src.production.config import load_production_config
from src.production.extraction import build_document_partitions


def main() -> int:
    parser = argparse.ArgumentParser(description="Build resumable cleaned-document partitions.")
    parser.add_argument("--config", default="configs/production_pipeline.yaml")
    parser.add_argument("--crawl-database", required=True)
    parser.add_argument("--archive-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    config = load_production_config(args.config)
    summary = build_document_partitions(
        crawl_database=args.crawl_database,
        archive_directory=args.archive_directory,
        extraction_config_path=config["processing"]["extraction_config"],
        output_directory=args.output_directory,
        checkpoint_path=args.checkpoint,
        rows_per_partition=int(config["processing"]["documents_per_partition"]),
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
