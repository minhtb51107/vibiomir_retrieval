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

from src.indexing.embedder import file_sha256
from src.production.config import load_production_config
from src.production.sampling import select_deterministic_sample


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Select a deterministic bounded Phase 8 URL sample.")
    parser.add_argument("--config", default="configs/production_pipeline.yaml")
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", default="vibiomir-phase8-bounded-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    config = load_production_config(args.config)
    ceiling = int(config["safety"]["maximum_bounded_crawl_records"])
    if args.size <= 0 or args.size > ceiling:
        parser.error(f"sample size must be between 1 and {ceiling}")
    records, summary = select_deterministic_sample(
        config["source"]["corpus"], sample_size=args.size, seed=args.seed
    )
    payload = {
        "format_version": 1,
        "source_corpus": config["source"]["corpus"],
        "source_sha256": file_sha256(config["source"]["corpus"]),
        "records": records,
    }
    _atomic_json(args.output, payload)
    summary.update(
        manifest=str(args.output),
        manifest_sha256=file_sha256(args.output),
        source_sha256=payload["source_sha256"],
    )
    _atomic_json(args.summary_out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
