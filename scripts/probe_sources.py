#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.ingestion.config import load_config
from src.probing.config import load_probe_config
from src.probing.probe import SourceProbe, write_probe_results
from src.probing.sampler import load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the bounded Phase 2B source probe.")
    parser.add_argument("--config", default="configs/source_probe.yaml")
    parser.add_argument("--crawler-config", default="configs/crawler.yaml")
    parser.add_argument(
        "--manifest", default="artifacts/source_probe/sample_manifest.json"
    )
    parser.add_argument(
        "--limit", type=int, help="Optional smaller cap; cannot exceed configured maximum"
    )
    parser.add_argument(
        "--ids", nargs="+", default=[], help="Probe only these manifest doc IDs"
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Replace matching doc IDs in existing results and recompute the summary",
    )
    args = parser.parse_args()
    probe_config = load_probe_config(args.config)
    records = load_manifest(args.manifest)
    if args.ids:
        selected_ids = {
            int(item)
            for value in args.ids
            for item in value.split(",")
            if item.strip()
        }
        records = [record for record in records if record.doc_id in selected_ids]
        missing = selected_ids - {record.doc_id for record in records}
        if missing:
            parser.error(f"doc IDs not present in manifest: {sorted(missing)}")
    if args.limit is not None:
        if args.limit <= 0 or args.limit > probe_config.maximum_live_urls:
            parser.error(
                f"--limit must be between 1 and {probe_config.maximum_live_urls}"
            )
        records = records[: args.limit]
    crawler_config = load_config(args.crawler_config)
    results = asyncio.run(SourceProbe(probe_config, crawler_config).run(records))
    json_path, csv_path, summary_path = write_probe_results(
        probe_config.output_dir, results, merge_existing=args.merge_existing
    )
    print(
        json.dumps(
            {
                "probed": len(results),
                "results_json": str(json_path),
                "results_csv": str(csv_path),
                "summary_json": str(summary_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
