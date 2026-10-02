#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.probing.config import load_probe_config
from src.probing.sampler import select_samples, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the reproducible Phase 2B sample.")
    parser.add_argument("--config", default="configs/source_probe.yaml")
    args = parser.parse_args()
    config = load_probe_config(args.config)
    records, counts = select_samples(config)
    json_path, csv_path = write_manifest(config, records, counts)
    print(
        json.dumps(
            {
                "selected_count": len(records),
                "manifest_json": str(json_path),
                "manifest_csv": str(csv_path),
                "rule_selected_counts": counts,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
