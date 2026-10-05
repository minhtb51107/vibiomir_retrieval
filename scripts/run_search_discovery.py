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

from src.search_discovery.pipeline import load_config, run_stage


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded Phase 10B0 search discovery.")
    parser.add_argument("--config", default="configs/search_discovery.yaml")
    parser.add_argument("--stage", choices=["A", "B"], required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    summary = run_stage(config, stage=args.stage)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "execution": summary["execution"],
                "combined_metrics": summary["combined_metrics"],
                "stage_a_passed": summary.get("stage_a_passed"),
                "viability": summary.get("viability"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
