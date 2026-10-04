#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.production.config import load_production_config
from src.production.safety import GIB, audit_disk, evaluate_disk_guard, validate_execution_gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 8 production safety gate and bounded launcher.")
    parser.add_argument("--config", default="configs/production_pipeline.yaml")
    parser.add_argument("--stage", choices=["8A", "8B", "8C", "8D", "8E", "8F"], required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--confirm-bounded")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--confirm-full-production")
    parser.add_argument("--projected-growth-bytes", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_production_config(args.config)
        validate_execution_gate(
            config,
            stage=args.stage,
            max_records=args.max_records,
            bounded_confirmation=args.confirm_bounded,
            full=args.full,
            full_confirmation=args.confirm_full_production,
        )
        audit = audit_disk(REPOSITORY_ROOT)
        threshold_key = "indexing" if args.stage in {"8B", "8E"} else "acquisition"
        safety = config["safety"]
        guard = evaluate_disk_guard(
            audit,
            projected_growth_bytes=args.projected_growth_bytes,
            minimum_free_bytes=int(safety["minimum_free_before_stage_gib"][threshold_key] * GIB),
            headroom_multiplier=float(safety["projected_growth_headroom_multiplier"]),
            maximum_projected_disk_fraction=float(safety["maximum_projected_disk_fraction"]),
        )
        if not guard["allowed"]:
            raise ValueError("disk safety guard rejected the requested stage")
        if args.execute:
            raise ValueError(
                "generic execution is intentionally unavailable; invoke the documented "
                "bounded stage command after this preflight succeeds"
            )
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "authorized": True,
                "dry_run": True,
                "stage": args.stage,
                "max_records": args.max_records,
                "disk_guard": guard,
                "message": "preflight only; no crawl or processing was started",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
