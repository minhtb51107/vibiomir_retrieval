#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_census.pipeline import atomic_json, load_config
from src.source_census.round2 import (
    finalize_round2,
    finalize_round2_report,
    package_round2_group,
    prepare_round2_groups,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Package Phase 10D Round-2 from cached scores only.")
    parser.add_argument("--config", default="configs/source_census.yaml")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--package-group", choices=[f"G{index}" for index in range(1, 8)])
    parser.add_argument("--finalize-report", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    state_path = Path("artifacts/runs/phase10d_round2/state.json")
    state = {
        "run_id": "phase10d_round2",
        "stage": "GROUP_ASSIGNMENT" if args.prepare_only else "PACKAGE",
        "status": "RUNNING",
        "new_model_inference": 0,
    }
    atomic_json(state_path, state)
    try:
        if args.prepare_only:
            result = prepare_round2_groups(config)
        elif args.package_group:
            result = package_round2_group(config, args.package_group)
        elif args.finalize_report:
            result = finalize_round2_report(config)
        else:
            result = finalize_round2(config)
    except Exception as exc:
        atomic_json(state_path, {**state, "status": "NEEDS_AGENT", "error": repr(exc)})
        raise
    terminal = "GROUPS_READY" if args.prepare_only else "GROUP_PACKAGED" if args.package_group else "WAITING_FOR_LEADERBOARD"
    atomic_json(state_path, {**state, "stage": terminal, "status": terminal, "group": args.package_group})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
