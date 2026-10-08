#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_census.pipeline import atomic_json, load_config, prepare_round1, score_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and advance the adaptive Phase 10D source census.")
    parser.add_argument("--config", default="configs/source_census.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    advance = sub.add_parser("advance")
    advance.add_argument("--round", type=int, required=True, choices=[1, 2])
    for group in "ABC":
        advance.add_argument(f"--score-{group}", type=float, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        result = prepare_round1(config)
    else:
        scores = {group: float(getattr(args, f"score_{group}")) for group in "ABC"}
        decision = score_gate(scores)
        result = {"round": args.round, "scores": scores, "decision": decision}
        destination = Path(config["outputs"]["artifacts"]) / f"round{args.round}_organizer_decision.json"
        atomic_json(destination, result)
        if decision["status"] == "AMBIGUOUS":
            print(json.dumps(result, indent=2))
            return 2
        result["next_round_prepared"] = False
        result["note"] = "Decision recorded; run the supervised next-round preparation to acquire only missing work."
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

