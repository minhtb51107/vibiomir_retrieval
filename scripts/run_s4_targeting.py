#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.s4_targeting.tournament import finalize, load_config, run_external_search, run_local


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the bounded Phase 10C0 S4 targeting tournament.")
    parser.add_argument("stage", choices=["local", "search-a", "search-b", "finalize"])
    parser.add_argument("--config", default="configs/s4_targeting.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "local": result = run_local(config)
    elif args.stage == "search-a": result = run_external_search(config, "A")
    elif args.stage == "search-b": result = run_external_search(config, "B")
    else: result = finalize(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
