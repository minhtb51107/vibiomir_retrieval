#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from src.source_census.union_experiment import run_union


def main() -> int:
    parser=argparse.ArgumentParser(description="Build a no-inference union from organizer-valid Phase 10E source caches.")
    parser.add_argument("--config",default="configs/phase10e_g1a10k_g6b15k_union.yaml")
    args=parser.parse_args()
    print(json.dumps(run_union(args.config),ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
