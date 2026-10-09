#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from src.source_census.union_experiment import run_probe_set


def main() -> int:
    parser=argparse.ArgumentParser(description="Build fixed-contract no-inference Phase 10E information probes.")
    parser.add_argument("--config",default="configs/phase10e_pre_reset_probes.yaml")
    args=parser.parse_args(); print(json.dumps(run_probe_set(args.config),ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
