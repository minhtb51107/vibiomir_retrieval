#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multilingual_retrieval.m31 import run_m31


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run M3.1 development-only retrieval repair")
    parser.parse_args()
    print(json.dumps(run_m31(ROOT), ensure_ascii=False, indent=2))
