#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multilingual_retrieval.m2 import run_vertical_slice


if __name__ == "__main__":
    print(json.dumps(run_vertical_slice(ROOT), ensure_ascii=False, indent=2))
