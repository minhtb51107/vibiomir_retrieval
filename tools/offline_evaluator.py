#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.silver import evaluate_rankings


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate rankings on the provenance-backed silver benchmark")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--official-corpus", type=Path, default=ROOT / "data/raw/links_corpus_vi_en_zh.parquet")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    official = set(map(int, pq.read_table(args.official_corpus, columns=["id"]).column(0).to_pylist()))
    result = evaluate_rankings(args.benchmark, args.rankings, official)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(payload, encoding="utf-8")
    print(payload, end=""); return 0


if __name__ == "__main__":
    raise SystemExit(main())
