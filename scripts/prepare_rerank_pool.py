#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.reranking.pipeline import prepare_candidate_pool, write_json
from src.reranking.reranker import load_reranking_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the unified Phase 7 candidate pool without loading CUDA."
    )
    parser.add_argument("--config", default="configs/reranking.yaml")
    args = parser.parse_args()
    config = load_reranking_config(args.config)
    _, summary = prepare_candidate_pool(config)
    destination = Path(config["outputs"]["artifacts"]) / "candidate_pool.json"
    write_json(destination, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
