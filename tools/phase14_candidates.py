#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_census.phase14_candidates import audit_embeddings, load_config, run


def main() -> int:
    parser = argparse.ArgumentParser(description="Dedicated resumable Phase 14 Medlatec candidate adapter")
    parser.add_argument("--config", default="configs/phase14_medlatec_coverage_3000.yaml")
    parser.add_argument("--run-state", default="artifacts/runs/phase14_medlatec_coverage_3000/state.json")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    result = audit_embeddings(config) if args.audit_only else run(config, Path(args.run_state))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
