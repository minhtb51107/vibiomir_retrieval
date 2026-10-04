#!/usr/bin/env python3
"""Generate the nine Phase 10A leaderboard-calibration submissions (offline)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.submission.calibration import build_calibration, load_calibration_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Phase 10A calibration submissions from persisted artifacts."
    )
    parser.add_argument("--config", default="configs/phase10a_calibration.yaml")
    parser.add_argument(
        "--phase9-manifest", default="artifacts/phase9_submission/submission_manifest.json"
    )
    parser.add_argument("--verify-determinism", action="store_true")
    parser.add_argument(
        "--only", nargs="*", help="Restrict to experiment ids (debugging; manifest is partial)"
    )
    args = parser.parse_args()
    config = load_calibration_config(args.config)
    manifest = build_calibration(
        config,
        config_path=args.config,
        verify_determinism=args.verify_determinism,
        phase9_manifest_path=args.phase9_manifest,
        only=args.only,
    )
    summary = [
        {
            "id": row["id"],
            "docs": row["documents_per_query"],
            "chunks": row["chunks_per_query"],
            "zip_sha256": row["zip_sha256"],
        }
        for row in manifest["experiments"]
    ]
    print(json.dumps({"determinism": manifest["determinism"], "experiments": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
