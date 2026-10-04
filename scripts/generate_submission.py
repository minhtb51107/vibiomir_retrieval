#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.submission.common import atomic_write_json
from src.submission.generator import (
    generate_all_variants,
    load_submission_config,
    manifest_fingerprint,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic Phase 9 organizer submissions."
    )
    parser.add_argument("--config", default="configs/submission.yaml")
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Generate a second isolated copy and require matching JSON and ZIP hashes.",
    )
    args = parser.parse_args()
    config = load_submission_config(args.config)
    manifest = generate_all_variants(config, config_path=args.config)
    determinism = {
        "verified": False,
        "variant_count": len(manifest["variants"]),
        "first_manifest_fingerprint": manifest_fingerprint(manifest),
    }
    if args.verify_determinism:
        with tempfile.TemporaryDirectory(prefix="phase9_determinism_") as temporary:
            second = generate_all_variants(
                config,
                config_path=args.config,
                output_directory=temporary,
                write_manifest=False,
            )
        first_rows = [
            (row["name"], row["json_sha256"], row["zip_sha256"])
            for row in manifest["variants"]
        ]
        second_rows = [
            (row["name"], row["json_sha256"], row["zip_sha256"])
            for row in second["variants"]
        ]
        if first_rows != second_rows:
            raise RuntimeError("determinism verification failed: output hashes differ")
        determinism.update(
            {
                "verified": True,
                "second_manifest_fingerprint": manifest_fingerprint(second),
                "byte_identical_json_and_zip": True,
            }
        )
    artifacts = Path(config["output"]["artifact_directory"])
    atomic_write_json(artifacts / "determinism.json", determinism)
    print(json.dumps({"manifest": manifest, "determinism": determinism}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
