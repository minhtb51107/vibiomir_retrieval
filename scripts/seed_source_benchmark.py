from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.acquisition_benchmark.seeding import seed_from_existing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--limit", type=int, required=True, choices=[200, 1000])
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/source_acquisition_benchmark.yaml").read_text(encoding="utf-8"))
    sample = json.loads(Path("artifacts/phase10b2_acquisition/sample_manifest.json").read_text(encoding="utf-8"))
    rows = sample["domains"][args.domain]["records"][: args.limit]
    selected = [row for row in rows if row["phase8_reusable"]]
    root = Path(config["pipeline"]["output_root"]) / args.domain
    result = seed_from_existing(
        selected,
        source_database=config["source"]["phase8_crawl_database"],
        source_archive=config["source"]["phase8_body_archive"],
        destination_database=root / "crawl.sqlite",
        destination_archive=root / "bodies",
    )
    print(json.dumps({"domain": args.domain, "limit": args.limit, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
