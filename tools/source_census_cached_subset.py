#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_census.cached_subsets import build_cached_subset_rankings, package_cached_subset
from src.source_census.pipeline import load_config
from src.source_census.round3 import (
    ROUND3_GROUPS,
    finalize_round3_report,
    prepare_depth_manifests,
    prepare_round3_groups,
)


def sources_argument(value: str) -> list[str]:
    result = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("at least one comma-separated source is required")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Package arbitrary cached Phase 10D source subsets.")
    parser.add_argument("--config", default="configs/source_census.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-round3")
    sub.add_parser("build-round3-rankings")
    package = sub.add_parser("package-round3-group")
    package.add_argument("--group", choices=list(ROUND3_GROUPS), required=True)
    sub.add_parser("finalize-round3")
    sub.add_parser("prepare-depth-manifests")
    subset = sub.add_parser("make-subset")
    subset.add_argument("--sources", type=sources_argument, required=True)
    subset.add_argument("--name", required=True)
    subset.add_argument("--work-dir", required=True)
    subset.add_argument("--output-dir", required=True)
    existing = sub.add_parser("package-existing")
    existing.add_argument("--ranking-root", required=True)
    existing.add_argument("--group", required=True)
    existing.add_argument("--submission-name", required=True)
    existing.add_argument("--output-dir", required=True)
    existing.add_argument("--marker", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "prepare-round3":
        result = prepare_round3_groups(config)
    elif args.command == "build-round3-rankings":
        result = build_cached_subset_rankings(config, ROUND3_GROUPS, "data/source_census/round3")
    elif args.command == "package-round3-group":
        result = package_cached_subset(
            config,
            ranking_root="data/source_census/round3",
            group=args.group,
            submission_name=f"phase10d_R3_{args.group}",
            output_dir=config["outputs"]["submissions"],
            marker_path=Path(config["outputs"]["artifacts"]) / f"round3_submission_{args.group}.json",
        )
    elif args.command == "finalize-round3":
        result = finalize_round3_report(config)
    elif args.command == "prepare-depth-manifests":
        result = prepare_depth_manifests(config)
    elif args.command == "package-existing":
        result = package_cached_subset(
            config,
            ranking_root=args.ranking_root,
            group=args.group,
            submission_name=args.submission_name,
            output_dir=args.output_dir,
            marker_path=args.marker,
        )
    else:
        groups = {args.name: args.sources}
        ranking_root = Path(args.work_dir) / "rankings"
        build_cached_subset_rankings(config, groups, ranking_root)
        result = package_cached_subset(
            config,
            ranking_root=ranking_root,
            group=args.name,
            submission_name=args.name,
            output_dir=args.output_dir,
            marker_path=Path(args.work_dir) / "submission_manifest.json",
        )
        result["sources"] = args.sources
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
