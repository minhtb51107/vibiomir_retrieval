#!/usr/bin/env python3
"""Resume Phase 10D depth1000 at RANKINGS after cache verification."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_census.cached_subsets import (
    _require_complete_score_cache,
    build_cached_subset_rankings,
)
from src.source_census.depth1000 import DEPTH_GROUPS, finalize_depth_report
from src.source_census.pipeline import _merge_source_parts, atomic_json, load_config


def stage(path: Path, name: str, **extra: object) -> None:
    atomic_json(path, {"stage": name, "status": "RUNNING", **extra})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/source_census_depth1000.yaml")
    parser.add_argument("--run-state", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    outputs = config["outputs"]
    state = Path(args.run_state)

    cache = _require_complete_score_cache(config)
    stage(state, "RANKINGS", cache=cache, resumed_from="RANKINGS")
    ranking_root = Path(outputs["root"]) / "rankings"
    build_cached_subset_rankings(config, DEPTH_GROUPS, ranking_root)

    canonical_root = Path(outputs["root"]) / "round1"
    _merge_source_parts(
        [Path(config["inputs"]["pilot_chunks"]), Path(outputs["chunks"])],
        canonical_root / "combined_chunks.parquet",
    )
    _merge_source_parts(
        [Path(config["inputs"]["pilot_documents"]), Path(outputs["documents"])],
        canonical_root / "combined_documents.parquet",
    )

    stage(state, "PACKAGE", completed=0, total=3, resumed_from="RANKINGS")
    submissions: dict[str, object] = {}
    for index, group in enumerate(DEPTH_GROUPS, 1):
        marker = Path(outputs["artifacts"]) / f"submission_{group}.json"
        subprocess.run(
            [
                sys.executable, "tools/source_census_cached_subset.py", "--config", args.config,
                "package-existing", "--ranking-root", str(ranking_root), "--group", group,
                "--submission-name", f"phase10d_DEPTH1000_{group}",
                "--output-dir", outputs["submissions"], "--marker", str(marker),
            ],
            cwd=ROOT,
            check=True,
        )
        submissions[group] = json.loads(marker.read_text(encoding="utf-8"))
        stage(state, "PACKAGE", completed=index, total=3, current_group=group)

    report = finalize_depth_report(config, submissions)
    atomic_json(
        state,
        {
            "stage": "WAITING_FOR_LEADERBOARD",
            "status": "WAITING_FOR_LEADERBOARD",
            "resumed_from": "RANKINGS",
            "rerank_scores_recomputed": 0,
            "report": str(Path(outputs["artifacts"]).parent / "depth1000_report.json"),
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
