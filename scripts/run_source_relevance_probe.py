#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_probe.pipeline import (
    build_retrieval, build_summary, embed_new_chunks, finalize_rankings,
    generate_submissions, load_config, prepare_inputs, resource_estimate,
    score_pending, seed_checkpoints, materialize_missing_pairs, distribute_novel_scores,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the offline Phase 10B3 source relevance probe.")
    parser.add_argument("--config", default="configs/source_relevance_probe.yaml")
    parser.add_argument("stage", choices=["prepare", "embed", "retrieve", "preflight", "novel-prepare", "novel-distribute", "score", "finalize", "submit", "summary"])
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_path = Path(config["outputs"]["artifact_directory"]) / "input_manifest.json"
    if args.stage == "prepare":
        result = prepare_inputs(config)
        resource_estimate(config, result)
    else:
        if not manifest_path.exists():
            raise FileNotFoundError("run the prepare stage first")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        functions = {
            "embed": embed_new_chunks, "retrieve": build_retrieval,
            "preflight": seed_checkpoints, "score": score_pending,
            "novel-prepare": materialize_missing_pairs,
            "novel-distribute": distribute_novel_scores,
            "finalize": finalize_rankings, "submit": generate_submissions,
            "summary": build_summary,
        }
        result = functions[args.stage](config, manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
