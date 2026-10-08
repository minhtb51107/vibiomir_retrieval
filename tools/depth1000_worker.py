#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_census.cached_subsets import build_cached_subset_rankings
from src.source_census.depth1000 import (
    DEPTH_GROUPS,
    assemble_depth_corpus,
    assemble_depth_embeddings,
    finalize_depth_report,
    seed_depth_scores,
)
from src.source_census.pipeline import (
    _merge_source_parts,
    atomic_json,
    build_source_candidates,
    load_config,
    merge_acquisition,
    score_database_status,
)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def stage(path: Path, name: str, **extra: object) -> None:
    atomic_json(path, {"stage": name, "status": "RUNNING", **extra})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/source_census_depth1000.yaml")
    parser.add_argument("--run-state", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    outputs = config["outputs"]
    depth = config["depth1000"]
    state = Path(args.run_state)
    python = sys.executable
    Path(outputs["artifacts"]).mkdir(parents=True, exist_ok=True)

    stage(state, "ACQUIRE", selected=json.loads(Path(depth["manifest"]).read_text(encoding="utf-8"))["total_incremental_ids"], completed=0)
    run([python, "tools/depth1000_acquire.py", "--config", args.config, "--run-state", args.run_state])

    stage(state, "MERGE_ACQUISITION")
    merge_acquisition(config)

    stage(state, "EXTRACT")
    run([
        python, "scripts/extract_production.py", "--crawl-database", outputs["crawl_database"],
        "--archive-directory", outputs["body_archive"], "--output-directory", outputs["processed_parts"],
        "--checkpoint", str(Path(outputs["root"]) / "extract.sqlite"),
        "--summary-out", str(Path(outputs["artifacts"]) / "extract.json"),
    ])
    run([
        python, "scripts/merge_document_partitions.py", "--input-directory", outputs["processed_parts"],
        "--output", depth["new_documents"],
        "--summary-out", str(Path(outputs["artifacts"]) / "new_document_merge.json"),
    ])

    stage(state, "CHUNK")
    run([
        python, "scripts/chunk_production.py", "--documents", depth["new_documents"],
        "--output-directory", outputs["chunks_parts"],
        "--checkpoint", str(Path(outputs["root"]) / "chunk.sqlite"),
        "--summary-out", str(Path(outputs["artifacts"]) / "chunk.json"),
    ])
    run([
        python, "scripts/merge_chunk_partitions.py", "--input-directory", outputs["chunks_parts"],
        "--output", depth["new_chunks"],
        "--summary-out", str(Path(outputs["artifacts"]) / "new_chunk_merge.json"),
    ])

    stage(state, "ASSEMBLE")
    assembly = assemble_depth_corpus(config)

    stage(state, "EMBEDDINGS", total=assembly["chunks"], done=0)
    embedding = assemble_depth_embeddings(config, state)

    stage(state, "SOURCE_CANDIDATES", completed_sources=0, total_sources=9)
    candidate_summary = build_source_candidates(config, run_state=state, groups=DEPTH_GROUPS)
    seed = seed_depth_scores(config)

    stage(state, "RERANK", **score_database_status(config), cached_pairs_reused=seed["reused"])
    restarts = 0
    model_loads = 0
    while True:
        status = score_database_status(config)
        if int(status["remaining"]) == 0:
            break
        model_loads += 1
        completed = subprocess.run([
            python, "tools/score_source_census_persistent.py", "--config", args.config,
            "--window-size", "2000", "--run-state", args.run_state,
            "--metrics-out", str(Path(outputs["artifacts"]) / "rerank_metrics.json"),
            "--model-load-count", str(model_loads), "--scorer-restarts", str(restarts),
        ], cwd=ROOT)
        if completed.returncode != 0:
            restarts += 1
            if restarts > 10:
                raise RuntimeError(f"depth1000 reranker exceeded restart limit: {completed.returncode}")
    integrity = score_database_status(config, verify_integrity=True)
    if integrity["integrity"] != "ok" or integrity["remaining"] != 0:
        raise RuntimeError(f"depth1000 reranker cache invalid: {integrity}")

    stage(state, "RANKINGS")
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

    stage(state, "PACKAGE", completed=0, total=3)
    submissions = {}
    for index, group in enumerate(DEPTH_GROUPS, 1):
        marker = Path(outputs["artifacts"]) / f"submission_{group}.json"
        run([
            python, "tools/source_census_cached_subset.py", "--config", args.config,
            "package-existing", "--ranking-root", str(ranking_root), "--group", group,
            "--submission-name", f"phase10d_DEPTH1000_{group}",
            "--output-dir", outputs["submissions"], "--marker", str(marker),
        ])
        submissions[group] = json.loads(marker.read_text(encoding="utf-8"))
        stage(state, "PACKAGE", completed=index, total=3, current_group=group)

    report = finalize_depth_report(config, submissions)
    atomic_json(state, {
        "stage": "WAITING_FOR_LEADERBOARD", "status": "WAITING_FOR_LEADERBOARD",
        "report": str(Path(outputs["artifacts"]).parent / "depth1000_report.json"),
        "candidate_summary": candidate_summary, "embedding_summary": embedding,
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
