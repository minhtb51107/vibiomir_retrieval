#!/usr/bin/env python3
"""Idempotent local worker for Phase 10D Round 1.

The expensive acquisition/extraction/chunking stages are delegated to the
existing production CLIs. Later cache/rerank/package stages deliberately fail
safe unless their concrete outputs are present; the supervisor records an
incident rather than fabricating a completed census.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_census.pipeline import (
    atomic_json, balanced_groups, build_source_candidates, finalize_round1,
    load_config, merge_acquisition, score_database_status, technical_triage,
)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def stage_state(path: Path, stage: str, status: str, **extra: object) -> None:
    atomic_json(path, {"stage": stage, "status": status, **extra})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/source_census.yaml")
    parser.add_argument("--run-state", required=True)
    parser.add_argument(
        "--resume-from", choices=["full", "source_candidates", "rerank", "package"], default="full"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    outputs = config["outputs"]
    python = sys.executable
    state = Path(args.run_state)
    Path(outputs["artifacts"]).mkdir(parents=True, exist_ok=True)
    if args.resume_from == "package":
        required = [
            Path(outputs["root"]) / "round1/source_candidates.parquet",
            Path(outputs["root"]) / "round1/source_scores.sqlite",
            Path(outputs["artifacts"]) / "source_candidate_summary.json",
            Path(outputs["artifacts"]) / "round1_groups.json",
            Path(outputs["artifacts"]) / "technical_triage.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"cannot resume PACKAGE; missing: {missing}")
        rerank = score_database_status(config, verify_integrity=True)
        if rerank != {"total": 582000, "done": 582000, "remaining": 0, "integrity": "ok"}:
            raise RuntimeError(f"cannot package incomplete/invalid rerank cache: {rerank}")
        candidate_summary = json.loads(required[2].read_text(encoding="utf-8"))
        stage_state(state, "PACKAGE", "RUNNING", candidate_summary=candidate_summary,
                    reranker_cache=rerank, resumed_from="durable_package_boundary")
        report = finalize_round1(config)
        stage_state(state, "WAITING_FOR_LEADERBOARD", "WAITING_FOR_LEADERBOARD",
                    report=str(Path(outputs["artifacts"]) / "round1_report.json"))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.resume_from == "full":
        stage_state(state, "ACQUIRE", "RUNNING")
        run([python, "tools/source_census_acquire.py", "--config", args.config, "--run-state", args.run_state])

        stage_state(state, "MERGE_ACQUISITION", "RUNNING")
        merge_acquisition(config)

        stage_state(state, "EXTRACT", "RUNNING")
        extract_summary = str(Path(outputs["artifacts"]) / "round1_extract.json")
        run([python, "scripts/extract_production.py", "--crawl-database", outputs["crawl_database"],
             "--archive-directory", outputs["body_archive"], "--output-directory", outputs["processed_parts"],
             "--checkpoint", str(Path(outputs["root"]) / "round1/extract.sqlite"), "--summary-out", extract_summary])
        run([python, "scripts/merge_document_partitions.py", "--input-directory", outputs["processed_parts"],
             "--output", outputs["documents"], "--summary-out", str(Path(outputs["artifacts"]) / "round1_document_merge.json")])

        stage_state(state, "CHUNK", "RUNNING")
        run([python, "scripts/chunk_production.py", "--documents", outputs["documents"],
             "--output-directory", outputs["chunks_parts"], "--checkpoint", str(Path(outputs["root"]) / "round1/chunk.sqlite"),
             "--summary-out", str(Path(outputs["artifacts"]) / "round1_chunk.json")])
        run([python, "scripts/merge_chunk_partitions.py", "--input-directory", outputs["chunks_parts"],
             "--output", outputs["chunks"], "--summary-out", str(Path(outputs["artifacts"]) / "round1_chunk_merge.json")])

        stage_state(state, "TRIAGE", "RUNNING")
        triage = technical_triage(config)
        groups = balanced_groups(config, triage)
        atomic_json(Path(outputs["artifacts"]) / "round1_groups.json", groups)
    elif args.resume_from == "source_candidates":
        required = [
            Path(outputs["documents"]), Path(outputs["chunks"]),
            Path(outputs["root"]) / "round1/chunk_embeddings.f32",
            Path(outputs["artifacts"]) / "technical_triage.json",
            Path(outputs["artifacts"]) / "round1_groups.json",
            Path(outputs["artifacts"]) / "candidate_cap_calibration.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"cannot resume SOURCE_CANDIDATES; missing: {missing}")
        triage = json.loads((Path(outputs["artifacts"]) / "technical_triage.json").read_text(encoding="utf-8"))
        groups = json.loads((Path(outputs["artifacts"]) / "round1_groups.json").read_text(encoding="utf-8"))
    else:
        required = [
            Path(outputs["root"]) / "round1/source_candidates.parquet",
            Path(outputs["root"]) / "round1/source_scores.sqlite",
            Path(outputs["artifacts"]) / "source_candidate_summary.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"cannot resume RERANK; missing: {missing}")
        candidate_summary = json.loads(required[2].read_text(encoding="utf-8"))

    if args.resume_from != "rerank":
        stage_state(state, "SOURCE_CANDIDATES", "RUNNING", healthy_sources=len(triage["healthy_sources"]), groups=groups)
        candidate_summary = build_source_candidates(
            config, run_state=state, groups=groups
        )

    stage_state(state, "RERANK", "RUNNING", **score_database_status(config))
    window_size = int(config["models"]["reranker"]["shard_size"])
    retry_history: list[dict[str, int]] = []
    scorer_restarts = 0
    model_load_count = 0
    while True:
        status = score_database_status(config)
        if int(status["remaining"]) == 0:
            break
        model_load_count += 1
        completed = subprocess.run(
            [python, "tools/score_source_census_persistent.py", "--config", args.config,
             "--window-size", str(window_size), "--run-state", args.run_state,
             "--metrics-out", str(Path(outputs["artifacts"]) / "persistent_rerank_metrics.json"),
             "--model-load-count", str(model_load_count),
             "--scorer-restarts", str(scorer_restarts)],
            cwd=ROOT,
        )
        if completed.returncode != 0:
            retry_history.append({"window_size": window_size, "exit_code": completed.returncode})
            scorer_restarts += 1
            if scorer_restarts <= 10:
                continue
            raise RuntimeError(f"persistent reranker exceeded restart limit; exit={completed.returncode}")
        stage_state(state, "RERANK", "RUNNING", shard_size=window_size,
                    model_load_count=model_load_count, scorer_restarts=scorer_restarts,
                    retry_history=retry_history, **score_database_status(config))

    final_integrity = score_database_status(config, verify_integrity=True)
    if final_integrity["integrity"] != "ok":
        raise RuntimeError("source-score checkpoint integrity failure")

    stage_state(state, "PACKAGE", "RUNNING", candidate_summary=candidate_summary)
    report = finalize_round1(config)
    stage_state(state, "WAITING_FOR_LEADERBOARD", "WAITING_FOR_LEADERBOARD", report=str(Path(outputs["artifacts"]) / "round1_report.json"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
