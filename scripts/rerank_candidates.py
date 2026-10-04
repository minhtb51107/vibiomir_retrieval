#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.reranking.pipeline import (
    build_multilingual_sanity,
    compact_benchmark,
    finalize_reranking,
    score_pool,
    write_json,
)
from src.reranking.reranker import load_reranking_config


def _score(config: dict, pool_summary: dict, pool_path: Path, artifacts: Path) -> None:
    from src.reranking.reranker import TransformerCrossEncoderReranker

    model = config["model"]
    load_started = time.perf_counter()
    reranker = TransformerCrossEncoderReranker(
        model_name=model["name"],
        revision=model["revision"],
        cache_dir=model["cache_dir"],
        device=model["device"],
        batch_size=int(model["batch_size"]),
        max_length=int(model["max_sequence_length"]),
        use_half_on_cuda=model["precision"] == "float16",
        seed=int(model["seed"]),
    )
    # Initialize CUDA/cuBLAS before large Arrow/Parquet allocations on Windows.
    reranker.score_pairs([("warmup", "warmup")])
    if model["device"] == "cuda":
        reranker._torch.cuda.synchronize()
        reranker._torch.cuda.reset_peak_memory_stats()
    model_load_seconds = round(time.perf_counter() - load_started, 6)
    metadata = reranker.metadata()
    metadata["model_load_seconds"] = model_load_seconds
    if model["device"] == "cuda":
        metadata["gpu"] = reranker._torch.cuda.get_device_name(0)
    write_json(artifacts / "model_metadata.json", metadata)
    scoring = score_pool(
        config=config,
        reranker=reranker,
        pool_path=pool_path,
        pool_summary=pool_summary,
    )
    scoring["model_load_seconds"] = model_load_seconds
    log_path = artifacts / "scoring_runs.json"
    runs = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    runs.append(scoring)
    write_json(log_path, runs)
    print(json.dumps(scoring, indent=2), flush=True)
    if not scoring["complete"]:
        raise SystemExit("scoring incomplete; rerun to resume from checkpoint")


def _finalize(config: dict, pool_summary: dict, pool_path: Path, artifacts: Path) -> None:
    metadata_path = artifacts / "model_metadata.json"
    runs_path = artifacts / "scoring_runs.json"
    if not metadata_path.exists() or not runs_path.exists():
        raise FileNotFoundError("run scoring first: model_metadata.json/scoring_runs.json missing")
    model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    scoring_runs = json.loads(runs_path.read_text(encoding="utf-8"))
    result = finalize_reranking(
        config=config,
        pool_path=pool_path,
        pool_summary=pool_summary,
        model_metadata=model_metadata,
        scoring_runs=scoring_runs,
    )
    benchmark = compact_benchmark(result)
    write_json(artifacts / "benchmark.json", benchmark)
    write_json(
        artifacts / "rank_movement.json",
        {
            "label": "DESCRIPTIVE DIAGNOSTICS — not relevance evaluation",
            "depth_experiments": benchmark["depth_experiments"],
        },
    )
    write_json(
        artifacts / "source_composition.json",
        {
            "label": "DESCRIPTIVE DIAGNOSTICS — not relevance evaluation",
            "depth_experiments": [
                {
                    "depth": row["depth"],
                    "top10_source_membership_before": row["top10_source_membership_before"],
                    "top10_source_membership_after": row["top10_source_membership_after"],
                    "promoted_into_top10_by_membership": row[
                        "promoted_into_top10_by_membership"
                    ],
                    "top10_language_signals_before": row["top10_language_signals_before"],
                    "top10_language_signals_after": row["top10_language_signals_after"],
                    "promoted_into_top10_by_language_signal": row[
                        "promoted_into_top10_by_language_signal"
                    ],
                }
                for row in benchmark["depth_experiments"]
            ],
        },
    )
    write_json(
        artifacts / "duplicate_control.json",
        {
            "label": "CONTROL ACTIVATION COUNTS — not relevance evaluation",
            "control_summary": benchmark["control_summary"],
        },
    )
    sanity = build_multilingual_sanity(
        pool_path=pool_path,
        reranked_path=(
            Path(config["outputs"]["directory"])
            / f"reranked_chunks_depth{config['candidate_pool']['default_depth']}.parquet"
        ),
        sample_count=int(config["diagnostics"]["sanity_query_count"]),
    )
    write_json(artifacts / "multilingual_sanity.json", sanity)
    print(json.dumps(benchmark, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerank Phase 6 candidate pools.")
    parser.add_argument("--config", default="configs/reranking.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--score-only",
        action="store_true",
        help="GPU stage only: resume scoring into the checkpoint, then exit.",
    )
    mode.add_argument(
        "--finalize-only",
        action="store_true",
        help="CPU-only stage: build outputs from a complete checkpoint (no model/CUDA).",
    )
    args = parser.parse_args()
    config = load_reranking_config(args.config)
    artifacts = Path(config["outputs"]["artifacts"])
    pool_summary_path = artifacts / "candidate_pool.json"
    if not pool_summary_path.exists():
        raise FileNotFoundError(
            "candidate pool summary is missing; run scripts/prepare_rerank_pool.py first"
        )
    pool_summary = json.loads(pool_summary_path.read_text(encoding="utf-8"))
    pool_path = Path(pool_summary["path"])
    if not args.finalize_only:
        _score(config, pool_summary, pool_path, artifacts)
    if not args.score_only:
        if not args.finalize_only:
            # Fresh interpreter state is safer for memory; prefer separate runs.
            print("note: finalizing in the same process as scoring", flush=True)
        _finalize(config, pool_summary, pool_path, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
