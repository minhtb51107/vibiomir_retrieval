#!/usr/bin/env python3
"""Launch independent, bounded GPU scorer processes until Phase 10B3 is complete."""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import yaml


def _state(database: Path) -> tuple[int, int, str]:
    connection = sqlite3.connect(database)
    total = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
    remaining = int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NULL").fetchone()[0])
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    connection.close()
    return total, remaining, integrity


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fresh 5,000-pair Phase 10B3 scorer shards.")
    parser.add_argument("--config", default="configs/source_relevance_probe.yaml")
    parser.add_argument("--shard-size", type=int, default=5000)
    args = parser.parse_args()
    if args.shard_size != 5000:
        raise ValueError("Phase 10B3 production shard size is fixed at 5,000")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    database = Path(config["outputs"]["work_directory"]) / "novel_pairs.sqlite"
    artifacts = Path(config["outputs"]["artifact_directory"])
    total, initial_remaining, integrity = _state(database)
    if integrity != "ok":
        raise RuntimeError("global scoring checkpoint failed integrity before sharding")
    expected_shards = math.ceil(initial_remaining / args.shard_size)
    runs = []
    environment = dict(os.environ)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    for shard in range(1, expected_shards + 1):
        _, before, integrity = _state(database)
        if before == 0: break
        if integrity != "ok": raise RuntimeError(f"checkpoint corrupt before shard {shard}")
        target = min(args.shard_size, before)
        label = f"shard_{shard:02d}"
        command = [
            sys.executable, "scripts/score_source_probe_pairs.py", "--config", args.config,
            "--max-new", str(target), "--report-every", "256", "--run-label", label,
        ]
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"fresh scorer {label} exited {completed.returncode}")
        _, after, integrity = _state(database)
        if integrity != "ok" or before - after != target:
            raise RuntimeError(f"checkpoint verification failed after {label}")
        result = json.loads((artifacts / f"fresh_scoring_{label}.json").read_text(encoding="utf-8"))
        runs.append({**result, "shard": shard, "checkpoint_remaining_after": after})
    _, final_remaining, final_integrity = _state(database)
    summary = {
        "total_unique_pairs": total, "initial_remaining": initial_remaining,
        "shard_size": args.shard_size, "shard_count": len(runs),
        "pairs_scored": initial_remaining - final_remaining,
        "final_remaining": final_remaining, "integrity_check": final_integrity,
        "total_model_load_seconds": round(sum(float(row["model_load_seconds"]) for row in runs), 6),
        "total_inference_seconds": round(sum(float(row["inference_seconds"]) for row in runs), 6),
        "weighted_pairs_per_second": round(
            sum(int(row["pairs_scored"]) for row in runs)
            / max(sum(float(row["inference_seconds"]) for row in runs), 1e-9), 6
        ),
        "runs": runs,
    }
    temporary = artifacts / "shard_scoring_summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary, artifacts / "shard_scoring_summary.json")
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}, indent=2))
    if final_remaining or final_integrity != "ok": return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
