#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indexing.phase14_embeddings import (
    BATCH_SIZE, DIMENSION, MAX_LENGTH, WINDOW_ROWS, audit_complete,
    embedding_paths, load_phase14_embedding_config, prepare_checkpoint,
    resource_preflight, validate_vector_rows,
)
from src.runtime.state_file import update_json, write_json


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def process_memory() -> dict[str, float]:
    import psutil
    info = psutil.Process().memory_info()
    return {
        "rss_mib": round(info.rss / 2**20, 2), "private_mib": round(info.private / 2**20, 2),
        "page_faults": int(getattr(info, "num_page_faults", -1)),
    }


def gpu_memory(torch) -> dict[str, float]:
    return {
        "gpu_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 2),
        "gpu_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 2),
    }


def run(config: dict, state_path: Path, *, max_new_rows: int, allow_production: bool) -> int:
    paths = embedding_paths(config)
    checkpoint = prepare_checkpoint(config)
    if checkpoint.get("complete"):
        audit = audit_complete(config, checkpoint)
        update_json(state_path, {
            "status": "READY_FOR_NEXT_STAGE", "stage": "EMBEDDINGS_COMPLETE",
            "total": audit["rows"], "done": audit["rows"], "remaining": 0,
            "checkpoint": str(paths["checkpoint"]), "embedding_audit": str(paths["audit"]),
            "embedding_sha256": audit["embedding_sha256"], "next_stage": "CANDIDATE_RETRIEVAL",
        }, preserve_terminal=False)
        return 0
    preflight = resource_preflight(config)
    write_json(Path(config["outputs"]["artifacts"]) / "embedding_resource_preflight.json", preflight)
    if not preflight["disk_gate_passed"]:
        update_json(state_path, {"status": "FAILED_SAFE", "stage": "EMBEDDINGS_DISK_WAIT", **preflight}, preserve_terminal=False)
        return 0
    if not (preflight["memory_gate_passed"] and preflight["commit_gate_passed"] and preflight["gpu_gate_passed"]):
        update_json(state_path, {
            "status": "EMBEDDINGS_MEMORY_WAIT", "stage": "EMBEDDINGS_MEMORY_WAIT",
            "total": preflight["rows"], "done": checkpoint["completed_rows"],
            "remaining": preflight["rows"] - checkpoint["completed_rows"],
            "checkpoint": str(paths["checkpoint"]), "last_successful_checkpoint": str(paths["checkpoint"]),
            "next_stage_blocked": "Measured incremental model-load/first-batch resource gate did not pass.",
            "next_action": "Resume the same dedicated worker when its measured physical/commit/VRAM gates pass.",
            **preflight,
        }, preserve_terminal=False)
        return 0

    if max_new_rows <= 0 and not allow_production:
        raise ValueError("full embedding requires --allow-production after a successful 64-row trial")
    if max_new_rows <= 0 and not bool((checkpoint.get("trial_gate") or {}).get("passed")):
        raise ValueError("full embedding requires a durable successful 64-row trial gate")
    if max_new_rows > 256:
        raise ValueError("bounded embedding trial may not exceed 256 new rows")
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from sentence_transformers import SentenceTransformer
    measurements = [{"phase": "before_model_load", **process_memory(), **gpu_memory(torch)}]
    load_started = time.perf_counter()
    model = SentenceTransformer(
        config["models"]["embedder"]["name"], revision=config["models"]["embedder"]["revision"],
        device="cuda", model_kwargs={"torch_dtype": torch.float16},
    )
    model.max_seq_length = MAX_LENGTH
    measurements.append({"phase": "model_loaded", "seconds": time.perf_counter() - load_started, **process_memory(), **gpu_memory(torch)})
    rows = int(checkpoint["signature"]["rows"])
    vectors = np.memmap(paths["partial"], dtype=np.float32, mode="r+", shape=(rows, DIMENSION))
    status = np.memmap(paths["status"], dtype=np.uint8, mode="r+", shape=(rows,))
    initial = int(status.sum()); position = 0; started = time.perf_counter()
    target_new = max_new_rows if max_new_rows > 0 else rows - initial
    update_json(state_path, {
        "status": "RUNNING", "stage": "EMBEDDINGS", "total": rows, "done": initial,
        "remaining": rows - initial, "checkpoint": str(paths["checkpoint"]), **measurements[-1],
    }, preserve_terminal=False)
    for batch in pq.ParquetFile(config["outputs"]["chunks"]).iter_batches(batch_size=WINDOW_ROWS, columns=["normalized_text"]):
        raw = batch.column(0).to_pylist()
        indices = [position + offset for offset in range(len(raw)) if not status[position + offset]]
        indices = indices[:max(0, target_new - (int(status.sum()) - initial))]
        if indices:
            pending = [normalize_text(str(raw[index - position])) for index in indices]
            encoded = model.encode(
                pending, batch_size=BATCH_SIZE, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            )
            encoded = np.asarray(encoded, dtype=np.float32)
            vector_gate = validate_vector_rows(encoded)
            if not vector_gate["passed"]:
                raise RuntimeError(f"Phase 14 embedding vector gate failed: {vector_gate}")
            vectors[indices] = encoded
            vectors.flush()
            status[indices] = 1
            status.flush()
        position += len(raw)
        completed = int(status.sum())
        memory = {**process_memory(), **gpu_memory(torch)}
        checkpoint.update({
            "completed_rows": completed, "remaining_rows": rows - completed,
            "last_scanned_row": position, "last_memory": memory,
        })
        if max_new_rows == 64 and checkpoint.get("trial_gate") is None and completed - initial >= 64:
            checkpoint["trial_gate"] = {
                "passed": True, "completed_rows": completed, "new_rows": completed - initial, "requested_rows": 64,
                "batch_size": BATCH_SIZE, "window_rows": WINDOW_ROWS, "memory": memory,
            }
        write_json(paths["checkpoint"], checkpoint)
        elapsed = max(time.perf_counter() - started, 1e-9)
        update_json(state_path, {
            "status": "RUNNING", "stage": "EMBEDDINGS", "total": rows, "done": completed,
            "remaining": rows - completed, "rows_per_second": round(max(0, completed - initial) / elapsed, 3),
            "eta_seconds": round((rows - completed) / max((completed - initial) / elapsed, 1e-9), 1),
            "checkpoint": str(paths["checkpoint"]), "trial_gate": checkpoint.get("trial_gate"), **memory,
        }, preserve_terminal=False)
        if completed - initial >= target_new:
            break
    completed_now = int(status.sum())
    if max_new_rows > 0:
        vectors.flush(); status.flush(); del vectors, status, model
        torch.cuda.empty_cache(); gc.collect()
        checkpoint.update({
            "completed_rows": completed_now, "remaining_rows": rows - completed_now,
            "bounded_trial_complete": True, "bounded_trial_new_rows": completed_now - initial,
        })
        write_json(paths["checkpoint"], checkpoint)
        update_json(state_path, {
            "status": "READY_FOR_NEXT_STAGE", "stage": "EMBEDDING_TRIAL_COMPLETE",
            "total": rows, "done": completed_now, "remaining": rows - completed_now,
            "checkpoint": str(paths["checkpoint"]), "trial_gate": checkpoint.get("trial_gate"),
            "production_continuation_allowed": bool(checkpoint.get("trial_gate", {}).get("passed")),
            "next_action": "Run the same worker with --allow-production to resume only missing rows.",
        }, preserve_terminal=False)
        return 0
    if completed_now != rows:
        raise RuntimeError("Phase 14 embedding loop ended before bitmap completion")
    vectors.flush(); status.flush(); del vectors, status, model
    torch.cuda.empty_cache(); gc.collect()
    if paths["final"].exists():
        raise FileExistsError("refusing to overwrite existing Phase 14 final embeddings")
    os.replace(paths["partial"], paths["final"])
    checkpoint.update({"complete": True, "completed_rows": rows, "remaining_rows": 0, "runtime_seconds": time.perf_counter() - started})
    write_json(paths["checkpoint"], checkpoint)
    audit = audit_complete(config, checkpoint)
    if not audit["passed"]:
        raise RuntimeError(f"Phase 14 final embedding audit failed: {audit}")
    update_json(state_path, {
        "status": "READY_FOR_NEXT_STAGE", "stage": "EMBEDDINGS_COMPLETE", "total": rows,
        "done": rows, "remaining": 0, "checkpoint": str(paths["checkpoint"]),
        "last_successful_checkpoint": str(paths["checkpoint"]), "embedding_audit": str(paths["audit"]),
        "embedding_sha256": audit["embedding_sha256"], "next_stage": "CANDIDATE_RETRIEVAL",
    }, preserve_terminal=False)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Dedicated resumable Phase 14 local embedding adapter")
    parser.add_argument("--config", default="configs/phase14_medlatec_coverage_3000.yaml")
    parser.add_argument("--run-state", default="artifacts/runs/phase14_medlatec_coverage_3000/state.json")
    parser.add_argument("--max-new-rows", type=int, default=0)
    parser.add_argument("--allow-production", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = load_phase14_embedding_config(args.config)
    checkpoint = prepare_checkpoint(config)
    if args.preflight_only:
        preflight = resource_preflight(config)
        write_json(Path(config["outputs"]["artifacts"]) / "embedding_resource_preflight.json", preflight)
        if not (preflight["memory_gate_passed"] and preflight["commit_gate_passed"] and preflight["gpu_gate_passed"]):
            paths = embedding_paths(config)
            update_json(args.run_state, {
                "status": "EMBEDDINGS_MEMORY_WAIT", "stage": "EMBEDDINGS_MEMORY_WAIT",
                "total": preflight["rows"], "done": checkpoint["completed_rows"],
                "remaining": preflight["rows"] - checkpoint["completed_rows"],
                "checkpoint": str(paths["checkpoint"]), "last_successful_checkpoint": str(paths["checkpoint"]),
                "next_stage_blocked": "Measured incremental model-load/first-batch resource gate did not pass.",
                "next_action": "Resume the same dedicated worker when its measured physical/commit/VRAM gates pass.",
                **preflight,
            }, preserve_terminal=False)
        print(json.dumps({"checkpoint": checkpoint, "preflight": preflight}, ensure_ascii=False, indent=2))
        return 0
    return run(config, Path(args.run_state), max_new_rows=args.max_new_rows, allow_production=args.allow_production)


if __name__ == "__main__":
    raise SystemExit(main())
