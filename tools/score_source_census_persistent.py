#!/usr/bin/env python3
"""Persistent, bounded-memory GPU scorer for Phase 10D source pairs."""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_census.pipeline import atomic_json, load_config
from src.reranking.reranker import TransformerCrossEncoderReranker
from src.reranking.equivalence import validate_score_equivalence


def process_memory() -> dict[str, float]:
    """Return Windows working-set/private memory without adding a dependency."""
    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = ctypes.c_void_p
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    get_memory.restype = ctypes.c_int
    ok = get_memory(get_process(), ctypes.byref(counters), ctypes.sizeof(counters))
    if not ok:
        return {"rss_mib": -1.0, "private_mib": -1.0}
    return {
        "rss_mib": round(counters.WorkingSetSize / 2**20, 2),
        "private_mib": round(counters.PrivateUsage / 2**20, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/source_census.yaml")
    parser.add_argument("--window-size", type=int, default=2000)
    parser.add_argument("--max-windows", type=int, default=0,
                        help="0 scores until no missing pairs remain")
    parser.add_argument("--run-state")
    parser.add_argument("--metrics-out")
    parser.add_argument("--model-load-count", type=int, default=1)
    parser.add_argument("--scorer-restarts", type=int, default=0)
    parser.add_argument("--equivalence-sample", type=int, default=8)
    parser.add_argument("--gate-only", action="store_true", help="validate scorer equivalence without scoring missing pairs")
    args = parser.parse_args()
    if not 1 <= args.window_size <= 2000:
        raise ValueError("persistent checkpoint window must be 1..2000")

    import torch

    config = load_config(args.config)
    model_config = config["models"]["reranker"]
    database = Path(config["outputs"]["root"]) / "round1/source_scores.sqlite"
    connection = sqlite3.connect(database, timeout=60)
    connection.execute("PRAGMA busy_timeout=60000")
    total = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
    initial_done = int(connection.execute(
        "SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL"
    ).fetchone()[0])
    process_started = time.perf_counter()
    load_started = time.perf_counter()
    reranker = TransformerCrossEncoderReranker(
        model_name=model_config["name"], revision=model_config["revision"],
        cache_dir=model_config["cache_dir"], device="cuda", batch_size=2,
        max_length=512, use_half_on_cuda=True, seed=2026,
    )
    reranker.score_pairs([("warmup", "warmup")])
    model_load_seconds = time.perf_counter() - load_started

    equivalence = {"checked": 0, "max_absolute_difference": 0.0, "passed": True}
    if args.equivalence_sample:
        control=json.loads(Path(model_config["equivalence"]["control_artifact"]).read_text(encoding="utf-8"))
        control_pairs=control["pairs"][:args.equivalence_sample]
        sample=[]
        for item in control_pairs:
            row=connection.execute(
                "SELECT query_id,chunk_id,query_text,chunk_text,rerank_score FROM pairs "
                "WHERE query_id=? AND chunk_id=?",
                (int(item["query_id"]),str(item["chunk_id"])),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"reranker control key missing from current cache: {item['query_id']}:{item['chunk_id']}")
            sample.append(row)
        observed = reranker.score_pairs([(str(r[2]), str(r[3])) for r in sample])
        rule=model_config.get("equivalence")
        if not rule:
            raise RuntimeError("reranker equivalence contract is not configured")
        investigation=json.loads(Path(rule["investigation_artifact"]).read_text(encoding="utf-8"))
        if control.get("contract") != investigation.get("contract"):
            raise RuntimeError("per-experiment reranker control contract differs from the validated investigation contract")
        sample_keys=[f"{int(row[0])}:{row[1]}" for row in sample]
        expected_keys=[f"{int(row['query_id'])}:{row['chunk_id']}" for row in control_pairs]
        observed_text_hashes=[
            (hashlib.sha256(str(row[2]).encode("utf-8")).hexdigest(),hashlib.sha256(str(row[3]).encode("utf-8")).hexdigest())
            for row in sample
        ]
        expected_text_hashes=[(str(row["query_text_sha256"]),str(row["chunk_text_sha256"])) for row in control_pairs]
        actual_contract={
            "model":model_config["name"],"revision":model_config["revision"],
            "tokenizer_revision":model_config["revision"],"batch_size":2,"max_length":512,
            "precision":"float16","eval_mode":not reranker.model.training,"padding":True,
            "truncation":"longest_first","score_extraction":"logits.reshape(-1).float().cpu().numpy()",
            "input_order":"query, chunk",
        }
        equivalence=validate_score_equivalence(
            [float(row["reference_score"]) for row in control_pairs],observed.tolist(),sample_keys,
            measured_absolute_tolerance=float(rule["measured_absolute_tolerance"]),
            expected_contract=investigation["contract"],actual_contract=actual_contract,
            expected_keys=expected_keys,expected_text_hashes=expected_text_hashes,
            observed_text_hashes=observed_text_hashes,
        )
        equivalence["stored_reference_scores_match_control"]=all(
            float(row[4])==float(item["reference_score"])
            for row,item in zip(sample,control_pairs,strict=True)
        )
        equivalence["passed"]=bool(equivalence["passed"] and equivalence["stored_reference_scores_match_control"])
        del sample, observed
        if not equivalence["passed"]:
            raise RuntimeError(f"reranker score equivalence failed: {equivalence}")
    if args.gate_only:
        connection.close()
        print(json.dumps({"equivalence": equivalence, "production_pairs_scored": 0}))
        return 0

    windows: list[dict[str, object]] = []
    cumulative_scored = 0
    window_number = 0
    while not args.max_windows or window_number < args.max_windows:
        rows = connection.execute(
            "SELECT query_id,chunk_id,query_text,chunk_text FROM pairs "
            "WHERE rerank_score IS NULL ORDER BY query_id,chunk_id LIMIT ?",
            (args.window_size,),
        ).fetchall()
        if not rows:
            break
        window_number += 1
        window_started = time.perf_counter()
        active_seconds = 0.0
        checkpoint_seconds = 0.0
        scored = 0
        for offset in range(0, len(rows), 64):
            batch = rows[offset:offset + 64]
            pairs = [(str(row[2]), str(row[3])) for row in batch]
            active_started = time.perf_counter()
            values = reranker.score_pairs(pairs)
            batch_active = time.perf_counter() - active_started
            checkpoint_started = time.perf_counter()
            with connection:
                connection.executemany(
                    "UPDATE pairs SET rerank_score=?,inference_ms=? "
                    "WHERE query_id=? AND chunk_id=? AND rerank_score IS NULL",
                    [(float(score), batch_active * 1000 / len(batch), int(row[0]), str(row[1]))
                     for row, score in zip(batch, values, strict=True)],
                )
            checkpoint_seconds += time.perf_counter() - checkpoint_started
            active_seconds += batch_active
            scored += len(batch)
            del batch, pairs, values

        cumulative_scored += scored
        done = initial_done + cumulative_scored
        del rows
        gc.collect()
        # The model stays resident; releasing allocator cache prevents window
        # intermediates from accumulating across hundreds of checkpoints.
        torch.cuda.empty_cache()
        memory = process_memory()
        gpu_allocated = round(torch.cuda.memory_allocated() / 2**20, 2)
        gpu_reserved = round(torch.cuda.memory_reserved() / 2**20, 2)
        window_wall = time.perf_counter() - window_started
        process_wall = time.perf_counter() - process_started
        active_rate = scored / max(active_seconds, 1e-9)
        effective_rate = cumulative_scored / max(process_wall, 1e-9)
        metric: dict[str, object] = {
            "window": window_number, "scored": scored,
            "active_seconds": round(active_seconds, 6),
            "checkpoint_seconds": round(checkpoint_seconds, 6),
            "wall_seconds": round(window_wall, 6),
            "active_pairs_per_second": round(active_rate, 3),
            "rss_mib": memory["rss_mib"], "private_mib": memory["private_mib"],
            "gpu_allocated_mib": gpu_allocated, "gpu_reserved_mib": gpu_reserved,
            "done": done,
        }
        windows.append(metric)
        result = {
            "model_load_seconds": round(model_load_seconds, 6),
            "model_load_count": args.model_load_count,
            "scorer_restarts": args.scorer_restarts,
            "window_size": args.window_size,
            "windows_completed": window_number,
            "pairs_scored": cumulative_scored,
            "initial_done": initial_done,
            "total": total,
            "done": done,
            "remaining": total - done,
            "effective_pairs_per_second": round(effective_rate, 3),
            "equivalence": equivalence,
            "windows": windows,
        }
        if args.metrics_out:
            atomic_json(args.metrics_out, result)
        if args.run_state:
            atomic_json(args.run_state, {
                "stage": "RERANK", "status": "RUNNING", "total": total,
                "done": done, "remaining": total - done,
                "integrity": "transactional-ok", "shard_size": args.window_size,
                "current_checkpoint_window": window_number,
                "model_load_seconds": round(model_load_seconds, 3),
                "model_load_count": args.model_load_count,
                "scorer_restarts": args.scorer_restarts,
                "active_pairs_per_second": round(active_rate, 3),
                "effective_pairs_per_second": round(effective_rate, 3),
                "rss_mib": memory["rss_mib"], "private_mib": memory["private_mib"],
                "gpu_allocated_mib": gpu_allocated, "gpu_reserved_mib": gpu_reserved,
                "checkpoint_seconds": round(checkpoint_seconds, 3),
                "eta_seconds": round((total - done) / max(effective_rate, 1e-9), 2),
                "last_successful_checkpoint": done,
                "equivalence": equivalence,
            })

    connection.close()
    final = {
        "model_load_seconds": round(model_load_seconds, 6),
        "model_load_count": args.model_load_count,
        "scorer_restarts": args.scorer_restarts,
        "window_size": args.window_size,
        "windows_completed": window_number,
        "pairs_scored": cumulative_scored,
        "initial_done": initial_done,
        "total": total,
        "done": initial_done + cumulative_scored,
        "remaining": total - initial_done - cumulative_scored,
        "effective_pairs_per_second": round(
            cumulative_scored / max(time.perf_counter() - process_started, 1e-9), 3
        ),
        "equivalence": equivalence,
        "windows": windows,
    }
    if args.metrics_out:
        atomic_json(args.metrics_out, final)
    print(json.dumps(final))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
