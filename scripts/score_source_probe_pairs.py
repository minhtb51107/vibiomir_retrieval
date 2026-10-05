#!/usr/bin/env python3
"""Isolated GPU scorer for the Phase 10B3 deduplicated missing-pair queue."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _process_memory() -> dict[str, int]:
    counters = _PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = ctypes.c_void_p
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    get_memory.restype = ctypes.c_int
    if not get_memory(get_process(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return {
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        "private_bytes": int(counters.PrivateUsage),
        "pagefile_bytes": int(counters.PagefileUsage),
        "peak_pagefile_bytes": int(counters.PeakPagefileUsage),
        "page_fault_count": int(counters.PageFaultCount),
    }


def _physical_bytes() -> int:
    status = _MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return int(status.ullTotalPhys)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score only missing Phase 10B3 pairs in a fresh GPU process.")
    parser.add_argument("--config", default="configs/source_relevance_probe.yaml")
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--report-every", type=int, default=256)
    parser.add_argument("--run-label", default=None)
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model = config["models"]["reranker"]
    if int(model["batch_size"]) != 2:
        raise ValueError("only the proven batch size 2 is permitted")

    # Import/load the model before opening the work queue: no Arrow, FAISS,
    # candidate pool, embedding, or large dataframe is imported in this process.
    from src.reranking.reranker import TransformerCrossEncoderReranker
    load_started = time.perf_counter()
    reranker = TransformerCrossEncoderReranker(
        model_name=model["name"], revision=model["revision"], cache_dir=model["cache_dir"],
        device="cuda", batch_size=2, max_length=512, use_half_on_cuda=True,
        seed=int(model["seed"]),
    )
    reranker.score_pairs([("warmup", "warmup")])
    torch = reranker._torch
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    model_load_seconds = time.perf_counter() - load_started

    database = Path(config["outputs"]["work_directory"]) / "novel_pairs.sqlite"
    connection = sqlite3.connect(database)
    initial_remaining = int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NULL").fetchone()[0])
    target = min(initial_remaining, args.max_new if args.max_new is not None else initial_remaining)
    physical = _physical_bytes()
    initial_memory = _process_memory()
    peak_private = initial_memory["private_bytes"]
    peak_working_set = initial_memory["working_set_bytes"]
    scored = 0
    inference_seconds = 0.0
    stopped_reason = None
    telemetry = []
    last_report_scored = 0
    last_report_seconds = 0.0
    low_intervals = 0
    while scored < target:
        limit = min(64, target - scored)
        rows = connection.execute(
            "SELECT query_id,chunk_id,query_text,chunk_text FROM pairs "
            "WHERE rerank_score IS NULL ORDER BY query_id,chunk_id LIMIT ?", (limit,),
        ).fetchall()
        if not rows: break
        started = time.perf_counter()
        scores = reranker.score_pairs([(str(row[2]), str(row[3])) for row in rows])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        per_pair_ms = elapsed * 1000 / len(rows)
        with connection:
            connection.executemany(
                "UPDATE pairs SET rerank_score=?,inference_ms=? WHERE query_id=? AND chunk_id=? AND rerank_score IS NULL",
                [(float(score), per_pair_ms, int(row[0]), str(row[1])) for row, score in zip(rows, scores, strict=True)],
            )
        scored += len(rows); inference_seconds += elapsed
        memory = _process_memory()
        peak_private = max(peak_private, memory["private_bytes"])
        peak_working_set = max(peak_working_set, memory["working_set_bytes"])
        throughput = scored / max(inference_seconds, 1e-9)
        if scored - last_report_scored >= args.report_every or scored == target:
            interval_pairs = scored - last_report_scored
            interval_seconds = inference_seconds - last_report_seconds
            interval_rate = interval_pairs / max(interval_seconds, 1e-9)
            checkpoint_count = int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0])
            snapshot = {
                "pairs_scored": scored, "checkpoint_count": checkpoint_count,
                "cumulative_pairs_per_second": round(throughput, 6),
                "interval_pairs_per_second": round(interval_rate, 6),
                **memory,
                "gpu_allocated_bytes": int(torch.cuda.memory_allocated()),
                "gpu_reserved_bytes": int(torch.cuda.memory_reserved()),
            }
            telemetry.append(snapshot)
            print(
                f"scored {scored}/{target}; cumulative={throughput:.2f}/s; "
                f"interval={interval_rate:.2f}/s; working_set={memory['working_set_bytes']/1073741824:.2f} GiB; "
                f"private={memory['private_bytes']/1073741824:.2f} GiB; faults={memory['page_fault_count']}; "
                f"checkpoint={checkpoint_count}", flush=True,
            )
            if interval_rate < float(config["safety"]["minimum_scoring_pairs_per_second"]):
                low_intervals += 1
            else:
                low_intervals = 0
            if low_intervals >= 2:
                stopped_reason = "sustained_throughput_gate"
                break
            last_report_scored = scored
            last_report_seconds = inference_seconds
    final_remaining = int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NULL").fetchone()[0])
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    connection.close()
    result = {
        "mode": "benchmark" if args.benchmark else "production",
        "initial_remaining": initial_remaining, "target_pairs": target,
        "pairs_scored": scored, "final_remaining": final_remaining,
        "model_load_seconds": round(model_load_seconds, 6),
        "inference_seconds": round(inference_seconds, 6),
        "pairs_per_second": round(scored / max(inference_seconds, 1e-9), 6),
        "peak_private_bytes": peak_private, "peak_working_set_bytes": peak_working_set,
        "physical_memory_bytes": physical,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "telemetry": telemetry,
        "integrity_check": integrity, "stopped_reason": stopped_reason,
        "gate_passed": (
            stopped_reason is None and scored == target
            and (not args.benchmark or scored / max(inference_seconds, 1e-9) >= 15.0)
        ),
    }
    artifacts = Path(config["outputs"]["artifact_directory"])
    artifacts.mkdir(parents=True, exist_ok=True)
    label = args.run_label or ("sustained_benchmark" if args.benchmark else "production")
    name = f"fresh_scoring_{label}.json"
    temporary = artifacts / (name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temporary, artifacts / name)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "telemetry"},
            indent=2,
        ),
        flush=True,
    )
    reranker.release()
    if not result["gate_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
