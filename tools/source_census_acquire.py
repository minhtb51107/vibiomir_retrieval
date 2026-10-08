#!/usr/bin/env python3
"""Bounded cross-domain acquisition scheduler for Phase 10D."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_census.pipeline import atomic_json, load_config


def slug(domain: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", domain.lower()).strip("_")


def completed_ids(database: Path) -> set[int]:
    if not database.exists():
        return set()
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=2)
    try:
        return {int(row[0]) for row in connection.execute("SELECT doc_id FROM crawl_results")}
    finally:
        connection.close()


def database_count(database: Path) -> int:
    if not database.exists():
        return 0
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=.5)
        value = int(connection.execute("SELECT COUNT(*) FROM crawl_results").fetchone()[0])
        connection.close()
        return value
    except sqlite3.Error:
        return 0


def update_state(path: Path, value: dict[str, Any]) -> None:
    atomic_json(path, {"stage": "ACQUIRE", "status": "RUNNING", **value})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/source_census.yaml")
    parser.add_argument("--run-state", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    outputs = config["outputs"]
    manifest = json.loads((Path(outputs["artifacts"]) / "round1_sample_manifest.json").read_text(encoding="utf-8"))
    total = int(manifest["scheduled_documents"])
    serial_ids = completed_ids(Path(outputs["serial_crawl_database"]))
    worker_root = Path(outputs["worker_root"]); worker_root.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for domain, info in sorted(manifest["domains"].items()):
        records = [row for row in info["records"] if int(row["doc_id"]) not in serial_ids]
        root = worker_root / slug(domain); root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "manifest.json"
        atomic_json(manifest_path, {"domain": domain, "records": records})
        jobs.append({"domain": domain, "root": root, "manifest": manifest_path, "selected": len(info["records"]), "remaining_manifest": len(records)})
    pending = [job for job in jobs if job["remaining_manifest"] and database_count(job["root"] / "crawl.sqlite") < job["remaining_manifest"]]
    completed_sources = {job["domain"] for job in jobs if not job["remaining_manifest"] or database_count(job["root"] / "crawl.sqlite") >= job["remaining_manifest"]}
    failed: dict[str, int] = {}
    retry_counts: dict[str, int] = {}
    active: dict[str, tuple[subprocess.Popen[Any], Any, dict[str, Any]]] = {}
    started = time.monotonic()
    logs = Path("artifacts/runs/phase10d_round1/logs/sources"); logs.mkdir(parents=True, exist_ok=True)
    max_workers = min(8, int(config["acquisition"]["global_concurrency"]))
    python = sys.executable

    def launch(job: dict[str, Any]) -> None:
        domain = str(job["domain"]); root = Path(job["root"])
        log = (logs / f"{slug(domain)}.log").open("a", encoding="utf-8")
        command = [python, "scripts/crawl_corpus.py", "--config", config["acquisition"]["crawler_config"], "--ids-file", str(job["manifest"]), "--full", "--output-db", str(root / "crawl.sqlite"), "--summary-out", str(root / "summary.json"), "--body-archive-dir", str(root / "bodies"), "--archive-documents-per-shard", str(config["acquisition"]["body_documents_per_shard"])]
        active[domain] = (subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log, job)

    while pending or active:
        while pending and len(active) < max_workers:
            launch(pending.pop(0))
        time.sleep(2)
        for domain, (process, log, job) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close(); del active[domain]
            if code == 0:
                completed_sources.add(domain)
            else:
                retry_counts[domain] = retry_counts.get(domain, 0) + 1
                if retry_counts[domain] <= 2:
                    pending.append(job)
                else:
                    failed[domain] = int(code)
        worker_done = sum(database_count(Path(job["root"]) / "crawl.sqlite") for job in jobs)
        done = min(total, len(serial_ids) + worker_done)
        elapsed_minutes = max((time.monotonic() - started) / 60.0, 1e-6)
        rate = max(0.0, worker_done / elapsed_minutes)
        eta = (total - done) / rate if rate > 0 else None
        update_state(Path(args.run_state), {
            "selected": total, "completed": done,
            "completed_official_ids": done,
            "serial_reused_ids": len(serial_ids),
            "active_sources": sorted(active),
            "active_source_count": len(active),
            "completed_sources": len(completed_sources),
            "total_sources": len(jobs),
            "failed_sources": failed,
            "retry_history": [
                {"source": domain, "retries": count}
                for domain, count in sorted(retry_counts.items())
            ],
            "urls_per_minute": round(rate, 3),
            "eta_minutes": round(eta, 2) if eta is not None else None,
            "free_disk_gib": round(shutil.disk_usage(ROOT).free / 2**30, 3),
            "last_successful_checkpoint": done,
        })
        if shutil.disk_usage(ROOT).free / 2**30 < float(config["acquisition"]["minimum_free_gib"]):
            for process, _, _ in active.values(): process.terminate()
            raise RuntimeError("20 GiB source-census disk floor reached")
    final_done = len(serial_ids) + sum(database_count(Path(job["root"]) / "crawl.sqlite") for job in jobs)
    atomic_json(Path(outputs["artifacts"]) / "round1_parallel_acquisition.json", {"selected": total, "completed": final_done, "remaining_after_bounded_workers": total-final_done, "serial_reused": len(serial_ids), "worker_completed": final_done-len(serial_ids), "completed_sources": len(completed_sources), "failed_sources": failed, "retry_counts": retry_counts, "maximum_concurrent_source_workers": max_workers})
    return 0


if __name__ == "__main__": raise SystemExit(main())
