#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def database_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1)
        value = int(connection.execute("SELECT COUNT(*) FROM crawl_results").fetchone()[0])
        connection.close()
        return value
    except sqlite3.Error:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/source_census_depth1000.yaml")
    parser.add_argument("--run-state", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    outputs = config["outputs"]
    manifest = json.loads(Path(config["depth1000"]["manifest"]).read_text(encoding="utf-8"))
    total = int(manifest["total_incremental_ids"])
    worker_root = Path(outputs["worker_root"]); worker_root.mkdir(parents=True, exist_ok=True)
    logs = Path(args.run_state).parent / "logs/sources"; logs.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for domain, info in sorted(manifest["sources"].items()):
        records = list(info["incremental_records"])
        if not records:
            continue
        root = worker_root / slug(domain); root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "manifest.json"
        atomic_json(manifest_path, {"domain": domain, "records": records})
        jobs.append({"domain": domain, "root": root, "manifest": manifest_path, "selected": len(records)})

    pending = [job for job in jobs if database_count(Path(job["root"]) / "crawl.sqlite") < int(job["selected"])]
    completed_sources = {
        str(job["domain"]) for job in jobs
        if database_count(Path(job["root"]) / "crawl.sqlite") >= int(job["selected"])
    }
    failed: dict[str, int] = {}
    retries: dict[str, int] = {}
    active: dict[str, tuple[subprocess.Popen[Any], Any, dict[str, Any]]] = {}
    started = time.monotonic()
    python = sys.executable
    max_workers = min(8, len(jobs), int(config["acquisition"]["global_concurrency"]))

    def launch(job: dict[str, Any]) -> None:
        domain = str(job["domain"]); root = Path(job["root"])
        log = (logs / f"{slug(domain)}.log").open("a", encoding="utf-8")
        command = [
            python, "scripts/crawl_corpus.py", "--config", config["acquisition"]["crawler_config"],
            "--ids-file", str(job["manifest"]), "--full", "--output-db", str(root / "crawl.sqlite"),
            "--summary-out", str(root / "summary.json"), "--body-archive-dir", str(root / "bodies"),
            "--archive-documents-per-shard", str(config["acquisition"]["body_documents_per_shard"]),
        ]
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
                retries[domain] = retries.get(domain, 0) + 1
                if retries[domain] <= 2:
                    pending.append(job)
                else:
                    failed[domain] = int(code)
        done = sum(database_count(Path(job["root"]) / "crawl.sqlite") for job in jobs)
        elapsed_minutes = max((time.monotonic() - started) / 60, 1e-9)
        rate = done / elapsed_minutes
        eta = (total - done) / rate if rate else None
        atomic_json(args.run_state, {
            "stage": "ACQUIRE", "status": "RUNNING", "selected": total, "completed": done,
            "completed_official_ids": done, "remaining": total - done,
            "active_sources": sorted(active), "active_source_count": len(active),
            "completed_sources": len(completed_sources), "total_sources": len(jobs),
            "failed_sources": failed, "retry_history": retries,
            "urls_per_minute": round(rate, 3), "eta_minutes": round(eta, 2) if eta else None,
            "free_disk_gib": round(shutil.disk_usage(ROOT).free / 2**30, 3),
            "last_successful_checkpoint": done,
        })
        if shutil.disk_usage(ROOT).free / 2**30 < float(config["acquisition"]["minimum_free_gib"]):
            for process, _, _ in active.values():
                process.terminate()
            raise RuntimeError("20 GiB depth1000 disk floor reached")

    final_done = sum(database_count(Path(job["root"]) / "crawl.sqlite") for job in jobs)
    result = {
        "selected": total, "completed": final_done, "remaining": total - final_done,
        "completed_sources": len(completed_sources), "failed_sources": failed,
        "retry_counts": retries, "maximum_concurrent_source_workers": max_workers,
    }
    atomic_json(Path(outputs["artifacts"]) / "parallel_acquisition.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
