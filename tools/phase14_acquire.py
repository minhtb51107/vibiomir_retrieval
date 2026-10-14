#!/usr/bin/env python3
"""Dedicated, resumable acquisition adapter for Phase 14 Medlatec coverage."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_census.phase14 import (
    acquisition_manifest,
    acquisition_progress,
    crawl_checkpoint_audit,
    load_acquisition_config,
)
from src.source_census.pipeline import atomic_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-state", required=True)
    args = parser.parse_args()
    config = load_acquisition_config(args.config)
    _, records, selection_sha = acquisition_manifest(config)
    total = len(records)
    output = config["outputs"]
    worker_root = Path(output["worker_root"]) / "medlatec_vn"
    worker_root.mkdir(parents=True, exist_ok=True)
    database = worker_root / "crawl.sqlite"
    body_archive = worker_root / "bodies"
    manifest_path = worker_root / "manifest.json"
    # The crawler accepts original_url directly and can therefore avoid a
    # corpus-wide lookup while retaining canonical IDs and URLs exactly.
    atomic_json(manifest_path, {"domain": "medlatec.vn", "selection_sha256": selection_sha, "records": records})
    before = crawl_checkpoint_audit(database, records)
    if not before["passed"]:
        raise RuntimeError(f"Phase 14 checkpoint audit failed before resume: {before}")
    run_dir = Path(args.run_state).parent
    source_log_path = run_dir / "logs" / "sources" / "medlatec_vn.log"
    source_log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "scripts/crawl_corpus.py",
        "--config", str(config["acquisition"]["crawler_config"]),
        "--ids-file", str(manifest_path), "--full",
        "--output-db", str(database),
        "--summary-out", str(worker_root / "summary.json"),
        "--body-archive-dir", str(body_archive),
        "--archive-documents-per-shard", str(config["acquisition"]["body_documents_per_shard"]),
    ]
    initial = int(before["rows"])
    started = time.monotonic()
    retries = 0
    while True:
        current = crawl_checkpoint_audit(database, records)
        if not current["passed"]:
            raise RuntimeError(f"Phase 14 checkpoint audit failed: {current}")
        if int(current["rows"]) >= total:
            break
        with source_log_path.open("a", encoding="utf-8", buffering=1) as source_log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=source_log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(2)
                current = crawl_checkpoint_audit(database, records)
                if not current["passed"]:
                    process.terminate()
                    raise RuntimeError(f"Phase 14 live checkpoint audit failed: {current}")
                elapsed_minutes = max((time.monotonic() - started) / 60, 1e-9)
                newly_committed = max(0, int(current["rows"]) - initial)
                rate = newly_committed / elapsed_minutes
                state = acquisition_progress(total, int(current["rows"]), rate=rate, database=database)
                state.update({
                    "run_id": config["run_id"], "selection_sha256": selection_sha,
                    "active_sources": ["medlatec.vn"], "active_source_count": 1,
                    "completed_sources": 0, "total_sources": 1,
                    "failed_sources": {}, "retry_history": {"medlatec.vn": retries},
                    "acquisition_child_pid": process.pid,
                    "free_disk_gib": round(shutil.disk_usage(ROOT).free / 2**30, 3),
                    "checkpoint_audit": current,
                })
                atomic_json(args.run_state, state)
                if shutil.disk_usage(ROOT).free / 2**30 < float(config["acquisition"]["minimum_free_gib"]):
                    process.terminate()
                    raise RuntimeError("20 GiB Phase 14 disk floor reached")
            exit_code = process.wait()
        if exit_code == 0:
            break
        retries += 1
        if retries > 2:
            raise RuntimeError(f"Phase 14 crawler exited {exit_code} after {retries} attempts")

    final = crawl_checkpoint_audit(database, records)
    if not final["passed"] or int(final["rows"]) != total:
        raise RuntimeError(f"Phase 14 acquisition closure failed: {final}")
    result = {
        "run_id": config["run_id"], "selection_sha256": selection_sha,
        "selected": total, "completed": final["rows"], "reused_checkpoint_rows": initial,
        "new_rows_this_resume": int(final["rows"]) - initial,
        "checkpoint_audit": final, "crawler_restarts": retries,
        "database": str(database), "body_archive": str(body_archive),
    }
    atomic_json(Path(output["artifacts"]) / "parallel_acquisition.json", result)
    terminal = acquisition_progress(total, total, rate=0.0, database=database)
    terminal.update({
        "run_id": config["run_id"], "status": "READY_FOR_NEXT_STAGE",
        "stage": "ACQUISITION_COMPLETE", "selection_sha256": selection_sha,
        "active_sources": [], "active_source_count": 0,
        "completed_sources": 1, "total_sources": 1,
        "checkpoint_audit": final,
    })
    atomic_json(args.run_state, terminal)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
