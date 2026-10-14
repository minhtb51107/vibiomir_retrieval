#!/usr/bin/env python3
"""Phase 14 adapter: audit, extract and chunk the completed bounded crawl."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.state_file import update_json, write_json
from src.source_census.phase14 import (
    acquisition_content_audit,
    acquisition_manifest,
    chunk_quality_audit,
    document_quality_audit,
    load_acquisition_config,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_count(path: Path) -> tuple[int, str]:
    if not path.exists():
        return 0, "not_created"
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True, timeout=5)
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        count = int(connection.execute("SELECT COUNT(*) FROM completed_items").fetchone()[0])
        return count, integrity
    finally:
        connection.close()


def write_artifact(path: Path, value: dict) -> None:
    write_json(path, value)


def run_partition_stage(
    *, command: list[str], state_path: Path, stage: str, total: int, checkpoint: Path
) -> None:
    process = subprocess.Popen(command, cwd=ROOT)
    started = time.monotonic()
    while process.poll() is None:
        done, integrity = checkpoint_count(checkpoint)
        elapsed = max(time.monotonic() - started, 1e-9)
        update_json(state_path, {
            "status": "RUNNING", "stage": stage, "total": total, "done": done,
            "remaining": max(0, total - done),
            "progress": {"stage": stage, "total": total, "done": done, "remaining": max(0, total - done)},
            "checkpoint": str(checkpoint), "checkpoint_integrity": integrity,
            "rows_per_second": round(done / elapsed, 3), "stage_child_pid": process.pid,
        }, preserve_terminal=False)
        time.sleep(2)
    exit_code = process.wait()
    done, integrity = checkpoint_count(checkpoint)
    if exit_code != 0:
        raise RuntimeError(f"{stage} child exited {exit_code}; durable rows={done}, integrity={integrity}")
    if done != total or integrity != "ok":
        raise RuntimeError(f"{stage} checkpoint incomplete: {done}/{total}, integrity={integrity}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-state", required=True)
    args = parser.parse_args()
    config = load_acquisition_config(args.config)
    _, records, selection_sha = acquisition_manifest(config)
    outputs = config["outputs"]
    artifacts = Path(outputs["artifacts"])
    artifacts.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.run_state)
    worker_root = Path(outputs["worker_root"]) / "medlatec_vn"
    crawl_database = worker_root / "crawl.sqlite"
    body_archive = worker_root / "bodies"
    total = len(records)

    update_json(state_path, {
        "run_id": config["run_id"], "status": "RUNNING", "stage": "ACQUISITION_AUDIT",
        "total": total, "done": 0, "remaining": total, "selection_sha256": selection_sha,
    }, preserve_terminal=False)
    acquisition = acquisition_content_audit(crawl_database, body_archive, records)
    write_artifact(artifacts / "acquisition_content_audit.json", acquisition)
    if not acquisition["passed"]:
        raise RuntimeError(f"Phase 14 acquisition content audit failed: {acquisition}")

    documents = Path(outputs["documents"])
    extraction_checkpoint = Path(outputs["extraction_checkpoint"])
    extraction_summary = artifacts / "extraction_summary.json"
    run_partition_stage(
        command=[sys.executable, "scripts/extract_production.py", "--config", "configs/production_pipeline.yaml",
                 "--crawl-database", str(crawl_database), "--archive-directory", str(body_archive),
                 "--output-directory", outputs["processed_parts"], "--checkpoint", str(extraction_checkpoint),
                 "--summary-out", str(extraction_summary)],
        state_path=state_path, stage="EXTRACT", total=total, checkpoint=extraction_checkpoint,
    )
    subprocess.run([
        sys.executable, "scripts/merge_document_partitions.py", "--input-directory", outputs["processed_parts"],
        "--output", str(documents), "--summary-out", str(artifacts / "document_merge.json")
    ], cwd=ROOT, check=True)
    document_audit = document_quality_audit(
        documents, records, int(config["phase14_medlatec"]["minimum_usable_documents"])
    )
    document_audit.update({"documents_path": str(documents), "sha256": sha256(documents)})
    write_artifact(artifacts / "extraction_quality_audit.json", document_audit)
    if not document_audit["passed"]:
        update_json(state_path, {
            "status": "FAILED_SAFE", "stage": "EXTRACTION_USABLE_GATE", "total": total,
            "done": document_audit["usable_documents"], "remaining": max(0, 2000-document_audit["usable_documents"]),
            "checkpoint": str(artifacts / "extraction_quality_audit.json"),
        }, preserve_terminal=False)
        return 0

    chunks = Path(outputs["chunks"])
    chunk_checkpoint = Path(outputs["chunk_checkpoint"])
    run_partition_stage(
        command=[sys.executable, "scripts/chunk_production.py", "--config", "configs/production_pipeline.yaml",
                 "--documents", str(documents), "--output-directory", outputs["chunks_parts"],
                 "--checkpoint", str(chunk_checkpoint), "--summary-out", str(artifacts / "chunk_summary.json")],
        state_path=state_path, stage="CHUNK", total=total, checkpoint=chunk_checkpoint,
    )
    subprocess.run([
        sys.executable, "scripts/merge_chunk_partitions.py", "--input-directory", outputs["chunks_parts"],
        "--output", str(chunks), "--summary-out", str(artifacts / "chunk_merge.json")
    ], cwd=ROOT, check=True)
    chunk_audit = chunk_quality_audit(chunks, documents)
    chunk_audit.update({"chunks_path": str(chunks), "sha256": sha256(chunks)})
    write_artifact(artifacts / "chunk_quality_audit.json", chunk_audit)
    if not chunk_audit["passed"]:
        raise RuntimeError(f"Phase 14 chunk quality audit failed: {chunk_audit}")

    chunk_rows = int(chunk_audit["chunk_rows"])
    embedding_bytes = chunk_rows * int(config["models"]["embedder"]["dimension"]) * 4
    disk = shutil.disk_usage(ROOT)
    memory = psutil.virtual_memory()
    historical = config["phase14_medlatec"]["historical_measurement"]
    historical_rate = float(historical["chunks"]) / float(historical["embedding_seconds"])
    equivalence_path = Path("artifacts/colab_reranker_poc/equivalence_report.json")
    equivalence = json.loads(equivalence_path.read_text(encoding="utf-8")) if equivalence_path.exists() else None
    pre_gpu = {
        "status": "PASS_EXTRACT_CHUNK_STOP_BEFORE_GPU",
        "selection_sha256": selection_sha,
        "usable_documents": document_audit["usable_documents"],
        "new_embedding_rows": chunk_rows,
        "embedding_dimension": int(config["models"]["embedder"]["dimension"]),
        "embedding_dtype": "float32",
        "embedding_vector_bytes": embedding_bytes,
        "transactional_two_vector_files_bytes": embedding_bytes * 2,
        "documents_bytes": documents.stat().st_size,
        "chunks_bytes": chunks.stat().st_size,
        "free_disk_bytes": disk.free,
        "disk_floor_bytes": 20 * 2**30,
        "available_physical_mib": round(memory.available / 2**20, 3),
        "local_historical_embedding_rows_per_second": round(historical_rate, 3),
        "local_projected_embedding_seconds": round(chunk_rows / historical_rate, 1),
        "colab_next_step": {
            "allowed_now": False,
            "reason": "A separate BGE-M3 embedding-equivalence proof and bounded export/import manifest are required first.",
            "reranker_import_allowed": False,
            "reranker_equivalence_status": equivalence.get("status") if equivalence else "MISSING",
            "reranker_max_absolute_difference": equivalence.get("maximum_absolute_difference") if equivalence else None,
            "reranker_ordering_changed_queries": equivalence.get("query_local_ordering_changed_count") if equivalence else None,
            "plan": [
                "Export exact chunk_id/text/hash rows in bounded shards without model scores.",
                "Benchmark pinned BGE-M3 revision on a deterministic Windows/Colab equivalence fixture.",
                "Only after embedding equivalence passes, encode shards and import by exact chunk_id/text hash.",
                "Keep reranker scores local until its separate A100 equivalence gate passes.",
            ],
        },
    }
    write_artifact(artifacts / "pre_gpu_resource_plan.json", pre_gpu)
    update_json(state_path, {
        "run_id": config["run_id"], "status": "READY_FOR_NEXT_STAGE", "stage": "EXTRACT_CHUNK_COMPLETE",
        "total": total, "done": total, "remaining": 0,
        "progress": {"stage": "EXTRACT_CHUNK", "total": total, "done": total, "remaining": 0},
        "usable_documents": document_audit["usable_documents"], "chunks": chunk_rows,
        "selection_sha256": selection_sha, "checkpoint": str(artifacts / "pre_gpu_resource_plan.json"),
        "next_stage_blocked": "GPU work requires explicit approval and an embedding equivalence gate",
    }, preserve_terminal=False)
    print(json.dumps({"acquisition": acquisition, "documents": document_audit, "chunks": chunk_audit, "pre_gpu": pre_gpu}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
