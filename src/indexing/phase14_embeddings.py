from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import shutil
from pathlib import Path
from typing import Any

import psutil
import yaml

from src.runtime.state_file import write_json


MODEL = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DIMENSION = 1024
BATCH_SIZE = 4
MAX_LENGTH = 512
WINDOW_ROWS = 64


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embedding_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["outputs"]["root"]) / "round1"
    root.mkdir(parents=True, exist_ok=True)
    return {
        "partial": root / "chunk_embeddings.streaming.partial",
        "status": root / "chunk_embeddings.streaming.status.u8",
        "checkpoint": root / "chunk_embeddings.streaming.checkpoint.json",
        "final": root / "chunk_embeddings.f32",
        "audit": Path(config["outputs"]["artifacts"]) / "embedding_audit.json",
    }


def load_phase14_embedding_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("phase") != "phase14_medlatec_coverage" or config.get("run_id") != "phase14_medlatec_coverage_3000":
        raise ValueError("unexpected dedicated Phase 14 embedding configuration")
    embedder = config["models"]["embedder"]
    expected = {
        "name": MODEL, "revision": REVISION, "dimension": DIMENSION,
        "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH,
    }
    if embedder != expected:
        raise ValueError(f"Phase 14 embedding contract mismatch: {embedder}")
    configured_window = int(config["embedding_recovery"].get("parquet_batch_rows", 0))
    if configured_window not in {64, 128}:
        raise ValueError("Phase 14 embedding window must be 64 or 128 rows")
    if int(config["embedding_recovery"].get("num_workers", -1)) != 0:
        raise ValueError("Phase 14 embedding must use zero DataLoader workers")
    if bool(config["embedding_recovery"].get("prefetching", True)):
        raise ValueError("Phase 14 embedding prefetching must remain disabled")
    return config


def signature(config: dict[str, Any]) -> dict[str, Any]:
    import pyarrow.parquet as pq
    chunks = Path(config["outputs"]["chunks"])
    rows = int(pq.ParquetFile(chunks).metadata.num_rows)
    return {
        "run_id": config["run_id"], "chunks_path": str(chunks),
        "chunks_sha256": file_sha256(chunks), "rows": rows,
        "dimension": DIMENSION, "dtype": "float32", "model": config["models"]["embedder"],
        "window_rows": WINDOW_ROWS, "normalization": True, "precision": "float16_cuda",
    }


def prepare_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    paths = embedding_paths(config)
    expected = signature(config)
    rows = int(expected["rows"])
    if paths["checkpoint"].exists():
        checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
        if checkpoint.get("signature") != expected:
            prior = checkpoint.get("signature", {})
            migration_allowed = (
                int(checkpoint.get("completed_rows", -1)) == 0
                and {key: prior.get(key) for key in prior if key != "window_rows"}
                == {key: expected.get(key) for key in expected if key != "window_rows"}
                and prior.get("window_rows") == 128 and expected.get("window_rows") == 64
            )
            if not migration_allowed:
                raise ValueError("Phase 14 embedding checkpoint signature mismatch")
            checkpoint["signature"] = expected
            checkpoint["checkpoint_migration"] = "empty_window_128_to_64"
            checkpoint["trial_rows_required"] = 64
        vector_path = paths["final"] if paths["final"].exists() and not paths["partial"].exists() else paths["partial"]
        if vector_path.stat().st_size != rows * DIMENSION * 4:
            raise ValueError("Phase 14 vector checkpoint size mismatch")
        if paths["status"].stat().st_size != rows:
            raise ValueError("Phase 14 completion bitmap size mismatch")
        status = np.memmap(paths["status"], dtype=np.uint8, mode="r", shape=(rows,))
        completed = int(status.sum())
        del status
        recorded = int(checkpoint.get("completed_rows", -1))
        if completed < recorded:
            raise ValueError("Phase 14 completion bitmap regressed behind checkpoint")
        if completed > recorded:
            checkpoint.update({
                "completed_rows": completed, "remaining_rows": rows - completed,
                "recovered_from_completion_bitmap": True,
            })
        if vector_path == paths["final"] and completed == rows:
            checkpoint.update({"complete": True, "remaining_rows": 0, "recovered_final_rename": True})
        write_json(paths["checkpoint"], checkpoint)
        return checkpoint
    existing = [path for key, path in paths.items() if key in {"partial", "status", "final"} and path.exists()]
    if existing:
        raise FileExistsError(f"unowned Phase 14 embedding artifacts require audit: {existing}")
    vector_tmp = paths["partial"].with_suffix(paths["partial"].suffix + ".initializing")
    status_tmp = paths["status"].with_suffix(paths["status"].suffix + ".initializing")
    vectors = np.memmap(vector_tmp, dtype=np.float32, mode="w+", shape=(rows, DIMENSION))
    status = np.memmap(status_tmp, dtype=np.uint8, mode="w+", shape=(rows,))
    status[:] = 0
    vectors.flush(); status.flush()
    del vectors, status
    os.replace(vector_tmp, paths["partial"])
    os.replace(status_tmp, paths["status"])
    checkpoint = {
        "signature": expected, "prepared": True, "complete": False,
        "completed_rows": 0, "remaining_rows": rows, "last_scanned_row": 0,
        "trial_rows_required": 64, "trial_gate": None,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    write_json(paths["checkpoint"], checkpoint)
    return checkpoint


def resource_preflight(config: dict[str, Any]) -> dict[str, Any]:
    import pyarrow.parquet as pq
    paths = embedding_paths(config)
    rows = int(pq.ParquetFile(config["outputs"]["chunks"]).metadata.num_rows)
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(Path.cwd())
    profile_path = Path(config["embedding_recovery"]["profile_artifact"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    before = next(row for row in profile["measurements"] if row["phase"] == "before_model_load")
    first = next(row for row in profile["measurements"] if row["phase"] == "first_batch_written")
    rss_delta = float(first["rss_mib"]) - float(before["rss_mib"])
    private_delta = float(first["private_mib"]) - float(before["private_mib"])
    vram_delta = float(first["gpu_allocated_mib"]) - float(before["gpu_allocated_mib"])
    required_mib = rss_delta * 1.25
    required_commit_mib = private_delta * 1.10
    required_vram_mib = vram_delta * 1.25
    try:
        gpu_free_mib = float(subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0])
    except Exception:
        gpu_free_mib = -1.0
    vector_bytes = rows * DIMENSION * 4
    equivalence_path = Path("artifacts/phase14_embedding_poc/equivalence_report.json")
    equivalence = json.loads(equivalence_path.read_text(encoding="utf-8")) if equivalence_path.exists() else {}
    return {
        "rows": rows, "dimension": DIMENSION, "vector_bytes": vector_bytes,
        "available_physical_mib": round(memory.available / 2**20, 3),
        "required_physical_mib": round(required_mib, 3),
        "physical_headroom_mib": round(memory.available / 2**20 - required_mib, 3),
        "memory_gate_passed": memory.available / 2**20 >= required_mib,
        "available_commit_mib": round(memory.available / 2**20 + memory.free / 2**20 + (memory.total - memory.available) / 2**20, 3),
        "available_virtual_mib": round(memory.available / 2**20 + psutil.swap_memory().free / 2**20, 3),
        "required_commit_headroom_mib": round(required_commit_mib, 3),
        "commit_gate_passed": (memory.available + psutil.swap_memory().free) / 2**20 >= required_commit_mib,
        "gpu_free_mib": gpu_free_mib, "required_vram_mib": round(required_vram_mib, 3),
        "gpu_gate_passed": gpu_free_mib >= required_vram_mib,
        "memory_gate_basis": {
            "profile": str(profile_path), "before_model_load_rss_mib": before["rss_mib"],
            "first_batch_rss_mib": first["rss_mib"], "measured_incremental_rss_mib": round(rss_delta, 3),
            "measured_incremental_private_mib": round(private_delta, 3),
            "measured_first_batch_vram_mib": round(vram_delta, 3),
            "physical_safety_multiplier": 1.25, "commit_safety_multiplier": 1.10,
            "rule": "measured incremental cost from pre-load to first committed batch plus explicit reserve",
        },
        "free_disk_gib": round(disk.free / 2**30, 3), "minimum_free_gib": 20.0,
        "disk_gate_passed": disk.free - vector_bytes >= 20 * 2**30,
        "colab_equivalence_status": equivalence.get("status", "MISSING"),
        "colab_equivalence_passed": bool(equivalence.get("passed", False)),
        "remote_embedding_import_allowed": False,
        "checkpoint_path": str(paths["checkpoint"]), "final_path": str(paths["final"]),
        "contract": {
            "model": MODEL, "revision": REVISION, "precision": "float16_cuda",
            "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH,
            "normalization": True, "window_rows": WINDOW_ROWS,
        },
    }


def validate_vector_rows(vectors: Any) -> dict[str, Any]:
    import numpy as np
    finite = bool(np.isfinite(vectors).all())
    norms = np.linalg.norm(vectors, axis=1)
    normalized = bool(np.all((norms >= 0.99) & (norms <= 1.01)))
    return {
        "rows": int(vectors.shape[0]), "finite": finite, "normalized": normalized,
        "norm_minimum": float(norms.min()), "norm_maximum": float(norms.max()),
        "passed": finite and normalized,
    }


def audit_complete(config: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    paths = embedding_paths(config)
    rows = int(checkpoint["signature"]["rows"])
    status = np.memmap(paths["status"], dtype=np.uint8, mode="r", shape=(rows,))
    completed = int(status.sum()); bitmap_values = sorted(set(np.asarray(status).tolist()))
    del status
    if completed != rows or bitmap_values != [1]:
        raise ValueError("Phase 14 completion bitmap is not complete")
    vectors = np.memmap(paths["final"], dtype=np.float32, mode="r", shape=(rows, DIMENSION))
    finite = True; norm_minimum = float("inf"); norm_maximum = float("-inf")
    for start in range(0, rows, 2048):
        block = np.asarray(vectors[start:start + 2048])
        result = validate_vector_rows(block)
        finite = finite and result["finite"] and result["normalized"]
        norm_minimum = min(norm_minimum, result["norm_minimum"])
        norm_maximum = max(norm_maximum, result["norm_maximum"])
    del vectors
    audit = {
        "passed": bool(finite), "rows": rows, "dimension": DIMENSION,
        "shape": [rows, DIMENSION], "dtype": "float32", "completed_bitmap_rows": completed,
        "bitmap_values": bitmap_values, "finite": bool(finite),
        "normalized": bool(norm_minimum >= 0.99 and norm_maximum <= 1.01),
        "norm_minimum": norm_minimum, "norm_maximum": norm_maximum,
        "chunks_sha256": checkpoint["signature"]["chunks_sha256"],
        "embedding_sha256": file_sha256(paths["final"]), "embedding_path": str(paths["final"]),
        "row_order": "exact chunks.parquet physical row order",
    }
    write_json(paths["audit"], audit)
    return audit
