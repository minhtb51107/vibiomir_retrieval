from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.indexing.phase14_embeddings import (
    DIMENSION, embedding_paths, load_phase14_embedding_config,
    prepare_checkpoint, validate_vector_rows,
)


def _config(tmp_path: Path) -> Path:
    chunks = tmp_path / "chunks.parquet"
    pq.write_table(pa.table({"chunk_id": ["a", "b", "c"], "normalized_text": ["A", "B", "C"]}), chunks)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "experiment": "phase14_medlatec_coverage_3000", "total_incremental_ids": 3000,
        "sources": {"medlatec.vn": {"incremental_records": [
            {"doc_id": index, "original_url": f"https://medlatec.vn/{index}"} for index in range(3000)
        ]}},
    }), encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    ids = list(range(3000))
    import hashlib
    digest = hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()
    preflight.write_text(json.dumps({"selected_id_sha256": digest}), encoding="utf-8")
    config = {
        "phase": "phase14_medlatec_coverage", "run_id": "phase14_medlatec_coverage_3000",
        "focused_scaling": {"phase_key": "phase14_medlatec"},
        "acquisition": {"minimum_free_gib": 20, "global_concurrency": 1},
        "models": {
            "embedder": {"name": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181", "dimension": 1024, "batch_size": 4, "max_length": 512},
            "reranker": {"batch_size": 2},
        },
        "embedding_recovery": {"parquet_batch_rows": 128, "num_workers": 0, "prefetching": False},
        "outputs": {"root": str(tmp_path / "output"), "artifacts": str(tmp_path / "artifacts"), "chunks": str(chunks)},
        "phase14_medlatec": {
            "manifest": str(manifest), "preflight": str(preflight), "sampling_seed": "vibiomir-phase14-medlatec-coverage-v1",
            "maximum_new_ids": 3000, "historical_measurement": {"required_embedding_physical_mib": 949.41},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_phase14_checkpoint_is_owned_resumable_and_row_aligned(tmp_path):
    config = load_phase14_embedding_config(_config(tmp_path))
    first = prepare_checkpoint(config)
    paths = embedding_paths(config)
    assert first["completed_rows"] == 0
    assert paths["partial"].stat().st_size == 3 * DIMENSION * 4
    assert paths["status"].stat().st_size == 3
    assert prepare_checkpoint(config) == first
    status = np.memmap(paths["status"], dtype=np.uint8, mode="r", shape=(3,))
    assert status.tolist() == [0, 0, 0]


def test_phase14_checkpoint_recovers_committed_bitmap_after_interruption(tmp_path):
    config = load_phase14_embedding_config(_config(tmp_path))
    prepare_checkpoint(config)
    paths = embedding_paths(config)
    status = np.memmap(paths["status"], dtype=np.uint8, mode="r+", shape=(3,))
    status[1] = 1; status.flush(); del status
    recovered = prepare_checkpoint(config)
    assert recovered["completed_rows"] == 1
    assert recovered["remaining_rows"] == 2
    assert recovered["recovered_from_completion_bitmap"] is True


def test_phase14_vector_gate_requires_finite_normalized_rows():
    valid = np.zeros((2, DIMENSION), dtype=np.float32); valid[:, 0] = 1
    assert validate_vector_rows(valid)["passed"] is True
    invalid = valid.copy(); invalid[1, 2] = np.nan
    assert validate_vector_rows(invalid)["passed"] is False
