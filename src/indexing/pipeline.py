from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .embedder import Embedder, file_sha256
from .faiss_index import ExactFaissIndex
from .metadata_store import build_metadata_store, verify_metadata_store


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def build_dense_index(
    *,
    chunks_path: str | Path,
    embedding_directory: str | Path,
    index_directory: str | Path,
    embedder: Embedder,
    add_batch_size: int = 4096,
    progress_every: int = 100,
) -> dict[str, Any]:
    chunks = Path(chunks_path)
    embedding_dir = Path(embedding_directory)
    index_dir = Path(index_directory)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(chunks)
    row_count = parquet.metadata.num_rows
    dimension = embedder.dimension
    model_metadata = embedder.metadata()
    embedding_signature = {
        key: model_metadata.get(key)
        for key in (
            "model_name",
            "resolved_revision",
            "embedding_dimension",
            "max_sequence_length",
            "pooling",
            "normalization",
            "inference_precision",
            "output_dtype",
        )
    }
    torch = None
    if model_metadata.get("device") == "cuda":
        import torch as torch_module

        torch = torch_module
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    source_hash = file_sha256(chunks)
    vectors_path = embedding_dir / "chunk_embeddings.f32"
    state_path = embedding_dir / "embedding_state.json"
    expected_bytes = row_count * dimension * np.dtype(np.float32).itemsize
    completed = 0
    if state_path.exists() and vectors_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected = {
            "source_sha256": source_hash,
            "row_count": row_count,
            "dimension": dimension,
            "embedding_signature": embedding_signature,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("embedding cache does not match source/model configuration")
        completed = int(state["completed_rows"])
    mode = "r+" if vectors_path.exists() else "w+"
    vectors = np.memmap(vectors_path, dtype=np.float32, mode=mode, shape=(row_count, dimension))
    if vectors_path.stat().st_size != expected_bytes:
        raise ValueError("embedding file has an unexpected size")
    initial_completed = completed
    started = time.perf_counter()
    position = 0
    for batch in parquet.iter_batches(batch_size=128, columns=["normalized_text"]):
        texts = batch.column(0).to_pylist()
        batch_end = position + len(texts)
        if batch_end <= completed:
            position = batch_end
            continue
        start_in_batch = max(0, completed - position)
        pending = texts[start_in_batch:]
        encoded = embedder.encode(pending)
        write_start = position + start_in_batch
        vectors[write_start:batch_end] = encoded
        vectors.flush()
        completed = batch_end
        state = {
            "source_sha256": source_hash,
            "row_count": row_count,
            "dimension": dimension,
            "embedding_signature": embedding_signature,
            "completed_rows": completed,
        }
        _atomic_json(state_path, state)
        if completed % progress_every < len(texts) or completed == row_count:
            elapsed = time.perf_counter() - started
            print(f"embedded {completed}/{row_count} rows ({completed / max(elapsed, 1e-9):.2f}/s)")
        position = batch_end
    embedding_seconds = time.perf_counter() - started

    mapping = build_metadata_store(chunks, index_dir / "chunk_metadata.sqlite")
    mapping_verification = verify_metadata_store(chunks, mapping["path"])
    index = ExactFaissIndex(dimension)
    for start in range(0, row_count, add_batch_size):
        index.add(vectors[start : min(start + add_batch_size, row_count)])
    index_path = index_dir / "chunks.index"
    index.save(index_path)
    if index.count != int(mapping["row_count"]) or mapping_verification["mismatches"]:
        raise RuntimeError("FAISS/index metadata row count mismatch")
    summary = {
        "model": model_metadata,
        "chunk_source": str(chunks),
        "chunk_source_sha256": source_hash,
        "chunk_count": row_count,
        "embedding_runtime_seconds": round(embedding_seconds, 3),
        "embedding_throughput_chunks_per_second": round(
            (row_count - initial_completed) / max(embedding_seconds, 1e-9),
            4,
        ),
        "embedding_rows_computed_this_run": row_count - initial_completed,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if torch else None,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if torch else None,
        "embedding_file": str(vectors_path),
        "embedding_size_bytes": vectors_path.stat().st_size,
        "index_type": "IndexFlatIP",
        "faiss_index": str(index_path),
        "faiss_index_size_bytes": index_path.stat().st_size,
        "mapping": mapping,
        "row_mapping_verification": mapping_verification,
        "row_mapping_verified": True,
    }
    _atomic_json(index_dir / "index_metadata.json", summary)
    return summary
