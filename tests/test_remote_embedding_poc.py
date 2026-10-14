from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.indexing import remote_embedding_poc as poc
from src.production.chunking import CHUNK_SCHEMA


def _fixture(tmp_path: Path, rows: int = 12) -> tuple[Path, Path, Path]:
    chunks = tmp_path / "chunks.parquet"
    values = []
    for index in range(rows):
        text = ("医学证据 " if index >= 8 else "Bằng chứng y khoa ") + ("x " * (index + 1))
        values.append({
            "chunk_id": f"chunk-{index}", "doc_id": index + 1, "chunk_index": 0,
            "raw_text": text, "normalized_text": text, "token_count": 10 + index * 30,
            "start_offset": 0, "end_offset": len(text), "section_type": "article",
            "heading_path": [], "source_url": "https://cancer.39.net/a" if index >= 8 else "https://medlatec.vn/a",
            "extraction_method": "semantic",
        })
    pq.write_table(pa.Table.from_pylist(values, schema=CHUNK_SCHEMA), chunks)
    vectors = np.zeros((rows, poc.DIMENSION), dtype=np.float32)
    for index in range(rows):
        vectors[index, index] = 1.0
    embeddings = tmp_path / "embeddings.f32"
    vectors.tofile(embeddings)
    notebook = tmp_path / "runner.py"
    notebook.write_text("print('runner')\n", encoding="utf-8")
    return chunks, embeddings, notebook


def _result(package: Path, output: Path, delta: float = 0.0) -> None:
    manifest, rows, vectors = poc.load_package(package)
    candidate = vectors.copy()
    candidate[0, 0] += delta
    vector_bytes = candidate.astype(np.float32).tobytes()
    result = {
        "input_package_sha256": poc.file_sha256(package), "contract": poc.CONTRACT,
        "rows": [{"sample_index": row["sample_index"], "chunk_id": row["chunk_id"], "text_sha256": row["text_sha256"]} for row in rows],
        "candidate_vectors_sha256": hashlib.sha256(vector_bytes).hexdigest(),
        "hardware": {"gpu": "fixture"}, "runtime": {"batch4_embedding_seconds": 1.0},
    }
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("result_manifest.json", json.dumps(result))
        archive.writestr("candidate_vectors.f32", vector_bytes)


def test_export_preserves_parquet_vector_alignment_and_batch_context(tmp_path):
    chunks, embeddings, notebook = _fixture(tmp_path)
    previous = tmp_path / "previous.parquet"
    pq.write_table(pq.read_table(chunks).slice(0, 2), previous)
    package = tmp_path / "poc.zip"
    report = poc.export_package(
        chunks, embeddings, notebook, package, previous_chunks_path=previous, count=8, seed="test",
    )
    manifest, rows, vectors = poc.load_package(package)
    source = np.memmap(embeddings, dtype=np.float32, mode="r", shape=(12, poc.DIMENSION))
    assert report["row_count"] == 8
    assert report["previous_reused_rows_excluded"] == 2
    assert report["newly_embedded_reference_pool_rows"] == 10
    assert not {"chunk-0", "chunk-1"} & {row["chunk_id"] for row in rows}
    assert all(row["reference_batch_offset"] == index % 4 for index, row in enumerate(rows))
    assert all(row["reference_pending_batch_start"] % 4 == 0 for row in rows)
    assert np.array_equal(vectors, source[[row["reference_row_index"] for row in rows]])
    assert manifest["production_import_allowed"] is False


def test_embedding_verifier_accepts_exact_and_rejects_measured_drift(tmp_path):
    chunks, embeddings, notebook = _fixture(tmp_path)
    package = tmp_path / "poc.zip"
    poc.export_package(chunks, embeddings, notebook, package, count=8, seed="test")
    exact = tmp_path / "exact.zip"
    _result(package, exact)
    assert poc.verify_results(package, exact)["passed"] is True
    drift = tmp_path / "drift.zip"
    _result(package, drift, delta=poc.REFERENCE_ABSOLUTE_TOLERANCE * 2)
    report = poc.verify_results(package, drift)
    assert report["passed"] is False
    assert report["rows_exceeding_absolute_tolerance"] == 1


def test_duplicate_analysis_separates_navigation_from_repeated_content(tmp_path):
    chunks, _, _ = _fixture(tmp_path, rows=8)
    table = pq.read_table(chunks).to_pylist()
    table[0]["raw_text"] = table[0]["normalized_text"] = "Tin cùng chuyên mục"
    table[0]["heading_path"] = ["Tin cùng chuyên mục"]
    table[1]["raw_text"] = table[1]["normalized_text"] = "Tin cùng chuyên mục"
    table[1]["heading_path"] = ["Tin cùng chuyên mục"]
    table[2]["raw_text"] = table[2]["normalized_text"] = "Nội dung y khoa lặp hợp lệ"
    table[3]["raw_text"] = table[3]["normalized_text"] = "Nội dung y khoa lặp hợp lệ"
    pq.write_table(pa.Table.from_pylist(table, schema=CHUNK_SCHEMA), chunks)
    report = poc.duplicate_chunk_analysis(chunks)
    assert report["exact_duplicate_groups"] == 2
    assert report["category_chunk_slots"]["related_article_navigation"] == 2
    assert report["category_chunk_slots"]["potential_legitimate_repetition"] == 2
    assert report["dataset_modified"] is False
