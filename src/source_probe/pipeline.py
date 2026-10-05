from __future__ import annotations

import hashlib
import gc
import json
import os
import sqlite3
import statistics
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.acquisition_benchmark.metrics import content_signals
from src.indexing.embedder import SentenceTransformerEmbedder, file_sha256, normalize_text
from src.indexing.faiss_index import ExactFaissIndex
from src.indexing.metadata_store import ChunkMetadataStore, build_metadata_store, verify_metadata_store
from src.indexing.tokenizer_validation import HuggingFaceOffsetTokenizer
from src.reranking.candidate_pool import build_pool_from_paths
from src.reranking.candidate_pool import build_unified_candidate_pool
from src.reranking.checkpoint import RerankCheckpoint
from src.reranking.pipeline import score_pool
from src.reranking.reranker import TransformerCrossEncoderReranker
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.ranking import aggregate_documents
from src.retrieval.sparse_retriever import SparseRetriever, build_sparse_index
from src.submission.common import file_sha256 as submission_sha256
from src.submission.common import write_deterministic_zip, write_submission_stream
from src.submission.expansion import SourceWindowExpander
from src.submission.validator import expected_query_ids, validate_submission


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config["phase"] != "phase10b3_source_relevance_probe":
        raise ValueError("unexpected phase configuration")
    if config["models"]["reranker"]["batch_size"] != 2:
        raise ValueError("Phase 10B3 must retain the proven reranker batch size 2")
    return config


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)


def write_parquet(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = list(rows)
    if not records:
        raise ValueError(f"refusing to write empty parquet: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(records), temporary)
    os.replace(temporary, destination)


def short_qa_eligible(row: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    text = str(row.get("normalized_text") or "")
    status = str(row.get("extraction_status") or "")
    return (
        status == str(rule["extraction_status"])
        and status not in set(rule["excluded_statuses"])
        and int(row.get("clean_char_count") or len(text)) >= int(rule["minimum_clean_characters"])
        and int(row.get("paragraph_count") or 0) >= int(rule["minimum_paragraphs"])
        and content_signals(text)["medical_terms"] >= int(rule["minimum_medical_signals"])
    )


def selected_document_ids(
    documents_path: str | Path,
    *,
    inclusion: str,
    short_rule: Mapping[str, Any],
    excluded_doc_ids: set[int],
) -> set[int]:
    selected: set[int] = set()
    for batch in pq.ParquetFile(documents_path).iter_batches(batch_size=512):
        for row in batch.to_pylist():
            doc_id = int(row["doc_id"])
            if doc_id in excluded_doc_ids:
                continue
            if inclusion == "extraction_success":
                eligible = row["extraction_status"] == "SUCCESS" and bool(str(row.get("normalized_text") or "").strip())
            elif inclusion == "short_qa":
                eligible = short_qa_eligible(row, short_rule)
            else:
                raise ValueError(f"unknown inclusion policy: {inclusion}")
            if eligible:
                selected.add(doc_id)
    return selected


def _filter_rows(path: str | Path, selected: set[int]) -> list[dict[str, Any]]:
    return [row for row in pq.read_table(path).to_pylist() if int(row["doc_id"]) in selected]


def prepare_inputs(config: dict[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    work = Path(config["outputs"]["work_directory"])
    pilot_docs = pq.read_table(inputs["pilot_documents"]).to_pylist()
    pilot_chunks = pq.read_table(inputs["pilot_chunks"]).to_pylist()
    pilot_ids = {int(row["doc_id"]) for row in pilot_docs}
    pilot_chunk_ids = {str(row["chunk_id"]) for row in pilot_chunks}
    summary: dict[str, Any] = {
        "pilot": {"document_rows": len(pilot_docs), "chunk_rows": len(pilot_chunks)},
        "variants": {},
    }
    for variant, source in config["sources"].items():
        selected = selected_document_ids(
            source["documents"], inclusion=source["inclusion"],
            short_rule=config["short_qa_rule"], excluded_doc_ids=pilot_ids,
        )
        docs = _filter_rows(source["documents"], selected)
        chunks = _filter_rows(source["chunks"], selected)
        duplicate_chunks = pilot_chunk_ids & {str(row["chunk_id"]) for row in chunks}
        if duplicate_chunks:
            raise ValueError(f"chunk ID collision in {variant}: {sorted(duplicate_chunks)[:3]}")
        represented = {int(row["doc_id"]) for row in chunks}
        docs = [row for row in docs if int(row["doc_id"]) in represented]
        selected = represented
        root = work / variant
        write_parquet(root / "source_documents.parquet", docs)
        write_parquet(root / "source_chunks.parquet", chunks)
        write_parquet(root / "combined_documents.parquet", [*pilot_docs, *docs])
        write_parquet(root / "combined_chunks.parquet", [*pilot_chunks, *chunks])
        summary["variants"][variant] = {
            "label": source["label"], "domain": source["domain"],
            "inclusion": source["inclusion"], "added_documents": len(selected),
            "added_chunks": len(chunks), "pilot_overlap_excluded": len(
                set(pq.read_table(source["documents"], columns=["doc_id"]).column(0).to_pylist()) & pilot_ids
            ),
            "combined_documents": len(pilot_docs) + len(docs),
            "combined_chunks": len(pilot_chunks) + len(chunks),
            "paths": {
                "source_documents": str(root / "source_documents.parquet"),
                "source_chunks": str(root / "source_chunks.parquet"),
                "combined_documents": str(root / "combined_documents.parquet"),
                "combined_chunks": str(root / "combined_chunks.parquet"),
            },
        }
    atomic_json(Path(config["outputs"]["artifact_directory"]) / "input_manifest.json", summary)
    return summary


def resource_estimate(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    dimension = int(config["models"]["embedder"]["dimension"])
    new_chunks = sum(int(row["added_chunks"]) for row in manifest["variants"].values())
    embedding_bytes = new_chunks * dimension * 4
    largest = max(int(row["combined_chunks"]) for row in manifest["variants"].values())
    index_bytes = sum(int(row["combined_chunks"]) * dimension * 4 for row in manifest["variants"].values())
    estimate = {
        "new_chunks_all_sources": new_chunks,
        "new_embedding_bytes": embedding_bytes,
        "combined_exact_indexes_bytes": index_bytes,
        "largest_variant_chunks": largest,
        "estimated_embedding_seconds_at_phase4_rate": round(new_chunks / 49.0083, 2),
        "maximum_new_storage_bytes": int(config["safety"]["maximum_new_storage_bytes"]),
        "storage_gate_passed": embedding_bytes + index_bytes < int(config["safety"]["maximum_new_storage_bytes"]),
        "note": "excludes small Parquet/SQLite retrieval outputs and deterministic submission ZIPs",
    }
    if not estimate["storage_gate_passed"]:
        raise RuntimeError("projected Phase 10B3 storage exceeds the configured safety gate")
    atomic_json(Path(config["outputs"]["artifact_directory"]) / "resource_estimate.json", estimate)
    return estimate


def _embedder(config: dict[str, Any], *, query: bool = False) -> SentenceTransformerEmbedder:
    model = config["models"]["embedder"]
    return SentenceTransformerEmbedder(
        model_name=model["name"], revision=model["revision"], device=model["device"],
        batch_size=int(model["query_batch_size"] if query else model["batch_size"]),
        max_length=int(model["max_length"]), normalize_embeddings=True,
        use_half_on_cuda=bool(model["use_half_on_cuda"]), seed=int(model["seed"]),
    )


def embed_new_chunks(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    work = Path(config["outputs"]["work_directory"])
    embedder = _embedder(config)
    started = time.perf_counter()
    result: dict[str, Any] = {"model": embedder.metadata(), "variants": {}}
    for variant, row in manifest["variants"].items():
        path = Path(row["paths"]["source_chunks"])
        texts = pq.read_table(path, columns=["normalized_text"]).column(0).to_pylist()
        vectors_path = work / variant / "source_embeddings.f32"
        state_path = work / variant / "source_embeddings.json"
        signature = {"source_sha256": file_sha256(path), "rows": len(texts), "dimension": embedder.dimension}
        if state_path.exists() and vectors_path.exists() and json.loads(state_path.read_text(encoding="utf-8")) == signature:
            reused = True
        else:
            vectors = embedder.encode([str(text) for text in texts])
            vectors.astype(np.float32).tofile(vectors_path)
            atomic_json(state_path, signature)
            reused = False
        result["variants"][variant] = {**signature, "path": str(vectors_path), "reused": reused}
    query_path = Path(config["inputs"]["queries"])
    query_vectors_path = work / "query_embeddings.f32"
    query_state_path = work / "query_embeddings.json"
    queries = pq.read_table(query_path, columns=["query"]).column(0).to_pylist()
    query_signature = {"source_sha256": file_sha256(query_path), "rows": len(queries), "dimension": embedder.dimension}
    if not (query_state_path.exists() and query_vectors_path.exists() and json.loads(query_state_path.read_text(encoding="utf-8")) == query_signature):
        embedder.encode([normalize_text(str(text)) for text in queries]).astype(np.float32).tofile(query_vectors_path)
        atomic_json(query_state_path, query_signature)
    result["queries"] = {**query_signature, "path": str(query_vectors_path)}
    result["runtime_seconds"] = round(time.perf_counter() - started, 3)
    atomic_json(Path(config["outputs"]["artifact_directory"]) / "embedding_summary.json", result)
    return result


def _write_rankings(path: Path, grouped: list[list[dict[str, Any]]]) -> None:
    write_parquet(path, (row for rows in grouped for row in rows))


class _ParquetSink:
    def __init__(self, path: Path, flush_rows: int = 5000) -> None:
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        if self.temporary.exists(): self.temporary.unlink()
        self.flush_rows = flush_rows
        self.buffer: list[dict[str, Any]] = []
        self.writer: pq.ParquetWriter | None = None

    def add(self, rows: Iterable[dict[str, Any]]) -> None:
        self.buffer.extend(rows)
        if len(self.buffer) >= self.flush_rows: self.flush()

    def flush(self) -> None:
        if not self.buffer: return
        table = pa.Table.from_pylist(self.buffer)
        if self.writer is None: self.writer = pq.ParquetWriter(self.temporary, table.schema)
        self.writer.write_table(table)
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is None: raise ValueError(f"empty parquet output: {self.path}")
        self.writer.close()
        os.replace(self.temporary, self.path)


def _query_groups(path: Path, depth: int) -> Iterable[list[dict[str, Any]]]:
    current_query: int | None = None
    current: list[dict[str, Any]] = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            query_id = int(row["query_id"])
            if current_query is not None and query_id != current_query:
                yield current
                current = []
            current_query = query_id
            if int(row["rank"]) <= depth:
                current.append(row)
    if current:
        yield current


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _stream_candidate_pool(
    *, dense_path: Path, sparse_path: Path, hybrid_path: Path, output_path: Path,
    dense_depth: int, sparse_depth: int, hybrid_depth: int, max_depth: int,
) -> dict[str, Any]:
    """Build one query at a time so 120k long-text rows never coexist in RAM."""
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists(): temporary.unlink()
    writer: pq.ParquetWriter | None = None
    buffer: list[dict[str, Any]] = []
    summaries = []
    row_count = 0
    try:
        iterators = (
            _query_groups(dense_path, dense_depth), _query_groups(sparse_path, sparse_depth),
            _query_groups(hybrid_path, hybrid_depth),
        )
        for dense_rows, sparse_rows, hybrid_rows in zip(*iterators, strict=True):
            query_ids = {int(rows[0]["query_id"]) for rows in (dense_rows, sparse_rows, hybrid_rows)}
            if len(query_ids) != 1:
                raise ValueError("candidate sources are not aligned by query")
            rows, summary = build_unified_candidate_pool(
                dense_rows=dense_rows, sparse_rows=sparse_rows, hybrid_rows=hybrid_rows,
                max_depth=max_depth,
            )
            summaries.append(summary)
            buffer.extend(rows)
            row_count += len(rows)
            if len(buffer) >= 5000:
                table = pa.Table.from_pylist(buffer)
                if writer is None: writer = pq.ParquetWriter(temporary, table.schema)
                writer.write_table(table)
                buffer.clear()
        if buffer:
            table = pa.Table.from_pylist(buffer)
            if writer is None: writer = pq.ParquetWriter(temporary, table.schema)
            writer.write_table(table)
        if writer is None:
            raise ValueError("empty candidate pool")
    finally:
        if writer is not None: writer.close()
    os.replace(temporary, output_path)
    union = [float(row["union_size"]["mean"]) for row in summaries]
    selected = [int(row["selected_size"]["max"]) for row in summaries]
    timings = [float(row["preparation_latency_ms_per_query"]["mean"]) for row in summaries]
    return {
        "query_count": len(summaries), "candidate_rows": row_count, "max_depth": max_depth,
        "source_rows_read": {
            source: sum(int(row["source_rows_read"][source]) for row in summaries)
            for source in ("dense", "sparse", "hybrid")
        },
        "duplicate_source_rows": sum(int(row["duplicate_source_rows"]) for row in summaries),
        "union_size": {"min": min(union), "median": statistics.median(union), "max": max(union), "mean": statistics.fmean(union)},
        "selected_size": {"min": min(selected), "max": max(selected)},
        "preparation_latency_ms_per_query": {"mean": statistics.fmean(timings), "p50": _percentile(timings, .5), "p95": _percentile(timings, .95)},
    }


def build_retrieval(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    work = Path(config["outputs"]["work_directory"])
    retrieval = config["retrieval"]
    queries = pq.read_table(config["inputs"]["queries"], columns=["id", "query"]).to_pylist()
    query_ids = [int(row["id"]) for row in queries]
    query_texts = [normalize_text(str(row["query"])) for row in queries]
    dimension = int(config["models"]["embedder"]["dimension"])
    query_vectors = np.memmap(work / "query_embeddings.f32", dtype=np.float32, mode="r", shape=(len(queries), dimension))
    pilot_count = pq.ParquetFile(config["inputs"]["pilot_chunks"]).metadata.num_rows
    pilot_vectors = np.memmap(config["inputs"]["pilot_embeddings"], dtype=np.float32, mode="r", shape=(pilot_count, dimension))
    summaries: dict[str, Any] = {}
    for variant, row in manifest["variants"].items():
        started = time.perf_counter()
        root = work / variant
        existing_pool_summary = root / "candidate_pool.json"
        existing_pool = root / "candidate_pool.parquet"
        if existing_pool_summary.exists() and existing_pool.exists():
            summaries[variant] = {"candidate_pool": json.loads(existing_pool_summary.read_text(encoding="utf-8")), "reused": True}
            continue
        combined_chunks = root / "combined_chunks.parquet"
        source_vectors = np.memmap(root / "source_embeddings.f32", dtype=np.float32, mode="r", shape=(row["added_chunks"], dimension))
        index_path = root / "chunks.index"
        dense_path = root / "dense_chunk_rankings.parquet"
        sparse_path = root / "bm25.sqlite"
        sparse_results = root / "sparse_chunk_rankings.parquet"
        hybrid_path = root / "hybrid_chunk_rankings.parquet"
        metadata_path = root / "chunk_metadata.sqlite"
        if not (dense_path.exists() and sparse_results.exists() and hybrid_path.exists()):
            mapping = build_metadata_store(combined_chunks, metadata_path)
            mapping_check = verify_metadata_store(combined_chunks, metadata_path)
            index = ExactFaissIndex(dimension)
            index.add(pilot_vectors); index.add(source_vectors); index.save(index_path)
            class VectorEmbedder:
                def __init__(self, vector_dimension: int) -> None: self.dimension = vector_dimension
                def encode(self, texts: list[str]) -> np.ndarray: raise AssertionError("query vectors must be reused")
                def metadata(self) -> dict[str, Any]: return {}
            with ChunkMetadataStore(metadata_path) as metadata:
                dense = DenseRetriever(embedder=VectorEmbedder(dimension), index=index, metadata=metadata).retrieve_vectors(
                    query_ids, query_texts, query_vectors, top_k=int(retrieval["dense_top_k"])
                )
            _write_rankings(dense_path, dense)
            del dense, index
            gc.collect()
            sparse_build = build_sparse_index(
                chunks_path=combined_chunks, output_path=sparse_path,
                k1=float(retrieval["bm25_k1"]), b=float(retrieval["bm25_b"]), dense_metadata_path=metadata_path,
            )
            sparse_sink = _ParquetSink(sparse_results)
            hybrid_sink = _ParquetSink(hybrid_path)
            hybrid_engine = HybridRetriever(rrf_constant=int(retrieval["rrf_constant"]))
            dense_groups = _query_groups(dense_path, int(retrieval["dense_top_k"]))
            with ChunkMetadataStore(metadata_path) as metadata, SparseRetriever(index_path=sparse_path, metadata=metadata) as sparse:
                for query_id, query_text, dense_rows in zip(query_ids, query_texts, dense_groups, strict=True):
                    sparse_rows = sparse.retrieve([query_id], [query_text], top_k=int(retrieval["sparse_top_k"]))[0]
                    sparse_sink.add(sparse_rows)
                    hybrid_sink.add(hybrid_engine.fuse([dense_rows], [sparse_rows], top_k=int(retrieval["hybrid_top_k"]))[0])
            sparse_sink.close(); hybrid_sink.close()
            gc.collect()
        else:
            mapping = {"path": str(metadata_path), "row_count": str(row["combined_chunks"])}
            mapping_check = verify_metadata_store(combined_chunks, metadata_path)
            sparse_build = {"path": str(sparse_path), "reused": True}
        pool_path = root / "candidate_pool.parquet"
        pool_summary = _stream_candidate_pool(
            dense_path=dense_path, sparse_path=sparse_results, hybrid_path=hybrid_path, output_path=pool_path,
            dense_depth=int(retrieval["dense_top_k"]), sparse_depth=int(retrieval["sparse_top_k"]),
            hybrid_depth=int(retrieval["hybrid_top_k"]), max_depth=int(retrieval["rerank_depth"]),
        )
        pool_summary.update({
            "path": str(pool_path), "sha256": file_sha256(pool_path),
            "input_hashes": {
                "dense_candidates_sha256": file_sha256(dense_path),
                "sparse_candidates_sha256": file_sha256(sparse_results),
                "hybrid_candidates_sha256": file_sha256(hybrid_path),
                "candidate_pool_sha256": file_sha256(pool_path),
            },
        })
        atomic_json(root / "candidate_pool.json", pool_summary)
        summaries[variant] = {
            "mapping": mapping, "mapping_verification": mapping_check,
            "index_bytes": index_path.stat().st_size, "sparse": sparse_build,
            "candidate_pool": pool_summary, "runtime_seconds": round(time.perf_counter() - started, 3),
        }
    atomic_json(Path(config["outputs"]["artifact_directory"]) / "retrieval_summary.json", summaries)
    return summaries


def reranker_metadata(config: dict[str, Any]) -> dict[str, Any]:
    model = config["models"]["reranker"]
    return {
        "model_name": model["name"], "resolved_revision": model["revision"],
        "max_sequence_length": int(model["max_sequence_length"]),
        "inference_precision": model["precision"], "batch_size": int(model["batch_size"]),
    }


def checkpoint_signature(pool_summary: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_hashes": dict(pool_summary["input_hashes"]),
        "model_name": metadata["model_name"], "resolved_revision": metadata["resolved_revision"],
        "max_sequence_length": metadata["max_sequence_length"],
        "inference_precision": metadata["inference_precision"], "batch_size": metadata["batch_size"],
    }


def seed_checkpoints(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    work = Path(config["outputs"]["work_directory"])
    source = sqlite3.connect(config["inputs"]["pilot_rerank_checkpoint"])
    metadata = reranker_metadata(config)
    summaries = {}
    for variant in manifest["variants"]:
        root = work / variant
        pool_summary = json.loads((root / "candidate_pool.json").read_text(encoding="utf-8"))
        signature = checkpoint_signature(pool_summary, metadata)
        with RerankCheckpoint(root / "rerank_scores.sqlite", signature=signature) as checkpoint:
            before = checkpoint.count
            existing = checkpoint.completed_keys()
            for batch in pq.ParquetFile(root / "candidate_pool.parquet").iter_batches(batch_size=500, columns=["query_id", "chunk_id"]):
                keys = [(int(row["query_id"]), str(row["chunk_id"])) for row in batch.to_pylist()]
                pending = [key for key in keys if key not in existing]
                rows = []
                for query_id, chunk_id in pending:
                    found = source.execute(
                        "SELECT rerank_score,inference_ms FROM scores WHERE query_id=? AND chunk_id=?",
                        (query_id, chunk_id),
                    ).fetchone()
                    if found is not None:
                        rows.append((query_id, chunk_id, float(found[0]), float(found[1])))
                checkpoint.write_scores(rows)
                existing.update((row[0], row[1]) for row in rows)
            after = checkpoint.count
            total = int(pool_summary["candidate_rows"])
            summaries[variant] = {"pool_rows": total, "reused_scores": after, "newly_seeded": after - before, "novel_scores": total - after}
    source.close()
    novel = sum(row["novel_scores"] for row in summaries.values())
    estimated = novel / float(config["safety"]["measured_rerank_pairs_per_second"])
    result = {
        "variants": summaries, "novel_pairs": novel,
        "estimated_novel_rerank_seconds": round(estimated, 2),
        "maximum_estimated_novel_rerank_seconds": int(config["safety"]["maximum_estimated_novel_rerank_seconds"]),
        "gpu_time_gate_passed": estimated <= int(config["safety"]["maximum_estimated_novel_rerank_seconds"]),
    }
    atomic_json(Path(config["outputs"]["artifact_directory"]) / "rerank_preflight.json", result)
    return result


NOVEL_SCHEMA = """
CREATE TABLE manifest (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE pairs (
    query_id INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    rerank_score REAL,
    inference_ms REAL,
    PRIMARY KEY(query_id, chunk_id)
);
CREATE TABLE memberships (
    variant TEXT NOT NULL,
    query_id INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY(variant, query_id, chunk_id)
);
CREATE INDEX idx_memberships_pair ON memberships(query_id, chunk_id);
"""


def _novel_signature(config: dict[str, Any]) -> dict[str, Any]:
    work = Path(config["outputs"]["work_directory"])
    return {
        "model": reranker_metadata(config),
        "candidate_pools": {
            variant: file_sha256(work / variant / "candidate_pool.parquet")
            for variant in sorted(config["sources"])
        },
    }


def materialize_missing_pairs(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """CPU-only: create one deduplicated SQLite work queue of missing pairs."""
    work = Path(config["outputs"]["work_directory"])
    destination = work / "novel_pairs.sqlite"
    signature = _novel_signature(config)
    if destination.exists():
        connection = sqlite3.connect(destination)
        row = connection.execute("SELECT value FROM manifest WHERE key='signature'").fetchone()
        if row is None or json.loads(row[0]) != signature:
            connection.close()
            raise ValueError("existing novel-pair manifest has incompatible inputs")
        total = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
        scored = int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0])
        memberships = int(connection.execute("SELECT COUNT(*) FROM memberships").fetchone()[0])
        connection.close()
        return {"path": str(destination), "unique_pairs": total, "memberships": memberships, "scored": scored, "remaining": total - scored, "reused": True}

    temporary = destination.with_suffix(".sqlite.tmp")
    if temporary.exists(): temporary.unlink()
    connection = sqlite3.connect(temporary)
    connection.executescript(NOVEL_SCHEMA)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("INSERT INTO manifest VALUES ('signature', ?)", (json.dumps(signature, sort_keys=True),))
    variant_missing: dict[str, int] = {}
    try:
        for variant in sorted(manifest["variants"]):
            root = work / variant
            pool_summary = json.loads((root / "candidate_pool.json").read_text(encoding="utf-8"))
            checkpoint_signature_value = checkpoint_signature(pool_summary, reranker_metadata(config))
            with RerankCheckpoint(root / "rerank_scores.sqlite", signature=checkpoint_signature_value) as checkpoint:
                completed = checkpoint.completed_keys()
            missing = 0
            for batch in pq.ParquetFile(root / "candidate_pool.parquet").iter_batches(
                batch_size=512, columns=["query_id", "query_text", "chunk_id", "chunk_text"]
            ):
                pair_rows = []
                membership_rows = []
                for row in batch.to_pylist():
                    key = (int(row["query_id"]), str(row["chunk_id"]))
                    if key in completed: continue
                    pair_rows.append((key[0], key[1], str(row["query_text"]), str(row["chunk_text"])))
                    membership_rows.append((variant, key[0], key[1]))
                    missing += 1
                with connection:
                    connection.executemany(
                        "INSERT INTO pairs(query_id,chunk_id,query_text,chunk_text) VALUES (?,?,?,?) "
                        "ON CONFLICT(query_id,chunk_id) DO NOTHING", pair_rows,
                    )
                    connection.executemany("INSERT INTO memberships VALUES (?,?,?)", membership_rows)
            variant_missing[variant] = missing
        conflicts = int(connection.execute(
            "SELECT COUNT(*) FROM memberships m JOIN pairs p USING(query_id,chunk_id) "
            "WHERE p.query_text='' OR p.chunk_text=''"
        ).fetchone()[0])
        if conflicts: raise ValueError("empty text found in novel-pair manifest")
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok": raise RuntimeError("novel-pair manifest integrity check failed")
        total = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
        membership_count = int(connection.execute("SELECT COUNT(*) FROM memberships").fetchone()[0])
    finally:
        connection.close()
    os.replace(temporary, destination)
    result = {
        "path": str(destination), "variant_missing": variant_missing,
        "memberships": membership_count, "unique_pairs": total,
        "deduplicated_pairs_saved": membership_count - total,
        "scored": 0, "remaining": total, "reused": False,
    }
    atomic_json(Path(config["outputs"]["artifact_directory"]) / "novel_pair_manifest.json", result)
    return result


def distribute_novel_scores(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """CPU-only: copy globally scored pairs into signed per-variant checkpoints."""
    work = Path(config["outputs"]["work_directory"])
    global_db = sqlite3.connect(work / "novel_pairs.sqlite")
    unscored = int(global_db.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NULL").fetchone()[0])
    if unscored:
        global_db.close()
        raise RuntimeError(f"global novel-pair checkpoint remains incomplete: {unscored}")
    result = {}
    for variant in sorted(manifest["variants"]):
        root = work / variant
        pool_summary = json.loads((root / "candidate_pool.json").read_text(encoding="utf-8"))
        signature = checkpoint_signature(pool_summary, reranker_metadata(config))
        with RerankCheckpoint(root / "rerank_scores.sqlite", signature=signature) as checkpoint:
            before = checkpoint.count
            cursor = global_db.execute(
                "SELECT p.query_id,p.chunk_id,p.rerank_score,p.inference_ms "
                "FROM memberships m JOIN pairs p USING(query_id,chunk_id) WHERE m.variant=?",
                (variant,),
            )
            while True:
                rows = cursor.fetchmany(512)
                if not rows: break
                checkpoint.write_scores((int(q), str(c), float(s), float(ms)) for q, c, s, ms in rows)
            result[variant] = {
                "before": before, "after": checkpoint.count,
                "pool_rows": int(pool_summary["candidate_rows"]),
                "complete": checkpoint.count == int(pool_summary["candidate_rows"]),
                "integrity": checkpoint.integrity_check(),
            }
    global_db.close()
    if not all(row["complete"] and row["integrity"] == "ok" for row in result.values()):
        raise RuntimeError("one or more variant checkpoints failed distribution validation")
    atomic_json(Path(config["outputs"]["artifact_directory"]) / "score_distribution.json", result)
    return result


def score_pending(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    preflight = seed_checkpoints(config, manifest)
    if not preflight["gpu_time_gate_passed"]:
        raise RuntimeError("estimated novel reranking exceeds the configured three-hour safety gate")
    model = config["models"]["reranker"]
    reranker = TransformerCrossEncoderReranker(
        model_name=model["name"], revision=model["revision"], cache_dir=model["cache_dir"],
        device=model["device"], batch_size=int(model["batch_size"]),
        max_length=int(model["max_sequence_length"]), use_half_on_cuda=True, seed=int(model["seed"]),
    )
    reranker.score_pairs([("warmup", "warmup")])
    results = {}
    work = Path(config["outputs"]["work_directory"])
    for variant in manifest["variants"]:
        root = work / variant
        pool_summary = json.loads((root / "candidate_pool.json").read_text(encoding="utf-8"))
        local_config = {"outputs": {"checkpoint": str(root / "rerank_scores.sqlite")}}
        results[variant] = score_pool(
            config=local_config, reranker=reranker,
            pool_path=root / "candidate_pool.parquet", pool_summary=pool_summary, progress_every=1000,
        )
    reranker.release()
    atomic_json(Path(config["outputs"]["artifact_directory"]) / "rerank_scoring.json", results)
    return results


def _all_query_groups(path: Path) -> Iterable[list[dict[str, Any]]]:
    current_id: int | None = None
    rows: list[dict[str, Any]] = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            query_id = int(row["query_id"])
            if current_id is not None and query_id != current_id:
                yield rows
                rows = []
            current_id = query_id
            rows.append(row)
    if rows:
        yield rows


def _write_baseline_final_rows(config: dict[str, Any]) -> None:
    root = Path(config["outputs"]["work_directory"]) / "B0"
    root.mkdir(parents=True, exist_ok=True)
    doc_sink = _ParquetSink(root / "final_documents_top10.parquet")
    source_docs = Path("data/retrieval/phase7_reranking/selected_pure_rerank_documents_best_chunk.parquet")
    for rows in _all_query_groups(source_docs):
        rows.sort(key=lambda row: (int(row["rank"]), int(row["doc_id"])))
        doc_sink.add(rows[:10])
    doc_sink.close()
    chunk_sink = _ParquetSink(root / "final_chunks_top20.parquet")
    for rows in _all_query_groups(Path(config["inputs"]["pilot_reranked_chunks"])):
        rows.sort(key=lambda row: (int(row["rerank_rank"]), str(row["chunk_id"])))
        chunk_sink.add(rows[:20])
    chunk_sink.close()


def _baseline_final_ids(config: dict[str, Any]) -> dict[str, dict[int, list[Any]]]:
    root = Path(config["outputs"]["work_directory"]) / "B0"
    documents = {
        int(rows[0]["query_id"]): [int(row["doc_id"]) for row in rows]
        for rows in _all_query_groups(root / "final_documents_top10.parquet")
    }
    chunks = {
        int(rows[0]["query_id"]): [str(row["chunk_id"]) for row in rows]
        for rows in _all_query_groups(root / "final_chunks_top20.parquet")
    }
    return {"documents": documents, "chunks": chunks}


def _finalize_one(config: dict[str, Any], variant: str, source_ids: set[int]) -> dict[str, Any]:
    root = Path(config["outputs"]["work_directory"]) / variant
    pool_summary = json.loads((root / "candidate_pool.json").read_text(encoding="utf-8"))
    signature = checkpoint_signature(pool_summary, reranker_metadata(config))
    baseline = _baseline_final_ids(config)
    reranked_sink = _ParquetSink(root / "reranked_chunks.parquet")
    documents_sink = _ParquetSink(root / "reranked_documents_best_chunk.parquet")
    top_chunks_sink = _ParquetSink(root / "final_chunks_top20.parquet")
    top_documents_sink = _ParquetSink(root / "final_documents_top10.parquet")
    query_source_docs = 0
    query_source_chunks = 0
    queries_changed = 0
    query_count = 0
    stage_entries: dict[str, set[int]] = {"dense": set(), "sparse": set(), "hybrid": set(), "reranker": set()}
    with RerankCheckpoint(root / "rerank_scores.sqlite", signature=signature) as checkpoint:
        if checkpoint.integrity_check() != "ok" or checkpoint.count != int(pool_summary["candidate_rows"]):
            raise RuntimeError(f"incomplete or corrupt rerank checkpoint for {variant}")
        for pool_rows in _all_query_groups(root / "candidate_pool.parquet"):
            query_id = int(pool_rows[0]["query_id"])
            score_map = {
                str(row[0]): float(row[1])
                for row in checkpoint.connection.execute(
                    "SELECT chunk_id,rerank_score FROM scores WHERE query_id=?", (query_id,)
                )
            }
            rows = []
            for original in pool_rows:
                row = dict(original)
                row["rerank_score"] = score_map[str(row["chunk_id"])]
                rows.append(row)
            rows.sort(key=lambda row: (-float(row["rerank_score"]), str(row["chunk_id"])))
            for rank, row in enumerate(rows, start=1):
                row["rerank_rank"] = rank
                row["score"] = float(row["rerank_score"])
            docs = aggregate_documents(rows, method="best_chunk", top_k_docs=100)
            top_doc_rows = docs[:10]
            top_chunks = rows[:20]
            reranked_sink.add(rows)
            documents_sink.add(docs)
            top_chunks_sink.add(top_chunks)
            top_documents_sink.add(top_doc_rows)
            top_docs = {int(row["doc_id"]) for row in top_doc_rows}
            query_source_docs += bool(top_docs & source_ids)
            query_source_chunks += any(int(row["doc_id"]) in source_ids for row in top_chunks)
            queries_changed += (
                [int(row["doc_id"]) for row in top_doc_rows] != baseline["documents"][query_id]
                or [str(row["chunk_id"]) for row in top_chunks] != baseline["chunks"][query_id]
            )
            for source in ("dense", "sparse", "hybrid"):
                if any(int(row["doc_id"]) in source_ids and row[f"{source}_rank"] is not None for row in top_chunks):
                    stage_entries[source].add(query_id)
            if any(int(row["doc_id"]) in source_ids for row in top_chunks):
                stage_entries["reranker"].add(query_id)
            query_count += 1
    for sink in (reranked_sink, documents_sink, top_chunks_sink, top_documents_sink):
        sink.close()
    threshold = int(config["submission"]["minimum_changed_queries_for_source_submission"])
    return {
        "query_count": query_count,
        "queries_changed": queries_changed,
        "queries_with_source_doc_top10": query_source_docs,
        "queries_with_source_chunk_top20": query_source_chunks,
        "stage_queries": {key: len(value) for key, value in stage_entries.items()},
        "meaningful_output_change": queries_changed >= threshold,
        "submission_threshold_changed_queries": threshold,
    }


def finalize_rankings(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    _write_baseline_final_rows(config)
    result = {}
    for variant, row in manifest["variants"].items():
        source_ids = set(pq.read_table(row["paths"]["source_documents"], columns=["doc_id"]).column(0).to_pylist())
        result[variant] = _finalize_one(config, variant, source_ids)
    atomic_json(Path(config["outputs"]["artifact_directory"]) / "local_diagnostics.json", result)
    return result


def _submission_records(
    query_ids: list[int], documents_path: Path, chunks_path: Path,
    chunk_lookup: dict[str, dict[str, Any]], document_text: dict[int, str], tokenizer: Any,
    *, document_depth: int, chunk_depth: int, target_tokens: int,
) -> Iterable[dict[str, Any]]:
    expander = SourceWindowExpander(document_text, tokenizer, target_tokens)
    document_groups = iter(_all_query_groups(documents_path))
    chunk_groups = iter(_all_query_groups(chunks_path))
    for query_id in query_ids:
        document_rows = next(document_groups, None)
        chunk_rows = next(chunk_groups, None)
        if not document_rows or int(document_rows[0]["query_id"]) != query_id:
            raise ValueError(f"missing or out-of-order document ranking for query {query_id}")
        if not chunk_rows or int(chunk_rows[0]["query_id"]) != query_id:
            raise ValueError(f"missing or out-of-order chunk ranking for query {query_id}")
        document_rows.sort(key=lambda row: (int(row["rank"]), int(row["doc_id"])))
        chunk_rows.sort(key=lambda row: (int(row["rerank_rank"]), str(row["chunk_id"])))
        docs: list[int] = []
        seen_docs: set[int] = set()
        for row in document_rows:
            doc_id = int(row["doc_id"])
            if doc_id not in seen_docs:
                docs.append(doc_id); seen_docs.add(doc_id)
            if len(docs) == document_depth: break
        emitted = []
        seen_objects: set[tuple[int, str]] = set()
        for row in chunk_rows:
            canonical = chunk_lookup[str(row["chunk_id"])]
            if int(row["doc_id"]) != int(canonical["doc_id"]) or str(row["chunk_text"]) != str(canonical["raw_text"]):
                raise ValueError("canonical chunk provenance mismatch")
            span = expander.expand(
                doc_id=int(canonical["doc_id"]), start=int(canonical["start_offset"]),
                end=int(canonical["end_offset"]), original_text=str(canonical["raw_text"]),
            )
            key = (int(canonical["doc_id"]), span.text)
            if key in seen_objects: continue
            seen_objects.add(key)
            emitted.append({"doc_id": key[0], "chunk_text": key[1]})
            if len(emitted) == chunk_depth: break
        yield {"id": query_id, "relevant_docs": docs, "relevant_chunks": emitted}
    if next(document_groups, None) is not None or next(chunk_groups, None) is not None:
        raise ValueError("ranking contains unexpected queries")


def _generate_submission(
    config: dict[str, Any], *, name: str, documents_path: Path, chunks_path: Path,
    canonical_path: Path, source_documents_path: Path, output: Path, tokenizer: Any,
) -> dict[str, Any]:
    query_ids = expected_query_ids(config["inputs"]["queries"])
    canonical_rows = pq.read_table(canonical_path).to_pylist()
    lookup = {str(row["chunk_id"]): row for row in canonical_rows}
    source_docs = {int(row["doc_id"]): str(row["normalized_text"]) for row in pq.read_table(source_documents_path, columns=["doc_id", "normalized_text"]).to_pylist()}
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f"{name}.json"
    zip_path = output / f"{name}.zip"
    args = dict(
        query_ids=query_ids, documents_path=documents_path, chunks_path=chunks_path, chunk_lookup=lookup,
        document_text=source_docs, tokenizer=tokenizer,
        document_depth=int(config["submission"]["documents_per_query"]),
        chunk_depth=int(config["submission"]["chunks_per_query"]),
        target_tokens=int(config["submission"]["chunk_expansion_tokens"]),
    )
    write_submission_stream(json_path, _submission_records(**args))
    write_deterministic_zip(json_path, zip_path)
    valid_ids = set(source_docs)
    valid_chunks = {(int(row["doc_id"]), str(row["raw_text"])) for row in canonical_rows}
    validation = validate_submission(
        zip_path, expected_queries=query_ids, valid_document_ids=valid_ids,
        valid_chunks=valid_chunks, source_documents=source_docs,
    )
    return {
        "name": name, "json_path": str(json_path), "zip_path": str(zip_path),
        "json_sha256": submission_sha256(json_path), "zip_sha256": submission_sha256(zip_path),
        "validation": validation,
    }


def generate_submissions(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    model = config["models"]["embedder"]
    tokenizer = HuggingFaceOffsetTokenizer(
        AutoTokenizer.from_pretrained(
            model["name"], revision=model["revision"], local_files_only=True
        )
    )
    output = Path(config["submission"]["output_directory"])
    artifacts = Path(config["outputs"]["artifact_directory"])
    diagnostics = json.loads((artifacts / "local_diagnostics.json").read_text(encoding="utf-8"))
    variants = {}
    b0_name = "phase10b3_B0_fixed_baseline"
    variants["B0"] = _generate_submission(
        config, name=b0_name,
        documents_path=Path("data/retrieval/phase7_reranking/selected_pure_rerank_documents_best_chunk.parquet"),
        chunks_path=Path(config["inputs"]["pilot_reranked_chunks"]),
        canonical_path=Path(config["inputs"]["pilot_chunks"]),
        source_documents_path=Path(config["inputs"]["pilot_documents"]), output=output, tokenizer=tokenizer,
    )
    filenames = {
        "S1": "phase10b3_S1_cnkang", "S2": "phase10b3_S2_familydoctor",
        "S3": "phase10b3_S3_ahospital", "S4": "phase10b3_S4_suckhoecongdong",
        "S5": "phase10b3_S5_120ask_shortqa", "S6": "phase10b3_S6_ask39_shortqa",
    }
    work = Path(config["outputs"]["work_directory"])
    selected = [variant for variant in filenames if diagnostics[variant]["meaningful_output_change"]]
    for variant in selected:
        name = filenames[variant]
        root = work / variant
        variants[variant] = _generate_submission(
            config, name=name, documents_path=root / "reranked_documents_best_chunk.parquet",
            chunks_path=root / "reranked_chunks.parquet", canonical_path=root / "combined_chunks.parquet",
            source_documents_path=root / "combined_documents.parquet", output=output, tokenizer=tokenizer,
        )
    with tempfile.TemporaryDirectory(prefix="phase10b3_determinism_") as temporary:
        second = {}
        temp = Path(temporary)
        second["B0"] = _generate_submission(
            config, name=b0_name,
            documents_path=Path("data/retrieval/phase7_reranking/selected_pure_rerank_documents_best_chunk.parquet"),
            chunks_path=Path(config["inputs"]["pilot_reranked_chunks"]), canonical_path=Path(config["inputs"]["pilot_chunks"]),
            source_documents_path=Path(config["inputs"]["pilot_documents"]), output=temp, tokenizer=tokenizer,
        )
        for variant in selected:
            name = filenames[variant]
            root = work / variant
            second[variant] = _generate_submission(
                config, name=name, documents_path=root / "reranked_documents_best_chunk.parquet",
                chunks_path=root / "reranked_chunks.parquet", canonical_path=root / "combined_chunks.parquet",
                source_documents_path=root / "combined_documents.parquet", output=temp, tokenizer=tokenizer,
            )
    deterministic = all(
        variants[key]["json_sha256"] == second[key]["json_sha256"] and variants[key]["zip_sha256"] == second[key]["zip_sha256"]
        for key in variants
    )
    result = {
        "variants": variants,
        "selection": {
            "generated_source_variants": selected,
            "skipped_source_variants": [variant for variant in filenames if variant not in selected],
            "rule": f"queries_changed >= {config['submission']['minimum_changed_queries_for_source_submission']}",
        },
        "determinism": {"verified": deterministic, "method": "independent full regeneration"},
    }
    if not deterministic: raise RuntimeError("submission determinism verification failed")
    atomic_json(Path(config["outputs"]["artifact_directory"]) / "submission_manifest.json", result)
    return result


def build_summary(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    artifacts = Path(config["outputs"]["artifact_directory"])
    diagnostics = json.loads((artifacts / "local_diagnostics.json").read_text(encoding="utf-8"))
    submissions = json.loads((artifacts / "submission_manifest.json").read_text(encoding="utf-8"))
    shard_scoring = json.loads((artifacts / "shard_scoring_summary.json").read_text(encoding="utf-8"))
    sustained_benchmark = json.loads((artifacts / "fresh_scoring_sustained_2048.json").read_text(encoding="utf-8"))
    score_distribution = json.loads((artifacts / "score_distribution.json").read_text(encoding="utf-8"))
    unmatched_doc_ids = {
        int(value)
        for value in pq.read_table(config["inputs"]["pilot_documents"], columns=["doc_id"]).column(0).to_pylist()
    }
    for source in manifest["variants"].values():
        unmatched_doc_ids.update(
            int(value)
            for value in pq.read_table(source["paths"]["source_documents"], columns=["doc_id"]).column(0).to_pylist()
        )
    checked_doc_ids = len(unmatched_doc_ids)
    for batch in pq.ParquetFile(config["inputs"]["official_corpus"]).iter_batches(columns=["id"], batch_size=131072):
        unmatched_doc_ids.difference_update(int(value) for value in batch.column(0).to_pylist())
    rows = {}
    for variant, source in manifest["variants"].items():
        diag = diagnostics[variant]
        changed = int(diag["queries_changed"])
        meaningful = bool(diag["meaningful_output_change"])
        rows[variant] = {
            "source": source["domain"], "added_documents": source["added_documents"], "added_chunks": source["added_chunks"],
            "queries_changed": changed,
            **diag,
            "local_signal": (
                "MEANINGFUL_OUTPUT_CHANGE" if meaningful
                else "OUTPUT_CHANGED_BELOW_THRESHOLD" if changed
                else "NO_IMPACT_LOCALLY"
            ),
            "recommend_submit": meaningful,
            "submission": submissions["variants"].get(variant),
        }
    summary = {
        "label": "LOCAL SIGNAL PROBE — not relevance evaluation; QUALITY UNKNOWN UNTIL ORGANIZER SCORE",
        "interpretation_predeclared": config["interpretation"],
        "fixed_policy": config["submission"], "sources": rows,
        "baseline": submissions["variants"]["B0"],
        "official_doc_id_audit": {
            "checked_local_doc_ids": checked_doc_ids,
            "invalid_official_doc_ids": len(unmatched_doc_ids),
        },
        "scoring": {
            "sustained_benchmark": sustained_benchmark,
            "sharded_run": shard_scoring,
            "distribution": score_distribution,
        },
        "submission_order": sorted(
            [key for key, row in rows.items() if row["recommend_submit"]],
            key=lambda key: (-rows[key]["queries_changed"], key),
        ),
    }
    atomic_json(artifacts / "source_probe_summary.json", summary)
    return summary
