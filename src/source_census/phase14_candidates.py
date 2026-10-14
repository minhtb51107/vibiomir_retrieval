from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.indexing.phase14_embeddings import audit_complete, file_sha256
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.lexical_tokenizer import tokenize_lexical
from src.runtime.state_file import update_json, write_json


SOURCE = "medlatec.vn"
DIMENSION = 1024
DENSE_BLOCK_ROWS = 2048
INDEX_BATCH_ROWS = 512


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the dedicated Phase 14 contract without weakening Phase 10 loaders."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("phase") != "phase14_medlatec_coverage":
        raise ValueError("unexpected dedicated Phase 14 candidate configuration")
    policy = config["candidate_cache"]
    expected = {
        "candidate_caps": [16], "dense_depth": 16, "sparse_depth": 16,
        "rrf_constant": 60, "bm25_k1": 1.2, "bm25_b": 0.75,
    }
    if policy != expected:
        raise ValueError(f"Phase 14 candidate policy mismatch: {policy}")
    return config


def _chunk_order_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for batch in pq.ParquetFile(path).iter_batches(columns=["chunk_id"], batch_size=8192):
        for chunk_id in batch.column(0).to_pylist():
            digest.update(str(chunk_id).encode("utf-8")); digest.update(b"\n")
            count += 1
    return count, digest.hexdigest()


def audit_embeddings(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["outputs"]["root"])
    checkpoint_path = root / "round1/chunk_embeddings.streaming.checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    audit = audit_complete(config, checkpoint)
    rows, order_hash = _chunk_order_sha256(Path(config["outputs"]["chunks"]))
    audit.update({
        "checkpoint_complete": bool(checkpoint.get("complete")),
        "checkpoint_completed_rows": int(checkpoint.get("completed_rows", -1)),
        "chunk_id_order_sha256": order_hash,
        "chunk_id_order_rows": rows,
        "checkpoint_path": str(checkpoint_path),
        "passed": bool(
            audit["passed"] and checkpoint.get("complete")
            and int(checkpoint.get("completed_rows", -1)) == rows
            and int(audit["completed_bitmap_rows"]) == rows
        ),
    })
    write_json(Path(config["outputs"]["artifacts"]) / "embedding_candidate_gate.json", audit)
    return audit


def _signature(config: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "old_chunks": Path(config["inputs"]["old_medlatec_chunks"]),
        "new_chunks": Path(config["outputs"]["chunks"]),
        "old_embeddings": Path(config["inputs"]["old_medlatec_embeddings"]),
        "new_embeddings": Path(config["outputs"]["root"]) / "round1/chunk_embeddings.f32",
        "queries": Path(config["inputs"]["queries"]),
        "query_embeddings": Path("data/source_relevance_probe/query_embeddings.f32"),
    }
    return {
        "format_version": 1,
        "run_id": config["run_id"],
        "source": SOURCE,
        "policy": config["candidate_cache"],
        "inputs": {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()},
        "union_order": ["old_medlatec_chunks", "phase14_new_chunks"],
    }


def candidate_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["outputs"]["root"])
    round1 = root / "round1"
    parts = round1 / "source_candidate_parts"
    parts.mkdir(parents=True, exist_ok=True)
    return {
        "database": round1 / "phase14_candidate_retrieval.sqlite",
        "dense": round1 / "phase14_dense_top16.npz",
        "part": parts / "medlatec_vn.parquet",
        "marker": parts / "medlatec_vn.json",
        "summary": Path(config["outputs"]["artifacts"]) / "source_candidate_summary.json",
        "noise": Path(config["outputs"]["artifacts"]) / "candidate_boilerplate_impact.json",
        "checkpoint": Path(config["outputs"]["artifacts"]) / "candidate_retrieval_checkpoint.json",
    }


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS chunks(
          row_index INTEGER PRIMARY KEY, corpus_part TEXT NOT NULL,
          local_index INTEGER NOT NULL, chunk_id TEXT NOT NULL UNIQUE,
          doc_id INTEGER NOT NULL, token_length INTEGER NOT NULL,
          suspicious INTEGER NOT NULL CHECK(suspicious IN (0,1))
        );
        CREATE TABLE IF NOT EXISTS postings(
          term TEXT NOT NULL, row_index INTEGER NOT NULL, term_frequency INTEGER NOT NULL,
          PRIMARY KEY(term,row_index)
        );
        CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term);
        CREATE TABLE IF NOT EXISTS sparse_top(
          query_position INTEGER NOT NULL, rank INTEGER NOT NULL,
          row_index INTEGER NOT NULL, score REAL NOT NULL,
          PRIMARY KEY(query_position,rank), UNIQUE(query_position,row_index)
        );
        """
    )
    return connection


def _meta(connection: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return str(row[0]) if row else default


def _set_meta(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value, ensure_ascii=False, sort_keys=True)),
    )


def _queries(config: dict[str, Any]) -> list[dict[str, Any]]:
    return pq.read_table(config["inputs"]["queries"], columns=["id", "query"]).to_pylist()


def _risk_texts(new_chunks: Path) -> set[str]:
    """Reproduce the audited duplicate-only navigation/boilerplate risk set."""
    groups: dict[str, dict[str, Any]] = {}
    for batch in pq.ParquetFile(new_chunks).iter_batches(
        columns=["raw_text", "heading_path"], batch_size=2048
    ):
        for row in batch.to_pylist():
            text = str(row["raw_text"])
            item = groups.setdefault(text, {"count": 0, "related": False})
            item["count"] += 1
            item["related"] = item["related"] or text.lower().strip() == "tin cùng chuyên mục" or any(
                "tin cùng chuyên mục" in str(value).lower()
                for value in (row.get("heading_path") or [])
            )
    markers = (
        "lựa chọn dịch vụ", "đặt lịch thăm khám", "lấy mẫu xét nghiệm tại nhà",
        "đăng ký khám và tư vấn", "bình luận (", "liên hệ ngay với số hotline",
        "quý khách hàng vui lòng",
    )
    return {
        text for text, item in groups.items()
        if item["count"] > 1 and (
            item["related"] or any(marker in text.lower() for marker in markers)
        )
    }


def _iter_union(config: dict[str, Any], columns: list[str]) -> Iterable[tuple[str, int, dict[str, Any]]]:
    paths = (
        ("old", Path(config["inputs"]["old_medlatec_chunks"])),
        ("new", Path(config["outputs"]["chunks"])),
    )
    for part, path in paths:
        local = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=INDEX_BATCH_ROWS):
            for row in batch.to_pylist():
                yield part, local, row
                local += 1


def build_sparse_index(
    config: dict[str, Any], connection: sqlite3.Connection, state_path: Path, signature: dict[str, Any]
) -> int:
    queries = _queries(config)
    query_terms = set()
    for row in queries:
        query_terms.update(tokenize_lexical(str(row["query"])))
    new_path = Path(config["outputs"]["chunks"])
    risk_texts = _risk_texts(new_path)
    completed = int(json.loads(_meta(connection, "indexed_rows", "0")))
    global_index = 0
    pending_chunks: list[tuple[Any, ...]] = []
    pending_postings: list[tuple[Any, ...]] = []
    started = time.perf_counter()
    columns = ["chunk_id", "doc_id", "normalized_text", "raw_text"]
    for part, local, row in _iter_union(config, columns):
        if global_index < completed:
            global_index += 1
            continue
        tokens = tokenize_lexical(str(row["normalized_text"]))
        counts = Counter(token for token in tokens if token in query_terms)
        suspicious = int(part == "new" and str(row["raw_text"]) in risk_texts)
        pending_chunks.append((global_index, part, local, str(row["chunk_id"]), int(row["doc_id"]), len(tokens), suspicious))
        pending_postings.extend((term, global_index, int(tf)) for term, tf in counts.items())
        global_index += 1
        if len(pending_chunks) >= INDEX_BATCH_ROWS:
            with connection:
                connection.executemany("INSERT INTO chunks VALUES(?,?,?,?,?,?,?)", pending_chunks)
                connection.executemany("INSERT INTO postings VALUES(?,?,?)", pending_postings)
                _set_meta(connection, "indexed_rows", global_index)
                _set_meta(connection, "signature", signature)
            pending_chunks.clear(); pending_postings.clear()
            update_json(state_path, {
                "status": "RUNNING", "stage": "CANDIDATE_INDEX",
                "done": global_index, "total": union_size(config),
                "checkpoint": str(candidate_paths(config)["database"]),
                "rows_per_second": round((global_index - completed) / max(time.perf_counter() - started, 1e-9), 2),
            }, preserve_terminal=False)
    if pending_chunks:
        with connection:
            connection.executemany("INSERT INTO chunks VALUES(?,?,?,?,?,?,?)", pending_chunks)
            connection.executemany("INSERT INTO postings VALUES(?,?,?)", pending_postings)
            _set_meta(connection, "indexed_rows", global_index)
            _set_meta(connection, "signature", signature)
    with connection:
        _set_meta(connection, "index_complete", True)
    return global_index


def union_size(config: dict[str, Any]) -> int:
    return sum(
        int(pq.ParquetFile(path).metadata.num_rows)
        for path in (config["inputs"]["old_medlatec_chunks"], config["outputs"]["chunks"])
    )


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_dense(config: dict[str, Any], state_path: Path, signature_hash: str) -> tuple[np.ndarray, np.ndarray]:
    paths = candidate_paths(config)
    queries = _queries(config)
    query_vectors = np.memmap(
        "data/source_relevance_probe/query_embeddings.f32", dtype=np.float32,
        mode="r", shape=(len(queries), DIMENSION),
    )
    total = union_size(config); top_k = int(config["candidate_cache"]["dense_depth"])
    best_scores = np.full((len(queries), top_k), -np.inf, dtype=np.float32)
    best_indices = np.full((len(queries), top_k), -1, dtype=np.int64)
    next_row = 0
    if paths["dense"].exists():
        saved = np.load(paths["dense"], allow_pickle=False)
        if str(saved["signature_hash"].item()) != signature_hash:
            raise ValueError("dense checkpoint signature mismatch")
        best_scores = saved["scores"]; best_indices = saved["indices"]
        next_row = int(saved["next_row"].item())
    old_rows = int(pq.ParquetFile(config["inputs"]["old_medlatec_chunks"]).metadata.num_rows)
    sources = (
        (0, old_rows, Path(config["inputs"]["old_medlatec_embeddings"])),
        (old_rows, total - old_rows, Path(config["outputs"]["root"]) / "round1/chunk_embeddings.f32"),
    )
    started = time.perf_counter(); initial = next_row
    for base, count, path in sources:
        vectors = np.memmap(path, dtype=np.float32, mode="r", shape=(count, DIMENSION))
        local_start = max(0, next_row - base)
        for start in range(local_start, count, DENSE_BLOCK_ROWS):
            stop = min(start + DENSE_BLOCK_ROWS, count)
            block = np.asarray(vectors[start:stop], dtype=np.float32)
            scores = np.asarray(query_vectors @ block.T, dtype=np.float32)
            block_indices = np.arange(base + start, base + stop, dtype=np.int64)
            for qpos in range(len(queries)):
                merged_scores = np.concatenate((best_scores[qpos], scores[qpos]))
                merged_indices = np.concatenate((best_indices[qpos], block_indices))
                order = sorted(
                    range(merged_scores.size),
                    key=lambda pos: (-float(merged_scores[pos]), int(merged_indices[pos])),
                )[:top_k]
                best_scores[qpos] = merged_scores[order]
                best_indices[qpos] = merged_indices[order]
            next_row = base + stop
            _atomic_npz(
                paths["dense"], scores=best_scores, indices=best_indices,
                next_row=np.asarray(next_row), signature_hash=np.asarray(signature_hash),
            )
            update_json(state_path, {
                "status": "RUNNING", "stage": "CANDIDATE_DENSE", "done": next_row,
                "total": total, "checkpoint": str(paths["dense"]),
                "rows_per_second": round((next_row - initial) / max(time.perf_counter() - started, 1e-9), 2),
            }, preserve_terminal=False)
        del vectors
    return best_indices, best_scores


def build_sparse(
    config: dict[str, Any], connection: sqlite3.Connection, state_path: Path
) -> None:
    queries = _queries(config); total = union_size(config)
    lengths = np.empty(total, dtype=np.int32)
    for row_index, length in connection.execute("SELECT row_index,token_length FROM chunks ORDER BY row_index"):
        lengths[int(row_index)] = int(length)
    average = float(lengths.mean()); k1 = float(config["candidate_cache"]["bm25_k1"])
    b = float(config["candidate_cache"]["bm25_b"]); depth = int(config["candidate_cache"]["sparse_depth"])
    completed = int(json.loads(_meta(connection, "sparse_queries", "0")))
    started = time.perf_counter()
    for qpos in range(completed, len(queries)):
        scores: dict[int, float] = {}
        for term in set(tokenize_lexical(str(queries[qpos]["query"]))):
            rows = connection.execute(
                "SELECT row_index,term_frequency FROM postings WHERE term=? ORDER BY row_index", (term,)
            ).fetchall()
            df = len(rows)
            if not df:
                continue
            ident = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            for row_index, tf in rows:
                index = int(row_index); frequency = int(tf)
                denominator = frequency + k1 * (1.0 - b + b * lengths[index] / max(average, 1e-9))
                scores[index] = scores.get(index, 0.0) + ident * frequency * (k1 + 1.0) / denominator
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:depth]
        with connection:
            connection.execute("DELETE FROM sparse_top WHERE query_position=?", (qpos,))
            connection.executemany(
                "INSERT INTO sparse_top(query_position,rank,row_index,score) VALUES(?,?,?,?)",
                ((qpos, rank, index, score) for rank, (index, score) in enumerate(ranked, 1)),
            )
            _set_meta(connection, "sparse_queries", qpos + 1)
        update_json(state_path, {
            "status": "RUNNING", "stage": "CANDIDATE_SPARSE", "done": qpos + 1,
            "total": len(queries), "checkpoint": str(candidate_paths(config)["database"]),
            "queries_per_second": round((qpos + 1 - completed) / max(time.perf_counter() - started, 1e-9), 3),
        }, preserve_terminal=False)


def _canonical_winners(config: dict[str, Any], selected: set[int]) -> dict[int, dict[str, Any]]:
    columns = [
        "chunk_id", "doc_id", "raw_text", "source_url", "chunk_index", "start_offset",
        "end_offset", "section_type", "heading_path", "extraction_method",
    ]
    result: dict[int, dict[str, Any]] = {}
    global_index = 0
    for _, _, row in _iter_union(config, columns):
        if global_index in selected:
            result[global_index] = row
        global_index += 1
    if set(result) != selected:
        raise ValueError(f"canonical winner closure incomplete: missing={len(selected-set(result))}")
    return result


def finalize(
    config: dict[str, Any], connection: sqlite3.Connection, dense_indices: np.ndarray,
    dense_scores: np.ndarray, state_path: Path, started: float,
) -> dict[str, Any]:
    queries = _queries(config); policy = config["candidate_cache"]
    chunk_rows = {
        int(row[0]): {"chunk_id": str(row[1]), "doc_id": int(row[2]), "suspicious": bool(row[3])}
        for row in connection.execute("SELECT row_index,chunk_id,doc_id,suspicious FROM chunks")
    }
    fused_by_query: list[list[dict[str, Any]]] = []
    selected: set[int] = set()
    for qpos in range(len(queries)):
        dense = [
            {"chunk_id": chunk_rows[int(index)]["chunk_id"], "row_index": int(index), "score": float(dense_scores[qpos, rank - 1]), "rank": rank}
            for rank, index in enumerate(dense_indices[qpos], 1)
        ]
        sparse = [
            {"chunk_id": chunk_rows[int(index)]["chunk_id"], "row_index": int(index), "score": float(score), "rank": int(rank)}
            for rank, index, score in connection.execute(
                "SELECT rank,row_index,score FROM sparse_top WHERE query_position=? ORDER BY rank", (qpos,)
            )
        ]
        fused = reciprocal_rank_fusion(
            [dense, sparse], rrf_constant=int(policy["rrf_constant"]),
            top_k=int(policy["candidate_caps"][0]),
        )
        for item in fused:
            # RRF preserves row_index from the first arm that introduced the chunk.
            selected.add(int(item["row_index"]))
        fused_by_query.append(fused)
    canonical = _canonical_winners(config, selected)
    output: list[dict[str, Any]] = []
    suspicious_rows = 0; affected_queries = 0; suspicious_docs: set[int] = set()
    rank_counts: Counter[int] = Counter()
    for qpos, fused in enumerate(fused_by_query):
        query_risk = False
        for item in fused:
            index = int(item["row_index"]); row = canonical[index]
            risky = bool(chunk_rows[index]["suspicious"])
            if risky:
                suspicious_rows += 1; query_risk = True; suspicious_docs.add(int(row["doc_id"])); rank_counts[int(item["rank"])] += 1
            output.append({
                "query_id": int(queries[qpos]["id"]), "query_text": str(queries[qpos]["query"]),
                "source": SOURCE, "source_rank": int(item["rank"]), "source_score": float(item["score"]),
                "chunk_id": str(row["chunk_id"]), "doc_id": int(row["doc_id"]),
                "chunk_text": str(row["raw_text"]), "source_url": str(row["source_url"]),
                "chunk_index": int(row["chunk_index"]), "start_offset": int(row["start_offset"]),
                "end_offset": int(row["end_offset"]), "section_type": str(row["section_type"]),
                "heading_path": row.get("heading_path"), "extraction_method": str(row["extraction_method"]),
            })
        affected_queries += int(query_risk)
    paths = candidate_paths(config)
    temporary = paths["part"].with_name(f".{paths['part'].name}.tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(output), temporary); os.replace(temporary, paths["part"])
    old_rows = int(pq.ParquetFile(config["inputs"]["old_medlatec_chunks"]).metadata.num_rows)
    new_candidate_rows = sum(int(item["row_index"]) >= old_rows for fused in fused_by_query for item in fused)
    new_candidate_chunks = {
        str(item["chunk_id"]) for fused in fused_by_query for item in fused if int(item["row_index"]) >= old_rows
    }
    marker = {
        "source": SOURCE, "query_count": len(queries), "chosen_m": int(policy["candidate_caps"][0]),
        "source_chunk_count": union_size(config), "candidate_rows": len(output),
        "candidate_part_sha256": file_sha256(paths["part"]),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(paths["marker"], marker)
    risk_total = int(connection.execute("SELECT COUNT(*) FROM chunks WHERE suspicious=1").fetchone()[0])
    noise = {
        "dataset_modified": False, "risk_chunks_in_phase14_corpus": risk_total,
        "expected_risk_chunks": 32762, "risk_count_matches_prior_audit": risk_total == 32762,
        "candidate_rows_flagged": suspicious_rows, "candidate_row_fraction": suspicious_rows / max(len(output), 1),
        "queries_affected": affected_queries, "unique_flagged_documents": len(suspicious_docs),
        "flagged_candidate_rank_distribution": {str(key): value for key, value in sorted(rank_counts.items())},
        "interpretation": "Diagnostic flag only; no chunk was deleted, modified, penalized, or reranked.",
    }
    write_json(paths["noise"], noise)
    summary = {
        **marker, "passed": len(output) == len(queries) * int(policy["candidate_caps"][0]),
        "old_source_chunks": old_rows, "new_source_chunks": union_size(config) - old_rows,
        "union_source_chunks": union_size(config), "new_phase14_candidate_rows": new_candidate_rows,
        "unique_new_phase14_candidate_chunks": len(new_candidate_chunks),
        "candidate_path": str(paths["part"]), "boilerplate_impact": str(paths["noise"]),
        "next_stage": "RERANK_SCORE_RECONCILIATION",
    }
    write_json(paths["summary"], summary)
    write_json(paths["checkpoint"], {"stage": "SOURCE_CANDIDATES_COMPLETE", "complete": True, **summary})
    update_json(state_path, {
        "status": "READY_FOR_NEXT_STAGE", "stage": "SOURCE_CANDIDATES_COMPLETE",
        "done": len(output), "total": len(output), "remaining": 0,
        "checkpoint": str(paths["checkpoint"]), "candidate_summary": str(paths["summary"]),
        "candidate_rows": len(output), "next_stage": "RERANK_SCORE_RECONCILIATION",
    }, preserve_terminal=False)
    return summary


def run(config: dict[str, Any], state_path: Path) -> dict[str, Any]:
    started = time.perf_counter(); gate = audit_embeddings(config)
    if not gate["passed"]:
        raise RuntimeError(f"Phase 14 embedding candidate gate failed: {gate}")
    paths = candidate_paths(config); signature = _signature(config)
    signature_hash = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    connection = _connect(paths["database"])
    try:
        prior = _meta(connection, "signature")
        if prior is not None and json.loads(prior) != signature:
            raise ValueError("Phase 14 candidate checkpoint signature mismatch")
        indexed = build_sparse_index(config, connection, state_path, signature)
        if indexed != union_size(config):
            raise RuntimeError("Phase 14 candidate index incomplete")
        dense_indices, dense_scores = build_dense(config, state_path, signature_hash)
        build_sparse(config, connection, state_path)
        return finalize(config, connection, dense_indices, dense_scores, state_path, started)
    finally:
        connection.close()
