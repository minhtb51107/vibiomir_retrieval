from __future__ import annotations

import gc
import json
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.acquisition_benchmark.sampling import deterministic_domain_samples
from src.indexing.embedder import normalize_text
from src.source_census.pipeline import atomic_json, score_database_status


DEPTH_GROUPS = {
    "G1A": ["medlatec.vn", "v.familydoctor.com.cn", "vinmec.com"],
    "G5A": ["benhviennhitrunguong.gov.vn", "zydcd.com", "hellobacsi.com"],
    "G6B": ["tiemchunglongchau.com.vn", "cancer.39.net", "suckhoedoisong.vn"],
}
SOURCES = sorted({source for values in DEPTH_GROUPS.values() for source in values})


def host(url: str) -> str:
    return (urlsplit(str(url)).hostname or "").lower().removeprefix("www.")


def _crawl_rows(database: Path, sources: set[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not database.exists():
        return result
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        for row in connection.execute("SELECT doc_id,original_url,status,http_status FROM crawl_results"):
            domain = host(str(row["original_url"]))
            if domain in sources:
                result[domain].append(dict(row))
    finally:
        connection.close()
    return result


def _usable_ids(path: Path, sources: set[str]) -> dict[str, set[int]]:
    result: dict[str, set[int]] = defaultdict(set)
    if not path.exists():
        return result
    for batch in pq.ParquetFile(path).iter_batches(
        columns=["doc_id", "original_url", "final_url", "extraction_status", "normalized_text"],
        batch_size=4096,
    ):
        for row in batch.to_pylist():
            domain = host(str(row.get("original_url") or row.get("final_url") or ""))
            if (
                domain in sources
                and str(row.get("extraction_status")) == "SUCCESS"
                and str(row.get("normalized_text") or "").strip()
            ):
                result[domain].add(int(row["doc_id"]))
    return result


def _source_worker_evidence(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    evidence: dict[str, dict[str, float]] = {}
    for path in Path(config["depth1000"]["existing_round1_worker_root"]).glob("*/summary.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            domains = value.get("run", {}).get("by_domain", {})
            if len(domains) != 1:
                continue
            source = next(iter(domains))
            completed = int(value["run"].get("completed", 0))
            retained = sum(item.stat().st_size for item in path.parent.rglob("*") if item.is_file())
            evidence[source] = {
                "completed": completed,
                "wall_seconds": float(value["run"].get("wall_seconds", 0.0)),
                "retained_bytes": retained,
            }
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return evidence


def prepare_depth1000_manifest(config: dict[str, Any]) -> dict[str, Any]:
    depth = config["depth1000"]
    artifact_path = Path(depth["manifest"])
    sources = set(SOURCES)
    crawl_sets = _crawl_rows(Path(depth["existing_round1_crawl_database"]), sources)
    c3_sets = _crawl_rows(Path(depth["existing_c3_crawl_database"]), sources)
    usable = _usable_ids(Path(depth["existing_round1_documents"]), sources)
    c3_usable = _usable_ids(Path(depth["existing_c3_documents"]), sources)
    for source, ids in c3_usable.items():
        usable[source].update(ids)

    samples, populations = deterministic_domain_samples(
        config["inputs"]["corpus"], domains=sources, sample_size=2200,
        seed=str(depth["sampling_seed"]),
    )
    evidence = _source_worker_evidence(config)
    source_rows: dict[str, Any] = {}
    total_incremental = 0
    projected_bytes = 0.0
    projected_seconds = 0.0
    for source in SOURCES:
        previous_rows = [*crawl_sets.get(source, []), *c3_sets.get(source, [])]
        previous_by_id = {int(row["doc_id"]): row for row in previous_rows}
        attempted_ids = set(previous_by_id)
        successful_ids = {
            doc_id for doc_id, row in previous_by_id.items() if str(row.get("status")) == "SUCCESS"
        }
        population = int(populations[source])
        target = min(1000, population)

        # Preserve the shallow experiment as a nested prefix.  If prior work
        # already exceeds the target (C3), choose a stable target-sized subset
        # of those attempted IDs and request nothing new.
        prior_rank = {
            int(row["doc_id"]): (int(row["rank"]), int(row["doc_id"]))
            for row in samples[source]
        }
        selected_existing = sorted(
            attempted_ids,
            key=lambda doc_id: prior_rank.get(doc_id, (2**65, doc_id)),
        )[:target]
        needed = max(0, target - len(selected_existing))
        incremental = [row for row in samples[source] if int(row["doc_id"]) not in attempted_ids][:needed]
        if len(incremental) != needed:
            raise ValueError(f"insufficient official IDs for {source}: {len(incremental)} != {needed}")
        target_ids = [*selected_existing, *(int(row["doc_id"]) for row in incremental)]
        if len(target_ids) != target or len(target_ids) != len(set(target_ids)):
            raise ValueError(f"invalid target set for {source}")

        observed = evidence.get(source, {})
        completed = int(observed.get("completed", 0))
        seconds_per_url = float(observed.get("wall_seconds", 0.0)) / completed if completed else 2.0
        bytes_per_url = float(observed.get("retained_bytes", 0.0)) / completed if completed else 250_000.0
        source_seconds = needed * seconds_per_url
        source_bytes = needed * bytes_per_url
        projected_seconds += source_seconds
        projected_bytes += source_bytes
        total_incremental += needed
        source_rows[source] = {
            "official_population": population,
            "previous_attempted_depth": len(attempted_ids),
            "previous_attempted_ids": sorted(attempted_ids),
            "previous_successful_depth": len(successful_ids),
            "previous_successful_ids": sorted(successful_ids),
            "previous_usable_documents": len(usable.get(source, set())),
            "previous_usable_ids": sorted(usable.get(source, set())),
            "target_depth": target,
            "target_ids": target_ids,
            "incremental_ids_required": needed,
            "incremental_records": incremental,
            "cannot_reach_1000": population < 1000,
            "projection": {
                "observed_completed": completed,
                "seconds_per_url": seconds_per_url,
                "retained_bytes_per_url": bytes_per_url,
                "incremental_source_seconds": source_seconds,
                "incremental_retained_bytes": source_bytes,
            },
        }
    free_bytes = shutil.disk_usage(Path.cwd()).free
    safety_floor_bytes = int(float(config["acquisition"]["minimum_free_gib"]) * 2**30)
    round2_path = Path(config["outputs"]["artifacts"]).parent / "round2_groups.json"
    round2 = json.loads(round2_path.read_text(encoding="utf-8"))
    source_metadata = round2["source_metadata"]
    projected_chunks = 0.0
    for source in SOURCES:
        metadata = source_metadata[source]
        usable_count = max(int(metadata.get("usable_documents", 0)), 1)
        chunks_per_usable = int(metadata.get("chunk_count", 0)) / usable_count
        expected_usable = source_rows[source]["target_depth"] * float(metadata.get("acquisition_yield", 1.0))
        if source == "suckhoedoisong.vn":
            c3_chunks = pq.ParquetFile(depth["existing_c3_chunks"]).metadata.num_rows
            c3_docs = max(pq.ParquetFile(depth["existing_c3_documents"]).metadata.num_rows, 1)
            chunks_per_usable = c3_chunks / c3_docs
            expected_usable = min(source_rows[source]["target_depth"], len(usable.get(source, set())))
        projected_chunks += expected_usable * chunks_per_usable
    round1_documents = Path(depth["existing_round1_documents"])
    round1_chunks = Path(depth["existing_round1_chunks"])
    document_bytes_per_row = round1_documents.stat().st_size / max(pq.ParquetFile(round1_documents).metadata.num_rows, 1)
    chunk_bytes_per_row = round1_chunks.stat().st_size / max(pq.ParquetFile(round1_chunks).metadata.num_rows, 1)
    embedding_bytes = projected_chunks * int(config["models"]["embedder"]["dimension"]) * 4
    document_chunk_bytes = (
        sum(row["target_depth"] for row in source_rows.values()) * document_bytes_per_row
        + projected_chunks * chunk_bytes_per_row
    ) * 3.0  # new partitions + canonical source data + combined/temporary copies
    old_score_path = Path(depth["existing_score_database"])
    old_connection = sqlite3.connect(f"file:{old_score_path}?mode=ro", uri=True)
    old_score_rows = int(old_connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
    old_connection.close()
    score_bytes = 9 * 1200 * 8 * (old_score_path.stat().st_size / max(old_score_rows, 1))
    downstream_bytes = embedding_bytes + document_chunk_bytes + score_bytes
    conservative_retained_bytes = (projected_bytes + downstream_bytes) * 1.35
    projected_rerank_seconds = (9 * 1200 * 8) / 20.0
    projected_embedding_seconds = projected_chunks / 35.0
    projected_processing_seconds = total_incremental * (908.628011 / 8173) + projected_chunks * (114.702867 / 21419)
    conservative_pipeline_seconds = (
        projected_seconds + projected_rerank_seconds + projected_embedding_seconds
        + projected_processing_seconds + 900.0
    )
    projected_free = free_bytes - conservative_retained_bytes
    result = {
        "format_version": 1,
        "created_for": "Phase 10D depth scaling from shallow (~100) to up to 1000 official IDs/source",
        "sampling_seed": depth["sampling_seed"],
        "sampling": "nested prior attempted IDs, then lowest SHA-256(seed:doc_id) ranks among unattempted official source URLs",
        "groups": DEPTH_GROUPS,
        "source_count": len(SOURCES),
        "total_incremental_ids": total_incremental,
        "free_disk_before_bytes": free_bytes,
        "free_disk_before_gib": free_bytes / 2**30,
        "projected_incremental_retained_bytes": projected_bytes,
        "projected_incremental_retained_gib": projected_bytes / 2**30,
        "projected_total_chunks": round(projected_chunks),
        "projected_embedding_bytes": round(embedding_bytes),
        "projected_downstream_bytes": round(downstream_bytes),
        "conservative_total_retained_bytes": round(conservative_retained_bytes),
        "conservative_total_retained_gib": conservative_retained_bytes / 2**30,
        "projected_source_serial_seconds": projected_seconds,
        "projected_source_serial_hours": projected_seconds / 3600,
        "conservative_end_to_end_seconds": conservative_pipeline_seconds,
        "conservative_end_to_end_hours": conservative_pipeline_seconds / 3600,
        "projected_free_disk_after_gib": projected_free / 2**30,
        "minimum_free_disk_gib": float(config["acquisition"]["minimum_free_gib"]),
        "disk_gate_passed": projected_free >= safety_floor_bytes,
        "sources": source_rows,
    }
    atomic_json(artifact_path, result)
    if not result["disk_gate_passed"]:
        raise RuntimeError("depth1000 projection violates the 20 GiB free-disk floor")
    return result


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    temporary.replace(path)


def assemble_depth_corpus(config: dict[str, Any]) -> dict[str, Any]:
    depth = config["depth1000"]
    manifest = json.loads(Path(depth["manifest"]).read_text(encoding="utf-8"))
    target_ids = {
        int(doc_id) for row in manifest["sources"].values() for doc_id in row["target_ids"]
    }
    document_paths = [
        Path(depth["existing_round1_documents"]), Path(depth["existing_c3_documents"]),
        Path(depth["new_documents"]),
    ]
    chunk_paths = [
        Path(depth["existing_round1_chunks"]), Path(depth["existing_c3_chunks"]),
        Path(depth["new_chunks"]),
    ]
    documents: dict[int, dict[str, Any]] = {}
    for path in document_paths:
        if not path.exists():
            continue
        for row in pq.read_table(path).to_pylist():
            doc_id = int(row["doc_id"])
            if doc_id in target_ids:
                documents.setdefault(doc_id, row)
    chunks: dict[str, dict[str, Any]] = {}
    for path in chunk_paths:
        if not path.exists():
            continue
        for row in pq.read_table(path).to_pylist():
            if int(row["doc_id"]) in target_ids:
                chunks.setdefault(str(row["chunk_id"]), row)
    document_rows = sorted(documents.values(), key=lambda row: int(row["doc_id"]))
    chunk_rows = sorted(chunks.values(), key=lambda row: (int(row["doc_id"]), int(row["chunk_index"]), str(row["chunk_id"])))
    _write_rows(Path(config["outputs"]["documents"]), document_rows)
    _write_rows(Path(config["outputs"]["chunks"]), chunk_rows)

    per_source: dict[str, Any] = {}
    for source in SOURCES:
        ids = set(int(value) for value in manifest["sources"][source]["target_ids"])
        source_docs = [row for row in document_rows if int(row["doc_id"]) in ids]
        source_chunks = [row for row in chunk_rows if int(row["doc_id"]) in ids]
        statuses = Counter(str(row.get("extraction_status") or "UNKNOWN") for row in source_docs)
        usable_ids = {
            int(row["doc_id"]) for row in source_docs
            if str(row.get("extraction_status")) == "SUCCESS" and str(row.get("normalized_text") or "").strip()
        }
        per_source[source] = {
            "target_depth": len(ids), "documents_present": len(source_docs),
            "usable_documents": len(usable_ids), "chunk_count": len(source_chunks),
            "status_counts": dict(sorted(statuses.items())),
        }
    triage = {
        "healthy_sources": SOURCES,
        "sources": {
            source: {
                "corpus_rows": manifest["sources"][source]["official_population"],
                "sampled": manifest["sources"][source]["target_depth"],
                "extracted_rows": per_source[source]["documents_present"],
                "usable_documents": per_source[source]["usable_documents"],
                "status_counts": per_source[source]["status_counts"],
                "likely_language": "zh" if source.endswith(".cn") or source in {"zydcd.com", "cancer.39.net", "jb39.com"} else "vi",
                "healthy": True,
                "technical_only": True,
            }
            for source in SOURCES
        },
    }
    atomic_json(Path(config["outputs"]["artifacts"]) / "technical_triage.json", triage)
    summary = {
        "target_official_ids": len(target_ids), "documents": len(document_rows),
        "chunks": len(chunk_rows), "sources": per_source,
    }
    atomic_json(Path(config["outputs"]["artifacts"]) / "corpus_assembly.json", summary)
    return summary


def assemble_depth_embeddings(config: dict[str, Any], run_state: Path | None = None) -> dict[str, Any]:
    chunks_path = Path(config["outputs"]["chunks"])
    chunks = pq.read_table(chunks_path, columns=["chunk_id", "normalized_text"]).to_pylist()
    output = Path(config["outputs"]["root"]) / "round1/chunk_embeddings.f32"
    output.parent.mkdir(parents=True, exist_ok=True)
    dimension = int(config["models"]["embedder"]["dimension"])
    expected = len(chunks) * dimension * 4
    state_path = Path(config["outputs"]["artifacts"]) / "embedding_summary.json"
    if output.exists() and output.stat().st_size == expected and state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))

    old_chunks = pq.read_table(config["depth1000"]["existing_round1_chunks"], columns=["chunk_id"]).column(0).to_pylist()
    old_vectors = np.memmap(
        config["depth1000"]["existing_round1_embeddings"], dtype=np.float32, mode="r",
        shape=(len(old_chunks), dimension),
    )
    old_lookup = {str(chunk_id): index for index, chunk_id in enumerate(old_chunks)}
    temporary = output.with_suffix(".f32.tmp")
    vectors = np.memmap(temporary, dtype=np.float32, mode="w+", shape=(len(chunks), dimension))
    missing: list[int] = []
    reused = 0
    for index, row in enumerate(chunks):
        old_index = old_lookup.get(str(row["chunk_id"]))
        if old_index is None:
            missing.append(index)
        else:
            vectors[index] = old_vectors[old_index]
            reused += 1
    del old_vectors, old_lookup

    started = time.perf_counter()
    if missing:
        import torch
        from sentence_transformers import SentenceTransformer

        model_cfg = config["models"]["embedder"]
        torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
        model = SentenceTransformer(model_cfg["name"], revision=model_cfg["revision"], device="cpu")
        model.max_seq_length = int(model_cfg["max_length"]); model.half(); model.to("cuda")
        block = 256
        for offset in range(0, len(missing), block):
            indices = missing[offset:offset + block]
            encoded = model.encode(
                [normalize_text(str(chunks[index]["normalized_text"])) for index in indices],
                batch_size=int(model_cfg["batch_size"]), show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            )
            vectors[indices] = np.asarray(encoded, dtype=np.float32)
            vectors.flush()
            if run_state is not None:
                atomic_json(run_state, {
                    "stage": "EMBEDDINGS", "status": "RUNNING", "total": len(missing),
                    "done": min(offset + len(indices), len(missing)), "reused_embeddings": reused,
                    "last_successful_checkpoint": min(offset + len(indices), len(missing)),
                })
        del model
        torch.cuda.empty_cache()
    vectors.flush(); del vectors
    temporary.replace(output)
    result = {
        "chunks": len(chunks), "dimension": dimension, "reused_embeddings": reused,
        "new_embeddings": len(missing), "runtime_seconds": time.perf_counter() - started,
        "path": str(output),
    }
    atomic_json(state_path, result)
    gc.collect()
    return result


def seed_exact_scores(destination_path: str | Path, old_path: str | Path) -> dict[str, Any]:
    """Reuse scores only for exact (query_id, chunk_id) keys."""
    destination_path = Path(destination_path)
    old_path = Path(old_path)
    connection = sqlite3.connect(destination_path)
    before = int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0])
    connection.execute("ATTACH DATABASE ? AS old", (str(old_path.resolve()),))
    expected_matches = int(connection.execute(
        "SELECT COUNT(*) FROM main.pairs AS dst JOIN old.pairs AS src "
        "ON src.query_id=dst.query_id AND src.chunk_id=dst.chunk_id "
        "WHERE src.rerank_score IS NOT NULL"
    ).fetchone()[0])
    with connection:
        connection.execute(
            "UPDATE main.pairs AS dst "
            "SET rerank_score=src.rerank_score, inference_ms=src.inference_ms "
            "FROM old.pairs AS src "
            "WHERE src.query_id=dst.query_id AND src.chunk_id=dst.chunk_id "
            "AND src.rerank_score IS NOT NULL AND dst.rerank_score IS NULL"
        )
    after = int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0])
    mismatched = int(connection.execute(
        "SELECT COUNT(*) FROM main.pairs AS dst JOIN old.pairs AS src "
        "ON src.query_id=dst.query_id AND src.chunk_id=dst.chunk_id "
        "WHERE src.rerank_score IS NOT NULL AND (dst.rerank_score IS NULL "
        "OR dst.rerank_score!=src.rerank_score OR dst.inference_ms!=src.inference_ms)"
    ).fetchone()[0])
    incorrectly_seeded = int(connection.execute(
        "SELECT COUNT(*) FROM main.pairs AS dst WHERE dst.rerank_score IS NOT NULL "
        "AND NOT EXISTS (SELECT 1 FROM old.pairs AS src WHERE src.query_id=dst.query_id "
        "AND src.chunk_id=dst.chunk_id AND src.rerank_score IS NOT NULL)"
    ).fetchone()[0])
    connection.execute("DETACH DATABASE old")
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    total = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
    distinct_scores = int(connection.execute(
        "SELECT COUNT(DISTINCT rerank_score) FROM pairs WHERE rerank_score IS NOT NULL"
    ).fetchone()[0])
    connection.close()
    result = {
        "total": total, "before": before, "expected_exact_matches": expected_matches,
        "reused": after - before, "after": after, "missing": total - after,
        "exact_match_mismatches": mismatched, "incorrectly_seeded_nonmatches": incorrectly_seeded,
        "distinct_seeded_scores": distinct_scores, "integrity": integrity,
    }
    if after - before != expected_matches or mismatched or incorrectly_seeded or integrity != "ok":
        raise RuntimeError(f"exact score seed invariant failed: {result}")
    return result


def seed_depth_scores(config: dict[str, Any]) -> dict[str, Any]:
    destination_path = Path(config["outputs"]["root"]) / "round1/source_scores.sqlite"
    old_path = Path(config["depth1000"]["existing_score_database"])
    result = seed_exact_scores(destination_path, old_path)
    atomic_json(Path(config["outputs"]["artifacts"]) / "rerank_seed_summary.json", result)
    return result


def finalize_depth_report(config: dict[str, Any], submissions: dict[str, Any]) -> dict[str, Any]:
    depth = config["depth1000"]
    manifest = json.loads(Path(depth["manifest"]).read_text(encoding="utf-8"))
    assembly = json.loads((Path(config["outputs"]["artifacts"]) / "corpus_assembly.json").read_text(encoding="utf-8"))
    candidates = json.loads((Path(config["outputs"]["artifacts"]) / "source_candidate_summary.json").read_text(encoding="utf-8"))
    seed = json.loads((Path(config["outputs"]["artifacts"]) / "rerank_seed_summary.json").read_text(encoding="utf-8"))
    crawl_summary_path = Path(config["outputs"]["artifacts"]) / "parallel_acquisition.json"
    crawl = json.loads(crawl_summary_path.read_text(encoding="utf-8")) if crawl_summary_path.exists() else {}
    extraction_summary_path = Path(config["outputs"]["artifacts"]) / "extract.json"
    extraction = (
        json.loads(extraction_summary_path.read_text(encoding="utf-8"))
        if extraction_summary_path.exists() else {}
    )
    status = score_database_status(config, verify_integrity=True)
    source_report = {}
    for source in SOURCES:
        source_manifest = manifest["sources"][source]
        final = assembly["sources"][source]
        source_report[source] = {
            "official_population": source_manifest["official_population"],
            "previous_attempted_depth": source_manifest["previous_attempted_depth"],
            "previous_successful_depth": source_manifest["previous_successful_depth"],
            "previous_usable_documents": source_manifest["previous_usable_documents"],
            "target_depth": source_manifest["target_depth"],
            "actual_attempted_incremental_ids": source_manifest["incremental_ids_required"],
            "final_usable_documents": final["usable_documents"],
            "final_chunk_count": final["chunk_count"],
        }
    report = {
        "terminal_state": "WAITING_FOR_LEADERBOARD",
        "scientific_variable": "source acquisition depth only",
        "groups": DEPTH_GROUPS,
        "manifest": depth["manifest"],
        "sources": source_report,
        "crawl": crawl,
        "extraction_status_counts": extraction.get("status_counts", {}),
        "extraction_failures_preserved": int(
            extraction.get("status_counts", {}).get("EXTRACTION_FAILED", 0)
        ),
        "candidate_summary": candidates,
        "reranker_cache": status,
        "cached_rerank_pairs_reused": seed["reused"],
        "new_rerank_pairs_scored": status["done"] - seed["after"],
        "submissions": submissions,
    }
    atomic_json(Path(config["outputs"]["artifacts"]).parent / "depth1000_report.json", report)
    return report
