#!/usr/bin/env python3
"""Recover only Phase 13A top3-mean document rankings and package frozen P11C chunks."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.submission.common import file_sha256, write_deterministic_zip, write_submission_stream
from src.submission.validator import expected_query_ids, normalize_validation_result
from src.source_census.union_experiment import _iter_json_array, _official_id_audit, _stream_validate
from tools.phase13a_fusion_replay import (
    OUT, P11_JSON, P12_JSON, P12_ROOT, aggregate_documents, atomic_json,
    candidate_ownership_audit, frozen_scores, load_embedded_review, query_groups,
    rank_global, sha256, update_run_state,
)

METHOD = "top3_mean_global_raw"
RUN_DB = OUT / "top3_mean_recovery.sqlite"
RANKINGS = OUT / "top3_mean_full_rankings.parquet"
MANIFEST = OUT / "top3_mean_full_rankings_manifest.json"
AUDIT = OUT / "top3_mean_controlled_submission_audit.json"
SUBMISSION_NAME = "phase13a_TOP3MEAN_DOCS20_PHASE11C_CHUNKS40"
SUBMISSION_ROOT = ROOT / "submissions/00_READY_TO_UPLOAD"
SUBMISSION_JSON = SUBMISSION_ROOT / f"{SUBMISSION_NAME}.json"
SUBMISSION_ZIP = SUBMISSION_ROOT / f"{SUBMISSION_NAME}.zip"
OFFICIAL = ROOT / "data/raw/links_corpus_vi_en_zh.parquet"
QUERIES = ROOT / "data/raw/query.parquet"
CANONICAL = P12_ROOT / "round1/canonical_text.sqlite"


def review_expectations(path: Path) -> dict[int, list[dict]]:
    review = load_embedded_review(path)
    result = {}
    for query_id, methods in review.items():
        if METHOD not in methods:
            raise ValueError(f"review query {query_id} lacks {METHOD}")
        result[query_id] = methods[METHOD]
    if len(result) != 60 or sum(map(len, result.values())) != 600:
        raise ValueError("review fixture must contain exactly 60 queries / 600 positions")
    return result


def verify_review_prefix(query_id: int, ranked: list[dict], expected: dict[int, list[dict]]) -> int:
    rows = expected.get(query_id)
    if rows is None:
        return 0
    for index, stored in enumerate(rows):
        observed = ranked[index]
        if int(observed["doc_id"]) != int(stored["doc_id"]):
            raise ValueError(
                f"review mismatch query={query_id} position={index + 1}: "
                f"doc {observed['doc_id']} != {stored['doc_id']}"
            )
        if round(float(observed["score"]), 6) != float(stored["score"]):
            raise ValueError(
                f"review score mismatch query={query_id} position={index + 1}: "
                f"{observed['score']} != serialized {stored['score']}"
            )
    return len(rows)


def open_checkpoint() -> sqlite3.Connection:
    OUT.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(RUN_DB)
    connection.executescript(
        "PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;"
        "CREATE TABLE IF NOT EXISTS rankings("
        "query_id INTEGER NOT NULL,rank INTEGER NOT NULL,doc_id INTEGER NOT NULL,score REAL NOT NULL,"
        "PRIMARY KEY(query_id,rank),UNIQUE(query_id,doc_id));"
        "CREATE TABLE IF NOT EXISTS completed(query_id INTEGER PRIMARY KEY,review_positions INTEGER NOT NULL);"
    )
    return connection


def recover_rankings(run_state: Path | None) -> dict:
    started = time.perf_counter(); process = psutil.Process()
    expected_review = review_expectations(OUT / "review_interface.html")
    chunk_arms, canonical, ownership = candidate_ownership_audit()
    scores = frozen_scores(chunk_arms)
    connection = open_checkpoint()
    completed = {int(row[0]) for row in connection.execute("SELECT query_id FROM completed")}
    total_queries = len(expected_query_ids(QUERIES)); peak = process.memory_info().rss
    update_run_state(run_state, status="RUNNING", stage="RECOVER_TOP3", done=len(completed), total=total_queries,
                     checkpoint=str(RUN_DB), integrity="transactional-ok")
    try:
        for rows in query_groups(P12_ROOT / "rankings/FULL/reranked_chunks.parquet"):
            query_id = int(rows[0]["query_id"])
            if query_id in completed:
                continue
            prepared = []
            for row in rows:
                chunk_id = str(row["chunk_id"]); key = (query_id, chunk_id)
                if key not in scores:
                    raise ValueError(f"missing frozen score {key}")
                doc_id = int(row["doc_id"])
                if doc_id not in canonical:
                    raise ValueError(f"missing canonical document ownership: {doc_id}")
                url, domain = canonical[doc_id]
                prepared.append({**row, "score": scores[key], "candidate_arms": sorted(chunk_arms[chunk_id]),
                                 "canonical_url": url, "canonical_domain": domain})
            ranked = rank_global(aggregate_documents(prepared, "top3_mean"))[:20]
            if len(ranked) != 20 or len({int(row["doc_id"]) for row in ranked}) != 20:
                raise ValueError(f"query {query_id} does not have 20 unique ranked documents")
            checked = verify_review_prefix(query_id, ranked, expected_review)
            with connection:
                connection.executemany(
                    "INSERT INTO rankings VALUES (?,?,?,?)",
                    [(query_id, rank, int(row["doc_id"]), float(row["score"])) for rank, row in enumerate(ranked, 1)],
                )
                connection.execute("INSERT INTO completed VALUES (?,?)", (query_id, checked))
            completed.add(query_id); peak = max(peak, process.memory_info().rss)
            update_run_state(run_state, stage="RECOVER_TOP3", done=len(completed), total=total_queries,
                             checkpoint=str(RUN_DB), integrity="transactional-ok")
    finally:
        connection.close()
    return {
        "ownership": ownership, "frozen_score_rows": len(scores), "query_count": len(completed),
        "runtime_seconds": time.perf_counter() - started, "peak_rss_bytes": peak,
    }


def consolidate(recovery: dict) -> dict:
    connection = sqlite3.connect(f"file:{RUN_DB}?mode=ro", uri=True)
    rows = connection.execute("SELECT query_id,rank,doc_id,score FROM rankings ORDER BY query_id,rank").fetchall()
    query_count = int(connection.execute("SELECT COUNT(*) FROM completed").fetchone()[0])
    review_count = int(connection.execute("SELECT COALESCE(SUM(review_positions),0) FROM completed").fetchone()[0])
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0]); connection.close()
    if integrity != "ok" or query_count != 1200 or len(rows) != 24000 or review_count != 600:
        raise ValueError(f"ranking checkpoint incomplete: integrity={integrity} queries={query_count} rows={len(rows)} review={review_count}")
    if any(not math.isfinite(float(row[3])) for row in rows):
        raise ValueError("non-finite recovered document score")
    official = set()
    for batch in pq.ParquetFile(OFFICIAL).iter_batches(columns=["id"], batch_size=131072):
        official.update(map(int, batch.column(0).to_pylist()))
    missing = sorted({int(row[2]) for row in rows} - official)
    if missing:
        raise ValueError(f"recovered rankings contain non-official IDs: {missing[:20]}")
    table = pa.table({
        "query_id": pa.array([row[0] for row in rows], pa.int64()),
        "rank": pa.array([row[1] for row in rows], pa.int32()),
        "doc_id": pa.array([row[2] for row in rows], pa.int64()),
        "score": pa.array([row[3] for row in rows], pa.float64()),
    })
    temporary = RANKINGS.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="zstd"); os.replace(temporary, RANKINGS)
    source_ranking = P12_ROOT / "rankings/FULL/reranked_chunks.parquet"
    score_cache = P12_ROOT / "round1/source_scores.sqlite"
    manifest = {
        "method": METHOD, "aggregation": "aggregate_documents(rows, 'top3_mean')", "ranking": "rank_global",
        "source_universe": "frozen Phase 12 FULL candidate universe", "matches_phase12_candidate_universe": True,
        "query_count": query_count, "rows": len(rows), "documents_per_query": 20,
        "review_positions_verified": review_count, "review_score_tolerance": "round(score, 6) exact equality",
        "ranking_sha256": file_sha256(RANKINGS), "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "implementation": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": file_sha256(__file__)},
        "frozen_inputs": {
            "phase12_ranked_chunks": {"path": str(source_ranking), "sha256": file_sha256(source_ranking)},
            "phase12_score_cache": {"path": str(score_cache), "sha256": file_sha256(score_cache)},
            "phase11c_submission": {"path": str(P11_JSON), "sha256": sha256(P11_JSON)},
            "phase12_submission": {"path": str(P12_JSON), "sha256": sha256(P12_JSON)},
            "review_interface": {"path": str(OUT / 'review_interface.html'), "sha256": sha256(OUT / 'review_interface.html')},
        },
        "runtime_seconds": recovery["runtime_seconds"], "peak_rss_bytes": recovery["peak_rss_bytes"],
        "checkpoint_integrity": integrity, "new_model_inference": 0,
    }
    atomic_json(MANIFEST, manifest)
    return manifest


def submission_records() -> object:
    table = pq.read_table(RANKINGS).to_pylist()
    by_query: dict[int, list[int]] = {}
    for row in table:
        by_query.setdefault(int(row["query_id"]), []).append(int(row["doc_id"]))
    for control in _iter_json_array(P11_JSON):
        query_id = int(control["id"])
        docs = by_query.get(query_id)
        if docs is None or len(docs) != 20:
            raise ValueError(f"missing recovered ranking for query {query_id}")
        yield {"id": query_id, "relevant_docs": docs, "relevant_chunks": control["relevant_chunks"]}


def package(manifest: dict) -> dict:
    SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)
    write_submission_stream(SUBMISSION_JSON, submission_records()); write_deterministic_zip(SUBMISSION_JSON, SUBMISSION_ZIP)
    first = (file_sha256(SUBMISSION_JSON), file_sha256(SUBMISSION_ZIP))
    write_submission_stream(SUBMISSION_JSON, submission_records()); write_deterministic_zip(SUBMISSION_JSON, SUBMISSION_ZIP)
    second = (file_sha256(SUBMISSION_JSON), file_sha256(SUBMISSION_ZIP))
    if first != second:
        raise ValueError("controlled submission regeneration is not deterministic")
    query_ids = expected_query_ids(QUERIES)
    validation = normalize_validation_result(_stream_validate(
        SUBMISSION_JSON, SUBMISSION_ZIP, query_ids, CANONICAL, document_depth=20, chunk_depth=40,
    ))
    official = _official_id_audit(SUBMISSION_JSON, OFFICIAL)
    chunk_queries = 0; chunk_arrays_equal = 0
    for control, candidate in zip(_iter_json_array(P11_JSON), _iter_json_array(SUBMISSION_JSON), strict=True):
        chunk_queries += 1
        chunk_arrays_equal += control["relevant_chunks"] == candidate["relevant_chunks"]
    with zipfile.ZipFile(SUBMISSION_ZIP) as archive:
        bad_crc = archive.testzip(); names = archive.namelist()
    frozen = json.loads((OUT / "baseline_replay_audit.json").read_text(encoding="utf-8"))
    frozen_ok = (frozen["passed"] and frozen["phase11c"]["frozen_json_sha256"] == sha256(P11_JSON)
                 and frozen["phase12"]["frozen_json_sha256"] == sha256(P12_JSON))
    passed = bool(validation["passed"] and official["passed"] and chunk_queries == 1200
                  and chunk_arrays_equal == 1200 and bad_crc is None and names == [SUBMISSION_JSON.name] and frozen_ok)
    audit = {
        "status": "READY_FOR_LEADERBOARD" if passed else "NEEDS_AGENT", "passed": passed,
        "method": METHOD, "source_universe": "frozen Phase 12 FULL", "phase12_universe_match": True,
        "scientific_comparison": {
            "versus_phase12_documents": "same candidate universe and frozen scores; document aggregation changes to top3 mean",
            "versus_phase11c_documents": "confounded by Phase12 corpus expansion plus aggregation change",
            "chunks": "copied exactly from frozen Phase11C for every query",
        },
        "ranking_manifest": str(MANIFEST), "ranking_sha256": manifest["ranking_sha256"],
        "review_verification": {"queries": 60, "ordered_positions": 600, "matches": 600, "missing_queries": 0, "ambiguous_provenance": 0},
        "validation": validation, "official_id_audit": official,
        "phase11c_chunk_arrays": {"queries_compared": chunk_queries, "exact_matches": chunk_arrays_equal, "changed": chunk_queries - chunk_arrays_equal},
        "frozen_controls": {"passed": frozen_ok, "phase11c_sha256": sha256(P11_JSON), "phase12_sha256": sha256(P12_JSON)},
        "zip": {"path": str(SUBMISSION_ZIP), "size_bytes": SUBMISSION_ZIP.stat().st_size, "sha256": second[1], "crc_error": bad_crc, "names": names},
        "json": {"path": str(SUBMISSION_JSON), "size_bytes": SUBMISSION_JSON.stat().st_size, "sha256": second[0]},
        "deterministic_regeneration": True, "new_model_inference": 0,
    }
    atomic_json(AUDIT, audit)
    if not passed:
        raise ValueError(f"controlled submission audit failed: {audit}")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-state", type=Path); args = parser.parse_args()
    update_run_state(args.run_state, status="RUNNING", stage="PREPARE", done=0, total=3)
    recovery = recover_rankings(args.run_state)
    update_run_state(args.run_state, stage="CONSOLIDATE", done=1, total=3, checkpoint=str(RUN_DB))
    manifest = consolidate(recovery)
    update_run_state(args.run_state, stage="PACKAGE", done=2, total=3, checkpoint=str(RANKINGS))
    audit = package(manifest)
    update_run_state(args.run_state, status="READY_FOR_LEADERBOARD", stage="READY_FOR_LEADERBOARD", done=3, total=3,
                     checkpoint=str(AUDIT), integrity="passed", submission_zip=str(SUBMISSION_ZIP), zip_sha256=audit["zip"]["sha256"])
    print(json.dumps(audit, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
