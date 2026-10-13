#!/usr/bin/env python3
"""M2A CPU-only PubMed lexical index, query bridge and hybrid candidate audit."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.pubmed_lexical import (
    ENGLISH_STOPWORDS, add_documents, bridge_terms, completed_ordinal, english_tokens,
    initialize_index, lexical_query, reciprocal_rank_union, search,
)
from src.evaluation.system_registry import query_groups
from src.runtime.state_file import update_json
from src.submission.common import atomic_write_json, file_sha256

CONFIG_PATH = ROOT / "configs/system_upgrade_m2a.yaml"


def percentile(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"min": float(array.min()), "p5": float(np.percentile(array, 5)),
            "median": float(np.median(array)), "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def state(path: Path | None, **updates) -> None:
    if path:
        update_json(path, {**updates, "worker_updated_at": dt.datetime.now(dt.timezone.utc).isoformat()})


def rows_from_pubmed(path: Path, start: int = 0, limit: int | None = None):
    ordinal = 0
    columns = ["doc_id", "original_url", "pmid", "title", "abstract_text"]
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=256):
        for row in batch.to_pylist():
            if ordinal >= start and (limit is None or ordinal < start + limit):
                yield {"ordinal": ordinal, "doc_id": int(row["doc_id"]), "official_url": str(row["original_url"]),
                       "pmid": int(row["pmid"]), "title": str(row.get("title") or ""),
                       "abstract": str(row.get("abstract_text") or "")}
            ordinal += 1
            if limit is not None and ordinal >= start + limit:
                return


def build_rows(connection: sqlite3.Connection, source: Path, start: int, total: int, batch_rows: int,
               stop_ordinal: int | None = None,
               state_path: Path | None = None) -> tuple[int, int]:
    pending = []; inserted = 0; peak = psutil.Process().memory_info().rss
    for row in rows_from_pubmed(source, start=start):
        if stop_ordinal is not None and int(row["ordinal"]) >= stop_ordinal:
            break
        pending.append(row)
        if len(pending) < batch_rows:
            continue
        inserted += add_documents(connection, pending); pending.clear()
        done = completed_ordinal(connection); peak = max(peak, psutil.Process().memory_info().rss)
        if done % 2048 < batch_rows:
            state(state_path, status="RUNNING", stage="BUILD_LEXICAL_INDEX", done=done, total=total,
                  checkpoint=f"sqlite-ordinal:{done}", rss_mib=peak / 2**20)
    if pending:
        inserted += add_documents(connection, pending)
    return inserted, peak


def preflight(config: dict, out: Path, data: Path) -> dict:
    report_path = out / "resource_report.json"
    source = ROOT / config["inputs"]["pubmed_documents"]
    sample_rows = int(config["resources"]["preflight_sample_rows"])
    sample_path = data / "preflight_sample.sqlite"
    total = pq.ParquetFile(source).metadata.num_rows
    if report_path.exists():
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        if prior.get("preflight_passed"):
            return prior
        if sample_path.exists():
            connection = sqlite3.connect(f"file:{sample_path.resolve()}?mode=ro", uri=True)
            count = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]; connection.close()
            if count == total and integrity == "ok":
                prior.update(preflight_passed=True, preflight_built_full_index=True,
                             measured_rows=count, preflight_index_path=str(sample_path.relative_to(ROOT)),
                             recovery_note="A missing benchmark stop bound built the full index; integrity passed and the bounded artifact is promoted instead of rebuilt.")
                atomic_write_json(report_path, prior)
                return prior
    for suffix in ("", "-wal", "-shm"):
        (Path(str(sample_path) + suffix)).unlink(missing_ok=True)
    before_rss = psutil.Process().memory_info().rss; before_available = psutil.virtual_memory().available
    started = time.perf_counter(); connection = initialize_index(sample_path)
    inserted, peak = build_rows(connection, source, 0, sample_rows, int(config["resources"]["build_batch_rows"]), stop_ordinal=sample_rows)
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)"); connection.close()
    elapsed = time.perf_counter() - started
    sample_bytes = sum(Path(str(sample_path) + suffix).stat().st_size for suffix in ("", "-wal", "-shm") if Path(str(sample_path) + suffix).exists())
    projected = math.ceil(sample_bytes / inserted * total * 1.30)
    rss_delta = max(0, peak - before_rss); required_headroom = max(128 * 2**20, int(rss_delta * 1.5))
    disk = shutil.disk_usage(ROOT)
    disk_ok = projected <= float(config["resources"]["maximum_additional_disk_gib"]) * 2**30 and disk.free - projected >= float(config["resources"]["minimum_free_disk_gib"]) * 2**30
    ram_ok = rss_delta <= float(config["resources"]["maximum_incremental_rss_mib"]) * 2**20 and before_available >= required_headroom
    report = {
        "preflight_passed": bool(inserted == sample_rows and disk_ok and ram_ok), "fts5_sample_rows": inserted,
        "sample_build_seconds": elapsed, "sample_documents_per_second": inserted / elapsed,
        "sample_index_bytes": sample_bytes, "projected_full_index_bytes_with_30pct_margin": projected,
        "rss_before_bytes": before_rss, "measured_incremental_peak_rss_bytes": rss_delta,
        "available_physical_before_bytes": before_available, "required_headroom_bytes": required_headroom,
        "free_disk_before_bytes": disk.free, "projected_remaining_disk_bytes": disk.free - projected,
        "disk_limit_bytes": int(float(config["resources"]["maximum_additional_disk_gib"]) * 2**30),
        "disk_reserve_bytes": int(float(config["resources"]["minimum_free_disk_gib"]) * 2**30),
        "cpu_only": True, "no_pagefile_change": True,
    }
    atomic_write_json(report_path, report)
    if not report["preflight_passed"]:
        raise RuntimeError(f"M2A resource preflight failed: {report}")
    return report


def finalize_index(connection: sqlite3.Connection, source: Path, output: Path, build_seconds: float, peak: int) -> dict:
    count = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    duplicate_pmids = int(connection.execute("SELECT COUNT(*) FROM (SELECT pmid FROM documents GROUP BY pmid HAVING COUNT(*)>1)").fetchone()[0])
    connection.execute("DROP TABLE IF EXISTS terms_vocab")
    connection.execute("CREATE VIRTUAL TABLE terms_vocab USING fts5vocab(documents_fts,'row')")
    connection.execute("DROP TABLE IF EXISTS term_df")
    connection.execute("CREATE TABLE term_df(term TEXT PRIMARY KEY,doc_count INTEGER NOT NULL)")
    connection.execute("INSERT INTO term_df SELECT term,doc FROM terms_vocab")
    connection.commit(); connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    digest = hashlib.sha256()
    for doc_id, value in connection.execute("SELECT doc_id,content_hash FROM documents ORDER BY doc_id"):
        digest.update(f"{doc_id}:{value}\n".encode())
    report = {
        "passed": count == 239896 and integrity == "ok" and duplicate_pmids == 0,
        "documents": count, "duplicate_pmids": duplicate_pmids, "sqlite_integrity": integrity,
        "source_path": str(source.relative_to(ROOT)), "source_sha256": file_sha256(source),
        "source_content_manifest_sha256": digest.hexdigest(), "index_path": str(output.relative_to(ROOT)),
        "index_size_bytes": output.stat().st_size, "build_seconds": build_seconds,
        "documents_per_second": count / max(build_seconds, .001), "peak_process_rss_bytes": peak,
        "tokenizer": "SQLite FTS5 unicode61 remove_diacritics=2", "title_bm25_weight": 4.0, "abstract_bm25_weight": 1.0,
        "incremental_resume": "ordinal checkpoint committed atomically with each FTS batch",
    }
    return report


HYBRID_SCHEMA = pa.schema([
    ("query_id", pa.int64()), ("doc_id", pa.int64()), ("hybrid_rank", pa.int64()),
    ("candidate_arms", pa.list_(pa.string())), ("dense_rank", pa.int64()), ("dense_score", pa.float64()),
    ("lexical_rank", pa.int64()), ("lexical_score", pa.float64()), ("rrf_score", pa.float64()),
])


def bridge_and_candidates(connection: sqlite3.Connection, config: dict, out: Path, data: Path, state_path: Path | None) -> tuple[Path, dict, list[dict]]:
    dense_path = ROOT / config["inputs"]["dense_top100"]
    queries = {int(row["id"]): str(row["query"]) for row in pq.read_table(ROOT / config["inputs"]["queries"], columns=["id", "query"]).to_pylist()}
    groups = query_groups(dense_path)
    destination = data / "hybrid_document_candidates.parquet"; temporary = destination.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(temporary, HYBRID_SCHEMA, compression="zstd")
    successes = empty = 0; additional = overlap_total = 0; latencies = []; pool_sizes = []; traces = []
    term_cache: dict[str, int] = {}
    try:
        for number, rows in enumerate(groups, 1):
            query_id = int(rows[0]["query_id"]); dense = [{"doc_id": int(row["doc_id"]), "rank": int(row["rank"]), "dense_score": float(row["dense_score"])} for row in rows[:int(config["retrieval"]["dense_depth"])] ]
            seed_ids = [row["doc_id"] for row in dense[:int(config["retrieval"]["dense_seed_documents"])]]
            placeholders = ",".join("?" for _ in seed_ids)
            seed_docs = [{"doc_id": int(row[0]), "title": row[1], "abstract": row[2]} for row in connection.execute(
                f"SELECT doc_id,title,abstract FROM documents WHERE doc_id IN ({placeholders})", seed_ids,
            )]
            tokens = sorted({token for row in seed_docs for token in english_tokens(row["title"] + " " + row["abstract"])})
            missing = [token for token in tokens if token not in term_cache]
            for start in range(0, len(missing), 800):
                subset = missing[start:start + 800]; marks = ",".join("?" for _ in subset)
                found = dict(connection.execute(f"SELECT term,doc_count FROM term_df WHERE term IN ({marks})", subset)) if subset else {}
                for token in subset: term_cache[token] = int(found.get(token, 0))
            terms = bridge_terms(seed_docs, {token: term_cache[token] for token in tokens}, 239896,
                                 maximum_terms=int(config["retrieval"]["expansion_terms"]),
                                 maximum_df_ratio=float(config["retrieval"]["maximum_term_document_frequency_ratio"]))
            expression = lexical_query(terms); started = time.perf_counter()
            lexical = search(connection, expression, int(config["retrieval"]["lexical_depth"])) if len(terms) >= 3 else []
            latencies.append(time.perf_counter() - started)
            if lexical: successes += 1
            else: empty += 1
            dense_ids = {row["doc_id"] for row in dense}; lexical_ids = {row["doc_id"] for row in lexical}
            overlap = len(dense_ids & lexical_ids); overlap_total += overlap; additional += len(lexical_ids - dense_ids)
            hybrid = reciprocal_rank_union(dense, lexical, int(config["retrieval"]["rrf_constant"]))[:int(config["retrieval"]["hybrid_depth"])]
            pool_sizes.append(len(hybrid))
            output = [{
                "query_id": query_id, "doc_id": row["doc_id"], "hybrid_rank": row["hybrid_rank"],
                "candidate_arms": row["candidate_arms"], "dense_rank": row["dense_rank"], "dense_score": row["dense_score"],
                "lexical_rank": row["lexical_rank"], "lexical_score": row["lexical_score"], "rrf_score": row["rrf_score"],
            } for row in hybrid]
            writer.write_table(pa.Table.from_pylist(output, schema=HYBRID_SCHEMA))
            if len(traces) < 30 or overlap < traces[-1].get("overlap", 101):
                trace = {"query_id": query_id, "vietnamese_query": queries[query_id], "seed_doc_ids": seed_ids,
                         "selected_terms": terms, "lexical_query": expression, "overlap": overlap,
                         "new_lexical_doc_ids_sample": sorted(lexical_ids - dense_ids)[:20],
                         "semantic_drift_warning": "Terms are pseudo-relevance signals, not verified translations or relevance labels."}
                traces.append(trace); traces = sorted(traces, key=lambda row: (row["overlap"], row["query_id"]))[:30]
            if number == 1 or number % 25 == 0:
                state(state_path, status="RUNNING", stage="QUERY_BRIDGE", done=number, total=1200, checkpoint=f"query:{query_id}")
        writer.close(); os.replace(temporary, destination)
    except Exception:
        writer.close(); temporary.unlink(missing_ok=True); raise
    report = {
        "passed": successes + empty == 1200, "queries": 1200, "successful_english_expansions": successes,
        "empty_or_degenerate_queries": empty, "dense_candidate_rows": 120000,
        "lexical_candidate_rows": successes * int(config["retrieval"]["lexical_depth"]),
        "dense_lexical_overlap": overlap_total, "additional_distinct_query_document_pairs": additional,
        "overlap_per_query": overlap_total / 1200, "additional_per_query": additional / 1200,
        "hybrid_pool_size": {"min": min(pool_sizes), "median": sorted(pool_sizes)[600], "max": max(pool_sizes)},
        "lexical_latency_seconds": percentile(latencies), "claim_boundary": "Novel candidates are UNKNOWN, not automatically relevant.",
        "candidate_path": str(destination.relative_to(ROOT)), "candidate_sha256": file_sha256(destination),
    }
    atomic_write_json(out / "query_bridge_report.json", report); atomic_write_json(out / "sample_query_traces.json", {"traces": traces})
    return destination, report, traces


EVIDENCE_SCHEMA = pa.schema([
    ("query_id", pa.int64()), ("doc_id", pa.int64()), ("hybrid_rank", pa.int64()), ("chunk_id", pa.string()),
    ("chunk_index", pa.int64()), ("chunk_text", pa.string()), ("start_offset", pa.int64()), ("end_offset", pa.int64()),
    ("section_type", pa.string()), ("source_url", pa.string()), ("existing_score", pa.float64()),
])


def evidence_pool(connection: sqlite3.Connection, candidate_path: Path, config: dict, out: Path, data: Path, state_path: Path | None) -> tuple[Path, dict]:
    candidate_docs = set(map(int, pq.read_table(candidate_path, columns=["doc_id"]).column(0).to_pylist()))
    connection.execute("DROP TABLE IF EXISTS representative_chunks")
    connection.execute("CREATE TABLE representative_chunks(doc_id INTEGER,slot INTEGER,chunk_id TEXT UNIQUE,chunk_index INTEGER,chunk_text TEXT,start_offset INTEGER,end_offset INTEGER,section_type TEXT,source_url TEXT,PRIMARY KEY(doc_id,slot))")
    counts = Counter(); pending = []
    chunk_path = ROOT / config["inputs"]["pubmed_chunks"]
    columns = ["doc_id", "chunk_id", "chunk_index", "raw_text", "start_offset", "end_offset", "section_type", "source_url"]
    for batch in pq.ParquetFile(chunk_path).iter_batches(columns=columns, batch_size=2048):
        for row in batch.to_pylist():
            doc_id = int(row["doc_id"])
            if doc_id not in candidate_docs or counts[doc_id] >= int(config["retrieval"]["evidence_chunks_per_document"]): continue
            slot = counts[doc_id]; counts[doc_id] += 1
            pending.append((doc_id, slot, str(row["chunk_id"]), int(row["chunk_index"]), str(row["raw_text"]),
                            int(row["start_offset"]), int(row["end_offset"]), str(row["section_type"]), str(row["source_url"])))
            if len(pending) >= 1024:
                with connection: connection.executemany("INSERT INTO representative_chunks VALUES(?,?,?,?,?,?,?,?,?)", pending)
                pending.clear()
    if pending:
        with connection: connection.executemany("INSERT INTO representative_chunks VALUES(?,?,?,?,?,?,?,?,?)", pending)
    destination = data / "bounded_evidence_candidates.parquet"; temporary = destination.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(temporary, EVIDENCE_SCHEMA, compression="zstd")
    score_db = sqlite3.connect(f"file:{(ROOT / config['inputs']['phase12_scores']).resolve()}?mode=ro", uri=True)
    reused = missing_scores = evidence_rows = ownership_mismatch = 0
    try:
        for number, group in enumerate(query_groups(candidate_path), 1):
            query_id = int(group[0]["query_id"]); doc_ids = [int(row["doc_id"]) for row in group]
            marks = ",".join("?" for _ in doc_ids)
            chunks = defaultdict(list)
            for row in connection.execute(f"SELECT doc_id,chunk_id,chunk_index,chunk_text,start_offset,end_offset,section_type,source_url FROM representative_chunks WHERE doc_id IN ({marks}) ORDER BY doc_id,slot", doc_ids):
                chunks[int(row[0])].append(row)
            scores = {str(chunk): float(score) for chunk, score in score_db.execute("SELECT chunk_id,rerank_score FROM pairs WHERE query_id=?", (query_id,))}
            output = []
            for candidate in group:
                doc_id = int(candidate["doc_id"])
                for chunk in chunks[doc_id]:
                    score = scores.get(str(chunk[1])); reused += score is not None; missing_scores += score is None
                    ownership_mismatch += int(doc_id != int(chunk[0]))
                    output.append({"query_id": query_id, "doc_id": doc_id, "hybrid_rank": int(candidate["hybrid_rank"]),
                                   "chunk_id": str(chunk[1]), "chunk_index": int(chunk[2]), "chunk_text": str(chunk[3]),
                                   "start_offset": int(chunk[4]), "end_offset": int(chunk[5]), "section_type": str(chunk[6]),
                                   "source_url": str(chunk[7]), "existing_score": score})
            writer.write_table(pa.Table.from_pylist(output, schema=EVIDENCE_SCHEMA)); evidence_rows += len(output)
            if number == 1 or number % 25 == 0:
                state(state_path, status="RUNNING", stage="EVIDENCE_POOL", done=number, total=1200, checkpoint=f"query:{query_id}")
        writer.close(); os.replace(temporary, destination)
    except Exception:
        writer.close(); temporary.unlink(missing_ok=True); raise
    finally:
        score_db.close()
    report = {"passed": ownership_mismatch == 0 and evidence_rows > 0, "candidate_documents": len(candidate_docs),
              "evidence_candidate_pairs": evidence_rows, "exact_scores_reused": reused,
              "genuinely_missing_reranker_pairs": missing_scores, "new_model_inference_performed": 0,
              "ownership_mismatches": ownership_mismatch, "maximum_chunks_per_document": int(config["retrieval"]["evidence_chunks_per_document"]),
              "selection_rule": "first two source-grounded PubMed evidence chunks by durable chunk_index; resource-control prototype, not relevance-optimal",
              "path": str(destination.relative_to(ROOT)), "sha256": file_sha256(destination)}
    atomic_write_json(out / "candidate_pool_manifest.json", report)
    return destination, report


def decision(out: Path, index: dict, bridge: dict, pool: dict, resource: dict) -> None:
    (out / "M2A_DECISION.md").write_text(f"""# M2A decision

- PubMed FTS5 index: **{'PASS' if index['passed'] else 'FAIL'}**, {index['documents']:,} documents, {index['index_size_bytes']/2**30:.2f} GiB.
- Operational VI→EN pseudo-relevance expansions: {bridge['successful_english_expansions']:,}/1,200; degenerate: {bridge['empty_or_degenerate_queries']}.
- Additional lexical-only query/document pairs: {bridge['additional_distinct_query_document_pairs']:,}; dense/lexical overlap: {bridge['dense_lexical_overlap']:,}.
- Bounded evidence pairs: {pool['evidence_candidate_pairs']:,}; existing scores: {pool['exact_scores_reused']:,}; missing scores: {pool['genuinely_missing_reranker_pairs']:,}.
- No relevance labels establish that lexical-only documents are useful. Pseudo-relevance terms may drift semantically; inspect `sample_query_traces.json` before M2B.

M2B is resource-feasible only as resumable candidate-only reranking at batch=2 using the exact missing-pair count above. It is not automatically authorized.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--state-path", type=Path); args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")); out = ROOT / config["outputs"]["artifacts"]; data = ROOT / config["outputs"]["root"]
    out.mkdir(parents=True, exist_ok=True); data.mkdir(parents=True, exist_ok=True)
    state(args.state_path, status="RUNNING", stage="RESOURCE_PREFLIGHT", done=0, total=1)
    resource = preflight(config, out, data)
    source = ROOT / config["inputs"]["pubmed_documents"]; index_path = ROOT / config["outputs"]["index"]
    if resource.get("preflight_built_full_index") and not index_path.exists():
        sample_path = ROOT / resource["preflight_index_path"]
        os.replace(sample_path, index_path)
    connection = initialize_index(index_path); start = completed_ordinal(connection); total = pq.ParquetFile(source).metadata.num_rows
    state(args.state_path, status="RUNNING", stage="BUILD_LEXICAL_INDEX", done=start, total=total, checkpoint=f"sqlite-ordinal:{start}")
    began = time.perf_counter(); _, peak = build_rows(connection, source, start, total, int(config["resources"]["build_batch_rows"]), state_path=args.state_path)
    build_seconds = time.perf_counter() - began
    state(args.state_path, status="RUNNING", stage="FINALIZE_INDEX", done=total, total=total, checkpoint=f"sqlite-ordinal:{total}")
    index_report = finalize_index(connection, source, index_path, build_seconds, peak)
    atomic_write_json(out / "index_build_report.json", index_report)
    if not index_report["passed"]: raise RuntimeError(f"index audit failed: {index_report}")
    candidate_path, bridge, traces = bridge_and_candidates(connection, config, out, data, args.state_path)
    evidence_path, pool = evidence_pool(connection, candidate_path, config, out, data, args.state_path)
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)"); connection.close()
    hybrid = {"passed": bridge["passed"] and pool["passed"], "control": "existing dense top100 unchanged",
              "new": "dense top100 union independent lexical top100 with rank-only RRF",
              "dense_lexical_overlap": bridge["dense_lexical_overlap"], "additional_distinct_documents": bridge["additional_distinct_query_document_pairs"],
              "candidate_diversity_boundary": "Counts measure independence, not relevance", "provenance_valid": pool["ownership_mismatches"] == 0,
              "exact_reproducibility": {"candidate_sha256": bridge["candidate_sha256"], "evidence_sha256": pool["sha256"]}}
    atomic_write_json(out / "hybrid_candidate_audit.json", hybrid)
    resource.update(actual_index_bytes=index_path.stat().st_size, final_free_disk_bytes=shutil.disk_usage(ROOT).free,
                    final_process_rss_bytes=psutil.Process().memory_info().rss)
    atomic_write_json(out / "resource_report.json", resource); decision(out, index_report, bridge, pool, resource)
    state(args.state_path, status="READY_FOR_HUMAN_REVIEW", stage="M2A_COMPLETE", done=1200, total=1200,
          checkpoint=str(evidence_path), integrity="passed", new_model_inference=0)
    print(json.dumps({"index": index_report, "bridge": bridge, "pool": pool}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
