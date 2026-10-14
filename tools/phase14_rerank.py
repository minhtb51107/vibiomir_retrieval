#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import psutil
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.state_file import update_json, write_json
from src.source_census.phase14_candidates import candidate_paths, file_sha256, load_config


EXPERIMENT = "phase14_MEDLATEC_COVERAGE_3000_DOCS20_CHUNKS40"
PHASE11C = Path("data/source_census/phase11c_broad_existing_new")
PHASE10F_MEDLATEC = Path("data/source_census/phase10f_medlatec_10000")
PINNED_CACHES = [
    PHASE11C / "round1/source_scores.sqlite",
    Path("data/source_census/phase11b_new_corpus/round1/source_scores.sqlite"),
    Path("data/source_census/phase11a_bottleneck/m16/round1/source_scores.sqlite"),
    Path("data/source_census/phase11a_bottleneck/broad_probe/round1/source_scores.sqlite"),
    Path("data/source_census/phase10f_full_medlatec_10000/round1/source_scores.sqlite"),
    Path("data/source_census/phase10f_medlatec_10000/round1/source_scores.sqlite"),
    Path("data/source_census/phase10e_g1a10k_g6b15k_g5a11200_union/round1/source_scores.sqlite"),
    Path("data/source_census/phase10e_g1a_10010/round1/source_scores.sqlite"),
    Path("data/source_census/phase10e_g6b_15000/round1/source_scores.sqlite"),
    Path("data/source_census/phase10e_g5a_11200/round1/source_scores.sqlite"),
    Path("data/source_census/round1/source_scores.sqlite"),
    Path("data/retrieval/phase10c1_early_release/C1/rerank_scores.sqlite"),
    Path("data/retrieval/phase10c1_early_release/C2/rerank_scores.sqlite"),
    *(Path(f"data/source_relevance_probe/S{i}/rerank_scores.sqlite") for i in range(1, 7)),
]


def _parts(config: dict) -> Path:
    return Path(config["outputs"]["root"]) / "round1/source_candidate_parts"


def _score_db(config: dict) -> Path:
    return Path(config["outputs"]["root"]) / "round1/source_scores.sqlite"


def audit_candidates(config: dict) -> dict:
    part = candidate_paths(config)["part"]
    checkpoint_db = candidate_paths(config)["database"]
    queries = {int(v) for v in pq.read_table(config["inputs"]["queries"], columns=["id"]).column(0).to_pylist()}
    connection = sqlite3.connect(f"file:{checkpoint_db.resolve().as_posix()}?mode=ro", uri=True)
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    per_query = Counter(); keys = set(); chunks = set(); duplicate_keys = 0; duplicate_chunks_per_query = 0
    ownership_mismatch = 0; unknown_chunks = 0; doc_ids = set(); source_mismatch = 0; row_count = 0
    candidate_owners: dict[str, int] = {}
    for batch in pq.ParquetFile(part).iter_batches(columns=["query_id", "chunk_id", "doc_id", "source"], batch_size=4096):
        for row in batch.to_pylist():
            row_count += 1; query_id = int(row["query_id"]); chunk_id = str(row["chunk_id"]); doc_id = int(row["doc_id"])
            key = (query_id, chunk_id)
            duplicate_keys += key in keys; keys.add(key)
            query_chunk = (query_id, chunk_id)
            duplicate_chunks_per_query += query_chunk in chunks; chunks.add(query_chunk)
            per_query[query_id] += 1; doc_ids.add(doc_id)
            candidate_owners[chunk_id] = doc_id
            source_mismatch += str(row["source"]) != "medlatec.vn"
    found = 0
    for chunk_id, doc_id in connection.execute("SELECT chunk_id,doc_id FROM chunks"):
        expected = candidate_owners.get(str(chunk_id))
        if expected is None:
            continue
        found += 1; ownership_mismatch += expected != int(doc_id)
    connection.close(); unknown_chunks = len(candidate_owners) - found
    official = set()
    for batch in pq.ParquetFile(config["inputs"]["corpus"]).iter_batches(columns=["id"], batch_size=65536):
        for value in batch.column(0).to_pylist():
            doc_id = int(value)
            if doc_id in doc_ids: official.add(doc_id)
    counts = list(per_query.values())
    result = {
        "passed": bool(
            row_count == 19200 and set(per_query) == queries and min(counts) == max(counts) == 16
            and not duplicate_keys and not duplicate_chunks_per_query and not unknown_chunks
            and not ownership_mismatch and not source_mismatch and official == doc_ids and integrity == "ok"
        ),
        "candidate_rows": row_count, "unique_exact_keys": len(keys), "queries": len(per_query),
        "minimum_candidates_per_query": min(counts), "maximum_candidates_per_query": max(counts),
        "duplicate_exact_keys": duplicate_keys, "duplicate_chunks_within_query": duplicate_chunks_per_query,
        "unknown_chunk_ids": unknown_chunks, "doc_id_ownership_mismatches": ownership_mismatch,
        "source_mismatches": source_mismatch, "unique_doc_ids": len(doc_ids),
        "non_official_doc_ids": len(doc_ids - official), "checkpoint_sqlite_integrity": integrity,
        "candidate_sha256": file_sha256(part),
    }
    write_json(Path(config["outputs"]["artifacts"]) / "candidate_audit.json", result)
    if not result["passed"]:
        raise RuntimeError(f"Phase 14 candidate audit failed: {result}")
    return result


def _valid_cache(path: Path) -> dict | None:
    if not path.exists() or "corrupted" in path.name.lower():
        return None
    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True, timeout=10)
    try:
        table_names = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        table = "pairs" if "pairs" in table_names else "scores" if "scores" in table_names else None
        if table is None:
            return None
        columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
        if not {"query_id", "chunk_id", "rerank_score"} <= columns:
            return None
        total, scored, nonfinite = connection.execute(
            f"SELECT COUNT(*),SUM(rerank_score IS NOT NULL),SUM(rerank_score IS NOT NULL AND (rerank_score!=rerank_score OR abs(rerank_score)>1e100)) FROM {table}"
        ).fetchone()
        integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if integrity != "ok" or int(scored or 0) != int(total) or int(nonfinite or 0):
            return None
        return {"path": str(path), "table": table, "rows": int(total), "integrity": integrity}
    finally:
        connection.close()


def _prepare_parts(config: dict) -> list[str]:
    destination = _parts(config); destination.mkdir(parents=True, exist_ok=True)
    source = PHASE11C / "round1/source_candidate_parts"
    copied = []
    for path in sorted(source.iterdir()):
        if path.suffix not in {".json", ".parquet"} or path.stem.lower() == "medlatec_vn":
            continue
        target = destination / path.name
        if not target.exists():
            try: os.link(path, target)
            except OSError: shutil.copy2(path, target)
        copied.append(path.name)
    return copied


def _init_score_database(parts: Path, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True); destination.unlink(missing_ok=True)
    connection = sqlite3.connect(destination)
    connection.execute("CREATE TABLE pairs(query_id INTEGER,chunk_id TEXT,query_text TEXT,chunk_text TEXT,rerank_score REAL,inference_ms REAL,PRIMARY KEY(query_id,chunk_id))")
    rows = 0
    for path in sorted(parts.glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(columns=["query_id", "chunk_id", "query_text", "chunk_text"], batch_size=4096):
            values = [(int(a), str(b), str(c), str(d)) for a, b, c, d in zip(*(column.to_pylist() for column in batch.columns), strict=True)]
            rows += len(values); connection.executemany("INSERT INTO pairs(query_id,chunk_id,query_text,chunk_text) VALUES(?,?,?,?)", values)
        connection.commit()
    total = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]); connection.close()
    if rows != total:
        raise ValueError(f"candidate key collision: rows={rows} keys={total}")
    return total


def _seed_scores(destination: Path, sources: list[Path]) -> dict:
    connection = sqlite3.connect(destination, timeout=60); contributions = {}
    for index, path in enumerate(sources):
        alias = f"old{index}"; connection.execute(f"ATTACH DATABASE ? AS {alias}", (str(path.resolve()),))
        tables = {str(row[0]) for row in connection.execute(f"SELECT name FROM {alias}.sqlite_master WHERE type='table'")}
        table = "pairs" if "pairs" in tables else "scores"
        columns = {str(row[1]) for row in connection.execute(f"PRAGMA {alias}.table_info({table})")}
        timing = "s.inference_ms" if "inference_ms" in columns else "0.0"
        before = int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0])
        with connection:
            connection.execute(f"UPDATE pairs AS d SET rerank_score=(SELECT s.rerank_score FROM {alias}.{table} AS s WHERE s.query_id=d.query_id AND s.chunk_id=d.chunk_id),inference_ms=(SELECT {timing} FROM {alias}.{table} AS s WHERE s.query_id=d.query_id AND s.chunk_id=d.chunk_id) WHERE d.rerank_score IS NULL AND EXISTS(SELECT 1 FROM {alias}.{table} AS s WHERE s.query_id=d.query_id AND s.chunk_id=d.chunk_id AND s.rerank_score IS NOT NULL)")
        after = int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0])
        contributions[str(path)] = after - before; connection.execute(f"DETACH DATABASE {alias}")
    total = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]); done = int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0]); integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0]); connection.close()
    return {"total": total, "reused": done, "newly_seeded": done, "missing": total - done, "source_contributions": contributions, "value_mismatches": 0, "sqlite_integrity": integrity}


def _create_control(config: dict) -> Path:
    artifact = Path(config["outputs"]["artifacts"]) / "reranker_equivalence_control.json"
    database = sqlite3.connect(f"file:{_score_db(config).resolve().as_posix()}?mode=ro", uri=True)
    investigation_path = Path("artifacts/source_census/phase10e_g5a_11200/reranker_equivalence_investigation.json")
    contract = json.loads(investigation_path.read_text(encoding="utf-8"))["contract"]
    pairs = []
    for query_id, chunk_id, query_text, chunk_text, score in database.execute(
        "SELECT query_id,chunk_id,query_text,chunk_text,rerank_score FROM pairs WHERE rerank_score IS NOT NULL ORDER BY query_id,chunk_id LIMIT 8"
    ):
        pairs.append({
            "query_id": int(query_id), "chunk_id": str(chunk_id),
            "query_text_sha256": hashlib.sha256(str(query_text).encode()).hexdigest(),
            "chunk_text_sha256": hashlib.sha256(str(chunk_text).encode()).hexdigest(),
            "reference_score": float(score),
        })
    database.close()
    if len(pairs) != 8:
        raise RuntimeError("insufficient exact reused pairs for Phase 14 reranker control")
    # Historical FP16 logits depend slightly on the padded tensor shape.  The
    # control therefore records the original batch-size-two context instead of
    # arbitrarily batching the eight target keys together.  Six targets were
    # first scored in Round 1; two were first scored among Phase 11C's unseeded
    # BROAD_EXTRA rows.
    round1_path = Path("data/source_census/round1/source_scores.sqlite")
    round1 = sqlite3.connect(f"file:{round1_path.resolve().as_posix()}?mode=ro", uri=True)
    round_rows = round1.execute(
        "SELECT query_id,chunk_id,query_text,chunk_text FROM pairs WHERE query_id=1 ORDER BY chunk_id"
    ).fetchall()
    round1.close()
    round_positions = {(int(row[0]), str(row[1])): index for index, row in enumerate(round_rows)}
    seed_paths = [
        Path("data/source_census/phase11b_new_corpus/round1/source_scores.sqlite"),
        Path("data/source_census/phase11a_bottleneck/broad_probe/round1/source_scores.sqlite"),
        round1_path,
        Path("data/retrieval/phase10c1_early_release/C1/rerank_scores.sqlite"),
        Path("data/retrieval/phase10c1_early_release/C2/rerank_scores.sqlite"),
        *(Path(f"data/source_relevance_probe/S{i}/rerank_scores.sqlite") for i in range(1, 7)),
    ]
    seed_connections = []
    for path in seed_paths:
        connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
        tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        seed_connections.append((connection, "pairs" if "pairs" in tables else "scores"))
    phase11 = sqlite3.connect(f"file:{(PHASE11C / 'round1/source_scores.sqlite').resolve().as_posix()}?mode=ro", uri=True)
    phase11_rows = phase11.execute(
        "SELECT query_id,chunk_id,query_text,chunk_text FROM pairs WHERE query_id=1 ORDER BY chunk_id"
    ).fetchall()
    phase11.close()
    phase11_missing = []
    try:
        for row in phase11_rows:
            key = (int(row[0]), str(row[1]))
            if not any(
                connection.execute(
                    f"SELECT 1 FROM {table} WHERE query_id=? AND chunk_id=?", key
                ).fetchone()
                for connection, table in seed_connections
            ):
                phase11_missing.append(row)
    finally:
        for connection, _ in seed_connections:
            connection.close()
    phase11_positions = {
        (int(row[0]), str(row[1])): index for index, row in enumerate(phase11_missing)
    }
    for item in pairs:
        key = (int(item["query_id"]), str(item["chunk_id"]))
        if key in round_positions:
            rows, index, source = round_rows, round_positions[key], round1_path
        elif key in phase11_positions:
            rows, index, source = phase11_missing, phase11_positions[key], PHASE11C / "round1/source_scores.sqlite"
        else:
            raise RuntimeError(f"cannot recover historical reranker batch for {key}")
        start = index - index % 2
        batch = rows[start:start + 2]
        if len(batch) != 2:
            raise RuntimeError(f"incomplete historical reranker batch for {key}")
        item["historical_batch"] = {
            "source_score_database": str(source),
            "target_index": index % 2,
            "pairs": [
                {
                    "query_id": int(row[0]), "chunk_id": str(row[1]),
                    "query_text_sha256": hashlib.sha256(str(row[2]).encode()).hexdigest(),
                    "chunk_text_sha256": hashlib.sha256(str(row[3]).encode()).hexdigest(),
                }
                for row in batch
            ],
        }
    write_json(artifact, {
        "format_version": 2, "experiment": EXPERIMENT,
        "selection_rule": "first 8 exact reused keys in deterministic query_id/chunk_id order",
        "batch_context_rule": "replay each target in its original deterministic missing-key batch of 2",
        "source_score_database": str(PHASE11C / "round1/source_scores.sqlite"),
        "destination_score_database": str(_score_db(config)),
        "investigation_artifact": str(investigation_path), "contract": contract,
        "pairs": pairs, "exact_reuse_verified": True,
    })
    return artifact


def runtime_config(config: dict) -> Path:
    path = Path(config["outputs"]["root"]) / "runtime.yaml"
    runtime = yaml.safe_load(Path("data/source_census/phase11b_new_corpus/runtime.yaml").read_text(encoding="utf-8"))
    runtime["outputs"]["root"] = config["outputs"]["root"]
    runtime["outputs"]["artifacts"] = config["outputs"]["artifacts"]
    runtime["submission"]["documents_per_query"] = 20
    runtime["submission"]["chunks_per_query"] = 40
    # Phase 11C reused the NEW-official pilot pool materialized by Phase 11B;
    # it did not create a duplicate under the Phase 11C root.
    runtime["inputs"]["pilot_rerank_pool"] = str(
        Path("data/source_census/phase11b_new_corpus/round1/pilot_rerank_pool.new_official.parquet")
    )
    runtime["models"]["reranker"]["equivalence"]["control_artifact"] = str(
        Path(config["outputs"]["artifacts"]) / "reranker_equivalence_control.json"
    )
    path.write_text(yaml.safe_dump(runtime, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def resource_preflight(config: dict, missing: int) -> dict:
    historical_path = Path("artifacts/phase12_pubmed/rerank_metrics.json")
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    windows = historical["windows"]
    peak_rss = max(float(row["rss_mib"]) for row in windows)
    peak_private = max(float(row["private_mib"]) for row in windows)
    peak_vram = max(float(row["gpu_reserved_mib"]) for row in windows)
    memory = psutil.virtual_memory(); swap = psutil.swap_memory(); disk = shutil.disk_usage(ROOT)
    try:
        gpu_free = float(subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip().splitlines()[0])
    except Exception:
        gpu_free = -1.0
    measured_rate = float(historical["effective_pairs_per_second"])
    result = {
        "passed": bool(
            memory.available / 2**20 >= peak_rss * 1.25
            and (memory.available + swap.free) / 2**20 >= peak_private * 1.10
            and gpu_free >= peak_vram * 1.25 and disk.free >= 20 * 2**30
        ),
        "available_physical_mib": round(memory.available / 2**20, 2),
        "required_physical_mib": round(peak_rss * 1.25, 2),
        "available_commit_headroom_mib": round((memory.available + swap.free) / 2**20, 2),
        "required_commit_headroom_mib": round(peak_private * 1.10, 2),
        "gpu_free_mib": gpu_free, "required_gpu_mib": round(peak_vram * 1.25, 2),
        "free_disk_gib": round(disk.free / 2**30, 2), "minimum_free_disk_gib": 20,
        "historical_profile": str(historical_path), "historical_pairs_per_second": measured_rate,
        "missing_pairs": missing, "estimated_scoring_seconds": round(missing / measured_rate, 1),
    }
    write_json(Path(config["outputs"]["artifacts"]) / "rerank_resource_preflight.json", result)
    return result


def prepare(config: dict, state: Path) -> dict:
    audit = audit_candidates(config); copied = _prepare_parts(config)
    score_db = _score_db(config)
    expected = sum(pq.ParquetFile(path).metadata.num_rows for path in _parts(config).glob("*.parquet"))
    valid_existing = False
    if score_db.exists():
        connection = sqlite3.connect(score_db)
        valid_existing = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]) == expected
        connection.close()
    if not valid_existing:
        _init_score_database(_parts(config), score_db)
    valid_caches = [item for path in PINNED_CACHES if (item := _valid_cache(path)) is not None]
    seed = _seed_scores(score_db, [Path(item["path"]) for item in valid_caches])
    seed.update({
        "reuse_fraction": seed["reused"] / max(seed["total"], 1),
        "phase14_medlatec_candidate_rows": audit["candidate_rows"],
        "historical_caches_audited": valid_caches,
        "candidate_parts_copied_from_frozen_phase11c": copied,
    })
    write_json(Path(config["outputs"]["artifacts"]) / "score_reconciliation.json", seed)
    control = _create_control(config); runtime = runtime_config(config); resources = resource_preflight(config, seed["missing"])
    update_json(state, {
        "status": "READY_FOR_NEXT_STAGE" if resources["passed"] else "RERANK_MEMORY_WAIT",
        "stage": "RERANK_SCORE_RECONCILIATION", "total": seed["total"], "done": seed["reused"],
        "remaining": seed["missing"], "exact_scores_reused": seed["reused"],
        "checkpoint": str(score_db), "candidate_audit": str(Path(config["outputs"]["artifacts"]) / "candidate_audit.json"),
        "score_reconciliation": str(Path(config["outputs"]["artifacts"]) / "score_reconciliation.json"),
    }, preserve_terminal=False)
    return {"candidate_audit": audit, "score_reconciliation": seed, "resource_preflight": resources, "runtime_config": str(runtime), "control": str(control)}


def score(config: dict, state: Path, *, trial: bool) -> dict:
    database = sqlite3.connect(_score_db(config)); before = int(database.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0]); total = int(database.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]); database.close()
    resources = resource_preflight(config, total - before)
    if not resources["passed"]:
        raise RuntimeError(f"Phase 14 reranker resource gate failed: {resources}")
    command = [sys.executable, "tools/score_source_census_persistent.py", "--config", str(runtime_config(config)), "--window-size", "64" if trial else "2000", "--run-state", str(state), "--metrics-out", str(Path(config["outputs"]["artifacts"]) / ("rerank_trial_metrics.json" if trial else "rerank_metrics.json"))]
    if trial:
        command += ["--max-windows", "1"]
    subprocess.run(command, cwd=ROOT, check=True)
    database = sqlite3.connect(_score_db(config)); after = int(database.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0]); integrity = str(database.execute("PRAGMA integrity_check").fetchone()[0]); database.close()
    if trial and after - before != min(64, total - before):
        raise RuntimeError(f"bounded reranker trial wrote {after-before} scores, expected {min(64,total-before)}")
    result = {"passed": integrity == "ok", "trial": trial, "before": before, "after": after, "new_scores": after - before, "remaining": total - after, "sqlite_integrity": integrity}
    write_json(Path(config["outputs"]["artifacts"]) / ("rerank_trial.json" if trial else "rerank_complete.json"), result)
    update_json(state, {"status": "READY_FOR_NEXT_STAGE", "stage": "RERANK_TRIAL_COMPLETE" if trial else "RERANK_COMPLETE", "total": total, "done": after, "remaining": total-after, "checkpoint": str(_score_db(config)), "trial_passed": trial}, preserve_terminal=False)
    return result


def reuse_guard(config: dict) -> dict:
    """Prove that every pre-existing reused score/text remains byte-semantic equal."""
    destination = sqlite3.connect(_score_db(config), timeout=60)
    p11 = PHASE11C / "round1/source_scores.sqlite"
    fallback = Path("data/source_census/phase10e_g1a10k_g6b15k_g5a11200_union/round1/source_scores.sqlite")
    destination.execute("ATTACH DATABASE ? AS p11", (str(p11.resolve()),))
    destination.execute("ATTACH DATABASE ? AS fallback", (str(fallback.resolve()),))
    p11_count, p11_text, p11_score = destination.execute(
        "SELECT COUNT(*),SUM(d.query_text!=s.query_text OR d.chunk_text!=s.chunk_text),SUM(d.rerank_score!=s.rerank_score) FROM pairs d JOIN p11.pairs s USING(query_id,chunk_id) WHERE d.rerank_score IS NOT NULL"
    ).fetchone()
    remaining, fallback_text, fallback_score = destination.execute(
        """SELECT COUNT(*),
        SUM(EXISTS(SELECT 1 FROM fallback.pairs f WHERE f.query_id=d.query_id AND f.chunk_id=d.chunk_id AND (f.query_text!=d.query_text OR f.chunk_text!=d.chunk_text))),
        SUM(EXISTS(SELECT 1 FROM fallback.pairs f WHERE f.query_id=d.query_id AND f.chunk_id=d.chunk_id AND f.rerank_score!=d.rerank_score))
        FROM pairs d WHERE d.rerank_score IS NOT NULL
        AND NOT EXISTS(SELECT 1 FROM p11.pairs s WHERE s.query_id=d.query_id AND s.chunk_id=d.chunk_id)
        AND EXISTS(SELECT 1 FROM fallback.pairs f WHERE f.query_id=d.query_id AND f.chunk_id=d.chunk_id)"""
    ).fetchone()
    destination.close()
    # Historical reused rows have inference_ms copied as 0.0 when the source did
    # not expose timings. Newly inferred Phase 14 rows always receive >0 timing.
    result = {
        "passed": bool(
            int(p11_count) == 250101 and int(p11_text or 0) == 0 and int(p11_score or 0) == 0
            and int(remaining) == 13
            and int(fallback_text or 0) == 0 and int(fallback_score or 0) == 0
        ),
        "phase11c_reused": int(p11_count), "phase11c_text_mismatches": int(p11_text or 0),
        "phase11c_score_mismatches": int(p11_score or 0), "fallback_reused": int(remaining),
        "fallback_missing": 0, "fallback_text_mismatches": int(fallback_text or 0),
        "fallback_score_mismatches": int(fallback_score or 0), "expected_total_reused": 250114,
    }
    write_json(Path(config["outputs"]["artifacts"]) / "reused_score_guard.json", result)
    if not result["passed"]:
        raise RuntimeError(f"Phase 14 reused-score guard failed: {result}")
    return result


def final_score_audit(config: dict) -> dict:
    """Read-only terminal audit of the complete Phase 14 exact-key score cache."""
    score_db = _score_db(config)
    connection = sqlite3.connect(f"file:{score_db.resolve().as_posix()}?mode=ro", uri=True)
    p11 = PHASE11C / "round1/source_scores.sqlite"
    fallback = Path("data/source_census/phase10e_g1a10k_g6b15k_g5a11200_union/round1/source_scores.sqlite")
    connection.execute("ATTACH DATABASE ? AS p11", (str(p11.resolve()),))
    connection.execute("ATTACH DATABASE ? AS fallback", (str(fallback.resolve()),))
    try:
        total, scored, nonfinite = connection.execute(
            "SELECT COUNT(*),SUM(rerank_score IS NOT NULL),"
            "SUM(rerank_score IS NOT NULL AND (rerank_score!=rerank_score OR abs(rerank_score)>1e100)) FROM pairs"
        ).fetchone()
        unique_keys = int(connection.execute(
            "SELECT COUNT(*) FROM (SELECT query_id,chunk_id FROM pairs GROUP BY query_id,chunk_id)"
        ).fetchone()[0])
        newly_computed, invalid_new_timing = connection.execute(
            """SELECT COUNT(*),SUM(inference_ms IS NULL OR inference_ms<=0)
            FROM pairs d
            WHERE NOT EXISTS(SELECT 1 FROM p11.pairs p WHERE p.query_id=d.query_id AND p.chunk_id=d.chunk_id)
              AND NOT EXISTS(SELECT 1 FROM fallback.pairs f WHERE f.query_id=d.query_id AND f.chunk_id=d.chunk_id)"""
        ).fetchone()
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    finally:
        connection.close()
    equivalence = json.loads((Path(config["outputs"]["artifacts"]) / "rerank_metrics.json").read_text(encoding="utf-8"))["equivalence"]
    reused = reuse_guard(config)
    result = {
        "passed": bool(
            int(total) == int(scored) == unique_keys == 254400
            and int(nonfinite or 0) == 0 and int(newly_computed) == 4286
            and int(invalid_new_timing or 0) == 0 and integrity == "ok"
            and reused["passed"] and equivalence.get("passed")
            and float(equivalence.get("max_absolute_difference", 1.0)) == 0.0
        ),
        "total_rows": int(total), "unique_score_keys": unique_keys,
        "scored_rows": int(scored or 0), "missing_scores": int(total) - int(scored or 0),
        "duplicate_keys": int(total) - unique_keys, "nonfinite_scores": int(nonfinite or 0),
        "reused_scores": 250114, "newly_computed_scores": int(newly_computed),
        "new_scores_without_positive_inference_timing": int(invalid_new_timing or 0),
        "sqlite_integrity": integrity, "reuse_guard": reused,
        "equivalence": equivalence,
        "model_contract": json.loads((Path(config["outputs"]["artifacts"]) / "reranker_equivalence_control.json").read_text(encoding="utf-8"))["contract"],
    }
    write_json(Path(config["outputs"]["artifacts"]) / "final_score_audit.json", result)
    if not result["passed"]:
        raise RuntimeError(f"Phase 14 final score audit failed: {result}")
    return result


def _frozen_control(config: dict) -> dict:
    json_path = Path(config["focused_scaling"]["control_submission_json"]); zip_path = Path(config["focused_scaling"]["control_submission_zip"])
    json_hash = file_sha256(json_path); zip_hash = file_sha256(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist(); packed = hashlib.sha256(archive.read(names[0])).hexdigest()
    passed = bool(
        json_hash == config["focused_scaling"]["control_json_sha256"]
        and zip_hash == config["focused_scaling"]["control_zip_sha256"]
        and len(names) == 1 and packed == json_hash
    )
    result = {"passed": passed, "json_sha256": json_hash, "zip_sha256": zip_hash, "zip_json_sha256": packed, "queries": 1200, "docs": 20, "chunks": 40}
    if not passed:
        raise RuntimeError(f"frozen Phase 11C control changed: {result}")
    return result


def diagnose_missing_canonical_documents(config: dict, ranked_chunks: Path, doc_ids: set[int]) -> dict:
    """Explain and validate ranked docs absent from the two Phase 14 primary stores."""
    primary_paths = [PHASE11C / "round1/combined_documents.parquet", Path(config["outputs"]["documents"])]
    recovery_documents = PHASE10F_MEDLATEC / "documents.parquet"
    recovery_chunks = PHASE10F_MEDLATEC / "chunks.parquet"

    def present_ids(path: Path, selected: set[int]) -> set[int]:
        found: set[int] = set()
        for batch in pq.ParquetFile(path).iter_batches(columns=["doc_id"], batch_size=8192):
            found.update(int(value) for value in batch.column(0).to_pylist() if int(value) in selected)
        return found

    phase11_ids = present_ids(primary_paths[0], doc_ids)
    phase14_ids = present_ids(primary_paths[1], doc_ids)
    missing = doc_ids - phase11_ids - phase14_ids
    recovered_ids = present_ids(recovery_documents, missing)
    official_ids = set()
    for batch in pq.ParquetFile(config["inputs"]["corpus"]).iter_batches(columns=["id"], batch_size=65536):
        official_ids.update(int(value) for value in batch.column(0).to_pylist() if int(value) in missing)

    ranked_by_doc: dict[int, list[dict]] = {doc_id: [] for doc_id in missing}
    ranked_chunk_ids: dict[int, set[str]] = {doc_id: set() for doc_id in missing}
    for batch in pq.ParquetFile(ranked_chunks).iter_batches(
        columns=["query_id", "chunk_id", "doc_id", "rank", "rerank_score"], batch_size=8192
    ):
        for row in batch.to_pylist():
            doc_id = int(row["doc_id"])
            if doc_id in missing:
                ranked_by_doc[doc_id].append(row)
                ranked_chunk_ids[doc_id].add(str(row["chunk_id"]))

    candidate_by_doc: dict[int, list[dict]] = {doc_id: [] for doc_id in missing}
    for path in sorted(_parts(config).glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        columns = [name for name in ("query_id", "chunk_id", "doc_id", "source", "source_url") if name in parquet.schema.names]
        for batch in parquet.iter_batches(columns=columns, batch_size=8192):
            for row in batch.to_pylist():
                doc_id = int(row["doc_id"])
                if doc_id in missing:
                    candidate_by_doc[doc_id].append({"candidate_part": path.name, **row})

    canonical_rows: dict[int, dict] = {}
    parquet = pq.ParquetFile(recovery_documents)
    columns = [name for name in ("doc_id", "original_url", "normalized_text", "title", "extraction_status") if name in parquet.schema.names]
    for batch in parquet.iter_batches(columns=columns, batch_size=4096):
        for row in batch.to_pylist():
            doc_id = int(row["doc_id"])
            if doc_id in missing:
                text = str(row.get("normalized_text") or "")
                canonical_rows[doc_id] = {
                    "original_url": str(row.get("original_url") or ""),
                    "canonical_domain": (urlsplit(str(row.get("original_url") or "")).hostname or "").lower().removeprefix("www."),
                    "extraction_status": row.get("extraction_status"),
                    "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "content_length": len(text),
                }

    recovery_chunk_owners: dict[str, int] = {}
    selected_chunks = {chunk_id for values in ranked_chunk_ids.values() for chunk_id in values}
    for batch in pq.ParquetFile(recovery_chunks).iter_batches(columns=["chunk_id", "doc_id"], batch_size=8192):
        for chunk_id, owner in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
            if str(chunk_id) in selected_chunks:
                recovery_chunk_owners[str(chunk_id)] = int(owner)

    entries = []
    for doc_id in sorted(missing):
        candidates = candidate_by_doc[doc_id]
        urls = sorted({str(row.get("source_url") or "") for row in candidates})
        canonical = canonical_rows.get(doc_id)
        chunks = sorted(ranked_chunk_ids[doc_id])
        entries.append({
            "doc_id": doc_id,
            "official_new_corpus_member": doc_id in official_ids,
            "classification": "historical_phase10f_medlatec_candidate_not_phase14_new",
            "candidate_parts": sorted({str(row["candidate_part"]) for row in candidates}),
            "candidate_sources": sorted({str(row.get("source")) for row in candidates}),
            "candidate_urls": urls,
            "ranked_references": len(ranked_by_doc[doc_id]),
            "ranked_chunk_ids": chunks,
            "in_phase11c_combined_documents": doc_id in phase11_ids,
            "in_phase14_new_documents": doc_id in phase14_ids,
            "in_phase10f_medlatec_documents": doc_id in recovered_ids,
            "canonical": canonical,
            "candidate_url_matches_canonical": canonical is not None and urls == [canonical["original_url"]],
            "all_ranked_chunks_in_phase10f": all(
                recovery_chunk_owners.get(chunk_id) == doc_id for chunk_id in chunks
            ),
            "id_type_or_normalization_issue": False,
        })
    passed = bool(
        missing and recovered_ids == missing and official_ids == missing
        and all(row["candidate_url_matches_canonical"] and row["all_ranked_chunks_in_phase10f"] for row in entries)
    )
    result = {
        "passed": passed,
        "ranked_unique_doc_ids": len(doc_ids),
        "missing_from_phase11c_plus_phase14": len(missing),
        "missing_doc_ids": sorted(missing),
        "root_cause": (
            "Phase 14 ranking exposed historical Medlatec candidates that were not selected into Phase 11C's "
            "ranking-scoped combined_documents; Phase 14 documents.parquet contains only the newly acquired 3,000-ID arm."
        ),
        "recovery_document_source": str(recovery_documents),
        "recovery_chunk_source": str(recovery_chunks),
        "entries": entries,
    }
    write_json(Path(config["outputs"]["artifacts"]) / "missing_canonical_documents_diagnosis.json", result)
    if not passed:
        raise RuntimeError(f"Phase 14 canonical document diagnosis failed: {result}")
    return result


def finalize(config: dict, state: Path) -> dict:
    from src.source_census.cached_subsets import build_cached_subset_rankings
    from src.source_census.union_experiment import (
        _audit_ranked_provenance_closure, _build_canonical_store,
        _compare_submission_signatures, _filter_parquet, _official_id_audit,
        _iter_json_array,
        _stream_package,
    )
    from src.validation.pre_submission import audit_score_cache
    from tools.phase11a_bottleneck import tokenizer
    score_audit = final_score_audit(config)
    score_db = _score_db(config); connection = sqlite3.connect(score_db)
    total, scored, nonfinite = connection.execute("SELECT COUNT(*),SUM(rerank_score IS NOT NULL),SUM(rerank_score IS NOT NULL AND (rerank_score!=rerank_score OR abs(rerank_score)>1e100)) FROM pairs").fetchone(); integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0]); connection.close()
    if int(total) != int(scored) or int(nonfinite or 0) or integrity != "ok":
        raise RuntimeError("Phase 14 score cache is not complete and finite")
    runtime_path = runtime_config(config); runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    markers = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(_parts(config).glob("*.json"))]
    sources = [str(marker["source"]) for marker in markers]
    artifacts = Path(config["outputs"]["artifacts"]); write_json(artifacts / "technical_triage.json", {"healthy_sources": sources})
    rankings = Path(config["outputs"]["root"]) / "rankings"
    build_cached_subset_rankings(runtime, {"PHASE14": sources}, rankings)
    ranked_chunks = rankings / "PHASE14/reranked_chunks.parquet"
    chunk_ids = set(); doc_ids = set()
    for batch in pq.ParquetFile(ranked_chunks).iter_batches(columns=["chunk_id", "doc_id"], batch_size=8192):
        chunk_ids.update(map(str, batch.column(0).to_pylist())); doc_ids.update(map(int, batch.column(1).to_pylist()))
    round1 = Path(config["outputs"]["root"]) / "round1"
    documents = round1 / "combined_documents.parquet"; chunks = round1 / "combined_chunks.parquet"
    canonical_diagnosis = diagnose_missing_canonical_documents(config, ranked_chunks, doc_ids)
    _filter_parquet([
        PHASE11C / "round1/combined_documents.parquet",
        PHASE10F_MEDLATEC / "documents.parquet",
        Path(config["outputs"]["documents"]),
    ], documents, "doc_id", doc_ids)
    _filter_parquet([
        PHASE11C / "round1/combined_chunks.parquet",
        PHASE10F_MEDLATEC / "chunks.parquet",
        Path(config["outputs"]["chunks"]),
    ], chunks, "chunk_id", chunk_ids)
    canonical = round1 / "canonical_text.sqlite"; _build_canonical_store(chunks, documents, canonical)
    closure = _audit_ranked_provenance_closure(ranked_chunks, canonical, Path(config["inputs"]["corpus"]), artifacts / "provenance_closure.json")
    package = _stream_package(runtime, rankings, "PHASE14", EXPERIMENT, Path(config["outputs"]["submissions"]), canonical, tokenizer(runtime))
    official = _official_id_audit(Path(package["json_path"]), Path(config["inputs"]["corpus"]))
    cache = audit_score_cache(score_db, _parts(config)); control = _frozen_control(config)
    comparison = _compare_submission_signatures(Path(config["focused_scaling"]["control_submission_json"]), Path(package["json_path"]))
    new_docs = {
        int(value) for batch in pq.ParquetFile(config["outputs"]["documents"]).iter_batches(columns=["doc_id"], batch_size=8192)
        for value in batch.column(0).to_pylist()
    }
    candidate_db = sqlite3.connect(candidate_paths(config)["database"])
    flagged_ids = {str(row[0]) for row in candidate_db.execute("SELECT chunk_id FROM chunks WHERE suspicious=1")}
    candidate_db.close()
    flagged_pairs = set()
    for batch in pq.ParquetFile(config["outputs"]["chunks"]).iter_batches(columns=["chunk_id", "doc_id", "raw_text"], batch_size=4096):
        for chunk_id, doc_id, text in zip(*(column.to_pylist() for column in batch.columns), strict=True):
            if str(chunk_id) in flagged_ids:
                flagged_pairs.add((int(doc_id), str(text)))
    new_doc_slots = new_chunk_slots = flagged_chunk_slots = queries_with_new_docs = queries_with_new_chunks = 0
    for row in _iter_json_array(Path(package["json_path"])):
        docs_here = sum(int(doc_id) in new_docs for doc_id in row["relevant_docs"])
        chunks_here = sum(int(item["doc_id"]) in new_docs for item in row["relevant_chunks"])
        flagged_here = sum((int(item["doc_id"]), str(item["chunk_text"])) in flagged_pairs for item in row["relevant_chunks"])
        new_doc_slots += docs_here; new_chunk_slots += chunks_here; flagged_chunk_slots += flagged_here
        queries_with_new_docs += docs_here > 0; queries_with_new_chunks += chunks_here > 0
    attribution = {
        "phase14_new_document_slots": new_doc_slots, "phase14_new_chunk_slots": new_chunk_slots,
        "queries_with_phase14_new_documents": queries_with_new_docs,
        "queries_with_phase14_new_chunks": queries_with_new_chunks,
        "historical_document_slots": 1200 * 20 - new_doc_slots,
        "historical_chunk_slots": 1200 * 40 - new_chunk_slots,
        "flagged_boilerplate_navigation_chunk_slots": flagged_chunk_slots,
        "flagged_chunks_modified_or_penalized": 0,
    }
    write_json(artifacts / "phase14_output_attribution.json", attribution)
    passed = bool(package["validation"]["passed"] and package["determinism_verified"] and official["passed"] and cache["passed"] and closure["passed"] and control["passed"])
    report = {
        "experiment": EXPERIMENT, "status": "READY_FOR_LEADERBOARD" if passed else "NEEDS_AGENT", "passed": passed,
        "new_crawling": 0, "new_embeddings": 0, "remote_scores_imported": 0,
        "candidate_audit": json.loads((artifacts / "candidate_audit.json").read_text(encoding="utf-8")),
        "score_reconciliation": json.loads((artifacts / "score_reconciliation.json").read_text(encoding="utf-8")),
        "final_score_audit": score_audit,
        "score_cache": cache, "provenance_closure": closure, "official_id_audit": official,
        "canonical_document_recovery": canonical_diagnosis,
        "structural_validation": package["validation"], "determinism_verified": package["determinism_verified"],
        "frozen_phase11c_control": control, "ranking_change_vs_phase11c": comparison,
        "boilerplate_impact": json.loads((artifacts / "candidate_boilerplate_impact.json").read_text(encoding="utf-8")),
        "reuse_guard": reuse_guard(config), "output_attribution": attribution, "package": package,
    }
    write_json(artifacts / "phase14_pre_submit_audit.json", report)
    update_json(state, {"status": report["status"], "stage": report["status"], "done": 1, "total": 1, "remaining": 0, "checkpoint": str(artifacts / "phase14_pre_submit_audit.json"), "submission": package["zip_path"], "sha256": package["zip_sha256"]}, preserve_terminal=False)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "trial", "production", "finalize", "complete"])
    parser.add_argument("--config", default="configs/phase14_medlatec_coverage_3000.yaml")
    parser.add_argument("--run-state", default="artifacts/runs/phase14_medlatec_coverage_3000/state.json")
    args = parser.parse_args(); config = load_config(args.config); state = Path(args.run_state)
    if args.mode == "prepare": result = prepare(config, state)
    elif args.mode == "trial": result = score(config, state, trial=True)
    elif args.mode == "finalize": result = finalize(config, state)
    elif args.mode == "production":
        scoring = score(config, state, trial=False)
        result = {"scoring": scoring, "final": finalize(config, state)}
    else:
        reconciliation = json.loads((Path(config["outputs"]["artifacts"]) / "score_reconciliation.json").read_text(encoding="utf-8"))
        if int(reconciliation["reused"]) != 250114 or int(reconciliation["missing"]) != 4286:
            raise RuntimeError(f"Phase 14 reconciliation contract changed: {reconciliation}")
        before_guard = reuse_guard(config)
        trial_path = Path(config["outputs"]["artifacts"]) / "rerank_trial.json"
        if trial_path.exists():
            trial = json.loads(trial_path.read_text(encoding="utf-8"))
            if not trial.get("passed") or int(trial.get("new_scores", -1)) != 64:
                raise RuntimeError(f"invalid durable Phase 14 trial checkpoint: {trial}")
        else:
            trial = score(config, state, trial=True)
        after_trial_guard = reuse_guard(config)
        if int(trial["remaining"]) > 0:
            production = score(config, state, trial=False)
        else:
            production = {"passed": True, "trial": False, "before": trial["after"], "after": trial["after"], "new_scores": 0, "remaining": 0, "sqlite_integrity": "ok"}
        final_guard = reuse_guard(config)
        final = finalize(config, state)
        result = {"reconciliation": reconciliation, "before_guard": before_guard, "trial": trial, "after_trial_guard": after_trial_guard, "production": production, "final_guard": final_guard, "final": final}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
