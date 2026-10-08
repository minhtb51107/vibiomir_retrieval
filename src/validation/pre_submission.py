from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import statistics
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pyarrow.parquet as pq


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score_distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(len(ordered) * .95) - 1) if ordered else 0
    return {
        "count": len(ordered), "distinct": len(set(ordered)),
        "min": ordered[0] if ordered else None,
        "median": statistics.median(ordered) if ordered else None,
        "p95": ordered[p95_index] if ordered else None,
        "max": ordered[-1] if ordered else None,
    }


def audit_score_cache(database: str | Path, candidate_parts: str | Path) -> dict[str, Any]:
    database = Path(database); candidate_parts = Path(candidate_parts)
    connection = sqlite3.connect(database)
    connection.execute("CREATE TEMP TABLE expected(query_id INTEGER,chunk_id TEXT,PRIMARY KEY(query_id,chunk_id)) WITHOUT ROWID")
    expected_rows = 0
    for path in sorted(candidate_parts.glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(columns=["query_id", "chunk_id"], batch_size=8192):
            rows = [(int(q), str(c)) for q, c in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True)]
            expected_rows += len(rows)
            connection.executemany("INSERT OR IGNORE INTO expected VALUES (?,?)", rows)
    unique_expected = int(connection.execute("SELECT COUNT(*) FROM expected").fetchone()[0])
    total = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
    scored = int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0])
    missing = int(connection.execute("SELECT COUNT(*) FROM expected e LEFT JOIN pairs p ON p.query_id=e.query_id AND p.chunk_id=e.chunk_id WHERE p.query_id IS NULL").fetchone()[0])
    unexpected = int(connection.execute("SELECT COUNT(*) FROM pairs p LEFT JOIN expected e ON e.query_id=p.query_id AND e.chunk_id=p.chunk_id WHERE e.query_id IS NULL").fetchone()[0])
    duplicate = expected_rows - unique_expected + int(connection.execute("SELECT COUNT(*) FROM (SELECT query_id,chunk_id,COUNT(*) n FROM pairs GROUP BY query_id,chunk_id HAVING n>1)").fetchone()[0])
    nonfinite = int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL AND (rerank_score!=rerank_score OR ABS(rerank_score)>1e100)").fetchone()[0])
    invalid_timing = int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL AND (inference_ms IS NULL OR inference_ms<0 OR inference_ms!=inference_ms)").fetchone()[0])
    scores = [float(row[0]) for row in connection.execute("SELECT rerank_score FROM pairs WHERE rerank_score IS NOT NULL")]
    distinct_timings = int(connection.execute("SELECT COUNT(DISTINCT inference_ms) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0])
    integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0]); connection.close()
    distribution = score_distribution(scores)
    failures = []
    if not (expected_rows == unique_expected == total): failures.append("candidate/stored key counts differ")
    if missing: failures.append(f"missing keys={missing}")
    if unexpected: failures.append(f"unexpected keys={unexpected}")
    if duplicate: failures.append(f"duplicate keys={duplicate}")
    if scored != total: failures.append(f"non-null scored={scored}, total={total}")
    if nonfinite: failures.append(f"non-finite scores={nonfinite}")
    if invalid_timing: failures.append(f"invalid inference timings={invalid_timing}")
    if integrity != "ok": failures.append(f"SQLite integrity={integrity}")
    if total >= 1000 and distribution["distinct"] <= 1: failures.append("degenerate rerank scores")
    if total >= 1000 and distinct_timings <= 1: failures.append("degenerate inference timings")
    return {
        "passed": not failures, "failures": failures,
        "expected_candidate_rows": expected_rows, "expected_unique_keys": unique_expected,
        "stored_key_count": total, "missing_keys": missing, "unexpected_keys": unexpected,
        "duplicate_keys": duplicate, "non_null_scored_count": scored,
        "non_finite_scores": nonfinite, "invalid_inference_timings": invalid_timing,
        "sqlite_integrity": integrity, "score_distribution": distribution,
        "distinct_inference_ms_count": distinct_timings,
    }


def audit_reuse(destination: str | Path, source: str | Path, seed: dict[str, Any]) -> dict[str, Any]:
    connection = sqlite3.connect(destination)
    connection.execute("ATTACH DATABASE ? AS srcdb", (str(Path(source).resolve()),))
    intersection = int(connection.execute("SELECT COUNT(*) FROM pairs d JOIN srcdb.pairs s ON s.query_id=d.query_id AND s.chunk_id=d.chunk_id WHERE s.rerank_score IS NOT NULL").fetchone()[0])
    mismatches = int(connection.execute("SELECT COUNT(*) FROM pairs d JOIN srcdb.pairs s ON s.query_id=d.query_id AND s.chunk_id=d.chunk_id WHERE s.rerank_score IS NOT NULL AND (d.rerank_score!=s.rerank_score OR d.inference_ms!=s.inference_ms)").fetchone()[0])
    total = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]); connection.close()
    actual = int(seed.get("reused", -1)); new_required = total - intersection
    nonmatches_seeded = int(seed.get("incorrectly_seeded_nonmatches", -1))
    failures = []
    if actual != intersection: failures.append(f"actual reuse={actual}, exact intersection={intersection}")
    if mismatches: failures.append(f"reused value mismatches={mismatches}")
    if nonmatches_seeded: failures.append(f"nonmatching keys seeded={nonmatches_seeded}")
    if new_required > 0 and int(seed.get("missing", -1)) == 0: failures.append("expanded corpus reported zero new inference")
    return {"passed":not failures,"failures":failures,"expected_reuse":intersection,"actual_reuse":actual,"reused_value_mismatches":mismatches,"nonmatching_keys_seeded_before_inference":nonmatches_seeded,"new_inference_required":new_required}


def semantic_submission_comparison(original: str | Path, replay: str | Path) -> dict[str, Any]:
    old=json.loads(Path(original).read_text(encoding="utf-8")); new=json.loads(Path(replay).read_text(encoding="utf-8"))
    old_by={int(r["id"]):r for r in old}; new_by={int(r["id"]):r for r in new}
    ids_equal=list(old_by)==list(new_by)
    docs=sum(old_by[q]["relevant_docs"]==new_by[q]["relevant_docs"] for q in old_by if q in new_by)
    chunks=sum(old_by[q]["relevant_chunks"]==new_by[q]["relevant_chunks"] for q in old_by if q in new_by)
    passed=ids_equal and docs==len(old_by) and chunks==len(old_by)
    return {"passed":passed,"query_ids_and_order_equal":ids_equal,"query_count":len(old_by),"identical_top10_docs":docs,"identical_top20_chunks":chunks,"original_sha256":sha256_file(original),"replay_sha256":sha256_file(replay)}


def audit_depth1000(config: dict[str, Any], submissions: dict[str, Any], fixed_rankings: str | Path) -> dict[str, Any]:
    root=Path(config["outputs"]["root"]); artifacts=Path(config["outputs"]["artifacts"])
    seed=json.loads((artifacts/"rerank_repair_seed.json").read_text(encoding="utf-8"))["seed"]
    cache=audit_score_cache(root/"round1/source_scores.sqlite",root/"round1/source_candidate_parts")
    reuse=audit_reuse(root/"round1/source_scores.sqlite",config["depth1000"]["existing_score_database"],seed)
    replay=semantic_submission_comparison("submissions/phase10d_R3_G1A.json","data/source_census/control_replay_g1a/phase10d_R3_G1A_control_replay.json")
    from src.source_census.pipeline import load_config
    baseline=load_config("configs/source_census.yaml")
    fixed_fields = {
        "query_set": config["inputs"]["queries"] == baseline["inputs"]["queries"],
        "control_documents": config["inputs"]["pilot_documents"] == baseline["inputs"]["pilot_documents"],
        "control_chunks": config["inputs"]["pilot_chunks"] == baseline["inputs"]["pilot_chunks"],
        "control_candidate_pool": config["inputs"]["pilot_rerank_pool"] == baseline["inputs"]["pilot_rerank_pool"],
        "control_scores": config["inputs"]["pilot_rerank_scores"] == baseline["inputs"]["pilot_rerank_scores"],
        "candidate_policy": (
            all(config["candidate_cache"][key] == baseline["candidate_cache"][key]
                for key in ("dense_depth","sparse_depth","rrf_constant","bm25_k1","bm25_b"))
            and 8 in config["candidate_cache"]["candidate_caps"]
            and 8 in baseline["candidate_cache"]["candidate_caps"]
        ),
        "embedder_model_revision": config["models"]["embedder"] == baseline["models"]["embedder"],
        "reranker_model_revision": config["models"]["reranker"] == baseline["models"]["reranker"],
        "top_k_and_provenance_policy": config["submission"] == baseline["submission"],
    }
    contract={
        "intended_variable":"source acquisition depth only",
        "query_set":config["inputs"]["queries"],"query_sha256":sha256_file(config["inputs"]["queries"]),
        "control_corpus":{"documents":config["inputs"]["pilot_documents"],"chunks":config["inputs"]["pilot_chunks"]},
        "candidate_policy":config["candidate_cache"],"embedder":config["models"]["embedder"],
        "reranker":config["models"]["reranker"],"submission_policy":config["submission"],
        "chunk_policy":{"content_tokens":508,"special_tokens":2,"overlap":64,"source_derived":True},
        "fixed_field_checks":fixed_fields,"passed":all(fixed_fields.values()),
    }
    shallow_docs=pq.read_table(config["depth1000"]["existing_round1_documents"],columns=["doc_id","original_url"]).to_pylist()
    new_docs=pq.read_table(config["outputs"]["documents"],columns=["doc_id","original_url"]).to_pylist()
    shallow_chunks=pq.read_table(config["depth1000"]["existing_round1_chunks"],columns=["chunk_id","doc_id"]).to_pylist()
    new_chunks=pq.read_table(config["outputs"]["chunks"],columns=["chunk_id","doc_id"]).to_pylist()
    target=set(sum(config.get("depth1000_groups",{}).values(),[]))
    # Depth config has fixed groups in its manifest; use their source membership.
    manifest=json.loads(Path(config["depth1000"]["manifest"]).read_text(encoding="utf-8")); target=set(s for g in manifest["groups"].values() for s in g)
    def host(u): return (urlsplit(str(u)).hostname or "").lower().removeprefix("www.")
    old_ids={int(r["doc_id"]) for r in shallow_docs if host(r["original_url"]) in target}; new_ids={int(r["doc_id"]) for r in new_docs if host(r["original_url"]) in target}
    old_chunk_ids={str(r["chunk_id"]) for r in shallow_chunks if int(r["doc_id"]) in old_ids}; new_chunk_ids={str(r["chunk_id"]) for r in new_chunks if int(r["doc_id"]) in new_ids}
    changes={}
    shallow_name={"G1A":"R3_G1A","G5A":"R3_G5A","G6B":"R3_G6B"}
    for group,row in submissions.items():
        fixed=json.loads(Path(row["json_path"]).read_text(encoding="utf-8")); old=json.loads(Path(f"submissions/phase10d_{shallow_name[group]}.json").read_text(encoding="utf-8"))
        f={int(x["id"]):x for x in fixed}; o={int(x["id"]):x for x in old}
        changes[group]={"queries_top10_changed":sum(o[q]["relevant_docs"]!=f[q]["relevant_docs"] for q in o),"queries_top20_changed":sum(o[q]["relevant_chunks"]!=f[q]["relevant_chunks"] for q in o)}
    accounting={"old_docs":len(old_ids),"new_depth_docs":len(new_ids-old_ids),"union_docs":len(old_ids|new_ids),"old_chunks":len(old_chunk_ids),"new_depth_chunks":len(new_chunk_ids-old_chunk_ids),"union_chunks":len(old_chunk_ids|new_chunk_ids),"old_candidate_pairs_retained":reuse["expected_reuse"],"new_candidate_pairs":reuse["new_inference_required"],"reused_rerank_pairs":reuse["actual_reuse"],"newly_inferred_pairs":cache["non_null_scored_count"]-reuse["actual_reuse"],"ranking_changes":changes}
    structural={g:{"passed":bool(r.get("validation",{}).get("valid")) and bool(r.get("determinism_verified")),"validation":r.get("validation"),"determinism_verified":r.get("determinism_verified")} for g,r in submissions.items()}
    failures=[]
    for name,item in (("score_cache",cache),("cache_reuse",reuse),("control_replay",replay)):
        if not item["passed"]: failures.append(name+": "+"; ".join(item.get("failures",["failed"])))
    if not contract["passed"]: failures.append("experiment contract changed outside source depth")
    if not all(x["passed"] for x in structural.values()): failures.append("structural submission validation failed")
    return {"status":"READY_FOR_LEADERBOARD" if not failures else "NEEDS_AGENT","passed":not failures,"failures":failures,"score_cache_integrity":cache,"score_distribution_sanity":{"passed":cache["passed"],"distribution":cache["score_distribution"],"distinct_inference_ms_count":cache["distinct_inference_ms_count"]},"cache_reuse_audit":reuse,"control_replay":replay,"experiment_contract":contract,"change_accounting":accounting,"structural_validation":structural}
