#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_census.cached_subsets import _require_complete_score_cache, build_cached_subset_rankings
from src.source_census.depth1000 import DEPTH_GROUPS, finalize_depth_report
from src.source_census.pipeline import atomic_json, load_config, score_database_status
from src.validation.pre_submission import audit_depth1000


def stage(path: Path, name: str, **values: object) -> None:
    atomic_json(path, {"stage": name, "status": "RUNNING", **values})


def compare_submissions(config: dict, fixed: dict[str, object]) -> dict[str, object]:
    docs = pq.read_table(config["outputs"]["documents"], columns=["doc_id", "original_url"]).to_pylist()
    source_by_doc = {
        int(row["doc_id"]): (urlsplit(str(row["original_url"])).hostname or "").lower().removeprefix("www.")
        for row in docs
    }
    comparison = {}
    for group in DEPTH_GROUPS:
        corrupt = json.loads((ROOT / "submissions" / f"phase10d_DEPTH1000_{group}.json").read_text(encoding="utf-8"))
        repaired = json.loads(Path(fixed[group]["json_path"]).read_text(encoding="utf-8"))
        old = {int(row["id"]): row for row in corrupt}; new = {int(row["id"]): row for row in repaired}
        doc_queries = chunk_queries = displaced = 0
        old_sources: Counter[str] = Counter(); new_sources: Counter[str] = Counter()
        for query_id in sorted(old):
            old_docs = list(map(int, old[query_id]["relevant_docs"])); new_docs = list(map(int, new[query_id]["relevant_docs"]))
            doc_queries += old_docs != new_docs
            old_chunks = [(int(x["doc_id"]), str(x["chunk_text"])) for x in old[query_id]["relevant_chunks"]]
            new_chunks = [(int(x["doc_id"]), str(x["chunk_text"])) for x in new[query_id]["relevant_chunks"]]
            chunk_queries += old_chunks != new_chunks
            displaced += len(set(old_docs) - set(new_docs))
            old_sources.update(source_by_doc.get(doc, "pilot/control") for doc in old_docs)
            new_sources.update(source_by_doc.get(doc, "pilot/control") for doc in new_docs)
        comparison[group] = {
            "queries_changed_top10_docs": doc_queries,
            "queries_changed_top20_chunks": chunk_queries,
            "displaced_document_slots": displaced,
            "corrupted_top10_source_distribution": dict(sorted(old_sources.items())),
            "fixed_top10_source_distribution": dict(sorted(new_sources.items())),
        }
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/source_census_depth1000.yaml")
    parser.add_argument("--run-state", required=True)
    args = parser.parse_args(); config = load_config(args.config)
    state = Path(args.run_state); outputs = config["outputs"]; python = sys.executable
    seed = json.loads((Path(outputs["artifacts"]) / "rerank_repair_seed.json").read_text(encoding="utf-8"))
    stage(state, "RERANK", **score_database_status(config), exact_reused=seed["seed"]["reused"])
    restarts = 0
    while score_database_status(config)["remaining"]:
        result = subprocess.run([
            python, "tools/score_source_census_persistent.py", "--config", args.config,
            "--window-size", "2000", "--run-state", args.run_state,
            "--metrics-out", str(Path(outputs["artifacts"]) / "rerank_repair_metrics.json"),
            "--model-load-count", str(restarts + 1), "--scorer-restarts", str(restarts),
        ], cwd=ROOT)
        if result.returncode:
            restarts += 1
            if restarts > 10:
                raise RuntimeError("repaired scorer exceeded restart limit")
    status = _require_complete_score_cache(config)
    database = Path(outputs["root"]) / "round1/source_scores.sqlite"
    connection = sqlite3.connect(database)
    audit = {
        "total": int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]),
        "scored": int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0]),
        "missing": int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NULL").fetchone()[0]),
        "duplicate_keys": int(connection.execute("SELECT COUNT(*) FROM (SELECT query_id,chunk_id,COUNT(*) n FROM pairs GROUP BY query_id,chunk_id HAVING n>1)").fetchone()[0]),
        "nonfinite_scores": int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score!=rerank_score OR ABS(rerank_score)>1e100").fetchone()[0]),
        "invalid_inference_timings": int(connection.execute("SELECT COUNT(*) FROM pairs WHERE inference_ms IS NULL OR inference_ms<0").fetchone()[0]),
        "distinct_scores": int(connection.execute("SELECT COUNT(DISTINCT rerank_score) FROM pairs").fetchone()[0]),
        "minimum_score": float(connection.execute("SELECT MIN(rerank_score) FROM pairs").fetchone()[0]),
        "mean_score": float(connection.execute("SELECT AVG(rerank_score) FROM pairs").fetchone()[0]),
        "maximum_score": float(connection.execute("SELECT MAX(rerank_score) FROM pairs").fetchone()[0]),
        "integrity": str(connection.execute("PRAGMA integrity_check").fetchone()[0]),
    }
    connection.close()
    if audit["total"] != 86400 or audit["scored"] != 86400 or audit["missing"] or audit["duplicate_keys"] or audit["nonfinite_scores"] or audit["invalid_inference_timings"] or audit["integrity"] != "ok" or audit["distinct_scores"] <= 1:
        raise RuntimeError(f"repaired score audit failed: {audit}")

    ranking_root = Path(outputs["root"]) / "rankings_fixed"
    stage(state, "RANKINGS_FIXED", score_audit=audit)
    build_cached_subset_rankings(config, DEPTH_GROUPS, ranking_root)
    stage(state, "PACKAGE_FIXED", completed=0, total=3)
    submissions = {}
    for index, group in enumerate(DEPTH_GROUPS, 1):
        marker = Path(outputs["artifacts"]) / f"submission_FIXED_{group}.json"
        subprocess.run([
            python, "tools/source_census_cached_subset.py", "--config", args.config,
            "package-existing", "--ranking-root", str(ranking_root), "--group", group,
            "--submission-name", f"phase10d_DEPTH1000_FIXED_{group}",
            "--output-dir", outputs["submissions"], "--marker", str(marker),
        ], cwd=ROOT, check=True)
        submissions[group] = json.loads(marker.read_text(encoding="utf-8"))
        stage(state, "PACKAGE_FIXED", completed=index, total=3, current_group=group)
    base = finalize_depth_report(config, submissions)
    report = {
        **base, "repair": seed, "score_audit": audit,
        "corrected_vs_corrupted": compare_submissions(config, submissions),
        "corrupted_submissions_status": "INVALID_EXPERIMENT_CACHE_CORRUPTION",
    }
    report_path = Path(outputs["artifacts"]).parent / "depth1000_fixed_report.json"
    atomic_json(report_path, report)
    submission_manifest = Path(outputs["artifacts"]) / "fixed_submission_manifest.json"
    atomic_json(submission_manifest, submissions)
    pre_submit = audit_depth1000(config, submissions, ranking_root)
    audit_path = Path("artifacts/validation/phase10d_depth1000_fixed_pre_submit_audit.json")
    atomic_json(audit_path, pre_submit)
    terminal = str(pre_submit["status"])
    atomic_json(state, {"stage":terminal,"status":terminal,"report":str(report_path),"pre_submit_audit":str(audit_path),"score_audit":audit})
    if terminal != "READY_FOR_LEADERBOARD":
        raise RuntimeError(f"pre-submission scientific gate failed: {pre_submit['failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
