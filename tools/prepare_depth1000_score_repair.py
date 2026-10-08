#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_census.depth1000 import seed_depth_scores
from src.source_census.pipeline import atomic_json, load_config


def key_text_digest(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for row in connection.execute(
        "SELECT query_id,chunk_id,query_text,chunk_text FROM pairs ORDER BY query_id,chunk_id"
    ):
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/source_census_depth1000.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["outputs"]["root"]) / "round1"
    database = root / "source_scores.sqlite"
    backup = root / "source_scores.corrupted_20261008.sqlite"
    connection = sqlite3.connect(database)
    before = {
        "total": int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]),
        "scored": int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0]),
        "distinct_scores": int(connection.execute("SELECT COUNT(DISTINCT rerank_score) FROM pairs").fetchone()[0]),
        "integrity": str(connection.execute("PRAGMA integrity_check").fetchone()[0]),
        "candidate_identity_sha256": key_text_digest(connection),
    }
    if before != {**before, "total": 86400, "scored": 86400, "distinct_scores": 1, "integrity": "ok"}:
        raise RuntimeError(f"unexpected corrupt-cache precondition: {before}")
    if not backup.exists():
        backup_connection = sqlite3.connect(backup)
        connection.backup(backup_connection)
        backup_connection.close()
    with connection:
        connection.execute("UPDATE pairs SET rerank_score=NULL,inference_ms=NULL")
    after_reset = int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0])
    identity_after_reset = key_text_digest(connection)
    connection.close()
    if after_reset or identity_after_reset != before["candidate_identity_sha256"]:
        raise RuntimeError("score reset changed candidate identity/text")

    seed = seed_depth_scores(config)
    connection = sqlite3.connect(database)
    seeded_distribution = connection.execute(
        "SELECT COUNT(*),COUNT(DISTINCT rerank_score),MIN(rerank_score),AVG(rerank_score),MAX(rerank_score) "
        "FROM pairs WHERE rerank_score IS NOT NULL"
    ).fetchone()
    identity_after_seed = key_text_digest(connection)
    connection.close()
    if identity_after_seed != before["candidate_identity_sha256"] or int(seeded_distribution[1]) <= 1:
        raise RuntimeError("repaired seed failed identity or non-degeneracy check")
    result = {
        "corrupt_cache_backup": str(backup), "before": before,
        "candidate_identity_preserved": True, "reset_scored_rows": after_reset,
        "seed": seed,
        "seeded_distribution": {
            "count": int(seeded_distribution[0]), "distinct": int(seeded_distribution[1]),
            "min": float(seeded_distribution[2]), "mean": float(seeded_distribution[3]),
            "max": float(seeded_distribution[4]),
        },
    }
    atomic_json(Path(config["outputs"]["artifacts"]) / "rerank_repair_seed.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
