from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


REQUIRED = {
    "silver_query_id", "query_text", "positive_doc_id", "source_url",
    "source_domain", "query_language", "document_language", "split",
    "provenance_type", "verbatim_source_query",
}


def normalize_query(text: str) -> str:
    return " ".join(re.findall(r"\w+", str(text).casefold(), flags=re.UNICODE))


def validate_benchmark(path: str | Path, official_ids: set[int]) -> dict[str, Any]:
    table = pq.read_table(path)
    missing = REQUIRED - set(table.column_names)
    if missing:
        raise ValueError(f"silver benchmark missing columns: {sorted(missing)}")
    rows = table.to_pylist()
    ids = [str(row["silver_query_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate silver_query_id")
    pairs = [(str(row["silver_query_id"]), int(row["positive_doc_id"])) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate silver query-positive pair")
    invalid = sorted({doc for _, doc in pairs if doc not in official_ids})
    if invalid:
        raise ValueError(f"silver positives are not NEW-official: {invalid[:20]}")
    if any(not str(row["source_url"]).startswith(("http://", "https://")) for row in rows):
        raise ValueError("silver positive provenance lacks source URL")
    normalized_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        normalized_splits[normalize_query(row["query_text"])].add(str(row["split"]))
    leakage = [query for query, splits in normalized_splits.items() if len(splits) > 1]
    if leakage:
        raise ValueError(f"normalized query leakage across splits: {leakage[:10]}")
    official_query_ids = [value for value in ids if value.isdigit() and 1 <= int(value) <= 1200]
    if official_query_ids:
        raise ValueError("silver IDs overlap the organizer query-ID namespace")
    return {
        "passed": True, "queries": len(rows), "positive_pairs": len(pairs),
        "official_positive_ids": len({doc for _, doc in pairs}),
        "normalized_cross_split_leakage": 0,
        "verbatim_source_queries": sum(bool(row["verbatim_source_query"]) for row in rows),
    }


def evaluate_rankings(benchmark: str | Path, rankings: str | Path, official_ids: set[int]) -> dict[str, Any]:
    """Evaluate known-source positives; never accepts organizer query IDs as silver IDs."""
    started = time.perf_counter()
    validation = validate_benchmark(benchmark, official_ids)
    positives: dict[str, set[int]] = defaultdict(set)
    metadata = {}
    for row in pq.read_table(benchmark).to_pylist():
        qid = str(row["silver_query_id"]); positives[qid].add(int(row["positive_doc_id"])); metadata[qid] = row
    ranked: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    table = pq.read_table(rankings)
    required = {"silver_query_id", "rank", "doc_id"}
    if not required <= set(table.column_names):
        raise ValueError(f"ranking file missing columns: {sorted(required-set(table.column_names))}")
    for row in table.to_pylist():
        qid = str(row["silver_query_id"])
        if qid not in positives:
            raise ValueError(f"ranking contains unknown silver query: {qid}")
        doc = int(row["doc_id"]); rank = int(row["rank"])
        if doc not in official_ids or rank < 1:
            raise ValueError(f"invalid ranked candidate: query={qid} rank={rank} doc={doc}")
        ranked[qid].append((rank, doc, str(row.get("source_domain") or "UNKNOWN")))
    reciprocal = []; hit20 = []; hit100 = []; coverage = []; diversity = []
    for qid, expected in positives.items():
        rows = sorted(ranked.get(qid, [])); docs = [doc for _, doc, _ in rows]
        if len(docs) != len(set(docs)):
            raise ValueError(f"duplicate ranked document for {qid}")
        matching = [rank for rank, doc, _ in rows if doc in expected]
        reciprocal.append(1.0 / min(matching) if matching else 0.0)
        hit20.append(any(rank <= 20 and doc in expected for rank, doc, _ in rows))
        hit100.append(any(rank <= 100 and doc in expected for rank, doc, _ in rows))
        coverage.append(bool(rows)); diversity.append(len({domain for _, _, domain in rows[:100]}))
    return {
        "benchmark_validation": validation,
        "known_positive_recall_at_20": sum(hit20) / len(hit20),
        "known_positive_recall_at_100": sum(hit100) / len(hit100),
        "mrr": sum(reciprocal) / len(reciprocal),
        "candidate_coverage": sum(coverage) / len(coverage),
        "mean_source_diversity_at_100": sum(diversity) / len(diversity),
        "queries": len(positives), "runtime_seconds": time.perf_counter() - started,
        "scope_warning": "Known-source proxy metrics only; not organizer Recall@K or hidden gold.",
    }
