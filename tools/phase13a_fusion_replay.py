#!/usr/bin/env python3
"""CPU-only Phase 13A aggregation/fusion replay over frozen reranker scores."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import numpy as np
import psutil
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_census.union_experiment import _iter_json_array, _submission_records_disk
from tools.phase12_worker import tokenizer


OUT = ROOT / "artifacts/phase13a_fusion"
P11_JSON = ROOT / "submissions/00_READY_TO_UPLOAD/phase11c_BROAD_EXISTING_NEW_DOCS20_CHUNKS40.json"
P12_JSON = ROOT / "submissions/00_READY_TO_UPLOAD/phase12_FULL_PUBMED_DENSE_DOCS20_CHUNKS40.json"
P11_ROOT = ROOT / "data/source_census/phase11c_broad_existing_new"
P12_ROOT = ROOT / "data/pubmed_phase12/retrieval"
PILOT_POOL = ROOT / "data/source_census/phase11b_new_corpus/round1/pilot_rerank_pool.new_official.parquet"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def update_run_state(path: Path | None, **updates: object) -> None:
    """Atomically publish worker progress without inventing supervisor state."""
    if path is None:
        return
    current: dict[str, Any] = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(updates)
    current["worker_updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_json(path, current)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: Iterable[float | int]) -> dict[str, float | int | None]:
    values = list(values)
    if not values:
        return {"count": 0, "min": None, "p5": None, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values), "min": float(array.min()), "p5": float(np.percentile(array, 5)),
        "median": float(np.median(array)), "p95": float(np.percentile(array, 95)), "max": float(array.max()),
    }


def normalized_evidence(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def near_duplicate(left: str, right: str, threshold: float = 0.85) -> bool:
    """Deterministic token-set overlap; exact normalized duplicates always match."""
    left = normalized_evidence(left); right = normalized_evidence(right)
    if left == right:
        return True
    a, b = set(left.split()), set(right.split())
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= threshold


def representative_chunks(rows: list[dict[str, Any]], *, top_k: int, robust: bool) -> tuple[list[dict], int]:
    ordered = sorted(rows, key=lambda row: (-float(row["score"]), str(row["chunk_id"])))
    selected: list[dict] = []; suppressed = 0
    for row in ordered:
        if robust and any(near_duplicate(str(row["chunk_text"]), str(old["chunk_text"])) for old in selected):
            suppressed += 1
            continue
        selected.append(row)
        if len(selected) == top_k:
            break
    return selected, suppressed


def aggregate_documents(rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    grouped_by_chunk: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        doc_id = int(row["doc_id"]); chunk_id = str(row["chunk_id"])
        previous = grouped_by_chunk[doc_id].get(chunk_id)
        if previous is None:
            grouped_by_chunk[doc_id][chunk_id] = {**row, "candidate_arms": sorted(set(row["candidate_arms"]))}
            continue
        identity = (float(row["score"]), str(row["chunk_text"]), str(row["canonical_url"]), str(row["canonical_domain"]))
        old_identity = (float(previous["score"]), str(previous["chunk_text"]), str(previous["canonical_url"]), str(previous["canonical_domain"]))
        if identity != old_identity:
            raise ValueError(f"conflicting exact chunk identity for document {doc_id}, chunk {chunk_id}")
        previous["candidate_arms"] = sorted(set(previous["candidate_arms"]) | set(row["candidate_arms"]))
    output = []
    for doc_id, by_chunk in grouped_by_chunk.items():
        chunks = list(by_chunk.values())
        if method == "max_chunk":
            chosen, suppressed = representative_chunks(chunks, top_k=1, robust=False)
        elif method == "top2_mean":
            chosen, suppressed = representative_chunks(chunks, top_k=2, robust=False)
        elif method == "top3_mean":
            chosen, suppressed = representative_chunks(chunks, top_k=3, robust=False)
        elif method == "robust_top3_mean":
            chosen, suppressed = representative_chunks(chunks, top_k=3, robust=True)
        else:
            raise ValueError(f"unknown aggregation method: {method}")
        if not chosen:
            raise ValueError(f"document {doc_id} has no scored representative chunk")
        domains = {str(row["canonical_domain"]) for row in chunks}
        urls = {str(row["canonical_url"]) for row in chunks}
        if len(domains) != 1 or len(urls) != 1:
            raise ValueError(f"conflicting canonical provenance for document {doc_id}: domains={domains} urls={urls}")
        candidate_arms = sorted({arm for row in chunks for arm in row["candidate_arms"]})
        output.append({
            "doc_id": doc_id, "canonical_domain": domains.pop(), "canonical_url": urls.pop(),
            "candidate_arms": candidate_arms,
            "score": float(np.mean([float(row["score"]) for row in chosen])),
            "best_chunk_id": str(chosen[0]["chunk_id"]), "scored_chunk_count": len(chunks),
            "representative_chunk_count": len(chosen), "near_duplicate_suppressed": suppressed,
            "chunks": sorted(chunks, key=lambda row: (-float(row["score"]), str(row["chunk_id"]))),
        })
    return output


def rank_global(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(documents, key=lambda row: (-float(row["score"]), str(row["best_chunk_id"]), int(row["doc_id"])))


def rank_rrf(documents: list[dict[str, Any]], constant: int = 60) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in documents:
        for arm in row["candidate_arms"]:
            by_source[str(arm)].append(row)
    contributions: dict[int, list[tuple[str, int, float]]] = defaultdict(list)
    rows_by_doc = {int(row["doc_id"]): row for row in documents}
    for source, rows in sorted(by_source.items()):
        for rank, row in enumerate(rank_global(rows), 1):
            contributions[int(row["doc_id"])].append((source, rank, 1.0 / (constant + rank)))
    output = []
    for doc_id, values in contributions.items():
        # Multi-arm discovery gets its best arm rank, never an additive duplicate bonus.
        best = max(value[2] for value in values)
        output.append({**rows_by_doc[doc_id], "arm_ranks": {arm: rank for arm, rank, _ in values}, "fusion_score": best})
    return sorted(output, key=lambda row: (-float(row["fusion_score"]), -float(row["score"]), str(row["canonical_domain"]), int(row["doc_id"])))


def rank_balanced_premerge(documents: list[dict[str, Any]], depth_per_arm: int = 40) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in documents:
        for arm in row["candidate_arms"]:
            by_source[str(arm)].append(row)
    eligible_ids = {int(row["doc_id"]) for source in sorted(by_source) for row in rank_global(by_source[source])[:depth_per_arm]}
    return rank_global([row for row in documents if int(row["doc_id"]) in eligible_ids])


def conservative_pubmed(documents: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    selected = []; pubmed = 0
    for row in rank_global(documents):
        is_pubmed = row["canonical_domain"] == "pubmed.ncbi.nlm.nih.gov"
        if is_pubmed and pubmed >= limit:
            continue
        selected.append(row)
        pubmed += is_pubmed
    return selected


def select_submission_chunks(
    candidate_rows: list[dict[str, Any]], selected_documents: list[dict[str, Any]], depth: int = 40,
) -> tuple[list[dict], dict]:
    """Select independent evidence chunks from the full valid candidate pool."""
    top_docs = {int(row["doc_id"]) for row in selected_documents}
    ordered = sorted(candidate_rows, key=lambda row: (-float(row["score"]), str(row["chunk_id"])))
    emitted: list[dict] = []; seen_ids: set[str] = set(); duplicate_suppressed = 0
    for row in ordered:
        chunk_id = str(row["chunk_id"])
        if chunk_id in seen_ids:
            duplicate_suppressed += 1
            continue
        seen_ids.add(chunk_id)
        if any(near_duplicate(str(row["chunk_text"]), str(old["chunk_text"])) for old in emitted):
            duplicate_suppressed += 1
            continue
        emitted.append(row)
        if len(emitted) == depth:
            break
    if len(emitted) < depth:
        query_id = int(candidate_rows[0]["query_id"]) if candidate_rows else None
        raise ValueError(
            f"full eligible candidate pool has only {len(emitted)} distinct valid chunks after deduplication; "
            f"expected={depth} query_id={query_id} candidate_rows={len(candidate_rows)}"
        )
    inside = sum(int(row["doc_id"]) in top_docs for row in emitted)
    return emitted, {
        "near_duplicate_chunks_suppressed": duplicate_suppressed,
        "chunk_owners_inside_top20": inside,
        "chunk_owners_outside_top20": depth - inside,
        "unique_chunk_owners": len({int(row["doc_id"]) for row in emitted}),
        "provenance_mismatches": 0,
    }


def document_aware_chunk_capacity(documents: list[dict[str, Any]]) -> int:
    """Diagnostic only: evidence capacity under the rejected Top-20-owner rule."""
    emitted: list[dict] = []
    for document in documents:
        for row in document["chunks"]:
            if not any(near_duplicate(str(row["chunk_text"]), str(old["chunk_text"])) for old in emitted):
                emitted.append(row)
    return len(emitted)


def relative_inversions(old: list[int], new: list[int]) -> int:
    new_set = set(new); retained = [value for value in old if value in new_set]
    position = {value: index for index, value in enumerate(new)}
    order = [position[value] for value in retained]
    return sum(order[i] > order[j] for i in range(len(order)) for j in range(i + 1, len(order)))


def query_groups(path: Path) -> Iterable[list[dict[str, Any]]]:
    current = None; rows = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
        for row in batch.to_pylist():
            query_id = int(row["query_id"])
            if current is not None and query_id != current:
                yield rows; rows = []
            current = query_id; rows.append(row)
    if rows:
        yield rows


def replay_one(label: str, rankings: Path, canonical: Path, frozen_json: Path, tok) -> dict:
    query_ids = pq.read_table(ROOT / "data/raw/query.parquet", columns=["id"]).column(0).to_pylist()
    generated = _submission_records_disk(
        list(map(int, query_ids)), rankings / "reranked_documents.parquet", rankings / "reranked_chunks.parquet",
        canonical, tok, document_depth=20, chunk_depth=40, target_tokens=1024,
    )
    mismatches = []
    count = 0
    for expected, observed in zip(_iter_json_array(frozen_json), generated, strict=True):
        count += 1
        if expected != observed:
            mismatches.append(int(expected["id"]))
    return {
        "label": label, "passed": count == 1200 and not mismatches, "query_count": count,
        "semantic_exact_queries": count - len(mismatches), "mismatch_count": len(mismatches),
        "mismatch_sample": mismatches[:20], "frozen_json": str(frozen_json), "frozen_json_sha256": sha256(frozen_json),
        "replay_mode": "actual _submission_records_disk reconstruction from frozen rankings and canonical provenance",
    }


def baseline_replay() -> dict:
    runtime = yaml.safe_load((P12_ROOT / "runtime.yaml").read_text(encoding="utf-8"))
    tok = tokenizer(runtime)
    phase11c = replay_one(
        "Phase 11C", P11_ROOT / "rankings/BROAD", P11_ROOT / "round1/canonical_text.sqlite", P11_JSON, tok,
    )
    phase12 = replay_one(
        "Phase 12", P12_ROOT / "rankings/FULL", P12_ROOT / "round1/canonical_text.sqlite", P12_JSON, tok,
    )
    result = {"passed": phase11c["passed"] and phase12["passed"], "phase11c": phase11c, "phase12": phase12}
    if not result["passed"]:
        raise ValueError(f"frozen baseline replay mismatch: {result}")
    return result


def baseline_replay_or_checkpoint() -> dict:
    path = OUT / "baseline_replay_audit.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        hashes_match = (
            prior.get("phase11c", {}).get("frozen_json_sha256") == sha256(P11_JSON)
            and prior.get("phase12", {}).get("frozen_json_sha256") == sha256(P12_JSON)
        )
        if prior.get("passed") is True and hashes_match:
            return {**prior, "checkpoint_reused": True, "checkpoint_integrity": "frozen hashes match"}
    return {**baseline_replay(), "checkpoint_reused": False}


def hostname(url: str) -> str:
    return (urlsplit(str(url)).hostname or "unknown").lower().removeprefix("www.")


def candidate_ownership_audit() -> tuple[dict[str, set[str]], dict[int, tuple[str, str]], dict]:
    """Separate canonical provenance from one-or-more candidate discovery arms."""
    local_urls: dict[int, tuple[str, str]] = {}
    document_path = P12_ROOT / "round1/combined_documents.parquet"
    for batch in pq.ParquetFile(document_path).iter_batches(columns=["doc_id", "original_url", "final_url"], batch_size=32768):
        for doc_id, original, final in zip(*(column.to_pylist() for column in batch.columns), strict=True):
            value = (str(original), str(final or original))
            previous = local_urls.setdefault(int(doc_id), value)
            if previous != value:
                raise ValueError(f"duplicate local document provenance conflict: doc_id={doc_id} {previous} vs {value}")
    canonical: dict[int, tuple[str, str]] = {}
    official_path = ROOT / "data/raw/links_corpus_vi_en_zh.parquet"
    for batch in pq.ParquetFile(official_path).iter_batches(columns=["id", "url"], batch_size=131072):
        for doc_id, url in zip(*(column.to_pylist() for column in batch.columns), strict=True):
            doc_id = int(doc_id)
            if doc_id in local_urls:
                canonical[doc_id] = (str(url), hostname(str(url)))
    if len(canonical) != len(local_urls):
        missing = sorted(set(local_urls) - set(canonical))
        raise ValueError(f"local documents absent from NEW official corpus: count={len(missing)} sample={missing[:20]}")
    original_url_conflicts = {doc: (urls[0], canonical[doc][0]) for doc, urls in local_urls.items() if urls[0] != canonical[doc][0]}
    if original_url_conflicts:
        raise ValueError(f"local original URL differs from NEW official URL: count={len(original_url_conflicts)} sample={dict(list(sorted(original_url_conflicts.items()))[:20])}")

    chunk_arms: dict[str, set[str]] = defaultdict(set)
    chunk_identity: dict[str, tuple[int, str]] = {}
    document_arms: dict[int, set[str]] = defaultdict(set)
    query_document_arms: dict[tuple[int, int], set[str]] = defaultdict(set)
    final_url_aliases: dict[int, set[str]] = defaultdict(set)
    unknown_url_conflicts: dict[int, set[str]] = defaultdict(set)
    arm_rows = Counter()
    paths = []
    for marker in sorted((P12_ROOT / "round1/source_candidate_parts").glob("*.json")):
        arm = str(json.loads(marker.read_text(encoding="utf-8"))["source"])
        paths.append((arm, marker.with_suffix(".parquet")))
    paths.append(("ORIGINAL_PILOT", PILOT_POOL))
    for arm, path in paths:
        for batch in pq.ParquetFile(path).iter_batches(
            columns=["query_id", "chunk_id", "doc_id", "chunk_text", "source_url"], batch_size=8192,
        ):
            for query_id, chunk_id, doc_id, text, url in zip(*(column.to_pylist() for column in batch.columns), strict=True):
                query_id = int(query_id); doc_id = int(doc_id); chunk_id = str(chunk_id)
                if doc_id not in canonical:
                    raise ValueError(f"candidate lacks canonical document record: arm={arm} doc_id={doc_id}")
                allowed_urls = set(local_urls[doc_id])
                if str(url) not in allowed_urls:
                    unknown_url_conflicts[doc_id].add(str(url))
                elif str(url) != canonical[doc_id][0]:
                    final_url_aliases[doc_id].add(str(url))
                identity = (doc_id, hashlib.sha256(str(text).encode("utf-8")).hexdigest())
                previous = chunk_identity.setdefault(chunk_id, identity)
                if previous != identity:
                    raise ValueError(f"conflicting canonical chunk provenance: chunk_id={chunk_id} {previous} vs {identity}")
                chunk_arms[chunk_id].add(arm); document_arms[doc_id].add(arm)
                query_document_arms[(query_id, doc_id)].add(arm); arm_rows[arm] += 1
    if unknown_url_conflicts:
        sample = {str(doc): sorted(urls) for doc, urls in list(sorted(unknown_url_conflicts.items()))[:20]}
        raise ValueError(f"candidate URL is neither recorded original_url nor final_url: count={len(unknown_url_conflicts)} sample={sample}")
    multi_docs = {doc: sorted(arms) for doc, arms in document_arms.items() if len(arms) > 1}
    multi_query_docs = {key: sorted(arms) for key, arms in query_document_arms.items() if len(arms) > 1}
    multi_chunks = {chunk: sorted(arms) for chunk, arms in chunk_arms.items() if len(arms) > 1}
    target_url, target_domain = canonical[495914]
    report = {
        "passed": True, "canonical_document_count": len(canonical), "candidate_rows_by_arm": dict(sorted(arm_rows.items())),
        "documents_with_multiple_candidate_arms": len(multi_docs),
        "query_document_pairs_with_multiple_candidate_arms": len(multi_query_docs),
        "chunks_present_in_multiple_candidate_arms": len(multi_chunks),
        "recorded_final_url_alias_documents": len(final_url_aliases),
        "cross_domain_redirect_documents": sum(hostname(next(iter(urls))) != canonical[doc][1] for doc, urls in final_url_aliases.items()),
        "final_url_alias_sample": {str(doc): sorted(urls) for doc, urls in list(sorted(final_url_aliases.items()))[:20]},
        "unknown_url_conflicts": 0, "canonical_chunk_conflicts": 0,
        "affected_candidate_arms": sorted({arm for arms in multi_docs.values() for arm in arms}),
        "document_495914": {
            "official_doc_id": 495914, "canonical_url": target_url, "canonical_domain": target_domain,
            "candidate_arms": sorted(document_arms[495914]),
            "candidate_chunk_ids": sorted(chunk for chunk, identity in chunk_identity.items() if identity[0] == 495914),
            "query_ids": sorted({query for query, doc in query_document_arms if doc == 495914}),
            "valid_canonical_provenance": True, "real_conflict": False,
        },
        "multi_arm_document_sample": [{"doc_id": doc, "canonical_domain": canonical[doc][1], "arms": arms} for doc, arms in list(sorted(multi_docs.items()))[:50]],
        "multi_arm_query_document_sample": [{"query_id": key[0], "doc_id": key[1], "arms": arms} for key, arms in list(sorted(multi_query_docs.items()))[:50]],
        "multi_arm_rule": "preserve the arm set; canonical URL/domain remains singular; exact query-chunk score is counted once",
    }
    return chunk_arms, canonical, report


def frozen_scores(chunk_arms: dict[str, set[str]]) -> dict[tuple[int, str], float]:
    lookup: dict[tuple[int, str], float] = {}
    connection = sqlite3.connect(P12_ROOT / "round1/source_scores.sqlite")
    try:
        for query_id, chunk_id, score in connection.execute("SELECT query_id,chunk_id,rerank_score FROM pairs"):
            key = (int(query_id), str(chunk_id)); lookup[key] = float(score)
    finally:
        connection.close()
    pilot_keys = set()
    for batch in pq.ParquetFile(PILOT_POOL).iter_batches(columns=["query_id", "chunk_id"], batch_size=32768):
        pilot_keys.update((int(query), str(chunk)) for query, chunk in zip(*(column.to_pylist() for column in batch.columns), strict=True))
    connection = sqlite3.connect(ROOT / "data/retrieval/phase7_reranking/rerank_scores.sqlite")
    try:
        for query_id, chunk_id, score in connection.execute("SELECT query_id,chunk_id,rerank_score FROM scores"):
            key = (int(query_id), str(chunk_id))
            if key in pilot_keys:
                previous = lookup.setdefault(key, float(score))
                if previous != float(score):
                    raise ValueError(f"same exact score key differs between caches: {key}: {previous} vs {score}")
    finally:
        connection.close()
    scored_chunks = {key[1] for key in lookup}
    missing = [chunk for chunk in chunk_arms if chunk not in scored_chunks]
    if missing:
        raise ValueError(f"candidate chunks without any frozen score keys: count={len(missing)} sample={missing[:20]}")
    return lookup


def summarize_variant(per_query: list[dict], runtime_seconds: float) -> dict:
    totals = Counter(); source_doc_positions: dict[str, list[int]] = defaultdict(list)
    source_chunk_positions: dict[str, list[int]] = defaultdict(list)
    for row in per_query:
        totals.update({
            "pubmed_document_slots": row["pubmed_document_slots"], "pubmed_chunk_slots": row["pubmed_chunk_slots"],
            "retained_phase11c_document_slots": row["retained_phase11c_documents"],
            "displaced_phase11c_document_slots": row["displaced_phase11c_documents"],
            "rank_inversions_vs_phase11c": row["rank_inversions_vs_phase11c"],
            "near_duplicate_chunks_suppressed": row["near_duplicate_chunks_suppressed"],
            "chunk_owners_inside_top20": row["chunk_owners_inside_top20"],
            "chunk_owners_outside_top20": row["chunk_owners_outside_top20"],
            "provenance_mismatches": row["provenance_mismatches"],
        })
        for position, source in enumerate(row["document_sources"], 1): source_doc_positions[source].append(position)
        for position, source in enumerate(row["chunk_sources"], 1): source_chunk_positions[source].append(position)
    return {
        "queries": len(per_query), **dict(totals),
        "queries_with_pubmed_documents": sum(row["pubmed_document_slots"] > 0 for row in per_query),
        "queries_with_pubmed_chunks": sum(row["pubmed_chunk_slots"] > 0 for row in per_query),
        "pubmed_documents_per_query": distribution(row["pubmed_document_slots"] for row in per_query),
        "pubmed_chunks_per_query": distribution(row["pubmed_chunk_slots"] for row in per_query),
        "scored_chunks_per_selected_document": distribution(value for row in per_query for value in row["selected_document_chunk_counts"]),
        "document_aware_chunk_capacity": distribution(row["document_aware_chunk_capacity"] for row in per_query),
        "queries_where_top20_owners_cannot_supply_40_distinct_chunks": sum(row["document_aware_chunk_capacity"] < 40 for row in per_query),
        "source_document_rank_distribution": {key: distribution(value) for key, value in sorted(source_doc_positions.items())},
        "source_chunk_rank_distribution": {key: distribution(value) for key, value in sorted(source_chunk_positions.items())},
        "runtime_seconds": round(runtime_seconds, 4),
        "diagnostic_only": True,
    }


def execute_variants() -> tuple[dict, dict, dict, dict[int, dict[str, list[dict]]], dict]:
    chunk_arms, canonical, ownership = candidate_ownership_audit()
    scores = frozen_scores(chunk_arms)
    old_outputs = {int(row["id"]): row for row in _iter_json_array(P11_JSON)}
    methods = {
        "max_chunk_global_raw": ("max_chunk", rank_global),
        "top2_mean_global_raw": ("top2_mean", rank_global),
        "top3_mean_global_raw": ("top3_mean", rank_global),
        "robust_top3_mean_global_raw": ("robust_top3_mean", rank_global),
        "max_chunk_rrf_by_candidate_arm": ("max_chunk", rank_rrf),
        "max_chunk_balanced_premerge_top40_per_arm": ("max_chunk", rank_balanced_premerge),
        "max_chunk_pubmed_cap5_diagnostic": ("max_chunk", conservative_pubmed),
    }
    details: dict[str, list[dict]] = {name: [] for name in methods}
    review: dict[int, dict[str, list[dict]]] = defaultdict(dict)
    review_ids = {int(row["query_id"]) for row in csv.DictReader((ROOT / "artifacts/phase13_diagnosis/human_review.csv").open(encoding="utf-8-sig"))}
    started = time.perf_counter(); process = psutil.Process(); peak_rss = process.memory_info().rss
    for rows in query_groups(P12_ROOT / "rankings/FULL/reranked_chunks.parquet"):
        query_id = int(rows[0]["query_id"]); prepared = []
        for row in rows:
            key = (query_id, str(row["chunk_id"]))
            if key not in scores or str(row["chunk_id"]) not in chunk_arms:
                raise ValueError(f"missing frozen score/source for {key}")
            doc_id = int(row["doc_id"])
            if doc_id not in canonical:
                raise ValueError(f"ranked row lacks canonical ownership: query={query_id} doc_id={doc_id}")
            canonical_url, canonical_domain = canonical[doc_id]
            prepared.append({
                **row, "score": float(scores[key]), "candidate_arms": sorted(chunk_arms[str(row["chunk_id"])]),
                "canonical_url": canonical_url, "canonical_domain": canonical_domain,
            })
        aggregated = {method: aggregate_documents(prepared, method) for method in {value[0] for value in methods.values()}}
        old_docs = list(map(int, old_outputs[query_id]["relevant_docs"]))
        for name, (aggregation, ranker) in methods.items():
            ranked = ranker(aggregated[aggregation]); selected_docs = ranked[:20]
            selected_chunks, consistency = select_submission_chunks(prepared, selected_docs, 40)
            document_aware_capacity = document_aware_chunk_capacity(selected_docs)
            docs = [int(row["doc_id"]) for row in selected_docs]; old_set = set(old_docs)
            row = {
                "query_id": query_id, "document_ids": docs,
                "document_sources": [str(item["canonical_domain"]) for item in selected_docs],
                "chunk_sources": [str(item["canonical_domain"]) for item in selected_chunks],
                "pubmed_document_slots": sum(item["canonical_domain"] == "pubmed.ncbi.nlm.nih.gov" for item in selected_docs),
                "pubmed_chunk_slots": sum(item["canonical_domain"] == "pubmed.ncbi.nlm.nih.gov" for item in selected_chunks),
                "retained_phase11c_documents": sum(doc in old_set for doc in docs),
                "displaced_phase11c_documents": sum(doc not in set(docs) for doc in old_docs),
                "rank_inversions_vs_phase11c": relative_inversions(old_docs, docs),
                "selected_document_chunk_counts": [int(item["scored_chunk_count"]) for item in selected_docs],
                "document_aware_chunk_capacity": document_aware_capacity,
                **consistency,
            }
            details[name].append(row)
            if query_id in review_ids:
                review[query_id][name] = [{
                    "doc_id": int(item["doc_id"]), "source": str(item["canonical_domain"]), "candidate_arms": item["candidate_arms"], "score": round(float(item["score"]), 6),
                    "scored_chunks": int(item["scored_chunk_count"]), "evidence": str(item["chunks"][0]["chunk_text"])[:1200],
                    "chunk_id": str(item["chunks"][0]["chunk_id"]),
                } for item in selected_docs[:10]]
        peak_rss = max(peak_rss, process.memory_info().rss)
    elapsed = time.perf_counter() - started
    summaries = {name: summarize_variant(rows, elapsed) for name, rows in details.items()}
    common = {
        "frozen_score_rows": len(scores), "new_model_inference": 0, "new_embeddings": 0, "new_crawling": 0,
        "cpu_only": True, "query_count": 1200, "peak_rss_bytes": peak_rss,
        "chunk_policy": "independent global raw-score chunk ranking over the same full frozen pool; canonical-ID and deterministic near-duplicate suppression",
        "human_labels_used": 0, "winner_selected": False,
    }
    aggregation = {"scope": common, "variants": {k: v for k, v in summaries.items() if "global_raw" in k}}
    fusion = {"scope": common, "variants": {k: v for k, v in summaries.items() if "global_raw" not in k}}
    consistency = {
        "scope": common,
        "variants": {name: {
            "document_slots": 24000, "chunk_slots": 48000,
            "chunk_owners_inside_top20": summary["chunk_owners_inside_top20"],
            "chunk_owners_outside_top20": summary["chunk_owners_outside_top20"],
            "queries_where_top20_owners_cannot_supply_40_distinct_chunks": summary["queries_where_top20_owners_cannot_supply_40_distinct_chunks"],
            "near_duplicate_chunks_suppressed": summary["near_duplicate_chunks_suppressed"],
            "top20_owner_slot_rate": summary["chunk_owners_inside_top20"] / 48000,
            "official_schema_allows_independent_chunk_owners": True,
            "provenance_mismatches": summary["provenance_mismatches"],
            "original_chunk_text_preserved": True, "provenance_inherited_from_passed_phase12_closure": True,
        } for name, summary in summaries.items()},
    }
    return aggregation, fusion, consistency, review, ownership


def build_review_interface(review: dict[int, dict[str, list[dict]]]) -> None:
    query_rows = pq.read_table(ROOT / "data/raw/query.parquet", columns=["id", "query"]).to_pylist()
    questions = {int(row["id"]): str(row["query"]) for row in query_rows}
    payload = []
    for query_id in sorted(review):
        methods = sorted(review[query_id])
        methods.sort(key=lambda method: hashlib.sha256(f"phase13a:{query_id}:{method}".encode()).hexdigest())
        payload.append({
            "query_id": query_id, "question": questions[query_id],
            "alternatives": [{"blind_label": f"System {chr(65 + index)}", "method": method, "documents": review[query_id][method]}
                             for index, method in enumerate(methods)],
        })
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html><html lang='vi'><head><meta charset='utf-8'><title>Phase 13A blind review</title>
<style>body{{font:15px system-ui;max-width:1450px;margin:auto;padding:20px}}.query{{border-top:3px solid #333;margin-top:35px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px}}.system{{border:1px solid #aaa;padding:10px}}.doc{{background:#f6f6f6;margin:8px 0;padding:8px}}small{{color:#555}}textarea{{width:100%}}</style></head><body>
<h1>Phase 13A — đánh giá mù</h1><p>Không xem tên phương pháp khi chấm. Nhãn cho từng tài liệu: RELEVANT / PARTIALLY_RELEVANT / IRRELEVANT / UNCLEAR. Bỏ trống nếu chưa đánh giá; kết quả chưa chấm là UNKNOWN, không phải IRRELEVANT.</p>
<button onclick='exportCsv()'>Xuất CSV đánh giá</button><div id='app'></div><script>const DATA={encoded}; const LABELS=['','RELEVANT','PARTIALLY_RELEVANT','IRRELEVANT','UNCLEAR'];
const app=document.getElementById('app'); for(const q of DATA){{let root=document.createElement('section');root.className='query';root.innerHTML=`<h2>Query ${{q.query_id}}</h2><p>${{esc(q.question)}}</p><div class="grid"></div>`;let grid=root.querySelector('.grid');for(const a of q.alternatives){{let box=document.createElement('div');box.className='system';box.innerHTML=`<h3>${{a.blind_label}}</h3>`;a.documents.forEach((d,i)=>{{let item=document.createElement('div');item.className='doc';item.innerHTML=`<b>#${{i+1}} · doc ${{d.doc_id}}</b><br><small>${{esc(d.source)}} · score ${{d.score}} · chunk ${{esc(d.chunk_id)}}</small><p>${{esc(d.evidence)}}</p><select data-q="${{q.query_id}}" data-system="${{a.blind_label}}" data-method="${{a.method}}" data-doc="${{d.doc_id}}">${{LABELS.map(x=>`<option>${{x}}</option>`).join('')}}</select>`;box.appendChild(item)}});grid.appendChild(box)}}app.appendChild(root)}}
function esc(x){{return String(x).replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]))}}function exportCsv(){{let lines=['query_id,blind_system,method,doc_id,judgment'];document.querySelectorAll('select').forEach(s=>lines.push([s.dataset.q,s.dataset.system,s.dataset.method,s.dataset.doc,s.value].map(x=>'"'+String(x).replaceAll('"','""')+'"').join(',')));let b=new Blob([lines.join('\n')],{{type:'text/csv'}}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='phase13a_human_judgments.csv';a.click()}}</script></body></html>"""
    path = OUT / "review_interface.html"; path.write_text(page, encoding="utf-8")


def write_summary(replay: dict, aggregation: dict, fusion: dict) -> None:
    a = aggregation["variants"]; f = fusion["variants"]
    lines = [
        "# Phase 13A — replay aggregation và fusion", "", "## Trạng thái", "",
        f"- Replay Phase 11C: **{'PASS' if replay['phase11c']['passed'] else 'FAIL'}**, {replay['phase11c']['semantic_exact_queries']}/1.200 query giống hệt.",
        f"- Replay Phase 12: **{'PASS' if replay['phase12']['passed'] else 'FAIL'}**, {replay['phase12']['semantic_exact_queries']}/1.200 query giống hệt.",
        "- CPU only; không crawl, embedding, reranker inference hoặc leaderboard submission.", "",
        "## Kết quả chẩn đoán (không phải relevance)", "",
        "| Variant | PubMed doc slots | PubMed chunk slots | Phase11C docs retained | Rank inversions |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in {**a, **f}.items():
        lines.append(f"| `{name}` | {row['pubmed_document_slots']} | {row['pubmed_chunk_slots']} | {row['retained_phase11c_document_slots']} | {row['rank_inversions_vs_phase11c']} |")
    lines += [
        "", "Không được chọn winner từ bảng này: giảm PubMed không đồng nghĩa tăng relevance. `review_interface.html` xáo thứ tự phương pháp theo query và giữ nhãn trống.",
        "", "## Còn thiếu", "", "Cần con người chấm pooled candidates bằng RELEVANT / PARTIALLY_RELEVANT / IRRELEVANT / UNCLEAR. Unjudged = UNKNOWN. Sau đó mới tính judged precision, nDCG/ranking quality trên tập đã chấm và báo coverage của judgment; tuyệt đối không gọi đó là official F2/recall.",
    ]
    (OUT / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_existing_outputs() -> dict[str, Any]:
    """Validate the completed CPU replay without executing any variant again."""
    required = {
        "baseline_replay_audit.json", "aggregation_comparison.json", "fusion_comparison.json",
        "document_chunk_consistency.json", "candidate_ownership_audit.json",
        "review_interface.html", "experiment_summary.md",
    }
    missing_files = sorted(name for name in required if not (OUT / name).is_file())
    if missing_files:
        raise ValueError(f"Phase 13A report set incomplete: {missing_files}")

    replay = json.loads((OUT / "baseline_replay_audit.json").read_text(encoding="utf-8"))
    aggregation = json.loads((OUT / "aggregation_comparison.json").read_text(encoding="utf-8"))
    fusion = json.loads((OUT / "fusion_comparison.json").read_text(encoding="utf-8"))
    consistency = json.loads((OUT / "document_chunk_consistency.json").read_text(encoding="utf-8"))
    ownership = json.loads((OUT / "candidate_ownership_audit.json").read_text(encoding="utf-8"))
    variants = {**aggregation["variants"], **fusion["variants"]}
    checks: dict[str, bool] = {
        "required_files_present": True,
        "phase11c_exact_replay": replay.get("phase11c", {}).get("semantic_exact_queries") == 1200
        and replay.get("phase11c", {}).get("mismatch_count") == 0,
        "phase12_exact_replay": replay.get("phase12", {}).get("semantic_exact_queries") == 1200
        and replay.get("phase12", {}).get("mismatch_count") == 0,
        "frozen_hashes_still_match": replay.get("phase11c", {}).get("frozen_json_sha256") == sha256(P11_JSON)
        and replay.get("phase12", {}).get("frozen_json_sha256") == sha256(P12_JSON),
        "seven_variants_present": len(variants) == 7,
        "all_variants_cover_1200_queries": all(row.get("queries") == 1200 for row in variants.values()),
        "all_variants_have_24000_docs": all(
            row.get("retained_phase11c_document_slots", 0) + row.get("displaced_phase11c_document_slots", 0) == 24000
            for row in variants.values()
        ),
        "all_variants_have_48000_chunks": all(
            row.get("chunk_owners_inside_top20", 0) + row.get("chunk_owners_outside_top20", 0) == 48000
            for row in variants.values()
        ),
        "zero_provenance_mismatches": all(row.get("provenance_mismatches") == 0 for row in variants.values()),
        "consistency_covers_all_variants": set(consistency.get("variants", {})) == set(variants),
        "ownership_audit_passed": ownership.get("passed") is True
        and ownership.get("unknown_url_conflicts") == 0
        and ownership.get("canonical_chunk_conflicts") == 0,
        "no_model_or_data_work": all(
            aggregation.get("scope", {}).get(field) == 0
            for field in ("new_model_inference", "new_embeddings", "new_crawling")
        ),
        "no_winner_selected": aggregation.get("scope", {}).get("winner_selected") is False
        and fusion.get("scope", {}).get("winner_selected") is False,
    }

    html_text = (OUT / "review_interface.html").read_text(encoding="utf-8")
    match = re.search(r"const DATA=(.*?); const LABELS=", html_text, flags=re.DOTALL)
    if not match:
        raise ValueError("review interface does not contain its deterministic DATA payload")
    review = json.loads(match.group(1).replace("<\\/", "</"))
    review_ids = [int(row["query_id"]) for row in review]
    alternatives = [alternative for row in review for alternative in row["alternatives"]]
    checks.update({
        "review_has_60_unique_queries": len(review_ids) == 60 and len(set(review_ids)) == 60,
        "review_has_seven_blinded_alternatives_per_query": all(len(row["alternatives"]) == 7 for row in review),
        "review_has_source_ids_scores_and_evidence": all(
            len(alt.get("documents", [])) == 10
            and all({"doc_id", "source", "candidate_arms", "score", "evidence", "chunk_id"} <= set(doc) for doc in alt["documents"])
            for alt in alternatives
        ),
        "review_starts_blank_and_exports": "const LABELS=['','RELEVANT'" in html_text
        and "phase13a_human_judgments.csv" in html_text,
        "display_uses_blind_label": "<h3>${a.blind_label}</h3>" in html_text,
    })
    failed = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "run_id": "phase13a_fusion_replay",
        "status": "READY_FOR_HUMAN_REVIEW" if not failed else "NEEDS_AGENT",
        "stage": "AWAITING_HUMAN_REVIEW" if not failed else "POST_RUN_AUDIT",
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "variant_count": len(variants), "review_query_count": len(review_ids),
        "report_sha256": {name: sha256(OUT / name) for name in sorted(required)},
        "frozen_control_sha256": {"phase11c": sha256(P11_JSON), "phase12": sha256(P12_JSON)},
        "candidate_score_contract": {
            "raw_candidate_rows": sum(ownership["candidate_rows_by_arm"].values()),
            "unique_exact_query_chunk_scores": aggregation["scope"]["frozen_score_rows"],
            "new_model_inference": 0,
        },
        "scientific_gate": "Human relevance labels are required before selecting a winning variant.",
    }
    atomic_json(OUT / "post_run_audit.json", report)
    if failed:
        raise ValueError(f"Phase 13A post-run validation failed: {failed}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-state", type=Path)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.finalize_existing:
        update_run_state(args.run_state, status="RUNNING", stage="POST_RUN_AUDIT", done=0, total=1)
        report = validate_existing_outputs()
        update_run_state(
            args.run_state, status=report["status"], stage=report["stage"], done=1, total=1,
            checkpoint=str(OUT / "post_run_audit.json"), integrity="passed",
            human_review_interface=str(OUT / "review_interface.html"), winner_selected=False,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    update_run_state(args.run_state, status="RUNNING", stage="BASELINE_REPLAY", done=0, total=3)
    replay = baseline_replay_or_checkpoint(); atomic_json(OUT / "baseline_replay_audit.json", replay)
    update_run_state(args.run_state, stage="VARIANT_REPLAY", done=1, total=3, checkpoint=str(OUT / "baseline_replay_audit.json"))
    aggregation, fusion, consistency, review, ownership = execute_variants()
    atomic_json(OUT / "candidate_ownership_audit.json", ownership)
    atomic_json(OUT / "aggregation_comparison.json", aggregation)
    atomic_json(OUT / "fusion_comparison.json", fusion)
    atomic_json(OUT / "document_chunk_consistency.json", consistency)
    build_review_interface(review); write_summary(replay, aggregation, fusion)
    update_run_state(args.run_state, stage="POST_RUN_AUDIT", done=2, total=3, checkpoint=str(OUT / "experiment_summary.md"))
    report = validate_existing_outputs()
    update_run_state(
        args.run_state, status=report["status"], stage=report["stage"], done=3, total=3,
        checkpoint=str(OUT / "post_run_audit.json"), integrity="passed",
        human_review_interface=str(OUT / "review_interface.html"), winner_selected=False,
    )
    print(json.dumps({
        "output": str(OUT), "baseline_replay": replay["passed"], "variants": len(aggregation["variants"]) + len(fusion["variants"]),
        "review_queries": len(review), "new_model_inference": 0, "winner_selected": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
