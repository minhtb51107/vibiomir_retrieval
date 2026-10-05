from __future__ import annotations

import hashlib
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_counts(values: list[int]) -> dict[str, int | float]:
    return {
        "min": min(values, default=0),
        "median": statistics.median(values) if values else 0,
        "p90": round(percentile(values, 0.9), 4),
        "max": max(values, default=0),
        "mean": round(statistics.fmean(values), 4) if values else 0.0,
    }


def _rows(connection, query_ids: list[int]) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in query_ids)
    sql = f"""
    SELECT a.query_id,a.variant,a.fingerprint,p.query_text,p.status AS request_status,
           p.http_status,p.error_type,p.error_message,r.rank,r.url,
           COALESCE(m.status,'unmapped') AS mapping_status,m.match_level,m.doc_id,
           m.official_url,m.official_domain
    FROM assignments a JOIN responses p USING(fingerprint)
    LEFT JOIN results r USING(fingerprint)
    LEFT JOIN mappings m ON m.result_url=r.url
    WHERE a.query_id IN ({placeholders})
    ORDER BY a.query_id,a.variant,r.rank
    """
    columns = [value[0] for value in connection.execute(sql, query_ids).description]
    return [dict(zip(columns, row, strict=True)) for row in connection.execute(sql, query_ids)]


def _metrics(rows: list[dict[str, Any]], query_ids: list[int]) -> dict[str, Any]:
    results = [row for row in rows if row["url"] is not None]
    by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_query[int(row["query_id"])].append(row)
    mapped_counts = []
    mapped_pairs: set[tuple[int, int]] = set()
    domains: Counter[str] = Counter()
    for query_id in query_ids:
        docs = {
            int(row["doc_id"])
            for row in by_query.get(query_id, [])
            if row["doc_id"] is not None
        }
        mapped_counts.append(len(docs))
        mapped_pairs.update((query_id, doc_id) for doc_id in docs)
    mapped_results = [row for row in results if row["doc_id"] is not None]
    domains.update(str(row["official_domain"]) for row in mapped_results)
    total = len(results)
    query_hits = sum(value > 0 for value in mapped_counts)
    return {
        "query_count": len(query_ids),
        "total_search_results": total,
        "unique_result_urls": len({str(row["url"]) for row in results}),
        "exact_official_matches": sum(row["mapping_status"] == "exact_official_match" for row in results),
        "normalized_official_matches": sum(row["mapping_status"] == "normalized_official_match" for row in results),
        "ambiguous_matches": sum(row["mapping_status"] == "ambiguous_match" for row in results),
        "mapped_result_rows": len(mapped_results),
        "mapped_result_fraction": round(len(mapped_results) / max(total, 1), 6),
        "queries_with_mapped_doc": query_hits,
        "query_map_rate": round(query_hits / max(len(query_ids), 1), 6),
        "mapped_docs_per_query": summarize_counts(mapped_counts),
        "unique_mapped_docs": len({doc_id for _, doc_id in mapped_pairs}),
        "unique_query_doc_pairs": len(mapped_pairs),
        "official_domains": len(domains),
        "mapped_domain_counts": dict(domains.most_common()),
        "top_domain_share": round(max(domains.values(), default=0) / max(sum(domains.values()), 1), 6),
        "top5_domain_share": round(sum(value for _, value in domains.most_common(5)) / max(sum(domains.values()), 1), 6),
    }


def _query_diagnostics(
    rows: list[dict[str, Any]], sample_rows: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["query_id"]), str(row["variant"]))].append(row)
    output = []
    for (query_id, variant), values in sorted(grouped.items()):
        results = [row for row in values if row["url"] is not None]
        mapped = [row for row in results if row["doc_id"] is not None]
        request = values[0]
        if request["request_status"] == "RATE_LIMITED":
            failure = "G_PROVIDER_RATE_LIMITED"
        elif request["request_status"] != "SUCCESS":
            failure = "A_PROVIDER_ERROR"
        elif not results:
            failure = "A_PROVIDER_RETURNED_NOTHING"
        elif mapped:
            failure = None
        elif any(row["mapping_status"] == "ambiguous_match" for row in results):
            failure = "C_AMBIGUOUS_CANONICALIZATION"
        else:
            failure = "B_RESULTS_OUTSIDE_OFFICIAL_CORPUS"
        output.append(
            {
                "query_id": query_id,
                "variant": variant,
                "query_characters": sample_rows[query_id]["query_characters"],
                "query_tokens": sample_rows[query_id]["query_tokens"],
                "length_quantile": sample_rows[query_id]["length_quantile"],
                "category": sample_rows[query_id]["category"],
                "evidence_proxy": sample_rows[query_id]["evidence_proxy"],
                "request_status": request["request_status"],
                "search_results": len(results),
                "mapped_results": len(mapped),
                "mapped_docs": len({int(row["doc_id"]) for row in mapped}),
                "distinct_mapped_domains": len({str(row["official_domain"]) for row in mapped}),
                "best_mapped_rank": min((int(row["rank"]) for row in mapped), default=None),
                "failure_reason": failure,
            }
        )
    by_query: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if row["doc_id"] is not None:
            by_query[int(row["query_id"])][str(row["variant"])].add(int(row["doc_id"]))
    for row in output:
        query = by_query[row["query_id"]]
        row["original_vi_succeeded"] = bool(query["Q0"])
        row["concise_vi_added_unique_docs"] = len(query["Q1"] - query["Q0"])
    return output


def _candidate_sets(path: str | Path, query_ids: set[int]) -> tuple[dict[int, set[int]], dict[int, set[int]], dict[int, set[int]]]:
    schema = pq.read_schema(path)
    rank_column = "rerank_rank" if "rerank_rank" in schema.names else "rank"
    table = pq.read_table(path, columns=["query_id", rank_column, "doc_id"])
    all_rows: dict[int, set[int]] = defaultdict(set)
    top10: dict[int, set[int]] = defaultdict(set)
    top50: dict[int, set[int]] = defaultdict(set)
    for row in table.to_pylist():
        query_id = int(row["query_id"])
        if query_id not in query_ids:
            continue
        rank = int(row[rank_column])
        doc_id = int(row["doc_id"])
        all_rows[query_id].add(doc_id)
        if rank <= 10:
            top10[query_id].add(doc_id)
        if rank <= 50:
            top50[query_id].add(doc_id)
    return all_rows, top10, top50


def _overlap_summary(discovered: dict[int, set[int]], candidates: dict[int, set[int]]) -> dict[str, Any]:
    overlaps = [len(discovered.get(query_id, set()) & docs) for query_id, docs in candidates.items()]
    return {
        "queries_with_any_overlap": sum(value > 0 for value in overlaps),
        "overlapping_query_doc_pairs": sum(overlaps),
        "queries_considered": len(candidates),
    }


def pilot_overlap(
    config: dict[str, Any], rows: list[dict[str, Any]], query_ids: list[int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    discovered: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        if row["doc_id"] is not None:
            discovered[int(row["query_id"])].add(int(row["doc_id"]))
    methods = {}
    candidate_sets = {}
    for method, key in (
        ("dense", "dense_candidates"), ("sparse", "sparse_candidates"),
        ("hybrid", "hybrid_candidates"), ("reranked", "reranked_candidates"),
    ):
        all_rows, top10, top50 = _candidate_sets(config["inputs"][key], set(query_ids))
        candidate_sets[method] = (all_rows, top10, top50)
        methods[method] = {
            "available": _overlap_summary(discovered, all_rows),
            "top10": _overlap_summary(discovered, top10),
            "top50": _overlap_summary(discovered, top50),
        }
    ids = pq.read_table(config["inputs"]["corpus"], columns=["id"]).column("id").combine_chunks().to_numpy()
    random_sets: dict[int, set[int]] = {}
    seed = int(config["sampling"]["seed"])
    for query_id in query_ids:
        count = len(discovered.get(query_id, set()))
        if count == 0:
            random_sets[query_id] = set()
            continue
        digest = hashlib.sha256(f"random:{seed}:{query_id}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        positions = rng.choice(len(ids), size=count, replace=False)
        random_sets[query_id] = {int(ids[position]) for position in positions}
    random = {}
    for method, (all_rows, top10, top50) in candidate_sets.items():
        random[method] = {
            "available": _overlap_summary(random_sets, all_rows),
            "top10": _overlap_summary(random_sets, top10),
            "top50": _overlap_summary(random_sets, top50),
        }
    return methods, random


def verdict(metrics: dict[str, Any], diagnostics: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    gates = config["stage_b_gates"]
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    combined_by_query: dict[int, bool] = defaultdict(bool)
    for row in diagnostics:
        by_category[row["category"]].append(row)
        combined_by_query[row["query_id"]] |= row["mapped_docs"] > 0
    category_rates = {}
    for category, rows in by_category.items():
        query_hits = {row["query_id"]: combined_by_query[row["query_id"]] for row in rows}
        category_rates[category] = {
            "queries": len(query_hits),
            "query_map_rate": round(sum(query_hits.values()) / max(len(query_hits), 1), 6),
        }
    promising = gates["promising"]
    borderline = gates["borderline"]
    strong_subgroup = any(
        value["queries"] >= int(promising["subgroup_minimum_size"])
        and value["query_map_rate"] >= float(promising["subgroup_minimum_query_map_rate"])
        for value in category_rates.values()
    )
    base_promising = (
        metrics["query_map_rate"] >= float(promising["minimum_query_map_rate"])
        and metrics["unique_mapped_docs"] <= int(promising["maximum_global_candidates"])
        and metrics["official_domains"] >= int(promising["minimum_official_domains"])
        and metrics["top_domain_share"] <= float(promising["maximum_top_domain_share"])
    )
    base_borderline = (
        metrics["query_map_rate"] >= float(borderline["minimum_query_map_rate"])
        and metrics["unique_mapped_docs"] <= int(borderline["maximum_global_candidates"])
        and metrics["official_domains"] >= int(borderline["minimum_official_domains"])
        and metrics["top_domain_share"] <= float(borderline["maximum_top_domain_share"])
    )
    value = "PROMISING" if base_promising or strong_subgroup else "BORDERLINE" if base_borderline else "NOT VIABLE"
    return {
        "verdict": value,
        "overall_promising_gate": base_promising,
        "strong_subgroup_gate": strong_subgroup,
        "borderline_gate": base_borderline,
        "category_query_map_rates": category_rates,
        "gates": gates,
    }


def analyze_stage(
    config: dict[str, Any], connection, sample: dict[str, Any], stage: str
) -> dict[str, Any]:
    query_ids = sample["stage_a_query_ids"] if stage == "A" else sample["stage_b_query_ids"]
    rows = _rows(connection, query_ids)
    sample_rows = {int(row["query_id"]): row for row in sample["queries"]}
    variants = {}
    for variant in config["query_variants"]["enabled"]:
        variants[variant] = _metrics([row for row in rows if row["variant"] == variant], query_ids)
    combined = _metrics(rows, query_ids)
    diagnostics = _query_diagnostics(rows, sample_rows)
    q0_docs: dict[int, set[int]] = defaultdict(set)
    q1_docs: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        if row["doc_id"] is not None:
            (q0_docs if row["variant"] == "Q0" else q1_docs)[int(row["query_id"])].add(int(row["doc_id"]))
    q1_added = sum(len(q1_docs[q] - q0_docs[q]) for q in query_ids)
    q1_global_added = len(
        set().union(*(q1_docs[q] for q in query_ids))
        - set().union(*(q0_docs[q] for q in query_ids))
    )
    combined_pairs = sum(len(q0_docs[q] | q1_docs[q]) for q in query_ids)
    domains = dict(connection.execute("SELECT domain,corpus_count FROM domain_stats"))
    mapped_domains = Counter()
    for row in rows:
        if row["doc_id"] is not None:
            mapped_domains[str(row["official_domain"])] += 1
    domain_analysis = {
        "full_official_domain_count": len(domains),
        "mapped_official_domain_count": len(mapped_domains),
        "mapped": [
            {
                "domain": domain,
                "mapped_result_rows": count,
                "mapped_share": round(count / max(sum(mapped_domains.values()), 1), 6),
                "corpus_count": int(domains.get(domain, 0)),
                "corpus_share": round(int(domains.get(domain, 0)) / max(sum(domains.values()), 1), 6),
            }
            for domain, count in mapped_domains.most_common()
        ],
    }
    pilot, random = pilot_overlap(config, rows, query_ids)
    failure_counts = Counter(
        row["failure_reason"] for row in diagnostics if row["failure_reason"] is not None
    )
    failures_by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in diagnostics:
        failures_by_query[int(row["query_id"])].append(row)
    query_level_failures: Counter[str] = Counter()
    for values in failures_by_query.values():
        if any(row["mapped_docs"] > 0 for row in values):
            continue
        if all(row["request_status"] != "SUCCESS" for row in values):
            query_level_failures["provider_error_all_variants"] += 1
        elif all(row["search_results"] == 0 for row in values):
            query_level_failures["provider_returned_nothing_all_variants"] += 1
        elif any(row["failure_reason"] == "C_AMBIGUOUS_CANONICALIZATION" for row in values):
            query_level_failures["ambiguous_only"] += 1
        else:
            query_level_failures["results_but_no_official_match"] += 1
    result = {
        "stage": stage,
        "query_count": len(query_ids),
        "provider": "bing_rss",
        "provider_result_depth": int(config["provider"]["result_depth"]),
        "variants_tested": list(config["query_variants"]["enabled"]),
        "multilingual_variants": {
            "english": config["query_variants"]["english"],
            "chinese": config["query_variants"]["chinese"],
        },
        "variant_metrics": variants,
        "combined_metrics": combined,
        "q1_incremental_unique_query_doc_pairs": q1_added,
        "q1_incremental_query_doc_pair_share": round(q1_added / max(combined_pairs, 1), 6),
        "q1_incremental_global_unique_docs": q1_global_added,
        "zero_map_failure_breakdown": dict(sorted(failure_counts.items())),
        "zero_map_query_breakdown": dict(sorted(query_level_failures.items())),
        "query_diagnostics": diagnostics,
        "domain_analysis": domain_analysis,
        "pilot_candidate_overlap": pilot,
        "random_baseline_overlap": random,
    }
    if stage == "B":
        result["viability"] = verdict(combined, diagnostics, config)
        full = int(config["projection"]["full_query_count"])
        scale = full / len(query_ids)
        projected_docs = min(4_394_718, math.ceil(combined["unique_mapped_docs"] * scale))
        requests = full * len(config["query_variants"]["enabled"])
        result["stage_c_projection"] = {
            "queries": full,
            "search_requests": requests,
            "unique_result_urls": math.ceil(combined["unique_result_urls"] * scale),
            "mapped_official_doc_ids": projected_docs,
            "crawl_candidate_count": projected_docs,
            "crawl_minutes_at_phase8_rate": round(projected_docs / float(config["projection"]["phase8_crawl_urls_per_minute"]), 2),
            "retained_storage_bytes": {
                "low": projected_docs * int(config["projection"]["low_retained_bytes_per_document"]),
                "expected": projected_docs * int(config["projection"]["expected_retained_bytes_per_usable_document"]),
                "high": projected_docs * int(config["projection"]["high_retained_bytes_per_document"]),
            },
        }
    return result
