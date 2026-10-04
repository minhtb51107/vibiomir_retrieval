from __future__ import annotations

import json
import os
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .selection import (
    domain_balanced_order,
    is_weak_query,
    select_combined_candidates,
)


CANDIDATE_SCHEMA = """
CREATE TABLE candidates (
    method TEXT NOT NULL,
    depth INTEGER NOT NULL,
    query_id INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    doc_id INTEGER NOT NULL,
    score REAL NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY(method, depth, query_id, rank)
) WITHOUT ROWID;
CREATE TABLE query_diagnostics (
    query_id INTEGER PRIMARY KEY,
    query_token_count INTEGER NOT NULL,
    lexical_candidate_count INTEGER NOT NULL,
    best_lexical_score REAL NOT NULL,
    weak_query INTEGER NOT NULL,
    top1000_distinct_domains INTEGER NOT NULL,
    top1000_lexical_evidence_fraction REAL NOT NULL
);
CREATE TABLE manifest (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _load_manifest(connection: sqlite3.Connection) -> dict[str, str]:
    return {str(key): str(value) for key, value in connection.execute("SELECT key, value FROM manifest")}


def _load_fallback(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT d.doc_id, d.domain, d.scheme, d.has_han, d.vietnamese_like, "
        "d.opaque, d.path_token_count, d.text_hash, s.corpus_count "
        "FROM fallback f JOIN documents d ON d.doc_id=f.doc_id "
        "JOIN domain_stats s ON s.domain=d.domain ORDER BY f.domain, f.slot"
    )
    return [
        {
            "doc_id": int(row[0]),
            "domain": str(row[1]),
            "scheme": str(row[2]),
            "has_han": int(row[3]),
            "vietnamese_like": int(row[4]),
            "opaque": int(row[5]),
            "path_token_count": int(row[6]),
            "text_hash": str(row[7]),
            "domain_corpus_count": int(row[8]),
            "score": 0.0,
        }
        for row in rows
    ]


def _retrieve_lexical(
    connection: sqlite3.Connection,
    *,
    tokens: list[str],
    term_map: dict[str, tuple[int, int]],
    corpus_size: int,
    average_length: float,
    k1: float,
    b: float,
    limit: int,
    maximum_query_terms: int,
) -> list[dict[str, Any]]:
    query_terms_by_rarity = [
        (document_frequency, term_id)
        for term in sorted(set(tokens))
        if (details := term_map.get(term)) is not None
        for term_id, document_frequency in [details]
        if document_frequency > 0
    ]
    query_terms = [
        term_id
        for _document_frequency, term_id in sorted(query_terms_by_rarity)[:maximum_query_terms]
    ]
    if not query_terms:
        return []
    if (k1, b) != (1.2, 0.75):
        raise ValueError("SQLite FTS5 BM25 uses fixed k1=1.2 and b=0.75")
    match_expression = " OR ".join(f"t{term_id}" for term_id in query_terms)
    rows = connection.execute(
        "SELECT f.rowid, -bm25(url_fts) AS score, d.domain, d.scheme, "
        "d.has_han, d.vietnamese_like, d.opaque, d.path_token_count, d.text_hash "
        "FROM url_fts f JOIN documents d ON d.doc_id=f.rowid "
        "WHERE url_fts MATCH ? ORDER BY bm25(url_fts), f.rowid LIMIT ?",
        (match_expression, limit),
    )
    return [
        {
            "doc_id": int(row[0]),
            "score": float(row[1]),
            "domain": str(row[2]),
            "scheme": str(row[3]),
            "has_han": int(row[4]),
            "vietnamese_like": int(row[5]),
            "opaque": int(row[6]),
            "path_token_count": int(row[7]),
            "text_hash": str(row[8]),
        }
        for row in rows
    ]


def _candidate_rows(
    method: str,
    depth: int,
    query_id: int,
    rows: list[dict[str, Any]],
) -> list[tuple[Any, ...]]:
    return [
        (
            method,
            depth,
            query_id,
            rank,
            int(row["doc_id"]),
            float(row.get("score", 0.0)),
            str(row.get("source", method)),
        )
        for rank, row in enumerate(rows, start=1)
    ]


def retrieve_all_queries(
    *,
    index_path: str | Path,
    output_path: str | Path,
    depths: list[int],
    lexical_pool: int,
    k1: float,
    b: float,
    weak_best_score: float,
    weak_minimum_candidates: int,
    maximum_domain_share: float,
    minimum_domains: int,
    weak_fallback_fraction: float,
    maximum_query_terms: int,
) -> dict[str, Any]:
    source = sqlite3.connect(f"file:{Path(index_path)}?mode=ro", uri=True)
    manifest = _load_manifest(source)
    corpus_size = int(manifest["corpus_rows"])
    average_length = float(manifest["average_document_length"])
    term_map = {
        str(term): (int(term_id), int(df))
        for term_id, term, df in source.execute(
            "SELECT term_id, term, document_frequency FROM terms"
        )
    }
    fallback_rows = _load_fallback(source)
    query_rows = list(source.execute("SELECT query_id, tokens_json, token_count FROM queries ORDER BY query_id"))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    output = sqlite3.connect(temporary)
    output.executescript(CANDIDATE_SCHEMA)
    output.execute("PRAGMA journal_mode=OFF")
    output.execute("PRAGMA synchronous=OFF")
    best_scores: list[float] = []
    lexical_counts: list[int] = []
    weak_count = 0
    started = time.perf_counter()
    for query_position, (query_id_raw, encoded_tokens, token_count) in enumerate(query_rows, start=1):
        query_id = int(query_id_raw)
        tokens = list(json.loads(str(encoded_tokens)))
        lexical = _retrieve_lexical(
            source,
            tokens=tokens,
            term_map=term_map,
            corpus_size=corpus_size,
            average_length=average_length,
            k1=k1,
            b=b,
            limit=lexical_pool,
            maximum_query_terms=maximum_query_terms,
        )
        weak = is_weak_query(
            lexical,
            minimum_best_score=weak_best_score,
            minimum_candidates=weak_minimum_candidates,
        )
        weak_count += int(weak)
        best_scores.append(float(lexical[0]["score"]) if lexical else 0.0)
        lexical_counts.append(len(lexical))
        fallback = domain_balanced_order(fallback_rows, query_id=query_id)
        combined_at_max: list[dict[str, Any]] = []
        batch_values: list[tuple[Any, ...]] = []
        for depth in depths:
            lexical_depth = lexical[:depth]
            domain_depth = [
                {**row, "source": "domain_prior"} for row in fallback[:depth]
            ]
            combined = select_combined_candidates(
                lexical,
                fallback,
                depth=depth,
                weak=weak,
                maximum_domain_share=maximum_domain_share,
                minimum_domains=minimum_domains,
                weak_fallback_fraction=weak_fallback_fraction,
            )
            if depth == max(depths):
                combined_at_max = combined
            batch_values.extend(_candidate_rows("lexical", depth, query_id, lexical_depth))
            batch_values.extend(_candidate_rows("domain_prior", depth, query_id, domain_depth))
            batch_values.extend(_candidate_rows("combined", depth, query_id, combined))
        with output:
            output.executemany(
                "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?)", batch_values
            )
            output.execute(
                "INSERT INTO query_diagnostics VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    query_id,
                    int(token_count),
                    len(lexical),
                    float(lexical[0]["score"]) if lexical else 0.0,
                    int(weak),
                    len({str(row["domain"]) for row in combined_at_max}),
                    round(
                        sum(str(row.get("source", "")).startswith("lexical") for row in combined_at_max)
                        / max(len(combined_at_max), 1),
                        6,
                    ),
                ),
            )
        if query_position % 100 == 0:
            elapsed = time.perf_counter() - started
            print(f"retrieved {query_position}/{len(query_rows)} queries ({query_position / max(elapsed, 1e-9):.2f}/s)")
    retrieval_seconds = time.perf_counter() - started
    output.execute("CREATE INDEX idx_candidates_lookup ON candidates(method, depth, query_id, doc_id)")
    output.execute("CREATE INDEX idx_candidates_global ON candidates(method, depth, doc_id)")
    output.executemany(
        "INSERT INTO manifest VALUES (?, ?)",
        {
            "source_index": str(index_path),
            "query_count": str(len(query_rows)),
            "depths": json.dumps(depths),
        }.items(),
    )
    output.commit()
    integrity = str(output.execute("PRAGMA integrity_check").fetchone()[0])
    output.close()
    source.close()
    os.replace(temporary, destination)
    return {
        "query_count": len(query_rows),
        "depths": depths,
        "weak_query_count": weak_count,
        "weak_query_fraction": round(weak_count / max(len(query_rows), 1), 6),
        "lexical_candidate_count": {
            "min": min(lexical_counts, default=0),
            "median": statistics.median(lexical_counts) if lexical_counts else 0,
            "max": max(lexical_counts, default=0),
        },
        "best_lexical_score": {
            "min": round(min(best_scores, default=0.0), 6),
            "median": round(statistics.median(best_scores), 6) if best_scores else 0,
            "p95": round(_percentile(best_scores, 0.95), 6),
            "max": round(max(best_scores, default=0.0), 6),
        },
        "maximum_query_terms_by_idf": maximum_query_terms,
        "retrieval_seconds": round(retrieval_seconds, 6),
        "mean_ms_per_query": round(retrieval_seconds * 1000 / max(len(query_rows), 1), 6),
        "output_path": str(destination),
        "output_size_bytes": destination.stat().st_size,
        "integrity_check": integrity,
    }


def _full_distribution(
    index: sqlite3.Connection, full_signal_stats: dict[str, Any]
) -> dict[str, Any]:
    manifest = _load_manifest(index)
    total = int(manifest["corpus_rows"])
    domains = list(index.execute("SELECT domain, corpus_count FROM domain_stats ORDER BY corpus_count DESC, domain"))
    result = {
        "rows": total,
        "unique_domains": len(domains),
        "top1_domain_share": round(int(domains[0][1]) / total, 6),
        "top5_domain_share": round(sum(int(row[1]) for row in domains[:5]) / total, 6),
    }
    for name in ("han_url_count", "vietnamese_like_url_count", "opaque_url_count"):
        count = int(full_signal_stats[name])
        result[name] = count
        result[name.replace("_count", "_share")] = round(count / max(total, 1), 6)
    result["path_signal_share"] = round(
        int(full_signal_stats["path_signal_count"]) / max(total, 1), 6
    )
    result["scheme_counts"] = full_signal_stats["scheme_counts"]
    return result


def candidate_diagnostics(
    *,
    index_path: str | Path,
    candidate_path: str | Path,
    depths: list[int],
    full_signal_stats: dict[str, Any],
) -> dict[str, Any]:
    index = sqlite3.connect(index_path)
    candidates = sqlite3.connect(candidate_path)
    candidates.execute("ATTACH DATABASE ? AS source", (str(Path(index_path).resolve()),))
    full = _full_distribution(index, full_signal_stats)
    output: dict[str, Any] = {"full_reference": full, "methods": {}}
    started = time.perf_counter()
    for method in ("lexical", "domain_prior", "combined"):
        method_rows: dict[str, Any] = {}
        for depth in depths:
            global_count = int(
                candidates.execute(
                    "SELECT COUNT(DISTINCT doc_id) FROM candidates WHERE method=? AND depth=?",
                    (method, depth),
                ).fetchone()[0]
            )
            domain_rows = list(
                candidates.execute(
                    "SELECT d.domain, COUNT(*) AS n FROM (SELECT DISTINCT doc_id FROM candidates "
                    "WHERE method=? AND depth=?) c JOIN source.documents d ON d.doc_id=c.doc_id "
                    "GROUP BY d.domain ORDER BY n DESC, d.domain",
                    (method, depth),
                )
            )
            signals = candidates.execute(
                "SELECT COALESCE(SUM(d.has_han),0), COALESCE(SUM(d.vietnamese_like),0), "
                "COALESCE(SUM(d.opaque),0), COALESCE(SUM(d.path_token_count>0),0), "
                "COALESCE(SUM(d.scheme='https'),0), COALESCE(SUM(d.scheme='http'),0) "
                "FROM (SELECT DISTINCT doc_id FROM candidates WHERE method=? AND depth=?) c "
                "JOIN source.documents d ON d.doc_id=c.doc_id",
                (method, depth),
            ).fetchone()
            duplicate_groups, repeated_rows = candidates.execute(
                "SELECT COUNT(*), COALESCE(SUM(n-1),0) FROM (SELECT d.text_hash, COUNT(*) n "
                "FROM (SELECT DISTINCT doc_id FROM candidates WHERE method=? AND depth=?) c "
                "JOIN source.documents d ON d.doc_id=c.doc_id GROUP BY d.text_hash HAVING n>1)",
                (method, depth),
            ).fetchone()
            per_query_rows = list(
                candidates.execute(
                    "SELECT COUNT(*), SUM(source LIKE 'lexical%') FROM candidates "
                    "WHERE method=? AND depth=? GROUP BY query_id",
                    (method, depth),
                )
            )
            per_query_counts = [int(row[0]) for row in per_query_rows]
            per_query_lexical_fractions = [
                int(row[1] or 0) / max(int(row[0]), 1) for row in per_query_rows
            ]
            method_rows[str(depth)] = {
                "global_unique_urls": global_count,
                "corpus_reduction_fraction": round(1 - global_count / max(full["rows"], 1), 6),
                "unique_domains": len(domain_rows),
                "domain_fraction_of_full": round(len(domain_rows) / max(full["unique_domains"], 1), 6),
                "top1_domain_share": round(int(domain_rows[0][1]) / max(global_count, 1), 6) if domain_rows else 0,
                "top5_domain_share": round(sum(int(row[1]) for row in domain_rows[:5]) / max(global_count, 1), 6),
                "han_url_count": int(signals[0]),
                "han_url_share": round(int(signals[0]) / max(global_count, 1), 6),
                "vietnamese_like_url_count": int(signals[1]),
                "vietnamese_like_url_share": round(int(signals[1]) / max(global_count, 1), 6),
                "opaque_url_count": int(signals[2]),
                "opaque_url_share": round(int(signals[2]) / max(global_count, 1), 6),
                "path_signal_share": round(int(signals[3]) / max(global_count, 1), 6),
                "https_count": int(signals[4]),
                "http_count": int(signals[5]),
                "duplicate_url_text_groups": int(duplicate_groups),
                "repeated_url_text_rows": int(repeated_rows),
                "per_query_candidate_count": {
                    "min": min(per_query_counts, default=0),
                    "median": statistics.median(per_query_counts)
                    if per_query_counts
                    else 0,
                    "max": max(per_query_counts, default=0),
                },
                "mean_query_lexical_evidence_fraction": round(
                    statistics.fmean(per_query_lexical_fractions), 6
                )
                if per_query_lexical_fractions
                else 0,
            }
        output["methods"][method] = method_rows
    diagnostics = list(
        candidates.execute(
            "SELECT query_token_count, lexical_candidate_count, best_lexical_score, weak_query, "
            "top1000_distinct_domains, top1000_lexical_evidence_fraction FROM query_diagnostics"
        )
    )
    ordered = sorted(diagnostics, key=lambda row: int(row[0]))
    quartiles = []
    for number in range(4):
        group = ordered[number * len(ordered) // 4 : (number + 1) * len(ordered) // 4]
        quartiles.append(
            {
                "query_token_min": min((int(row[0]) for row in group), default=0),
                "query_token_max": max((int(row[0]) for row in group), default=0),
                "queries": len(group),
                "weak_queries": sum(int(row[3]) for row in group),
                "mean_distinct_domains_at_1000": round(
                    statistics.fmean(int(row[4]) for row in group), 3
                ) if group else 0,
                "mean_lexical_evidence_fraction_at_1000": round(
                    statistics.fmean(float(row[5]) for row in group), 6
                ) if group else 0,
            }
        )
    output["query_length_quartiles"] = quartiles
    output["dedupe_diagnostics_seconds"] = round(time.perf_counter() - started, 6)
    candidates.close()
    index.close()
    return output


def candidate_retention(
    *,
    candidate_path: str | Path,
    sources: dict[str, str | Path],
    depths: list[int],
) -> dict[str, Any]:
    connection = sqlite3.connect(candidate_path)
    connection.execute(
        "CREATE TEMP TABLE content_candidates(method TEXT, query_id INTEGER, doc_id INTEGER, rank INTEGER, "
        "PRIMARY KEY(method, query_id, doc_id)) WITHOUT ROWID"
    )
    source_depths: dict[str, int] = {}
    for method, path_value in sources.items():
        parquet = pq.ParquetFile(path_value)
        names = parquet.schema_arrow.names
        rank_column = "rerank_rank" if "rerank_rank" in names else "rank"
        maximum_rank = 0
        for batch in parquet.iter_batches(columns=["query_id", "doc_id", rank_column]):
            values = []
            for row in batch.to_pylist():
                rank = int(row[rank_column])
                maximum_rank = max(maximum_rank, rank)
                values.append((method, int(row["query_id"]), int(row["doc_id"]), rank))
            with connection:
                connection.executemany(
                    "INSERT OR IGNORE INTO content_candidates VALUES (?, ?, ?, ?)", values
                )
        source_depths[method] = maximum_rank
    output: dict[str, Any] = {}
    for method in sources:
        denominator = int(
            connection.execute(
                "SELECT COUNT(*) FROM content_candidates WHERE method=?", (method,)
            ).fetchone()[0]
        )
        top10_denominator = int(
            connection.execute(
                "SELECT COUNT(*) FROM content_candidates WHERE method=? AND rank<=10", (method,)
            ).fetchone()[0]
        )
        by_depth = {}
        for depth in depths:
            retained = int(
                connection.execute(
                    "SELECT COUNT(*) FROM content_candidates p WHERE p.method=? AND EXISTS ("
                    "SELECT 1 FROM candidates c WHERE c.method='combined' AND c.depth=? "
                    "AND c.query_id=p.query_id AND c.doc_id=p.doc_id)",
                    (method, depth),
                ).fetchone()[0]
            )
            retained_top10 = int(
                connection.execute(
                    "SELECT COUNT(*) FROM content_candidates p WHERE p.method=? AND p.rank<=10 AND EXISTS ("
                    "SELECT 1 FROM candidates c WHERE c.method='combined' AND c.depth=? "
                    "AND c.query_id=p.query_id AND c.doc_id=p.doc_id)",
                    (method, depth),
                ).fetchone()[0]
            )
            by_depth[str(depth)] = {
                "retained": retained,
                "total": denominator,
                "retention_fraction": round(retained / max(denominator, 1), 6),
                "top10_retained": retained_top10,
                "top10_total": top10_denominator,
                "top10_retention_fraction": round(
                    retained_top10 / max(top10_denominator, 1), 6
                ),
            }
        output[method] = {"source_depth": source_depths[method], "by_url_depth": by_depth}
    connection.close()
    return output


def evaluate_viability(
    *,
    index_summary: dict[str, Any],
    retrieval_summary: dict[str, Any],
    diagnostics: dict[str, Any],
    retention: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    full = index_summary["full_corpus"]
    combined = diagnostics["methods"]["combined"]["1000"]
    full_han_share = full["han_url_count"] / max(index_summary["corpus_rows"], 1)
    full_opaque_share = full["opaque_url_count"] / max(index_summary["corpus_rows"], 1)
    signals = {
        "global_urls_at_1000": combined["global_unique_urls"],
        "reranked_retention_at_1000": retention["reranked"]["by_url_depth"]["1000"]["retention_fraction"],
        "weak_query_fraction": retrieval_summary["weak_query_fraction"],
        "domain_fraction_at_1000": combined["domain_fraction_of_full"],
        "han_share_retention": round(combined["han_url_share"] / max(full_han_share, 1e-12), 6),
        "opaque_share_retention": round(combined["opaque_url_share"] / max(full_opaque_share, 1e-12), 6),
    }

    def passes(level: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
        checks = {
            "global_size": signals["global_urls_at_1000"] <= int(level["maximum_global_urls_at_1000"]),
            "reranked_retention": signals["reranked_retention_at_1000"] >= float(level["minimum_reranked_retention_at_1000"]),
            "weak_queries": signals["weak_query_fraction"] <= float(level["maximum_weak_query_fraction"]),
            "domain_diversity": signals["domain_fraction_at_1000"] >= float(level["minimum_domain_fraction_at_1000"]),
            "han_preservation": signals["han_share_retention"] >= float(level["minimum_han_share_retention"]),
            "opaque_preservation": signals["opaque_share_retention"] >= float(level["minimum_opaque_share_retention"]),
        }
        return all(checks.values()), checks

    promising, promising_checks = passes(gates["promising"])
    borderline, borderline_checks = passes(gates["borderline"])
    verdict = "PROMISING" if promising else "BORDERLINE" if borderline else "NOT VIABLE"
    return {
        "verdict": verdict,
        "signals": signals,
        "promising_checks": promising_checks,
        "borderline_checks": borderline_checks,
        "bounded_10k_crawl_justified": verdict in {"PROMISING", "BORDERLINE"},
        "policy": "engineering feasibility only; no relevance claim",
    }
