from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit

import pyarrow.parquet as pq
import yaml

from src.search_discovery.provider import BingRssProvider
from src.search_discovery.query_variants import concise_vietnamese
from src.search_discovery.url_mapping import canonical_key, normalize_url
from src.url_reduction.selection import bm25_term_score
from src.url_reduction.url_text import extract_url_features, lexical_tokens


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)


def stable_key(value: int | str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def stable_fold(doc_id: int, folds: int, seed: str) -> int:
    return int(stable_key(doc_id, seed)[:16], 16) % folds


def s4_rows(corpus_path: str | Path, domain: str) -> list[dict[str, Any]]:
    output = []
    parquet = pq.ParquetFile(corpus_path)
    for batch in parquet.iter_batches(columns=["id", "url"], batch_size=131072):
        for doc_id, url in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
            host = (urlsplit(str(url)).hostname or "").casefold().removeprefix("www.")
            if host == domain:
                output.append({"doc_id": int(doc_id), "url": str(url)})
    output.sort(key=lambda row: row["doc_id"])
    return output


def proxy_sets(source_documents: str | Path, final_documents: str | Path) -> dict[str, Any]:
    source_ids = set(pq.read_table(source_documents, columns=["doc_id"]).column(0).to_pylist())
    rows = pq.read_table(final_documents, columns=["query_id", "rank", "doc_id"]).to_pylist()
    by_depth = {
        str(depth): sorted(
            {int(row["doc_id"]) for row in rows if int(row["rank"]) <= depth and int(row["doc_id"]) in source_ids}
        )
        for depth in (10, 5, 3)
    }
    frequencies = Counter(int(row["doc_id"]) for row in rows if int(row["doc_id"]) in source_ids)
    multiple = sorted(doc_id for doc_id, count in frequencies.items() if count >= 2)
    return {
        "primary_top10": by_depth["10"], "stronger_top5": by_depth["5"],
        "stronger_top3": by_depth["3"], "selected_multiple_queries": multiple,
        "counts": {
            "primary_top10": len(by_depth["10"]), "stronger_top5": len(by_depth["5"]),
            "stronger_top3": len(by_depth["3"]), "selected_multiple_queries": len(multiple),
        },
        "warning": "retrieval-selected proxy sets, not gold relevance labels",
    }


def path_structure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    depths: Counter[int] = Counter()
    prefixes: Counter[str] = Counter()
    for row in rows:
        segments = [unquote(value) for value in urlsplit(row["url"]).path.split("/") if value]
        depths[len(segments)] += 1
        if len(segments) >= 2:
            prefixes[segments[0].casefold()] += 1
    repeated = {key: value for key, value in prefixes.items() if value >= 2}
    viable = bool(repeated)
    return {
        "url_count": len(rows), "path_depth_counts": dict(sorted(depths.items())),
        "repeated_first_segment_families": dict(sorted(repeated.items(), key=lambda item: (-item[1], item[0]))[:50]),
        "multi_segment_url_count": sum(value for depth, value in depths.items() if depth >= 2),
        "verdict": "VIABLE" if viable else "NOT_VIABLE",
        "reason": (
            "repeated path families exist" if viable
            else "all article URLs are root-level unique slugs; no observed category path can rank URLs"
        ),
    }


def build_token_state(rows: list[dict[str, Any]], stopwords: frozenset[str]) -> tuple[list[Counter[str]], Counter[str], dict[int, int], dict[str, Any]]:
    token_rows: list[Counter[str]] = []
    df: Counter[str] = Counter()
    id_to_index = {}
    opaque = 0
    informative = 0
    for index, row in enumerate(rows):
        feature = extract_url_features(row["url"], stopwords=stopwords)
        counts = Counter(dict(feature.token_counts))
        token_rows.append(counts)
        df.update(counts.keys())
        id_to_index[row["doc_id"]] = index
        opaque += int(feature.opaque)
        informative += int(bool(counts) and not feature.opaque)
    return token_rows, df, id_to_index, {"opaque_urls": opaque, "informative_slug_urls": informative}


def _complete_ranking(scored: dict[int, float], rows: list[dict[str, Any]], seed: str) -> list[int]:
    ranked = sorted(scored, key=lambda doc_id: (-scored[doc_id], doc_id))
    included = set(ranked)
    ranked.extend(
        row["doc_id"] for row in sorted(rows, key=lambda row: stable_key(row["doc_id"], seed))
        if row["doc_id"] not in included
    )
    return ranked


def lexical_ranking(
    rows: list[dict[str, Any]], token_rows: list[Counter[str]], df: Counter[str],
    queries_path: str | Path, stopwords: frozenset[str], config: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    lengths = []
    for index, counts in enumerate(token_rows):
        lengths.append(max(sum(counts.values()), 1))
        for term, frequency in counts.items():
            postings[term].append((index, frequency))
    average_length = sum(lengths) / len(lengths)
    lexical = config["lexical"]
    global_rrf: defaultdict[int, float] = defaultdict(float)
    query_rows = pq.read_table(queries_path, columns=["id", "query"]).to_pylist()
    matched_queries = 0
    started = time.perf_counter()
    for query in query_rows:
        terms = set(lexical_tokens(concise_vietnamese(str(query["query"])), stopwords=stopwords))
        scores: defaultdict[int, float] = defaultdict(float)
        for term in terms:
            term_postings = postings.get(term, ())
            for index, frequency in term_postings:
                scores[index] += bm25_term_score(
                    corpus_size=len(rows), document_frequency=len(term_postings),
                    term_frequency=frequency, document_length=lengths[index],
                    average_document_length=average_length, k1=float(lexical["bm25_k1"]),
                    b=float(lexical["bm25_b"]),
                )
        ranked = sorted(scores, key=lambda index: (-scores[index], rows[index]["doc_id"]))[: int(lexical["per_query_depth"])]
        matched_queries += int(bool(ranked))
        for rank, index in enumerate(ranked, start=1):
            global_rrf[rows[index]["doc_id"]] += 1.0 / (float(lexical["rrf_constant"]) + rank)
    ranking = _complete_ranking(global_rrf, rows, "s4-url-lexical-fallback")
    return ranking, {
        "query_count": len(query_rows), "queries_with_lexical_matches": matched_queries,
        "globally_scored_urls": len(global_rrf), "runtime_seconds": round(time.perf_counter() - started, 3),
        "method": "per-query URL-path BM25 top-1000 aggregated with RRF; hostname excluded",
    }


def _seed_scores(
    seed_ids: set[int], rows: list[dict[str, Any]], token_rows: list[Counter[str]],
    df: Counter[str], id_to_index: dict[int, int], maximum_fraction: float,
) -> dict[int, float]:
    seed_frequency: Counter[str] = Counter()
    for doc_id in seed_ids:
        seed_frequency.update(token_rows[id_to_index[doc_id]].keys())
    allowed = {
        term: frequency for term, frequency in seed_frequency.items()
        if df[term] / len(rows) <= maximum_fraction
    }
    scores = {}
    for index, counts in enumerate(token_rows):
        score = 0.0
        for term in counts:
            if term in allowed:
                idf = math.log(1.0 + (len(rows) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * (1.0 + math.log(allowed[term]))
        if score:
            scores[rows[index]["doc_id"]] = score
    return scores


def seed_expand_rankings(
    rows: list[dict[str, Any]], token_rows: list[Counter[str]], df: Counter[str],
    id_to_index: dict[int, int], primary: set[int], config: dict[str, Any],
) -> tuple[list[int], dict[int, list[int]], dict[str, Any]]:
    settings = config["seed_expand"]
    folds = int(settings["folds"])
    seed = str(settings["fold_seed"])
    maximum_fraction = float(settings["maximum_token_document_fraction"])
    fold_rankings = {}
    fold_sizes = {}
    started = time.perf_counter()
    for fold in range(folds):
        held_out = {doc_id for doc_id in primary if stable_fold(doc_id, folds, seed) == fold}
        training = primary - held_out
        scores = _seed_scores(training, rows, token_rows, df, id_to_index, maximum_fraction)
        fold_rankings[fold] = _complete_ranking(scores, rows, f"s4-seed-fold-{fold}")
        fold_sizes[str(fold)] = {"seeds": len(training), "held_out": len(held_out), "scored_urls": len(scores)}
    full_scores = _seed_scores(primary, rows, token_rows, df, id_to_index, maximum_fraction)
    full_ranking = _complete_ranking(full_scores, rows, "s4-seed-all-fallback")
    return full_ranking, fold_rankings, {
        "evaluation": "five-fold stable-hash held-out proxy evaluation",
        "exploratory_full_seed_ranking": "not a valid proxy evaluation result",
        "folds": fold_sizes, "globally_scored_urls": len(full_scores),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }


def _retention(
    ranking: list[int], positive_ids: set[int], budgets: list[int], *, missing_rank: int
) -> dict[str, float]:
    positions = {doc_id: rank for rank, doc_id in enumerate(ranking, start=1) if doc_id in positive_ids}
    return {
        str(budget): round(sum(positions.get(doc_id, missing_rank) <= budget for doc_id in positive_ids) / max(len(positive_ids), 1), 6)
        for budget in budgets
    }


def evaluate_ranking(ranking: list[int], proxies: dict[str, Any], budgets: list[int], corpus_size: int) -> dict[str, Any]:
    return {
        "primary_retention": _retention(ranking, set(proxies["primary_top10"]), budgets, missing_rank=corpus_size + 1),
        "top5_retention": _retention(ranking, set(proxies["stronger_top5"]), budgets, missing_rank=corpus_size + 1),
        "top3_retention": _retention(ranking, set(proxies["stronger_top3"]), budgets, missing_rank=corpus_size + 1),
        "random_expected": {str(budget): round(budget / corpus_size, 6) for budget in budgets},
    }


def evaluate_fold_rankings(
    fold_rankings: dict[int, list[int]], proxies: dict[str, Any], budgets: list[int], corpus_size: int,
    folds: int, seed: str,
) -> dict[str, Any]:
    output = {}
    for name, key in (("primary_retention", "primary_top10"), ("top5_retention", "stronger_top5"), ("top3_retention", "stronger_top3")):
        positives = set(proxies[key])
        hits = {budget: 0 for budget in budgets}
        for fold, ranking in fold_rankings.items():
            held = {doc_id for doc_id in positives if stable_fold(doc_id, folds, seed) == fold}
            positions = {doc_id: rank for rank, doc_id in enumerate(ranking, start=1) if doc_id in held}
            for budget in budgets:
                hits[budget] += sum(positions.get(doc_id, corpus_size + 1) <= budget for doc_id in held)
        output[name] = {str(budget): round(hits[budget] / max(len(positives), 1), 6) for budget in budgets}
    output["random_expected"] = {str(budget): round(budget / corpus_size, 6) for budget in budgets}
    output["evaluation"] = "held-out only; each positive evaluated under the fold where it was not a seed"
    return output


def verdict(metrics: dict[str, Any]) -> str:
    primary = float(metrics["primary_retention"]["20000"])
    top5 = float(metrics["top5_retention"]["20000"])
    top3 = float(metrics["top3_retention"]["20000"])
    random = float(metrics["random_expected"]["20000"])
    if primary >= 0.70 and top5 >= 0.70 and top3 >= 0.70 and primary > random:
        return "STRONG"
    if primary >= 0.40 and primary > random:
        return "PROMISING"
    return "WEAK"


def rrf_rank(rankings: list[list[int]], constant: int, rows: list[dict[str, Any]], seed: str) -> list[int]:
    scores: defaultdict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (constant + rank)
    return _complete_ranking(scores, rows, seed)


def run_local(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = Path(config["outputs"]["artifacts"])
    work = Path(config["outputs"]["work_directory"])
    artifacts.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
    rows = s4_rows(config["inputs"]["corpus"], config["domain"])
    proxies = proxy_sets(config["inputs"]["s4_documents"], config["inputs"]["s4_final_documents"])
    structure = path_structure(rows)
    stopwords = frozenset(config["tokenization"]["stopwords"])
    started = time.perf_counter()
    token_rows, df, id_to_index, token_summary = build_token_state(rows, stopwords)
    preprocessing_seconds = time.perf_counter() - started
    lexical, lexical_summary = lexical_ranking(rows, token_rows, df, config["inputs"]["queries"], stopwords, config)
    seed_full, seed_folds, seed_summary = seed_expand_rankings(
        rows, token_rows, df, id_to_index, set(proxies["primary_top10"]), config
    )
    budgets = [int(value) for value in config["budgets"]]
    lexical_metrics = evaluate_ranking(lexical, proxies, budgets, len(rows))
    seed_metrics = evaluate_fold_rankings(
        seed_folds, proxies, budgets, len(rows), int(config["seed_expand"]["folds"]), str(config["seed_expand"]["fold_seed"])
    )
    methods = {
        "A_url_lexical": {"metrics": lexical_metrics, "verdict": verdict(lexical_metrics), **lexical_summary},
        "B_path_category": {"metrics": None, "verdict": "NOT_VIABLE", **structure},
        "C_seed_expand": {"metrics": seed_metrics, "verdict": verdict(seed_metrics), **seed_summary},
        "E_native_search": {
            "metrics": None, "verdict": "NOT_VIABLE", "network_requests": 0,
            "reason": "existing Phase 10B1 preflight: 0/5 official mappings and only 40% content-like results",
        },
    }
    viable = [name for name in ("A_url_lexical", "C_seed_expand") if methods[name]["verdict"] in {"STRONG", "PROMISING"}]
    ensemble_result = {"ran": False, "reason": "fewer than two PROMISING/STRONG local methods"}
    ensemble_full = None
    if len(viable) >= 2:
        ensemble_full = rrf_rank([lexical, seed_full], int(config["ensemble"]["rrf_constant"]), rows, "s4-ensemble-full")
        fold_ensembles = {
            fold: rrf_rank([lexical, ranking], int(config["ensemble"]["rrf_constant"]), rows, f"s4-ensemble-fold-{fold}")
            for fold, ranking in seed_folds.items()
        }
        ensemble_metrics = evaluate_fold_rankings(
            fold_ensembles, proxies, budgets, len(rows), int(config["seed_expand"]["folds"]), str(config["seed_expand"]["fold_seed"])
        )
        ensemble_result = {
            "ran": True, "methods": viable, "fusion": "RRF without learned weights",
            "metrics": ensemble_metrics, "verdict": verdict(ensemble_metrics),
        }
    rankings = {"A_url_lexical": lexical, "C_seed_expand": seed_full}
    if ensemble_full is not None: rankings["ensemble"] = ensemble_full
    for name, ranking in rankings.items():
        path = work / f"{name}_ranking.json"
        atomic_json(path, {"method": name, "doc_ids": ranking})
    structure.update({"tokenization": token_summary, "preprocessing_seconds": round(preprocessing_seconds, 3)})
    atomic_json(artifacts / "s4_structure_summary.json", structure)
    atomic_json(artifacts / "proxy_positive_summary.json", proxies)
    atomic_json(artifacts / "method_results.json", methods)
    atomic_json(artifacts / "ensemble_results.json", ensemble_result)
    return {"s4_urls": len(rows), "proxy_counts": proxies["counts"], "methods": methods, "ensemble": ensemble_result}


SEARCH_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses(query_id INTEGER PRIMARY KEY, query_text TEXT NOT NULL, status TEXT NOT NULL, http_status INTEGER, attempt_count INTEGER NOT NULL, error_type TEXT, error_message TEXT, results_json TEXT NOT NULL);
"""


def _mapping_maps(rows: list[dict[str, Any]]) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]]:
    exact: defaultdict[str, list[int]] = defaultdict(list)
    normalized: defaultdict[str, list[int]] = defaultdict(list)
    canonical: defaultdict[str, list[int]] = defaultdict(list)
    for row in rows:
        exact[row["url"]].append(row["doc_id"])
        normalized[normalize_url(row["url"])].append(row["doc_id"])
        canonical[canonical_key(row["url"])].append(row["doc_id"])
    return exact, normalized, canonical


def map_s4_url(url: str, maps: tuple[dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]]) -> tuple[int | None, str]:
    exact, normalized, canonical = maps
    levels = ((exact.get(url, []), "exact"), (normalized.get(normalize_url(url), []), "normalized"), (canonical.get(canonical_key(url), []), "canonical"))
    for candidates, level in levels:
        if len(candidates) == 1: return int(candidates[0]), level
        if len(candidates) > 1: return None, "ambiguous"
    return None, "no_match"


def run_external_search(config: dict[str, Any], stage: str = "A") -> dict[str, Any]:
    settings = config["external_search"]
    rows = s4_rows(config["inputs"]["corpus"], config["domain"])
    maps = _mapping_maps(rows)
    sample = json.loads(Path(config["inputs"]["phase10b0_sample"]).read_text(encoding="utf-8"))
    query_rows = {int(row["id"]): str(row["query"]) for row in pq.read_table(config["inputs"]["queries"]).to_pylist()}
    selected = sample["stage_a_query_ids"] if stage == "A" else sample["stage_b_query_ids"]
    if stage == "B":
        prior = json.loads((Path(config["outputs"]["artifacts"]) / "external_search_stage_a.json").read_text(encoding="utf-8"))
        if not prior["gate_passed"]: raise RuntimeError("Stage A gate did not pass")
    provider = BingRssProvider(settings)
    cache_path = Path(settings["cache"]); cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path); connection.executescript(SEARCH_SCHEMA)
    network_requests = 0
    for query_id in selected:
        if connection.execute("SELECT 1 FROM responses WHERE query_id=?", (query_id,)).fetchone(): continue
        concise = concise_vietnamese(query_rows[query_id])
        search_text = f"site:{config['domain']} {concise}"
        results, metadata = provider.search(search_text, int(settings["result_depth"]))
        with connection:
            connection.execute(
                "INSERT INTO responses VALUES (?,?,?,?,?,?,?,?)",
                (query_id, search_text, metadata["status"], metadata["http_status"], metadata["attempt_count"], metadata["error_type"], metadata["error_message"], json.dumps(results, ensure_ascii=False)),
            )
        network_requests += 1
    result_rows = []
    for query_id in selected:
        row = connection.execute("SELECT query_text,status,results_json FROM responses WHERE query_id=?", (query_id,)).fetchone()
        if row is None: continue
        for result in json.loads(row[2]):
            doc_id, level = map_s4_url(result["url"], maps)
            result_rows.append({"query_id": query_id, "rank": int(result["rank"]), "url": result["url"], "doc_id": doc_id, "match_level": level})
    connection.close()
    by_query: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    scores: defaultdict[int, float] = defaultdict(float)
    for row in result_rows:
        by_query[row["query_id"]].append(row)
        if row["doc_id"] is not None: scores[row["doc_id"]] += 1.0 / (60 + row["rank"])
    mapped_queries = sum(any(row["doc_id"] is not None for row in by_query[query_id]) for query_id in selected)
    ranking = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    rate = mapped_queries / len(selected)
    summary = {
        "stage": stage, "query_count": len(selected), "network_requests_this_run": network_requests,
        "result_rows": len(result_rows), "unique_result_urls": len({row["url"] for row in result_rows}),
        "mapped_rows": sum(row["doc_id"] is not None for row in result_rows),
        "unique_mapped_docs": len(scores), "queries_with_mapped_docs": mapped_queries,
        "query_map_rate": round(rate, 6), "ranking_doc_ids": ranking,
        "gate_passed": rate >= float(settings["stage_a_minimum_query_map_rate"]) if stage == "A" else None,
        "gate": ">=30% of 50 queries map at least one official S4 URL",
        "match_levels": dict(Counter(row["match_level"] for row in result_rows)),
    }
    atomic_json(Path(config["outputs"]["artifacts"]) / f"external_search_stage_{stage.lower()}.json", summary)
    return summary


def finalize(config: dict[str, Any]) -> dict[str, Any]:
    artifacts = Path(config["outputs"]["artifacts"])
    methods = json.loads((artifacts / "method_results.json").read_text(encoding="utf-8"))
    ensemble = json.loads((artifacts / "ensemble_results.json").read_text(encoding="utf-8"))
    proxies = json.loads((artifacts / "proxy_positive_summary.json").read_text(encoding="utf-8"))
    structure = json.loads((artifacts / "s4_structure_summary.json").read_text(encoding="utf-8"))
    external = json.loads((artifacts / "external_search_stage_a.json").read_text(encoding="utf-8"))
    budgets = [int(value) for value in config["budgets"]]
    external_ranking = external["ranking_doc_ids"]
    external_metrics = evaluate_ranking(external_ranking, proxies, budgets, structure["url_count"])
    methods["D_external_site_search"] = {
        "verdict": "WEAK" if external["unique_mapped_docs"] else "NOT_VIABLE",
        "metrics": external_metrics, **{key: value for key, value in external.items() if key != "ranking_doc_ids"},
    }
    atomic_json(artifacts / "method_results.json", methods)
    budget_summary = {
        "s4_url_count": structure["url_count"], "budgets": budgets,
        "methods": {name: {"verdict": row["verdict"], "metrics": row.get("metrics")} for name, row in methods.items()},
        "ensemble": ensemble,
    }
    atomic_json(artifacts / "candidate_budget_summary.json", budget_summary)
    return budget_summary
