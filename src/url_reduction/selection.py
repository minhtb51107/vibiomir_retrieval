from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable


def bm25_term_score(
    *,
    corpus_size: int,
    document_frequency: int,
    term_frequency: int,
    document_length: int,
    average_document_length: float,
    k1: float,
    b: float,
) -> float:
    if corpus_size <= 0 or document_frequency <= 0 or term_frequency <= 0:
        return 0.0
    inverse_document_frequency = math.log(
        1.0 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
    )
    denominator = term_frequency + k1 * (
        1.0 - b + b * document_length / max(average_document_length, 1e-12)
    )
    return inverse_document_frequency * term_frequency * (k1 + 1.0) / denominator


def is_weak_query(
    lexical_candidates: list[dict[str, Any]],
    *,
    minimum_best_score: float,
    minimum_candidates: int,
) -> bool:
    return (
        not lexical_candidates
        or float(lexical_candidates[0]["score"]) < minimum_best_score
        or len(lexical_candidates) < minimum_candidates
    )


def domain_balanced_order(
    rows: Iterable[dict[str, Any]], *, query_id: int
) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_domain.setdefault(str(row["domain"]), []).append(dict(row))
    domains = sorted(
        by_domain,
        key=lambda domain: (-int(by_domain[domain][0].get("domain_corpus_count", 0)), domain),
    )
    if not domains:
        return []
    offset = query_id % len(domains)
    domains = domains[offset:] + domains[:offset]
    maximum = max(len(values) for values in by_domain.values())
    ordered = []
    for slot in range(maximum):
        for domain in domains:
            values = by_domain[domain]
            if slot < len(values):
                ordered.append(values[(slot + query_id) % len(values)])
    return ordered


def select_combined_candidates(
    lexical: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
    *,
    depth: int,
    weak: bool,
    maximum_domain_share: float,
    minimum_domains: int,
    weak_fallback_fraction: float,
) -> list[dict[str, Any]]:
    if depth <= 0:
        raise ValueError("depth must be positive")
    domain_cap = max(
        1,
        min(
            math.ceil(depth * maximum_domain_share),
            math.ceil(depth / max(minimum_domains, 1)),
        ),
    )
    fallback_reserve = math.ceil(depth * weak_fallback_fraction) if weak else 0
    lexical_target = depth - fallback_reserve
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    domain_counts: Counter[str] = Counter()
    deferred: list[dict[str, Any]] = []

    def add(row: dict[str, Any], source: str, enforce_cap: bool = True) -> bool:
        doc_id = int(row["doc_id"])
        domain = str(row["domain"])
        if doc_id in selected_ids:
            return False
        if enforce_cap and domain_counts[domain] >= domain_cap:
            return False
        value = dict(row)
        value["source"] = source
        selected.append(value)
        selected_ids.add(doc_id)
        domain_counts[domain] += 1
        return True

    for row in lexical:
        if len(selected) >= lexical_target:
            break
        if not add(row, "lexical"):
            deferred.append(row)
    for row in fallback:
        if len(selected) >= depth:
            break
        add(row, "fallback")
    for row in lexical:
        if len(selected) >= depth:
            break
        if int(row["doc_id"]) not in selected_ids:
            add(row, "lexical_deferred", enforce_cap=False)
    for row in fallback:
        if len(selected) >= depth:
            break
        add(row, "fallback_deferred", enforce_cap=False)
    return selected[:depth]


def global_deduplicate(rows: Iterable[tuple[int, int]]) -> set[int]:
    return {int(doc_id) for _, doc_id in rows}
