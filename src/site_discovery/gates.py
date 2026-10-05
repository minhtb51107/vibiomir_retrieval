from __future__ import annotations


def classify_g0(weighted_share: float, specialist_share: float = 0.0) -> str:
    if weighted_share >= 0.30:
        return "STRONG"
    if weighted_share >= 0.10:
        return "PARTIAL"
    if specialist_share >= 0.05:
        return "PARTIAL_SPECIALIST"
    return "WEAK"


def classify_g1(
    *, query_coverage: float, subgroup_coverage: float,
    candidates_per_query_multiplier_vs_bing: float,
    reliable_mapping: bool, operationally_cheap: bool, concentrated_duplicates: bool,
) -> str:
    promising_signal = (
        query_coverage >= 0.30
        or subgroup_coverage >= 0.40
        or candidates_per_query_multiplier_vs_bing >= 5.0
    )
    if promising_signal and reliable_mapping and operationally_cheap and not concentrated_duplicates:
        return "PROMISING"
    if query_coverage >= 0.15 or subgroup_coverage >= 0.40:
        return "BORDERLINE"
    return "NOT_VIABLE"


def discovery_efficiency(unique_docs: int, requests: int) -> float:
    return unique_docs / requests if requests else 0.0


def g2_metrics_complete(metrics: dict[str, object]) -> bool:
    required = {
        "query_coverage", "unique_mapped_docs", "mapped_docs_per_query",
        "contributing_domains", "corpus_weighted_domain_coverage",
        "novelty_vs_bing", "novelty_vs_pilot", "duplicate_concentration",
        "projected_1200_candidates", "discovery_efficiency", "new_doc_efficiency",
    }
    return required.issubset(metrics)
