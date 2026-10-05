from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.submission.common import atomic_write_json

from .analysis import analyze_stage
from .cache import SearchCache, request_fingerprint
from .provider import BingRssProvider, SearchProvider
from .query_variants import build_variants
from .sampling import build_stratified_sample
from .url_mapping import map_result_urls


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"version", "inputs", "outputs", "sampling", "provider", "query_variants", "url_mapping", "stage_a_gate", "stage_b_gates", "projection"}
    if not isinstance(config, dict) or required - set(config):
        raise ValueError(f"search discovery config missing: {sorted(required - set(config or {}))}")
    if config["provider"]["kind"] != "bing_rss":
        raise ValueError("only the bounded bing_rss provider is implemented")
    return config


def _write_sample(config: dict[str, Any], sample: dict[str, Any]) -> None:
    artifacts = Path(config["outputs"]["artifacts"])
    atomic_write_json(artifacts / "query_sample.json", sample)


def _stage_a_passes(summary: dict[str, Any], config: dict[str, Any]) -> bool:
    diagnostics = summary["query_diagnostics"]
    completed = sum(row["request_status"] == "SUCCESS" for row in diagnostics)
    completion_fraction = completed / max(len(diagnostics), 1)
    gate = config["stage_a_gate"]
    hard_failure = (
        summary["combined_metrics"]["query_map_rate"] <= float(gate["hard_failure_maximum_query_map_rate"])
        and summary["combined_metrics"]["unique_mapped_docs"] <= int(gate["hard_failure_maximum_total_mapped_docs"])
    )
    return completion_fraction >= float(gate["minimum_completed_request_fraction"]) and not hard_failure


def run_stage(
    config: dict[str, Any],
    *,
    stage: str,
    provider: SearchProvider | None = None,
) -> dict[str, Any]:
    if stage not in {"A", "B"}:
        raise ValueError("stage must be A or B")
    sample = build_stratified_sample(config)
    _write_sample(config, sample)
    query_ids = set(sample["stage_a_query_ids"] if stage == "A" else sample["stage_b_query_ids"])
    query_rows = {int(row["query_id"]): row for row in sample["queries"] if int(row["query_id"]) in query_ids}
    client = provider or BingRssProvider(config["provider"])
    cache = SearchCache(config["outputs"]["cache"])
    requested = reused = 0
    try:
        depth = int(config["provider"]["result_depth"])
        for query_id in sorted(query_ids):
            for variant, query_text in build_variants(query_rows[query_id]["query_text"]).items():
                fingerprint = request_fingerprint(client.name, query_text, depth)
                cached_status = cache.response_status(fingerprint)
                if cached_status is not None:
                    cache.assign(query_id, variant, fingerprint)
                    reused += 1
                    continue
                results, metadata = client.search(query_text, depth)
                cache.write_response(
                    fingerprint=fingerprint,
                    provider=client.name,
                    query_text=query_text,
                    depth=depth,
                    results=results,
                    **metadata,
                )
                cache.assign(query_id, variant, fingerprint)
                requested += 1
        unmapped = cache.unmapped_urls()
        if unmapped:
            mappings, domain_counts = map_result_urls(
                unmapped,
                corpus_path=config["inputs"]["corpus"],
                tracking_parameters=config["url_mapping"]["strip_tracking_parameters"],
            )
            cache.write_mappings(mappings)
            cache.write_domain_stats(domain_counts)
        summary = analyze_stage(config, cache.connection, sample, stage)
        summary["execution"] = {
            "new_external_requests": requested,
            "cached_assignments_reused": reused,
            "new_result_urls_mapped": len(unmapped),
            "cache_path": config["outputs"]["cache"],
            "cache_integrity_check": cache.connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0],
        }
        if stage == "A":
            summary["stage_a_passed"] = _stage_a_passes(summary, config)
        artifacts = Path(config["outputs"]["artifacts"])
        atomic_write_json(artifacts / f"stage_{stage.lower()}_summary.json", summary)
        if stage == "A":
            atomic_write_json(
                artifacts / "provider_access.json",
                {
                    "provider": summary["provider"],
                    "result_depth": summary["provider_result_depth"],
                    "execution": summary["execution"],
                    "request_status_counts": {
                        status: count
                        for status, count in cache.connection.execute(
                            "SELECT status,COUNT(*) FROM responses GROUP BY status"
                        )
                    },
                    "stage_a_passed": summary["stage_a_passed"],
                },
            )
        else:
            atomic_write_json(artifacts / "viability_verdict.json", summary["viability"])
            atomic_write_json(artifacts / "scale_projection.json", summary["stage_c_projection"])
            atomic_write_json(
                artifacts / "pilot_overlap.json",
                {
                    "label": "DESCRIPTIVE CANDIDATE OVERLAP — not relevance recall",
                    "search": summary["pilot_candidate_overlap"],
                    "random_equivalent_size": summary["random_baseline_overlap"],
                },
            )
            atomic_write_json(artifacts / "domain_analysis.json", summary["domain_analysis"])
        return summary
    finally:
        cache.close()
