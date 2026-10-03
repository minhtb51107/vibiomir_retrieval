#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Phase 2C capacity evidence.")
    parser.add_argument("--dataset-stats", default="artifacts/dataset_stats.json")
    parser.add_argument(
        "--benchmark-summary", default="artifacts/crawl_benchmark/benchmark_summary.json"
    )
    parser.add_argument(
        "--resume-summary", default="artifacts/crawl_benchmark/resume_summary.json"
    )
    parser.add_argument("--crawler-config", default="configs/crawler.yaml")
    parser.add_argument("--out", default="artifacts/crawl_benchmark/results.json")
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset_stats).read_text(encoding="utf-8"))
    benchmark = json.loads(Path(args.benchmark_summary).read_text(encoding="utf-8"))
    resume = json.loads(Path(args.resume_summary).read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(args.crawler_config).read_text(encoding="utf-8"))

    total_rows = int(dataset["corpus"]["total_rows"])
    database = resume["database"]
    per_domain = database["per_domain"]
    top_domains = {
        item["domain"]: int(item["count"])
        for item in dataset["corpus"]["domains"]["top_30"]
    }
    measured_response_rows = sum(
        int(values["completed"])
        for values in per_domain.values()
        if int(values["downloaded_bytes"]) > 0
    )
    measured_bytes = sum(int(values["downloaded_bytes"]) for values in per_domain.values())
    fallback_average = measured_bytes / measured_response_rows
    covered_rows = sum(top_domains.values())
    weighted_network_bytes = sum(
        count
        * (
            per_domain[domain]["downloaded_bytes"] / per_domain[domain]["completed"]
            if domain in per_domain
            else fallback_average
        )
        for domain, count in top_domains.items()
    ) + (total_rows - covered_rows) * fallback_average

    main_database_bytes = int(database["main_database_bytes"])
    bytes_per_row = main_database_bytes / int(database["rows"])
    measured_urls_per_minute = float(benchmark["run"]["urls_completed_per_minute"])
    timed_wall_seconds = float(benchmark["run"]["wall_seconds"])
    timed_domain_throughput = {
        domain: round(int(count) * 60 / timed_wall_seconds, 4)
        for domain, count in benchmark["run"]["by_domain"].items()
    }
    throughput_days = total_rows / measured_urls_per_minute / 60 / 24

    default_policy = config["concurrency"]
    overrides = default_policy.get("domain_overrides") or {}
    domain_duration_estimates: dict[str, dict[str, float | int]] = {}
    for domain, count in list(top_domains.items())[:10]:
        if domain in config.get("safety", {}).get("access_restricted_domains", {}):
            domain_duration_estimates[domain] = {
                "corpus_rows": count,
                "estimated_days": 0.0,
                "basis": "metadata-only ACCESS_RESTRICTED classification; no page requests",
            }
            continue
        if domain == "zysjonline.com":
            domain_duration_estimates[domain] = {
                "corpus_rows": count,
                "estimated_days": 0.0,
                "basis": "cached robots denial; rows recorded without per-URL page requests",
            }
            continue
        policy = overrides.get(domain, {})
        concurrency = int(policy.get("concurrency", default_policy["per_domain_limit"]))
        delay = float(policy.get("delay_seconds", default_policy["default_domain_delay_seconds"]))
        values = per_domain.get(domain)
        mean_elapsed_seconds = (
            float(values["elapsed_ms_sum"]) / max(1, int(values["completed"])) / 1000
            if values
            else 1.0
        )
        start_rate = (1 / delay) if delay else float("inf")
        latency_rate = concurrency / max(mean_elapsed_seconds, 0.001)
        effective_rate = min(start_rate, latency_rate)
        domain_duration_estimates[domain] = {
            "corpus_rows": count,
            "measured_mean_result_elapsed_seconds": round(mean_elapsed_seconds, 4),
            "configured_concurrency": concurrency,
            "configured_delay_seconds": delay,
            "estimated_days": round(count / effective_rate / 86400, 2),
            "basis": "configured start-rate limit and measured mean result latency",
        }

    results = {
        "phase": "2C",
        "measured": {
            "selected_unique_urls": int(database["rows"]),
            "timed_resume_wall_seconds": timed_wall_seconds,
            "timed_resume_scheduled": benchmark["run"]["scheduled"],
            "timed_resume_completed": benchmark["run"]["completed"],
            "timed_resume_skipped": benchmark["run"]["skipped_existing"],
            "urls_completed_per_minute": measured_urls_per_minute,
            "attempt_requests_per_second": benchmark["run"]["attempt_requests_per_second"],
            "http_request_operations_per_second": benchmark["run"][
                "http_request_operations_per_second"
            ],
            "timed_domain_urls_per_minute": timed_domain_throughput,
            "status_counts": database["status_counts"],
            "http_status_counts": database["http_status_counts"],
            "success_rate": round(database["status_counts"]["SUCCESS"] / database["rows"], 6),
            "retry_count": database["retry_count"],
            "redirect_count": database["redirect_count"],
            "downloaded_bytes": database["downloaded_bytes"],
            "content_type_counts": database["content_type_counts"],
            "declared_http_encoding_counts": database["declared_http_encoding_counts"],
            "response_size_buckets": database["response_size_buckets"],
            "tiny_html_count": database["tiny_html_count"],
            "js_shell_candidate_count": database["js_shell_candidate_count"],
            "main_sqlite_bytes": main_database_bytes,
            "main_sqlite_bytes_per_row": round(bytes_per_row, 2),
            "per_domain": per_domain,
        },
        "estimated": {
            "full_corpus_rows": total_rows,
            "wall_clock_days_at_measured_balanced_throughput": round(throughput_days, 2),
            "planning_wall_clock_range_days": [25, 35],
            "main_sqlite_bytes": round(bytes_per_row * total_rows),
            "domain_weighted_download_bytes": round(weighted_network_bytes),
            "raw_body_storage_uncompressed_bytes": round(weighted_network_bytes),
            "compressed_html_storage_range_bytes": [
                round(weighted_network_bytes * 0.25),
                round(weighted_network_bytes * 0.40),
            ],
            "extracted_text_storage_range_bytes": [
                round(weighted_network_bytes * 0.10),
                round(weighted_network_bytes * 0.25),
            ],
            "fallback_average_download_bytes_for_unmeasured_domains": round(
                fallback_average, 2
            ),
            "largest_domain_duration_estimates": domain_duration_estimates,
            "caveats": [
                "The benchmark is intentionally balanced rather than corpus-proportional.",
                "Network estimates weight measured domain averages by Phase 1 top-30 counts and use the benchmark response average elsewhere.",
                "Compression and extracted-text ratios are planning assumptions, not measured outputs.",
                "No benchmark response reached the 2 MiB download cap.",
            ],
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "measured_rows": database["rows"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
