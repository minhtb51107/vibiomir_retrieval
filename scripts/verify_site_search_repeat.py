from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.search_discovery.url_mapping import map_result_urls
from src.site_discovery.cache import ProbeCache
from src.site_discovery.gates import classify_g0
from src.site_discovery.probe import (
    ConservativeFetcher, SearchMechanism, build_search_url, decode_body, extract_result_urls, page_flags,
)


def main() -> int:
    config = yaml.safe_load(Path("configs/site_discovery.yaml").read_text(encoding="utf-8"))
    inventory = json.loads(Path("artifacts/phase10b1_site_discovery/domain_inventory.json").read_text(encoding="utf-8"))
    first = json.loads(Path("artifacts/phase10b1_site_discovery/stage0_probe.json").read_text(encoding="utf-8"))
    provisional = [row for row in first["domains"] if row["native_search"] and row["mapping"]["exact"] + row["mapping"]["normalized"] > 0]
    checks = []
    all_urls = []
    probe_config = config["probing"]
    with ProbeCache(Path(config["outputs"]["cache_dir"]) / "stage0.sqlite") as cache:
        fetcher = ConservativeFetcher(
            cache, user_agent=probe_config["user_agent"], timeout=float(probe_config["request_timeout_seconds"]),
            delay_seconds=float(probe_config["delay_seconds"]), max_new_requests=len(provisional),
        )
        try:
            for row in provisional:
                mechanism_data = row["search_mechanisms"][0]
                mechanism = SearchMechanism(
                    mechanism_data["template_url"], mechanism_data["query_parameter"], mechanism_data["evidence"]
                )
                url = build_search_url(mechanism, "糖尿病")
                response, error, cached = fetcher.get(url)
                urls = []
                flags = {}
                if response and response.status_code < 400:
                    html = decode_body(response)
                    flags = page_flags(html)
                    urls = extract_result_urls(html, response.final_url, row["domain"])
                sample_limit = int(probe_config["max_result_urls_per_mechanism"])
                urls = urls[:sample_limit]
                first_urls = row["candidate_urls"][:sample_limit]
                all_urls.extend(urls)
                checks.append({
                    "domain": row["domain"], "request_url": url,
                    "status": response.status_code if response else None,
                    "cached": cached, "error": error,
                    "result_title": flags.get("title"),
                    "first_probe_urls": first_urls, "second_probe_urls": urls,
                    "result_sets_identical": set(first_urls) == set(urls),
                    "new_urls_in_second_probe": len(set(urls) - set(first_urls)),
                })
        finally:
            fetcher.close()
        new_requests = fetcher.new_requests
    mappings, _ = map_result_urls(
        list(dict.fromkeys(all_urls)), corpus_path=config["inputs"]["corpus_path"],
        tracking_parameters=["utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid"],
    ) if all_urls else ([], {})
    by_url = {row["result_url"]: row for row in mappings}
    for check in checks:
        mapped = [by_url[url] for url in check["second_probe_urls"] if url in by_url]
        check["second_probe_mapped"] = sum(row["doc_id"] is not None for row in mapped)
        check["query_sensitive_and_mappable"] = (
            not check["result_sets_identical"] and check["new_urls_in_second_probe"] > 0
            and check["second_probe_mapped"] > 0
        )
    usable = {check["domain"] for check in checks if check["query_sensitive_and_mappable"]}
    lookup = {row["domain"]: row for row in inventory["domains"]}
    share = sum(float(lookup[domain]["corpus_share"]) for domain in usable)
    payload = {
        "purpose": "Assumption E: reject mechanisms returning an unchanged popular page set for different queries.",
        "new_http_requests": new_requests,
        "usable_query_sensitive_domains": sorted(usable),
        "usable_weighted_corpus_share": round(share, 8),
        "g0_verdict_after_repeat_probe": classify_g0(share),
        "stage1_allowed": classify_g0(share) != "WEAK",
        "checks": checks,
    }
    output = Path(config["outputs"]["artifact_dir"]) / "stage0_repeat_probe.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "checks"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
