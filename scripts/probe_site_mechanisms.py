from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.search_discovery.url_mapping import map_result_urls
from src.site_discovery.cache import ProbeCache
from src.site_discovery.gates import classify_g0
from src.site_discovery.inventory import canonical_domain
from src.site_discovery.probe import (
    ConservativeFetcher, build_search_url, decode_body, detect_search_mechanisms,
    classify_discovery_url, extract_result_urls, extract_sitemap_locs, page_flags, robots_policy, sitemap_urls,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cheap Stage 0 probe of selected official domains.")
    parser.add_argument("--config", default="configs/site_discovery.yaml")
    parser.add_argument("--inventory", default="artifacts/phase10b1_site_discovery/domain_inventory.json")
    parser.add_argument("--offline-cache-only", action="store_true", help="Reuse cached responses and make zero HTTP requests.")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    inventory_path = Path(args.inventory)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    selected = inventory["selected_domains"]
    row_lookup = {row["domain"]: row for row in inventory["domains"]}
    probe_config = config["probing"]
    records = []
    all_candidates = []
    with ProbeCache(Path(config["outputs"]["cache_dir"]) / "stage0.sqlite") as cache:
        fetcher = ConservativeFetcher(
            cache, user_agent=probe_config["user_agent"],
            timeout=float(probe_config["request_timeout_seconds"]),
            delay_seconds=float(probe_config["delay_seconds"]),
            max_new_requests=0 if args.offline_cache_only else int(probe_config["max_total_stage0_requests"]),
        )
        try:
            for domain in selected:
                base = "https://" + domain + "/"
                record = {
                    "domain": domain, "requests": [], "native_search": False,
                    "sitemap": False, "category_index": False,
                    "search_mechanisms": [], "candidate_urls": [], "errors": [],
                }
                robots_url = base + "robots.txt"
                robots_response, error, cached = fetcher.get(robots_url)
                record["requests"].append({"url": robots_url, "cached": cached, "error": error,
                    "status": robots_response.status_code if robots_response else None})
                robots_text = decode_body(robots_response) if robots_response and robots_response.status_code < 400 else ""
                sitemaps = sitemap_urls(robots_text)
                record["sitemap"] = bool(sitemaps)
                record["sitemap_urls"] = sitemaps[:5]
                policy = robots_policy(robots_text, base, probe_config["user_agent"]) if robots_text else None

                if policy and not policy.can_fetch(probe_config["user_agent"], base):
                    record["errors"].append("robots_disallowed_home")
                    records.append(record)
                    continue
                home_response, error, cached = fetcher.get(base)
                record["requests"].append({"url": base, "cached": cached, "error": error,
                    "status": home_response.status_code if home_response else None})
                if not home_response:
                    record["errors"].append(error or "home_request_failed")
                    records.append(record)
                    continue
                html = decode_body(home_response)
                flags = page_flags(html)
                record.update(flags)
                record["category_index"] = int(flags["category_link_count"]) > 0
                mechanisms = detect_search_mechanisms(html, home_response.final_url)
                if not mechanisms and flags["wordpress"]:
                    from src.site_discovery.probe import SearchMechanism
                    mechanisms = [SearchMechanism(home_response.final_url, "s", "WordPress generator on home page")]
                for mechanism in mechanisms[:1]:
                    language = row_lookup[domain]["likely_language"]
                    probe_query = "健康" if language == "zh" else "sức khỏe" if language == "vi" else "health"
                    search_url = build_search_url(mechanism, probe_query)
                    if policy and not policy.can_fetch(probe_config["user_agent"], search_url):
                        record["errors"].append("robots_disallowed_search")
                        continue
                    response, search_error, search_cached = fetcher.get(search_url)
                    record["requests"].append({"url": search_url, "cached": search_cached,
                        "error": search_error, "status": response.status_code if response else None})
                    mechanism_record = {
                        "template_url": mechanism.template_url,
                        "query_parameter": mechanism.query_parameter,
                        "evidence": mechanism.evidence,
                        "probe_url": search_url,
                        "status": response.status_code if response else None,
                    }
                    if response and response.status_code < 400:
                        result_html = decode_body(response)
                        result_flags = page_flags(result_html)
                        urls = extract_result_urls(result_html, response.final_url, domain)
                        mechanism_record["challenge"] = result_flags["challenge"]
                        mechanism_record["result_title"] = result_flags["title"]
                        mechanism_record["result_url_count"] = len(urls)
                        query_echoed = (
                            probe_query.casefold() in str(result_flags["title"]).casefold()
                            or any(marker in str(result_flags["title"]).casefold() for marker in ("search", "tìm kiếm", "搜索"))
                        )
                        mechanism_record["query_page_evidence"] = query_echoed
                        if urls and query_echoed and not result_flags["challenge"]:
                            record["native_search"] = True
                            record["candidate_urls"].extend(urls)
                    else:
                        record["errors"].append(search_error or "search_http_error")
                    record["search_mechanisms"].append(mechanism_record)

                if not record["native_search"] and sitemaps and len(record["requests"]) < int(probe_config["max_requests_per_domain"]):
                    sitemap_url = sitemaps[0]
                    if policy is None or policy.can_fetch(probe_config["user_agent"], sitemap_url):
                        response, sitemap_error, sitemap_cached = fetcher.get(sitemap_url)
                        record["requests"].append({"url": sitemap_url, "cached": sitemap_cached,
                            "error": sitemap_error, "status": response.status_code if response else None})
                        if response and response.status_code < 400:
                            record["candidate_urls"].extend(extract_sitemap_locs(decode_body(response)))
                sample_limit = int(probe_config["max_result_urls_per_mechanism"])
                record["candidate_urls"] = list(dict.fromkeys(record["candidate_urls"]))[:sample_limit]
                all_candidates.extend(record["candidate_urls"])
                records.append(record)
        finally:
            fetcher.close()
        new_requests = fetcher.new_requests

    unique_candidates = list(dict.fromkeys(all_candidates))
    mappings, _ = map_result_urls(
        unique_candidates, corpus_path=config["inputs"]["corpus_path"],
        tracking_parameters=["utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid"],
    ) if unique_candidates else ([], {})
    by_url = {row["result_url"]: row for row in mappings}
    for record in records:
        mapped = [by_url[url] for url in record["candidate_urls"] if url in by_url]
        record["mapping"] = {
            "sampled_urls": len(mapped),
            "exact": sum(row["status"] == "exact_official_match" for row in mapped),
            "normalized": sum(row["status"] == "normalized_official_match" for row in mapped),
            "ambiguous": sum(row["status"] == "ambiguous_match" for row in mapped),
            "no_match": sum(row["status"] == "no_official_match" for row in mapped),
        }
        kinds = [classify_discovery_url(url) for url in record["candidate_urls"]]
        content_like = kinds.count("CONTENT_LIKE")
        record["page_type_probe"] = {
            "content_like": content_like,
            "category_or_index": kinds.count("CATEGORY_OR_INDEX"),
            "article_page_rate": round(content_like / len(kinds), 6) if kinds else None,
        }
    native_domains = {record["domain"] for record in records if record["native_search"] and record["mapping"]["exact"] + record["mapping"]["normalized"] > 0}
    native_share = sum(float(row_lookup[domain]["corpus_share"]) for domain in native_domains)
    g0 = classify_g0(native_share)
    existing_output = Path(config["outputs"]["artifact_dir"]) / "stage0_probe.json"
    initial_new_requests = new_requests
    if args.offline_cache_only and existing_output.exists():
        previous = json.loads(existing_output.read_text(encoding="utf-8"))
        initial_new_requests = int(previous.get("observed_initial_new_http_requests", previous.get("new_http_requests", 0)))
    result = {
        "stage": "0A_0B", "selected_domain_count": len(selected),
        "selected_corpus_share": inventory["selected_corpus_share"],
        "new_http_requests_this_run": new_requests,
        "observed_initial_new_http_requests": initial_new_requests,
        "unique_candidate_urls_probed": len(unique_candidates),
        "provisional_native_search_domains_before_repeat_probe": sorted(native_domains),
        "provisional_weighted_corpus_share_before_repeat_probe": round(native_share, 8),
        "provisional_g0_before_repeat_probe": g0,
        "repeat_probe_required": bool(native_domains),
        "stage1_allowed": False,
        "domains": records,
    }
    output = existing_output
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "domains"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
