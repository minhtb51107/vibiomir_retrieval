from __future__ import annotations

from typing import Any


def _recommend(row: dict[str, Any], probe: dict[str, Any] | None, usable: set[str]) -> str:
    domain = str(row["domain"])
    evidence = row.get("phase2_accessibility") or {}
    statuses = evidence.get("statuses", {}) if isinstance(evidence, dict) else {}
    templates = evidence.get("template_classes", {}) if isinstance(evidence, dict) else {}
    if domain in usable:
        return "SEARCH"
    if statuses and set(statuses).issubset({"ROBOTS_BLOCKED", "HTTP_ERROR"}):
        return "BLOCKED"
    if templates and set(templates).issubset({"ACCESS_RESTRICTED", "JAVASCRIPT_HEAVY"}):
        return "BLOCKED"
    if probe and probe.get("sitemap"):
        return "SITEMAP"
    if probe and probe.get("category_index"):
        return "CATEGORY"
    if statuses.get("SUCCESS", 0) or row.get("present_in_pilot"):
        return "DIRECT_CRAWL"
    return "UNKNOWN"


def enrich_source_map(
    inventory_rows: list[dict[str, Any]], probe_rows: list[dict[str, Any]],
    repeat_checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    probes = {row["domain"]: row for row in probe_rows}
    repeats = {row["domain"]: row for row in repeat_checks}
    usable = {domain for domain, row in repeats.items() if row.get("query_sensitive_and_mappable")}
    output = []
    for source in inventory_rows:
        row = dict(source)
        probe = probes.get(row["domain"])
        repeat = repeats.get(row["domain"])
        if probe:
            detected = bool(probe.get("search_mechanisms"))
            if row["domain"] in usable:
                row["native_search"] = "USABLE_QUERY_SENSITIVE"
            elif repeat and repeat.get("result_sets_identical"):
                row["native_search"] = "QUERY_INSENSITIVE"
            elif detected:
                row["native_search"] = "PRESENT_NOT_MAPPABLE_OR_NOT_FUNCTIONAL"
            else:
                row["native_search"] = "NOT_FOUND"
            row["sitemap"] = "DECLARED" if probe.get("sitemap") else "NOT_FOUND"
            row["category_index"] = "OBSERVED" if probe.get("category_index") else "NOT_FOUND"
            mapping = probe.get("mapping", {})
            mapped = int(mapping.get("exact", 0)) + int(mapping.get("normalized", 0))
            row["mapping_works"] = "YES" if mapped else "NO_SAMPLED_MATCH"
            row["search_quality_tested"] = bool(repeat)
            row["stage0_request_count"] = len(probe.get("requests", []))
        row["recommended_acquisition"] = _recommend(row, probe, usable)
        output.append(row)
    return output


def source_map_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Official Source Discovery Map", "",
        "This map covers all 97 official corpus domains. Language labels are source/domain associations unless Phase 2 content measurements are cited; they are not full-corpus language measurements. `UNKNOWN` means the source was not safely characterized, not that it lacks a mechanism.", "",
        "| Domain | Rows | Share | Lang | Phase 2 / pilot evidence | Native search | Sitemap | Category | Mapping | Recommended |",
        "|---|---:|---:|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        evidence = row.get("phase2_accessibility")
        if evidence:
            status = ", ".join(f"{key}:{value}" for key, value in evidence.get("statuses", {}).items())
        elif row.get("present_in_pilot"):
            status = f"pilot docs:{row.get('pilot_document_count', 0)}"
        else:
            status = "none"
        lines.append(
            f"| {row['domain']} | {int(row['corpus_rows']):,} | {100 * float(row['corpus_share']):.3f}% | "
            f"{row['likely_language']} | {status} | {row['native_search']} | {row['sitemap']} | "
            f"{row['category_index']} | {row['mapping_works']} | {row['recommended_acquisition']} |"
        )
    lines.extend([
        "", "## Interpretation", "",
        "Only `a-hospital.com` passed both query-sensitivity and official-URL mapping in the bounded Stage 0 probe. Sitemap and category mechanisms remain useful acquisition metadata, but they are not relevance search and were not counted toward G0 searchable coverage. Recommendations are conservative: unprobed sources remain `UNKNOWN`; no private endpoint or anti-bot workaround was inferred.", "",
    ])
    return "\n".join(lines)
