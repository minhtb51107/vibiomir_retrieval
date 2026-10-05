from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.site_discovery.reporting import enrich_source_map, source_map_markdown


def main() -> int:
    root = Path("artifacts/phase10b1_site_discovery")
    inventory = json.loads((root / "domain_inventory.json").read_text(encoding="utf-8"))
    probe = json.loads((root / "stage0_probe.json").read_text(encoding="utf-8"))
    repeat = json.loads((root / "stage0_repeat_probe.json").read_text(encoding="utf-8"))
    rows = enrich_source_map(inventory["domains"], probe["domains"], repeat["checks"])
    recommendations = Counter(row["recommended_acquisition"] for row in rows)
    mechanisms = {
        "native_search_forms_or_links_detected": sum(
            bool(row.get("search_mechanisms")) for row in probe["domains"]
        ),
        "usable_query_sensitive_native_search": len(repeat["usable_query_sensitive_domains"]),
        "robots_declared_sitemap": sum(bool(row.get("sitemap")) for row in probe["domains"]),
        "category_index_observed": sum(bool(row.get("category_index")) for row in probe["domains"]),
        "no_usable_mechanism_selected": sum(
            not row.get("native_search") and not row.get("sitemap") and not row.get("category_index")
            for row in probe["domains"]
        ),
    }
    summary = {
        "phase": "10B1",
        "network_stage_completed": "STAGE_0_ONLY",
        "observed_new_http_requests": int(probe["observed_initial_new_http_requests"]) + int(repeat["new_http_requests"]),
        "candidate_content_requests": 0,
        "inventory": {
            "domains": inventory["total_domains"],
            "corpus_rows": inventory["total_corpus_rows"],
            "selected_domains": len(inventory["selected_domains"]),
            "selected_weighted_corpus_share": inventory["selected_corpus_share"],
        },
        "mechanisms": mechanisms,
        "g0": {
            "usable_domains": repeat["usable_query_sensitive_domains"],
            "usable_weighted_corpus_share": repeat["usable_weighted_corpus_share"],
            "verdict": repeat["g0_verdict_after_repeat_probe"],
            "stage1_allowed": False,
            "stop_reason": "3.853% is below the predeclared 10% PARTIAL threshold and 5% specialist exception.",
        },
        "translation": "NOT_RUN_NO_CACHED_MT_AND_G0_FAILED",
        "stage1": {"ran": False, "queries": 0, "requests": 0},
        "stage2": {"ran": False, "queries": 0, "requests": 0},
        "g1": "NOT_REACHED",
        "g2": "NOT_REACHED",
        "recommendation_counts": dict(sorted(recommendations.items())),
        "final_verdict": "NOT_VIABLE_AS_MAJOR_DISCOVERY_CHANNEL",
        "recommended_next_action": "Use the source map for bounded source-specific acquisition planning; do not run query-scale site-native search. Sitemap/category sources may be assessed separately as acquisition feeds, not relevance discovery.",
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "official_source_discovery_map.json").write_text(json.dumps({"domains": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path("docs/OFFICIAL_SOURCE_DISCOVERY_MAP.md").write_text(source_map_markdown(rows), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
