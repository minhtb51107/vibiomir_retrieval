from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.search_discovery.url_mapping import map_result_urls
from src.site_discovery.cache import CachedResponse, ProbeCache
from src.site_discovery.gates import classify_g0, classify_g1, discovery_efficiency, g2_metrics_complete
from src.site_discovery.inventory import (
    build_inventory, canonical_domain, likely_language, mark_selection,
    select_representative_domains, weighted_share,
)
from src.site_discovery.probe import (
    SearchMechanism, build_search_url, detect_search_mechanisms,
    classify_discovery_url, extract_result_urls, extract_sitemap_locs, sitemap_urls,
)
from src.site_discovery.reporting import enrich_source_map, source_map_markdown
from src.site_discovery.routing import dedupe_candidates, route_query_variants


def test_domain_inventory_and_language(tmp_path):
    corpus = tmp_path / "corpus.parquet"
    chunks = tmp_path / "chunks.parquet"
    pq.write_table(pa.table({"id": [1, 2, 3], "url": [
        "http://www.120ask.com/a", "https://example.vn/b", "https://unknown.org/c",
    ]}), corpus)
    pq.write_table(pa.table({"doc_id": [1], "source_url": ["https://120ask.com/a"]}), chunks)
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps([{"original_url": "https://120ask.com/a", "status": "SUCCESS",
                                  "template_classification": "Q_AND_A_TEMPLATE", "language_signal": "zh-Han-script"}]))
    bing = tmp_path / "bing.json"
    bing.write_text(json.dumps({"mapped": [{"domain": "www.example.vn"}]}))
    rows = build_inventory(corpus_path=corpus, chunks_path=chunks, probe_path=probe, bing_path=bing)
    assert len(rows) == 3
    lookup = {row["domain"]: row for row in rows}
    assert lookup["120ask.com"]["present_in_pilot"] is True
    assert lookup["example.vn"]["bing_b0_discovered"] is True
    assert likely_language("120ask.com")[0] == "zh"
    assert likely_language("example.vn")[0] == "vi"
    assert canonical_domain("WWW.Example.vn") == "example.vn"


def test_representative_selection_and_weighted_coverage():
    rows = [
        {"domain": "cnkang.com", "corpus_share": 0.4, "likely_language": "zh"},
        {"domain": "suckhoedoisong.vn", "corpus_share": 0.2, "likely_language": "vi"},
        {"domain": "unknown.org", "corpus_share": 0.1, "likely_language": "unknown"},
    ]
    for row in rows:
        row.update(selected_stage0=False, selection_reason=None)
    selected = select_representative_domains(rows, count=3)
    mark_selection(rows, selected)
    assert set(selected) == {"cnkang.com", "suckhoedoisong.vn", "unknown.org"}
    assert weighted_share(rows, set(selected)) == pytest.approx(0.7)


def test_adapter_search_generation_and_parsing():
    html = """<html><head><title>Search</title></head><body>
      <form action='/find' method='get'><input type='search' name='q'></form>
      <div class='search-result-title'><a href='/article/123456.html'>A useful article title here</a></div>
      <nav><a href='/category'>Category</a></nav></body></html>"""
    mechanisms = detect_search_mechanisms(html, "https://example.org/")
    assert mechanisms == [SearchMechanism("https://example.org/find", "q", "public GET search form on home page")]
    assert build_search_url(mechanisms[0], "tim mạch") == "https://example.org/find?q=tim+m%E1%BA%A1ch"
    assert extract_result_urls(html, "https://example.org/find", "example.org", 1) == [
        "https://example.org/article/123456.html"
    ]


def test_sitemap_and_result_deduplication():
    robots = "Sitemap: https://example.org/sitemap.xml\nSitemap: https://example.org/sitemap.xml"
    assert sitemap_urls(robots) == ["https://example.org/sitemap.xml"]
    assert extract_sitemap_locs("<loc>https://e/a</loc><loc>https://e/a</loc>") == ["https://e/a"]
    assert classify_discovery_url("https://e/article-123456.html") == "CONTENT_LIKE"
    assert classify_discovery_url("https://e/category/health") == "CATEGORY_OR_INDEX"
    rows = [{"domain": "e", "url": "u"}, {"domain": "e", "url": "u"}, {"domain": "e", "url": "v"}]
    assert [row["url"] for row in dedupe_candidates(rows)] == ["u", "v"]


def test_mapping_exact_normalized_and_ambiguity(tmp_path):
    corpus = tmp_path / "corpus.parquet"
    pq.write_table(pa.table({
        "id": [1, 2, 3, 4],
        "url": ["https://example.org/a", "https://example.org/b/", "https://example.org/duplicate", "https://example.org/duplicate"],
    }), corpus)
    rows, _ = map_result_urls(
        ["https://example.org/a", "https://example.org/b", "https://example.org/duplicate", "https://example.org/missing"],
        corpus_path=corpus, tracking_parameters=[],
    )
    assert [row["status"] for row in rows] == [
        "exact_official_match", "normalized_official_match", "ambiguous_match", "no_official_match"
    ]


def test_cache_resume_is_deterministic(tmp_path):
    path = tmp_path / "cache.sqlite"
    response = CachedResponse("https://e/", "https://e/", 200, "text/html", b"ok", False)
    with ProbeCache(path) as cache:
        cache.put(response)
        cache.put(response)
        assert cache.get("https://e/") == response
        assert cache.connection.execute("SELECT count(*) FROM responses").fetchone()[0] == 1
    with ProbeCache(path) as resumed:
        assert resumed.get("https://e/") == response


def test_multilingual_routing_without_translation():
    assert route_query_variants("vi", translation_available=False) == ("Q0", "Q1")
    assert route_query_variants("zh", translation_available=False) == ("Q1_CONTROL_ONLY",)
    assert route_query_variants("zh", translation_available=True) == ("Q3", "Q1")


def test_predeclared_gate_logic_and_efficiency():
    assert classify_g0(0.31) == "STRONG"
    assert classify_g0(0.15) == "PARTIAL"
    assert classify_g0(0.04, specialist_share=0.06) == "PARTIAL_SPECIALIST"
    assert classify_g0(0.04) == "WEAK"
    assert classify_g1(query_coverage=0.31, subgroup_coverage=0.0,
                       candidates_per_query_multiplier_vs_bing=1.0,
                       reliable_mapping=True, operationally_cheap=True,
                       concentrated_duplicates=False) == "PROMISING"
    assert classify_g1(query_coverage=0.1, subgroup_coverage=0.1,
                       candidates_per_query_multiplier_vs_bing=1.0,
                       reliable_mapping=True, operationally_cheap=True,
                       concentrated_duplicates=False) == "NOT_VIABLE"
    assert discovery_efficiency(10, 5) == 2.0
    assert discovery_efficiency(10, 0) == 0.0


def test_g2_requires_all_value_metrics():
    metrics = {key: 1 for key in (
        "query_coverage", "unique_mapped_docs", "mapped_docs_per_query",
        "contributing_domains", "corpus_weighted_domain_coverage",
        "novelty_vs_bing", "novelty_vs_pilot", "duplicate_concentration",
        "projected_1200_candidates", "discovery_efficiency", "new_doc_efficiency",
    )}
    assert g2_metrics_complete(metrics)
    metrics.pop("new_doc_efficiency")
    assert not g2_metrics_complete(metrics)


def test_source_map_generation():
    inventory = [{
        "domain": "a-hospital.com", "corpus_rows": 10, "corpus_share": 0.5,
        "likely_language": "zh", "phase2_accessibility": None,
        "present_in_pilot": True, "pilot_document_count": 2,
        "native_search": "UNKNOWN", "sitemap": "UNKNOWN", "category_index": "UNKNOWN",
        "mapping_works": "UNTESTED", "search_quality_tested": False,
    }]
    probes = [{"domain": "a-hospital.com", "search_mechanisms": [{}], "sitemap": True,
               "category_index": False, "mapping": {"exact": 1, "normalized": 0}, "requests": [1]}]
    repeats = [{"domain": "a-hospital.com", "query_sensitive_and_mappable": True}]
    rows = enrich_source_map(inventory, probes, repeats)
    assert rows[0]["recommended_acquisition"] == "SEARCH"
    assert "a-hospital.com" in source_map_markdown(rows)
