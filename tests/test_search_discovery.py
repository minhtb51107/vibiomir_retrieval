from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.search_discovery.analysis import _query_diagnostics, summarize_counts, verdict
from src.search_discovery.cache import SearchCache, request_fingerprint
from src.search_discovery.pipeline import run_stage
from src.search_discovery.provider import parse_bing_rss
from src.search_discovery.query_variants import build_variants, concise_vietnamese
from src.search_discovery.sampling import build_stratified_sample
from src.search_discovery.url_mapping import canonical_key, map_result_urls, normalize_url


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_url_normalization_fragment_tracking_percent_and_trailing_slash() -> None:
    tracking = frozenset({"utm_source"})
    first = normalize_url(
        "HTTP://Example.COM:80/a/%7Eb/?x=1&utm_source=z#frag",
        strip_tracking=tracking,
    )
    second = normalize_url("http://example.com/a/~b/?x=1")
    assert first == "http://example.com/a/~b?x=1"
    assert first == second
    assert canonical_key("https://example.com/a") == "//example.com/a"


def test_mapping_exact_normalized_ambiguous_and_no_match(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.parquet"
    _write(
        corpus,
        [
            {"id": 1, "url": "http://example.com/a?utm_source=x"},
            {"id": 2, "url": "https://example.com/a"},
            {"id": 3, "url": "https://other.test/b/"},
        ],
    )
    urls = [
        "http://example.com/a?utm_source=x",
        "HTTP://EXAMPLE.COM:80/a",
        "//example.com/a",
        "https://missing.test/x",
    ]
    rows, domains = map_result_urls(
        urls, corpus_path=corpus, tracking_parameters=["utm_source"]
    )
    assert [row["status"] for row in rows] == [
        "exact_official_match",
        "normalized_official_match",
        "ambiguous_match",
        "no_official_match",
    ]
    assert rows[0]["doc_id"] == rows[1]["doc_id"] == 1
    assert rows[2]["doc_id"] is None
    assert domains == {"example.com": 2, "other.test": 1}


def test_query_variants_are_deterministic_and_remove_scaffolding() -> None:
    text = "Xin hỏi bác sĩ, tôi muốn hỏi thuốc nào điều trị viêm gan?"
    assert build_variants(text) == build_variants(text)
    concise = concise_vietnamese(text)
    assert "xin" not in concise and "bác sĩ" not in concise
    assert "thuốc" in concise and "viêm gan" in concise


def test_search_cache_resume_and_result_deduplication(tmp_path: Path) -> None:
    cache = SearchCache(tmp_path / "cache.sqlite")
    fingerprint = request_fingerprint("mock", "đau đầu", 10)
    cache.write_response(
        fingerprint=fingerprint,
        provider="mock",
        query_text="đau đầu",
        depth=10,
        status="SUCCESS",
        http_status=200,
        attempt_count=1,
        error_type=None,
        error_message=None,
        completed_at="2026-10-05T00:00:00Z",
        results=[{"rank": 1, "title": "A", "url": "https://x.test/a", "snippet": "s"}],
    )
    cache.assign(1, "Q0", fingerprint)
    assert cache.response_status(fingerprint) == "SUCCESS"
    assert cache.unmapped_urls() == ["https://x.test/a"]
    cache.write_mappings(
        [{"result_url": "https://x.test/a", "status": "no_official_match", "match_level": None, "doc_id": None, "official_url": None, "official_domain": None, "candidate_count": 0}]
    )
    assert cache.unmapped_urls() == []
    cache.close()
    reopened = SearchCache(tmp_path / "cache.sqlite")
    assert reopened.response_status(fingerprint) == "SUCCESS"
    reopened.close()


def test_provider_result_parser_deduplicates_and_respects_depth() -> None:
    payload = b"""<?xml version='1.0'?><rss><channel>
      <item><title>A</title><link>https://x.test/a</link><description>one</description></item>
      <item><title>A again</title><link>https://x.test/a</link><description>duplicate</description></item>
      <item><title>B</title><link>https://x.test/b</link><description>two</description></item>
      <item><title>C</title><link>https://x.test/c</link><description>three</description></item>
    </channel></rss>"""
    rows = parse_bing_rss(payload, 2)
    assert [row["url"] for row in rows] == ["https://x.test/a", "https://x.test/b"]
    assert [row["rank"] for row in rows] == [1, 2]


def test_failure_classification_distinguishes_provider_and_mapping_failures() -> None:
    sample = {
        1: {"query_characters": 10, "query_tokens": 2, "length_quantile": 1, "category": "general", "evidence_proxy": "weak"},
        2: {"query_characters": 20, "query_tokens": 4, "length_quantile": 2, "category": "drug", "evidence_proxy": "mixed"},
        3: {"query_characters": 30, "query_tokens": 6, "length_quantile": 3, "category": "symptom", "evidence_proxy": "strong"},
    }
    base = {"variant": "Q0", "fingerprint": "x", "query_text": "q", "http_status": 200, "error_type": None, "error_message": None, "rank": None, "url": None, "mapping_status": "unmapped", "match_level": None, "doc_id": None, "official_url": None, "official_domain": None}
    rows = [
        {**base, "query_id": 1, "request_status": "SUCCESS"},
        {**base, "query_id": 2, "request_status": "SUCCESS", "rank": 1, "url": "https://outside.test"},
        {**base, "query_id": 3, "request_status": "RATE_LIMITED", "http_status": 429},
    ]
    diagnostics = _query_diagnostics(rows, sample)
    assert [row["failure_reason"] for row in diagnostics] == [
        "A_PROVIDER_RETURNED_NOTHING",
        "B_RESULTS_OUTSIDE_OFFICIAL_CORPUS",
        "G_PROVIDER_RATE_LIMITED",
    ]


def _fixture(tmp_path: Path) -> dict:
    queries = [
        {"id": index, "query": f"Xin hỏi bệnh đau đầu và thuốc điều trị số {index}?"}
        for index in range(1, 13)
    ]
    corpus = [
        {"id": 100 + index, "url": f"https://official.test/article-{index}"}
        for index in range(1, 13)
    ]
    candidates = []
    sparse = []
    for query in queries:
        qid = query["id"]
        candidates.extend(
            [
                {"query_id": qid, "rank": 1, "doc_id": 100 + qid},
                {"query_id": qid, "rank": 2, "doc_id": 101 + (qid % 11)},
            ]
        )
        sparse.extend(
            [
                {"query_id": qid, "rank": 1, "doc_id": 100 + qid},
                {"query_id": qid, "rank": 2, "doc_id": 999},
            ]
        )
    _write(tmp_path / "queries.parquet", queries)
    _write(tmp_path / "corpus.parquet", corpus)
    for name, rows in (("dense", candidates), ("sparse", sparse), ("hybrid", candidates), ("reranked", candidates)):
        _write(tmp_path / f"{name}.parquet", rows)
    return {
        "version": "test",
        "inputs": {
            "queries": str(tmp_path / "queries.parquet"),
            "corpus": str(tmp_path / "corpus.parquet"),
            "dense_candidates": str(tmp_path / "dense.parquet"),
            "sparse_candidates": str(tmp_path / "sparse.parquet"),
            "hybrid_candidates": str(tmp_path / "hybrid.parquet"),
            "reranked_candidates": str(tmp_path / "reranked.parquet"),
        },
        "outputs": {"cache": str(tmp_path / "cache.sqlite"), "artifacts": str(tmp_path / "artifacts")},
        "sampling": {"seed": 7, "stage_a_queries": 4, "stage_b_queries": 8, "length_quantiles": 2, "evidence_top_k": 2},
        "provider": {"kind": "bing_rss", "result_depth": 10},
        "query_variants": {"enabled": ["Q0", "Q1"], "english": "NOT_TESTED", "chinese": "NOT_TESTED"},
        "url_mapping": {"strip_tracking_parameters": []},
        "stage_a_gate": {"minimum_completed_request_fraction": 0.9, "hard_failure_maximum_query_map_rate": 0.02, "hard_failure_maximum_total_mapped_docs": 0},
        "stage_b_gates": {
            "promising": {"minimum_query_map_rate": 0.5, "maximum_global_candidates": 100, "minimum_official_domains": 1, "maximum_top_domain_share": 1.0, "subgroup_minimum_size": 2, "subgroup_minimum_query_map_rate": 0.75, "q1_minimum_incremental_share": 0.2},
            "borderline": {"minimum_query_map_rate": 0.2, "maximum_global_candidates": 100, "minimum_official_domains": 1, "maximum_top_domain_share": 1.0},
        },
        "projection": {"full_query_count": 12, "phase8_crawl_urls_per_minute": 122, "expected_retained_bytes_per_usable_document": 100, "low_retained_bytes_per_document": 50, "high_retained_bytes_per_document": 150},
    }


def test_sampling_is_deterministic_and_stratified(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    first = build_stratified_sample(config)
    second = build_stratified_sample(config)
    assert first == second
    assert len(first["stage_a_query_ids"]) == 4
    assert len(first["stage_b_query_ids"]) == 8
    assert {row["evidence_proxy"] for row in first["queries"]} == {"strong"}


class MockProvider:
    name = "mock"
    calls = 0

    def search(self, query: str, depth: int):
        self.calls += 1
        number = int(query.rstrip("?").split()[-1])
        return (
            [
                {"rank": 1, "title": "official", "url": f"https://official.test/article-{number}", "snippet": "s"},
                {"rank": 2, "title": "outside", "url": f"https://outside.test/{number}", "snippet": "s"},
            ],
            {"status": "SUCCESS", "http_status": 200, "attempt_count": 1, "error_type": None, "error_message": None, "completed_at": "2026-10-05T00:00:00Z"},
        )


def test_pipeline_checkpoint_resume_mapping_metrics_and_no_network(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    provider = MockProvider()
    first = run_stage(config, stage="A", provider=provider)
    calls = provider.calls
    assert calls > 0
    assert first["stage_a_passed"] is True
    assert first["combined_metrics"]["query_map_rate"] == 1.0
    assert first["combined_metrics"]["official_domains"] == 1
    second = run_stage(config, stage="A", provider=provider)
    assert provider.calls == calls
    assert second["execution"]["new_external_requests"] == 0
    assert second["combined_metrics"] == first["combined_metrics"]


def test_count_summary_and_gate_logic() -> None:
    assert summarize_counts([0, 1, 4, 10])["p90"] == 8.2
    config = {
        "stage_b_gates": {
            "promising": {"minimum_query_map_rate": 0.5, "maximum_global_candidates": 100, "minimum_official_domains": 2, "maximum_top_domain_share": 0.7, "subgroup_minimum_size": 2, "subgroup_minimum_query_map_rate": 0.8, "q1_minimum_incremental_share": 0.2},
            "borderline": {"minimum_query_map_rate": 0.2, "maximum_global_candidates": 200, "minimum_official_domains": 1, "maximum_top_domain_share": 0.9},
        }
    }
    metrics = {"query_map_rate": 0.6, "unique_mapped_docs": 20, "official_domains": 3, "top_domain_share": 0.5}
    diagnostics = [
        {"category": "drug", "query_id": 1, "mapped_docs": 1},
        {"category": "drug", "query_id": 2, "mapped_docs": 0},
    ]
    assert verdict(metrics, diagnostics, config)["verdict"] == "PROMISING"
