from __future__ import annotations

import json
import socket
import sqlite3
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.url_reduction.index import build_query_conditioned_index
from src.url_reduction.pipeline import candidate_retention, retrieve_all_queries
from src.url_reduction.selection import (
    bm25_term_score,
    domain_balanced_order,
    global_deduplicate,
    is_weak_query,
    select_combined_candidates,
)
from src.url_reduction.url_text import extract_url_features, lexical_tokens


def test_url_parsing_percent_decoding_and_query_preservation() -> None:
    value = extract_url_features(
        "https://Example.COM/viem-da-day/%C4%91i%E1%BB%81u-tr%E1%BB%8B?q=thuoc_1#fragment"
    )
    assert value.domain == "example.com"
    assert "điều" in value.normalized_text
    assert "thuoc" in value.normalized_text
    assert "fragment" not in value.normalized_text


def test_hostname_tld_does_not_become_lexical_evidence() -> None:
    value = extract_url_features("https://example.com/opaque-123")
    assert "w:com" not in value.tokens
    assert value.domain == "example.com"
    assert value.path_token_count > 0


def test_unicode_nfc_and_accent_folded_tokens() -> None:
    decomposed = unicodedata.normalize("NFD", "điều trị")
    tokens = lexical_tokens(decomposed)
    assert "w:điều" in tokens
    assert "w:dieu" in tokens
    assert "w:trị" in tokens
    assert "w:tri" in tokens


def test_han_unigrams_and_bigrams() -> None:
    tokens = lexical_tokens("胃炎治疗")
    assert tokens[:4] == ["h1:胃", "h1:炎", "h1:治", "h1:疗"]
    assert {"h2:胃炎", "h2:炎治", "h2:治疗"} <= set(tokens)


def test_bm25_score_rewards_term_frequency() -> None:
    one = bm25_term_score(
        corpus_size=100,
        document_frequency=10,
        term_frequency=1,
        document_length=10,
        average_document_length=10,
        k1=1.2,
        b=0.75,
    )
    two = bm25_term_score(
        corpus_size=100,
        document_frequency=10,
        term_frequency=2,
        document_length=10,
        average_document_length=10,
        k1=1.2,
        b=0.75,
    )
    assert two > one > 0


def test_domain_balanced_fallback_is_deterministic() -> None:
    rows = [
        {"doc_id": index, "domain": f"d{index % 3}", "domain_corpus_count": 100 - index}
        for index in range(12)
    ]
    first = domain_balanced_order(rows, query_id=7)
    second = domain_balanced_order(rows, query_id=7)
    assert [row["doc_id"] for row in first] == [row["doc_id"] for row in second]
    assert len({row["domain"] for row in first[:3]}) == 3


def test_combined_selection_preserves_domains_and_deduplicates() -> None:
    lexical = [
        {"doc_id": index, "domain": "dominant.test", "score": 10 - index}
        for index in range(8)
    ]
    fallback = [
        {"doc_id": 100 + index, "domain": f"fallback-{index}.test", "score": 0.0}
        for index in range(10)
    ]
    selected = select_combined_candidates(
        lexical,
        fallback,
        depth=10,
        weak=True,
        maximum_domain_share=0.2,
        minimum_domains=5,
        weak_fallback_fraction=0.5,
    )
    assert len(selected) == 10
    assert len({row["doc_id"] for row in selected}) == 10
    assert len({row["domain"] for row in selected}) >= 5


def test_weak_query_detection_and_global_dedupe() -> None:
    assert is_weak_query([], minimum_best_score=2, minimum_candidates=2)
    assert is_weak_query(
        [{"score": 1.5}, {"score": 1.0}], minimum_best_score=2, minimum_candidates=2
    )
    assert not is_weak_query(
        [{"score": 3.0}, {"score": 1.0}], minimum_best_score=2, minimum_candidates=2
    )
    assert global_deduplicate([(1, 10), (2, 10), (2, 11)]) == {10, 11}


def _tiny_inputs(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus.parquet"
    queries = tmp_path / "queries.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [2, 1, 3, 4],
                "url": [
                    "https://a.test/alpha-treatment",
                    "https://b.test/alpha-treatment",
                    "http://c.test/%E8%83%83%E7%82%8E",
                    "https://d.test/123456",
                ],
            }
        ),
        corpus,
    )
    pq.write_table(
        pa.table({"id": [1], "query": ["zzgone zzmissing alpha"]}), queries
    )
    return corpus, queries


def test_index_and_ranking_ties_are_deterministic_and_offline(
    tmp_path: Path, monkeypatch
) -> None:
    corpus, queries = _tiny_inputs(tmp_path)

    class DeniedSocket(socket.socket):
        def connect(self, address: object) -> None:
            raise AssertionError(f"network attempted: {address}")

    monkeypatch.setattr(socket, "socket", DeniedSocket)
    index = tmp_path / "index.sqlite"
    summary = build_query_conditioned_index(
        corpus_path=corpus,
        query_path=queries,
        output_path=index,
        hash_path=tmp_path / "hashes.u64",
        stopwords=frozenset(),
        batch_rows=2,
        fallback_rows_per_domain=1,
        fallback_seed="fixed",
        sqlite_cache_mib=1,
        maximum_url_characters=1000,
    )
    assert summary["corpus_rows"] == 4
    candidates = tmp_path / "candidates.sqlite"
    retrieve_all_queries(
        index_path=index,
        output_path=candidates,
        depths=[2],
        lexical_pool=2,
        k1=1.2,
        b=0.75,
        weak_best_score=0,
        weak_minimum_candidates=1,
        maximum_domain_share=1.0,
        minimum_domains=1,
        weak_fallback_fraction=0.0,
        maximum_query_terms=1,
    )
    connection = sqlite3.connect(candidates)
    ranked = [
        row[0]
        for row in connection.execute(
            "SELECT doc_id FROM candidates WHERE method='lexical' ORDER BY rank"
        )
    ]
    connection.close()
    assert ranked == [1, 2]


def test_candidate_retention_calculation(tmp_path: Path) -> None:
    candidate_db = tmp_path / "candidates.sqlite"
    connection = sqlite3.connect(candidate_db)
    connection.execute(
        "CREATE TABLE candidates(method TEXT, depth INTEGER, query_id INTEGER, rank INTEGER, "
        "doc_id INTEGER, score REAL, source TEXT)"
    )
    connection.executemany(
        "INSERT INTO candidates VALUES ('combined', ?, 1, ?, ?, 1.0, 'lexical')",
        [(50, 1, 10), (100, 1, 10), (100, 2, 20)],
    )
    connection.commit()
    connection.close()
    sources = {}
    for name in ("dense", "sparse", "hybrid", "reranked"):
        path = tmp_path / f"{name}.parquet"
        rank_name = "rerank_rank" if name == "reranked" else "rank"
        pq.write_table(
            pa.table({"query_id": [1, 1], "doc_id": [10, 30], rank_name: [1, 2]}),
            path,
        )
        sources[name] = path
    retention = candidate_retention(
        candidate_path=candidate_db, sources=sources, depths=[50, 100]
    )
    assert retention["dense"]["by_url_depth"]["50"]["retention_fraction"] == 0.5
    assert retention["reranked"]["by_url_depth"]["100"]["retained"] == 1


def test_opaque_and_script_signals() -> None:
    assert extract_url_features("https://x.test/123456789").opaque
    assert extract_url_features("https://x.test/%E8%83%83%E7%82%8E").has_han
    assert extract_url_features("https://x.vn/benh-viem-da-day").vietnamese_like
