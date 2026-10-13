from __future__ import annotations

import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from src.retrieval.pubmed_lexical import (
    add_documents, bridge_terms, completed_ordinal, initialize_index, lexical_query,
    reciprocal_rank_union, search,
)


def sample_rows():
    return [
        {"ordinal": 0, "doc_id": 10, "official_url": "https://pubmed.ncbi.nlm.nih.gov/10/", "pmid": 10,
         "title": "Insulin therapy for diabetes", "abstract": "Insulin treatment improves glycemic control."},
        {"ordinal": 1, "doc_id": 20, "official_url": "https://pubmed.ncbi.nlm.nih.gov/20/", "pmid": 20,
         "title": "Hypertension therapy", "abstract": "Blood pressure treatment and cardiovascular outcomes."},
    ]


def test_disk_index_is_deterministic_and_resume_safe(tmp_path):
    path = tmp_path / "index.sqlite"
    connection = initialize_index(path)
    assert add_documents(connection, sample_rows()) == 2
    assert completed_ordinal(connection) == 2
    assert add_documents(connection, sample_rows()) == 0
    first = search(connection, '"insulin" OR "diabetes"')
    second = search(connection, '"insulin" OR "diabetes"')
    assert first == second and first[0]["doc_id"] == 10
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    connection.close()


def test_query_bridge_excludes_generic_high_df_terms():
    seeds = [{"title": "Rare insulin biomarker", "abstract": "rare insulin biomarker biomarker"}]
    terms = bridge_terms(seeds, {"rare": 2, "insulin": 50, "biomarker": 3}, 1000, maximum_terms=2)
    assert [row["term"] for row in terms] == ["biomarker", "rare"]
    assert lexical_query(terms) == '"biomarker" OR "rare"'


def test_rank_union_deduplicates_documents_and_is_reproducible():
    dense = [{"doc_id": 1, "rank": 1, "dense_score": .9}, {"doc_id": 2, "rank": 2, "dense_score": .8}]
    lexical = [{"doc_id": 2, "rank": 1, "bm25_score": -4.0}, {"doc_id": 3, "rank": 2, "bm25_score": -3.0}]
    one = reciprocal_rank_union(dense, lexical)
    two = reciprocal_rank_union(dense, lexical)
    assert one == two
    assert len(one) == 3
    assert one[0]["doc_id"] == 2
    assert one[0]["candidate_arms"] == ["dense", "lexical"]


def test_index_preserves_official_identity(tmp_path):
    connection = initialize_index(tmp_path / "index.sqlite")
    add_documents(connection, sample_rows()[:1])
    assert connection.execute("SELECT doc_id,official_url,pmid FROM documents").fetchone() == (
        10, "https://pubmed.ncbi.nlm.nih.gov/10/", 10,
    )
    connection.close()


def test_module_is_cpu_only():
    import src.retrieval.pubmed_lexical as module
    source = open(module.__file__, encoding="utf-8").read()
    assert "torch" not in source and "transformers" not in source


def test_preflight_build_respects_stop_ordinal(tmp_path):
    from tools.system_upgrade_m2a import build_rows
    source = tmp_path / "documents.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"doc_id": value, "original_url": f"https://pubmed.ncbi.nlm.nih.gov/{value}/", "pmid": value,
         "title": f"title {value}", "abstract_text": "bounded abstract"}
        for value in range(1, 11)
    ]), source)
    connection = initialize_index(tmp_path / "bounded.sqlite")
    inserted, _ = build_rows(connection, source, 0, 10, 2, stop_ordinal=3)
    assert inserted == 3
    assert completed_ordinal(connection) == 3
    assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 3
    connection.close()
