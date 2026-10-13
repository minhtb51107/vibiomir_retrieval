from __future__ import annotations

import sqlite3

from src.retrieval.grounded_query_bridge import anchored_fts_query, grounded_anchors, normalize_vi, repaired_search
from src.retrieval.pubmed_lexical import add_documents, initialize_index


LEXICON = {
    "policy": "local_source_explicit_bilingual_pairs_only",
    "entries": [{"concept_id": "tonsillitis", "kind": "disease", "vietnamese": ["viêm amidan", "viêm a-mi-đan"],
                 "english": ["tonsillitis"], "source_doc_id": 1, "source_url": "https://example.test/1",
                 "source_excerpt": "Viêm amidan (Tonsillitis)"}],
    "literal_medical_anchors": ["hiv"],
}


def test_accent_and_hyphen_variants_map_to_provenanced_anchor():
    for query in ("Có nên cắt viêm amidan?", "Em bị viêm a-mi-đan"):
        rows = grounded_anchors(query, LEXICON)
        assert rows[0]["english"] == ["tonsillitis"]
        assert rows[0]["provenance"]["doc_id"] == 1
    assert normalize_vi("A-mi-đan") == "a mi dan"


def test_seed_specific_terms_cannot_enter_anchored_query():
    rows = grounded_anchors("Viêm amidan có nên cắt không?", LEXICON)
    expression = anchored_fts_query(rows)
    assert expression == '"tonsillitis"'
    assert "hymen" not in expression and "endometriosis" not in expression


def test_missing_anchor_falls_back_exactly_to_dense(tmp_path):
    connection = initialize_index(tmp_path / "index.sqlite")
    dense = [{"doc_id": 9, "rank": 1, "dense_score": .8}]
    result = repaired_search(connection, "Một câu hỏi chưa có thuật ngữ", LEXICON, dense)
    assert result["fallback"] is True
    assert result["candidates"] == dense
    connection.close()


def test_grounded_query_retrieves_only_using_verified_terms(tmp_path):
    connection = initialize_index(tmp_path / "index.sqlite")
    add_documents(connection, [
        {"ordinal": 0, "doc_id": 1, "official_url": "https://pubmed.ncbi.nlm.nih.gov/1/", "pmid": 1,
         "title": "Tonsillitis treatment", "abstract": "Management of inflamed tonsils."},
        {"ordinal": 1, "doc_id": 2, "official_url": "https://pubmed.ncbi.nlm.nih.gov/2/", "pmid": 2,
         "title": "Endometriosis surgery", "abstract": "Pelvic surgery."},
    ])
    result = repaired_search(connection, "Viêm a-mi-đan có nên cắt không?", LEXICON, [], 100)
    repeated = repaired_search(connection, "Viêm a-mi-đan có nên cắt không?", LEXICON, [], 100)
    assert result == repeated
    assert result["fallback"] is False
    assert [row["doc_id"] for row in result["lexical"]] == [1]
    assert result["required_anchors_preserved"] is True
    connection.close()


def test_query_literal_is_traceable_to_original_query():
    rows = grounded_anchors("Xét nghiệm HIV có chính xác không?", LEXICON)
    assert rows[0]["english"] == ["HIV"]
    assert rows[0]["provenance"]["type"] == "original_query_literal"


def test_longer_literal_suppresses_redundant_subset():
    lexicon = {**LEXICON, "literal_medical_anchors": ["covid", "covid-19"]}
    rows = grounded_anchors("Tiêm COVID-19", lexicon)
    assert [row["english"] for row in rows] == [["covid-19"]]
