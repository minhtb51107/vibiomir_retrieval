from __future__ import annotations

import math

import pytest

from src.evaluation.system_registry import aggregate_candidate_rows, content_digest, summarize_ledger


def test_candidate_ledger_deduplicates_exact_chunk_across_arms():
    rows = aggregate_candidate_rows(
        7,
        [
            ("dense", [{"doc_id": 10, "chunk_id": "c1", "source_rank": 2}, {"doc_id": 10, "chunk_id": "c2", "source_rank": 4}]),
            ("sparse", [{"doc_id": 10, "chunk_id": "c1", "source_rank": 1}]),
        ],
        {"c1": 0.8, "c2": 0.4}, {10}, set(), {10: "example.org"},
    )
    assert len(rows) == 1
    assert rows[0]["candidate_arms"] == ["dense", "sparse"]
    assert rows[0]["candidate_chunk_count"] == 2
    assert rows[0]["duplicate_chunk_competition"] == 1
    assert rows[0]["best_reranker_score"] == 0.8
    assert rows[0]["mean_top2_score"] == pytest.approx(0.6)


def test_candidate_ledger_rejects_missing_or_nonfinite_score():
    candidate = [("dense", [{"doc_id": 10, "chunk_id": "c1", "source_rank": 1}])]
    with pytest.raises(ValueError, match="missing/nonfinite"):
        aggregate_candidate_rows(1, candidate, {}, set(), set(), {10: "example.org"})
    with pytest.raises(ValueError, match="missing/nonfinite"):
        aggregate_candidate_rows(1, candidate, {"c1": math.nan}, set(), set(), {10: "example.org"})


def test_summary_does_not_treat_structural_loss_as_relevance():
    rows = [{"candidate_arms": ["a", "b"], "duplicate_chunk_competition": 1,
             "reached_phase11c_docs20": True, "reached_phase12_docs20": False}]
    summary = summarize_ledger(rows)
    assert summary["query_document_pairs"] == 1
    assert summary["multi_arm_query_document_pairs"] == 1
    assert summary["candidate_documents_by_arm"] == {"a": 1, "b": 1}


def test_content_hash_is_whitespace_stable():
    assert content_digest("a  b\n c") == content_digest("a b c")
