import pytest

from tools.phase13a_top3_recovery import verify_review_prefix


def ranked(doc_id=7, score=1.2345674):
    return [{"doc_id": doc_id, "score": score}]


def test_review_prefix_requires_exact_document_and_rounded_score():
    expected = {3: [{"doc_id": 7, "score": 1.234567}]}
    assert verify_review_prefix(3, ranked(), expected) == 1


def test_review_prefix_hard_fails_first_document_mismatch():
    with pytest.raises(ValueError, match="review mismatch query=3 position=1"):
        verify_review_prefix(3, ranked(), {3: [{"doc_id": 8, "score": 1.234567}]})


def test_review_prefix_hard_fails_score_outside_serialization_rounding():
    with pytest.raises(ValueError, match="review score mismatch"):
        verify_review_prefix(3, ranked(score=1.234560), {3: [{"doc_id": 7, "score": 1.234567}]})


def test_non_review_query_requires_no_fixture_position():
    assert verify_review_prefix(99, ranked(), {}) == 0
