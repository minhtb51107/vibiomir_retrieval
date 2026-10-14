from src.source_census.phase14 import deterministic_new_records


def test_phase14_selection_is_deterministic_and_excludes_prior_ids():
    records = [{"doc_id": value, "url": f"https://medlatec.vn/{value}"} for value in range(20)]
    excluded = {1, 3, 5, 7}
    first = deterministic_new_records(records, excluded, 8, "fixed-seed")
    second = deterministic_new_records(list(reversed(records)), excluded, 8, "fixed-seed")
    assert first == second
    assert len(first) == 8
    assert not ({row["doc_id"] for row in first} & excluded)


def test_phase14_selection_fails_if_budget_exceeds_unacquired_population():
    records = [{"doc_id": value, "url": f"https://medlatec.vn/{value}"} for value in range(3)]
    try:
        deterministic_new_records(records, {0, 1}, 2, "fixed-seed")
    except ValueError as error:
        assert "only 1 unacquired" in str(error)
    else:
        raise AssertionError("insufficient deterministic population must fail")
