from __future__ import annotations

from src.s4_targeting.tournament import (
    evaluate_ranking, map_s4_url, path_structure, rrf_rank, stable_fold, verdict,
)


def test_path_structure_rejects_unique_root_slugs() -> None:
    rows = [{"doc_id": 1, "url": "https://example.vn/benh-a"}, {"doc_id": 2, "url": "https://example.vn/benh-b"}]
    result = path_structure(rows)
    assert result["verdict"] == "NOT_VIABLE"
    assert result["multi_segment_url_count"] == 0


def test_stable_fold_is_deterministic() -> None:
    assert stable_fold(123, 5, "seed") == stable_fold(123, 5, "seed")
    assert 0 <= stable_fold(123, 5, "seed") < 5


def test_mapping_rejects_ambiguity_and_accepts_exact() -> None:
    maps = ({"https://x/a": [1], "https://x/b": [2, 3]}, {}, {})
    assert map_s4_url("https://x/a", maps) == (1, "exact")
    assert map_s4_url("https://x/b", maps) == (None, "ambiguous")


def test_retention_and_verdict_gates() -> None:
    proxies = {"primary_top10": [1, 2], "stronger_top5": [1], "stronger_top3": [1]}
    result = evaluate_ranking([1, 2, 3], proxies, [20000], 100000)
    assert result["primary_retention"]["20000"] == 1.0
    assert verdict(result) == "STRONG"
    empty = evaluate_ranking([], proxies, [20000], 100000)
    assert empty["primary_retention"]["20000"] == 0.0


def test_rrf_is_deterministic_and_rewards_agreement() -> None:
    rows = [{"doc_id": value, "url": f"https://x/{value}"} for value in (1, 2, 3)]
    first = rrf_rank([[1, 2, 3], [2, 1, 3]], 60, rows, "seed")
    second = rrf_rank([[1, 2, 3], [2, 1, 3]], 60, rows, "seed")
    assert first == second
    assert set(first) == {1, 2, 3}
