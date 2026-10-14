import tools.phase15_ranking_optimization as phase15
from tools.phase15_ranking_optimization import diversity_order, rank_top2_mean


def _row(chunk: str, doc: int, score: float, rank: int, text: str | None = None) -> dict:
    return {"query_id": 1, "chunk_id": chunk, "doc_id": doc, "score": score,
            "rerank_rank": rank, "chunk_text": text or chunk}


def test_top2_mean_uses_unique_best_two_chunks_and_deterministic_ties():
    rows = [_row("a", 1, 0.9, 1), _row("b", 1, 0.3, 3), _row("a", 1, 0.9, 1),
            _row("c", 2, 0.7, 2), _row("d", 3, 0.7, 4)]
    ranked = rank_top2_mean(rows)
    assert [row["doc_id"] for row in ranked] == [2, 3, 1]
    assert ranked[2]["score"] == 0.6
    assert ranked[2]["representative_chunk_count"] == 2


def test_top2_mean_single_chunk_preserves_its_score():
    ranked = rank_top2_mean([_row("x", 7, -1.25, 1)])
    assert ranked[0]["score"] == -1.25
    assert ranked[0]["representative_chunk_count"] == 1


def test_chunk_diversity_caps_first_pass_then_fills_without_padding():
    rows = [_row(f"a{i}", 1, 10-i, i) for i in range(1, 6)] + [_row("b", 2, 4, 6), _row("c", 3, 3, 7)]
    ordered, audit = diversity_order(rows, depth=6, cap=2)
    assert [row["chunk_id"] for row in ordered[:4]] == ["a1", "a2", "b", "c"]
    assert len({row["chunk_id"] for row in ordered}) == 7
    assert audit["first_pass_chunks"] == 4
    assert audit["fill_chunks_needed"] == 2


def test_chunk_diversity_deduplicates_exact_chunk_identity():
    rows = [_row("a", 1, 1, 1), _row("a", 1, 1, 2), _row("b", 2, 0, 3)]
    ordered, audit = diversity_order(rows, depth=2)
    assert [row["chunk_id"] for row in ordered] == ["a", "b"]
    assert audit["duplicate_chunk_ids_suppressed"] == 1


def test_chunk_diversity_fails_when_full_pool_has_insufficient_capacity():
    try:
        diversity_order([_row("a", 1, 1, 1)], depth=2)
    except ValueError as error:
        assert "only 1 distinct valid chunks" in str(error)
    else:
        raise AssertionError("expected a hard capacity failure")


def test_submission_tokenizer_passes_the_phase_config(monkeypatch):
    observed = []
    sentinel = object()
    monkeypatch.setattr(phase15, "_historical_tokenizer", lambda config: observed.append(config) or sentinel)
    config = {"models": {"embedder": {"name": "BAAI/bge-m3", "revision": "pinned"}}}
    assert phase15.submission_tokenizer(config) is sentinel
    assert observed == [config]
