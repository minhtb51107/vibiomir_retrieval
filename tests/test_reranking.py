from __future__ import annotations

from pathlib import Path

import numpy as np

from src.reranking.candidate_pool import build_unified_candidate_pool
from src.reranking.checkpoint import RerankCheckpoint
from src.reranking.pipeline import _rerank_depth, build_multilingual_sanity
from src.reranking.selection import (
    aggregate_reranked_documents,
    select_candidates,
)


def _row(
    source: str,
    *,
    chunk_id: str,
    doc_id: int,
    rank: int,
    score: float,
    text: str,
    section_type: str = "content",
) -> dict[str, object]:
    row: dict[str, object] = {
        "query_id": 7,
        "query_text": "đau tim",
        "rank": rank,
        "score": score,
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "chunk_text": text,
        "source_url": f"https://example.test/{doc_id}",
        "chunk_index": rank - 1,
        "start_offset": 0,
        "end_offset": len(text),
        "section_type": section_type,
        "heading_path": [],
        "extraction_method": "semantic_container",
    }
    if source == "hybrid":
        row["fusion_method"] = "rrf"
    return row


def test_unified_pool_deduplicates_and_preserves_rank_metadata_and_text() -> None:
    exact_text = "Nội dung nguồn không được viết lại."
    dense = [_row("dense", chunk_id="a", doc_id=1, rank=1, score=0.8, text=exact_text)]
    sparse = [
        _row("sparse", chunk_id="a", doc_id=1, rank=2, score=9.0, text=exact_text),
        _row("sparse", chunk_id="b", doc_id=2, rank=1, score=10.0, text="Khác"),
    ]
    hybrid = [_row("hybrid", chunk_id="a", doc_id=1, rank=1, score=0.03, text=exact_text)]
    first, summary = build_unified_candidate_pool(
        dense_rows=dense,
        sparse_rows=sparse,
        hybrid_rows=hybrid,
        max_depth=10,
    )
    second, _ = build_unified_candidate_pool(
        dense_rows=dense,
        sparse_rows=sparse,
        hybrid_rows=hybrid,
        max_depth=10,
    )
    assert first == second
    assert len(first) == 2
    assert first[0]["chunk_id"] == "a"
    assert first[0]["dense_rank"] == 1
    assert first[0]["sparse_rank"] == 2
    assert first[0]["hybrid_rank"] == 1
    assert first[0]["source_membership"] == ["dense", "sparse", "hybrid"]
    assert first[0]["chunk_text"] == exact_text
    assert summary["candidate_rows"] == 2


def test_rerank_ordering_and_deterministic_tie_break() -> None:
    rows = [
        {**_row("dense", chunk_id="b", doc_id=2, rank=1, score=0.1, text="B"), "pool_rank": 1, "rerank_score": 2.0},
        {**_row("dense", chunk_id="a", doc_id=1, rank=2, score=0.2, text="A"), "pool_rank": 2, "rerank_score": 2.0},
        {**_row("dense", chunk_id="c", doc_id=3, rank=3, score=0.3, text="C"), "pool_rank": 3, "rerank_score": 1.0},
    ]
    reranked, _ = _rerank_depth({7: rows}, 3)
    assert [row["chunk_id"] for row in reranked[7]] == ["a", "b", "c"]
    assert [row["rerank_rank"] for row in reranked[7]] == [1, 2, 3]
    assert all(row["score"] == row["rerank_score"] for row in reranked[7])


def test_selection_controls_and_no_text_mutation() -> None:
    original_text = "Tiêu đề ngắn"
    rows = [
        {**_row("hybrid", chunk_id="a", doc_id=1, rank=1, score=0.1, text=original_text, section_type="title"), "rerank_score": 4.0},
        {**_row("hybrid", chunk_id="b", doc_id=1, rank=2, score=0.1, text="Nội dung B"), "rerank_score": 3.0},
        {**_row("hybrid", chunk_id="c", doc_id=1, rank=3, score=0.1, text="Nội dung C"), "rerank_score": 2.0},
        {**_row("hybrid", chunk_id="d", doc_id=2, rank=4, score=0.1, text=original_text), "rerank_score": 1.0},
    ]
    selected, stats = select_candidates(
        rows,
        top_k=3,
        exact_text_suppression=True,
        max_chunks_per_doc=2,
        boilerplate_penalty_enabled=True,
        boilerplate_max_characters=80,
        boilerplate_section_types={"title"},
        boilerplate_penalty=1.0,
    )
    assert stats["exact_text_suppressed"] == 1
    assert stats["max_chunks_per_doc_suppressed"] == 1
    assert stats["boilerplate_penalized"] == 1
    assert len([row for row in selected if row["doc_id"] == 1]) <= 2
    assert rows[0]["chunk_text"] == original_text
    assert all(row["chunk_text"] in {r["chunk_text"] for r in rows} for row in selected)


def test_document_aggregation_uses_rerank_scores_and_deduplicates_docs() -> None:
    rows = [
        {**_row("hybrid", chunk_id="a", doc_id=1, rank=1, score=0.1, text="A"), "rerank_score": 5.0},
        {**_row("hybrid", chunk_id="b", doc_id=1, rank=2, score=0.1, text="B"), "rerank_score": 3.0},
        {**_row("hybrid", chunk_id="c", doc_id=2, rank=3, score=0.1, text="C"), "rerank_score": 4.5},
    ]
    best = aggregate_reranked_documents(rows, method="best_chunk", top_k_docs=2)
    mean = aggregate_reranked_documents(
        rows, method="top_n_mean", top_k_docs=2, top_n=2
    )
    assert [row["doc_id"] for row in best] == [1, 2]
    assert [row["doc_id"] for row in mean] == [2, 1]
    assert len({row["doc_id"] for row in best}) == len(best)
    assert best[0]["rerank_score"] == 5.0


def test_checkpoint_resume_signature_and_score_reproducibility(tmp_path: Path) -> None:
    path = tmp_path / "scores.sqlite"
    signature = {"model": "mock", "revision": "1", "source": "abc"}
    with RerankCheckpoint(path, signature=signature) as checkpoint:
        checkpoint.write_scores([(1, "a", 0.5, 2.0), (1, "b", -1.0, 3.0)])
        assert checkpoint.count == 2
        assert checkpoint.integrity_check() == "ok"
    with RerankCheckpoint(path, signature=signature) as checkpoint:
        assert checkpoint.completed_keys() == {(1, "a"), (1, "b")}
        assert checkpoint.read_scores() == {
            (1, "a"): (0.5, 2.0),
            (1, "b"): (-1.0, 3.0),
        }


class MockReranker:
    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        return np.asarray([len(set(query.split()) & set(text.split())) for query, text in pairs], dtype=np.float32)

    def metadata(self) -> dict[str, object]:
        return {
            "model_name": "mock",
            "resolved_revision": "test",
            "max_sequence_length": 32,
            "inference_precision": "float32",
            "batch_size": 2,
        }


def test_mock_reranker_interface_is_deterministic() -> None:
    reranker = MockReranker()
    pairs = [("đau tim", "đau tim nhiều"), ("đau tim", "bệnh gan")]
    first = reranker.score_pairs(pairs)
    second = reranker.score_pairs(pairs)
    np.testing.assert_array_equal(first, second)
    assert first.tolist() == [2.0, 0.0]


def _tiny_config(tmp_path: Path) -> dict:
    return {
        "candidate_pool": {"rerank_depths": [2, 3], "default_depth": 3},
        "selection": {
            "final_top_k_chunks": 3,
            "final_top_k_docs": 2,
            "max_chunks_per_doc": 2,
            "boilerplate_penalty": {
                "max_characters": 80,
                "section_types": ["title"],
                "penalty": 1.0,
            },
        },
        "documents": {"top_n_mean": 2},
        "outputs": {
            "directory": str(tmp_path / "out"),
            "checkpoint": str(tmp_path / "out" / "scores.sqlite"),
        },
    }


def _tiny_pool(tmp_path: Path) -> tuple[Path, dict]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for index, (chunk_id, doc_id) in enumerate([("a", 1), ("b", 1), ("c", 2)], start=1):
        row = _row("dense", chunk_id=chunk_id, doc_id=doc_id, rank=index, score=0.1, text=f"đau tim {chunk_id}")
        row.update(
            pool_rank=index,
            source_membership_key="dense",
            dense_rank=index,
            sparse_rank=None,
            sparse_score=None,
            hybrid_rank=None,
            hybrid_score=None,
            dense_score=0.1,
        )
        rows.append(row)
    path = tmp_path / "pool.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    summary = {
        "input_hashes": {"pool": "x"},
        "preparation_latency_ms_per_query": {"mean": 1.0},
    }
    return path, summary


def test_score_resume_then_cpu_only_finalize(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    from src.reranking.pipeline import finalize_reranking, score_pool

    config = _tiny_config(tmp_path)
    path, summary = _tiny_pool(tmp_path)
    reranker = MockReranker()
    metadata = reranker.metadata()
    # Simulate an interrupted run: one score already persisted.
    from src.reranking.pipeline import _checkpoint_signature

    with RerankCheckpoint(
        config["outputs"]["checkpoint"],
        signature=_checkpoint_signature(summary, metadata),
    ) as checkpoint:
        checkpoint.write_scores([(7, "a", 99.0, 5.0)])
    run = score_pool(config=config, reranker=reranker, pool_path=path, pool_summary=summary)
    assert run["initial_rows"] == 1
    assert run["rows_scored_this_run"] == 2
    assert run["complete"] is True
    # Pre-existing score must be preserved, not recomputed.
    with RerankCheckpoint(
        config["outputs"]["checkpoint"],
        signature=_checkpoint_signature(summary, metadata),
    ) as checkpoint:
        assert checkpoint.read_scores()[(7, "a")] == (99.0, 5.0)
    result = finalize_reranking(
        config=config,
        pool_path=path,
        pool_summary=summary,
        model_metadata=metadata,
        scoring_runs=[run],
    )
    assert result["checkpoint"]["final_rows"] == 3
    depth3 = pq.read_table(tmp_path / "out" / "reranked_chunks_depth3.parquet").to_pylist()
    assert depth3[0]["chunk_id"] == "a"  # preserved 99.0 score ranks first
    assert (tmp_path / "out" / "selected_pure_rerank_documents_best_chunk.parquet").exists()


def test_multilingual_sanity_is_deterministic_and_descriptive(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pool_path, _ = _tiny_pool(tmp_path)
    pool_rows = pq.read_table(pool_path).to_pylist()
    for row in pool_rows:
        row["sparse_rank"] = row["pool_rank"]
        row["sparse_score"] = 1.0
        row["hybrid_rank"] = row["pool_rank"]
        row["hybrid_score"] = 0.01
    pq.write_table(pa.Table.from_pylist(pool_rows), pool_path)
    reranked_rows = []
    for rank, row in enumerate(reversed(pool_rows), start=1):
        reranked_rows.append({**row, "rerank_rank": rank, "rerank_score": float(rank)})
    reranked_path = tmp_path / "reranked.parquet"
    pq.write_table(pa.Table.from_pylist(reranked_rows), reranked_path)

    first = build_multilingual_sanity(
        pool_path=pool_path, reranked_path=reranked_path, sample_count=1
    )
    second = build_multilingual_sanity(
        pool_path=pool_path, reranked_path=reranked_path, sample_count=1
    )
    assert first == second
    assert first["query_count"] == 1
    assert first["label"].endswith("not relevance evaluation")
    assert first["rows"][0]["reranked"]["chunk_id"] == "c"
    assert first["rows"][0]["reranked"]["text_preview"] == "đau tim c"
