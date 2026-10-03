from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.indexing.metadata_store import build_metadata_store
from src.retrieval.fusion import (
    deduplicate_chunks,
    reciprocal_rank_fusion,
    reciprocal_rank_fusion_documents,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.lexical_tokenizer import tokenize_lexical
from src.retrieval.ranking import aggregate_documents
from src.retrieval.sparse_retriever import (
    SparseRetriever,
    build_sparse_index,
    verify_sparse_mapping,
)
from src.indexing.metadata_store import ChunkMetadataStore


REQUIRED_RESULT_FIELDS = {
    "query_id",
    "query_text",
    "rank",
    "score",
    "chunk_id",
    "doc_id",
    "chunk_text",
    "source_url",
}


def _write_chunks(path: Path) -> None:
    rows = [
        {
            "chunk_id": "heart-vi",
            "doc_id": 10,
            "chunk_index": 0,
            "raw_text": "Điều trị bệnh tim và huyết áp.",
            "normalized_text": "Điều trị bệnh tim và huyết áp.",
            "token_count": 9,
            "start_offset": 0,
            "end_offset": 32,
            "section_type": "content",
            "heading_path": ["Tim"],
            "source_url": "https://example.test/heart",
            "extraction_method": "semantic_container",
        },
        {
            "chunk_id": "liver-vi",
            "doc_id": 20,
            "chunk_index": 0,
            "raw_text": "Bệnh gan và men gan cao.",
            "normalized_text": "Bệnh gan và men gan cao.",
            "token_count": 7,
            "start_offset": 0,
            "end_offset": 24,
            "section_type": "answer",
            "heading_path": [],
            "source_url": "https://example.test/liver",
            "extraction_method": "density_scored",
        },
        {
            "chunk_id": "heart-zh",
            "doc_id": 30,
            "chunk_index": 0,
            "raw_text": "心脏病治疗方法",
            "normalized_text": "心脏病治疗方法",
            "token_count": 8,
            "start_offset": 0,
            "end_offset": 8,
            "section_type": "answer",
            "heading_path": ["治疗"],
            "source_url": "https://example.test/zh-heart",
            "extraction_method": "semantic_container",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    chunks = tmp_path / "chunks.parquet"
    metadata = tmp_path / "metadata.sqlite"
    sparse = tmp_path / "bm25.sqlite"
    _write_chunks(chunks)
    build_metadata_store(chunks, metadata)
    build_sparse_index(
        chunks_path=chunks,
        output_path=sparse,
        k1=1.2,
        b=0.75,
        dense_metadata_path=metadata,
    )
    return chunks, metadata, sparse


def test_multilingual_tokenizer_preserves_vietnamese_and_english_words() -> None:
    tokens = tokenize_lexical("  ĐIỀU trị Tim, HEART  ")
    assert tokens == ["w:điều", "w:trị", "w:tim", "w:heart"]


def test_chinese_tokenizer_emits_unigrams_and_bigrams() -> None:
    tokens = tokenize_lexical("心脏病")
    assert tokens == ["h1:心", "h1:脏", "h1:病", "h2:心脏", "h2:脏病"]


def test_bm25_ranking_empty_query_mapping_schema_and_determinism(tmp_path: Path) -> None:
    _, metadata_path, sparse_path = _build_fixture(tmp_path)
    assert verify_sparse_mapping(sparse_path, metadata_path) == {
        "rows_checked": 3,
        "mismatches": 0,
    }
    with ChunkMetadataStore(metadata_path) as metadata:
        with SparseRetriever(index_path=sparse_path, metadata=metadata) as retriever:
            first = retriever.retrieve([1], ["điều trị tim"], top_k=3)[0]
            second = retriever.retrieve([1], ["điều trị tim"], top_k=3)[0]
            chinese = retriever.retrieve([2], ["心脏病"], top_k=3)[0]
            empty = retriever.retrieve([3], ["!!!"], top_k=3)[0]
    assert first == second
    assert first[0]["chunk_id"] == "heart-vi"
    assert chinese[0]["chunk_id"] == "heart-zh"
    assert empty == []
    assert REQUIRED_RESULT_FIELDS <= set(first[0])
    assert first[0]["chunk_text"] == "Điều trị bệnh tim và huyết áp."
    assert first[0]["doc_id"] == 10


def _row(chunk_id: str, doc_id: int, rank: int, score: float) -> dict[str, object]:
    return {
        "query_id": 1,
        "query_text": "tim",
        "rank": rank,
        "score": score,
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "chunk_text": chunk_id,
        "source_url": f"https://example.test/{doc_id}",
    }


def test_rrf_correctness_tie_breaking_and_duplicate_removal() -> None:
    dense = [_row("a", 1, 1, 0.9), _row("b", 2, 2, 0.8)]
    sparse = [_row("b", 2, 1, 5.0), _row("a", 1, 2, 4.0)]
    fused = reciprocal_rank_fusion([dense, sparse], rrf_constant=60)
    assert [row["chunk_id"] for row in fused] == ["a", "b"]
    assert fused[0]["score"] == fused[1]["score"]
    assert fused[0]["source_ranks"] == [1, 2]
    duplicated = reciprocal_rank_fusion(
        [[dense[0], dense[0]], [sparse[1]]], rrf_constant=60
    )
    assert len(duplicated) == 1
    assert duplicated[0]["score"] == 1 / 61 + 1 / 62
    assert len(deduplicate_chunks([dense[0], {**dense[0], "score": 0.1}])) == 1


def test_hybrid_repeat_runs_schema_and_document_aggregation() -> None:
    dense = [[_row("a", 1, 1, 0.9), _row("b", 2, 2, 0.8)]]
    sparse = [[_row("b", 2, 1, 5.0), _row("c", 1, 2, 4.0)]]
    retriever = HybridRetriever(rrf_constant=60)
    first = retriever.fuse(dense, sparse, top_k=3)
    second = retriever.fuse(dense, sparse, top_k=3)
    assert first == second
    assert REQUIRED_RESULT_FIELDS <= set(first[0][0])
    best = aggregate_documents(first[0], method="best_chunk", top_k_docs=2)
    mean = aggregate_documents(
        first[0], method="top_n_mean", top_k_docs=2, top_n=2
    )
    assert len({row["doc_id"] for row in best}) == len(best)
    assert len({row["doc_id"] for row in mean}) == len(mean)
    document_fused = reciprocal_rank_fusion_documents(
        [best, mean], rrf_constant=60, top_k=2
    )
    assert len({row["doc_id"] for row in document_fused}) == len(document_fused)
    assert all(row["aggregation_method"] == "rrf_after_document_aggregation" for row in document_fused)
