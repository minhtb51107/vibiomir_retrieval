from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.indexing.embedder import l2_normalize, normalize_text
from src.indexing.faiss_index import ExactFaissIndex
from src.indexing.metadata_store import ChunkMetadataStore
from src.indexing.pipeline import build_dense_index
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.ranking import aggregate_documents


class MockEmbedder:
    dimension = 2

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            if "tim" in text.casefold():
                vectors.append([1.0, 0.0])
            elif "gan" in text.casefold():
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 1.0])
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": "BAAI/bge-m3",
            "embedding_dimension": self.dimension,
            "normalization": "L2",
            "device": "mock",
        }


def _chunks(path: Path) -> None:
    rows = [
        {
            "chunk_id": "c-heart-1", "doc_id": 10, "chunk_index": 0,
            "raw_text": "Bệnh tim cần khám.", "normalized_text": "Bệnh tim cần khám.",
            "token_count": 5, "start_offset": 0, "end_offset": 18,
            "section_type": "content", "heading_path": ["Tim"],
            "source_url": "https://example.test/10", "extraction_method": "semantic_container",
        },
        {
            "chunk_id": "c-heart-2", "doc_id": 10, "chunk_index": 1,
            "raw_text": "Điều trị tim.", "normalized_text": "Điều trị tim.",
            "token_count": 4, "start_offset": 20, "end_offset": 32,
            "section_type": "content", "heading_path": ["Tim"],
            "source_url": "https://example.test/10", "extraction_method": "semantic_container",
        },
        {
            "chunk_id": "c-liver", "doc_id": 20, "chunk_index": 0,
            "raw_text": "Bệnh gan.", "normalized_text": "Bệnh gan.",
            "token_count": 3, "start_offset": 0, "end_offset": 9,
            "section_type": "answer", "heading_path": [],
            "source_url": "https://example.test/20", "extraction_method": "density_scored",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_normalization_and_embedding_interface() -> None:
    assert normalize_text("  Tiếng\n  Việt  ") == "Tiếng Việt"
    vectors = MockEmbedder().encode(["tim", "gan"])
    assert vectors.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])


def test_exact_index_mapping_top_k_and_provenance(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.parquet"
    _chunks(chunks)
    summary = build_dense_index(
        chunks_path=chunks,
        embedding_directory=tmp_path / "embeddings",
        index_directory=tmp_path / "index",
        embedder=MockEmbedder(),
        add_batch_size=2,
        progress_every=1,
    )
    assert summary["index_type"] == "IndexFlatIP"
    assert summary["chunk_count"] == 3
    assert summary["row_mapping_verified"]
    assert summary["row_mapping_verification"] == {"rows_checked": 3, "mismatches": 0}
    resumed = build_dense_index(
        chunks_path=chunks,
        embedding_directory=tmp_path / "embeddings",
        index_directory=tmp_path / "index",
        embedder=MockEmbedder(),
        add_batch_size=2,
        progress_every=1,
    )
    assert resumed["embedding_rows_computed_this_run"] == 0

    index = ExactFaissIndex.load(tmp_path / "index" / "chunks.index")
    with ChunkMetadataStore(tmp_path / "index" / "chunk_metadata.sqlite") as metadata:
        assert [row["chunk_id"] for row in metadata.get_rows([0, 1, 2])] == [
            "c-heart-1", "c-heart-2", "c-liver"
        ]
        retriever = DenseRetriever(
            embedder=MockEmbedder(), index=index, metadata=metadata
        )
        first = retriever.retrieve([7], [" đau tim "], top_k=2)[0]
        second = retriever.retrieve([7], [" đau tim "], top_k=2)[0]
        assert first == second
        assert len(first) == 2
        assert {row["chunk_id"] for row in first} == {"c-heart-1", "c-heart-2"}
        assert first[0]["chunk_text"] in {"Bệnh tim cần khám.", "Điều trị tim."}
        assert first[0]["source_url"] == "https://example.test/10"
        assert {
            "query_id", "query_text", "rank", "score", "chunk_id", "doc_id",
            "chunk_text", "source_url",
        } <= set(first[0])


def test_document_deduplication_and_aggregations() -> None:
    chunks = [
        {"query_id": 1, "query_text": "q", "score": 0.9, "doc_id": 1,
         "chunk_id": "a", "chunk_text": "a", "source_url": "u1"},
        {"query_id": 1, "query_text": "q", "score": 0.7, "doc_id": 1,
         "chunk_id": "b", "chunk_text": "b", "source_url": "u1"},
        {"query_id": 1, "query_text": "q", "score": 0.85, "doc_id": 2,
         "chunk_id": "c", "chunk_text": "c", "source_url": "u2"},
    ]
    best = aggregate_documents(chunks, method="best_chunk", top_k_docs=2)
    mean = aggregate_documents(chunks, method="top_n_mean", top_k_docs=2, top_n=2)
    assert [row["doc_id"] for row in best] == [1, 2]
    assert [row["doc_id"] for row in mean] == [2, 1]
    assert len({row["doc_id"] for row in best}) == len(best)
    assert best[0]["score"] == 0.9
    assert mean[1]["score"] == 0.8
