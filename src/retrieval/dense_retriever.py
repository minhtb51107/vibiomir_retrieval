from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.indexing.embedder import Embedder, normalize_text
from src.indexing.faiss_index import ExactFaissIndex
from src.indexing.metadata_store import ChunkMetadataStore


class DenseRetriever:
    def __init__(
        self,
        *,
        embedder: Embedder,
        index: ExactFaissIndex,
        metadata: ChunkMetadataStore,
    ):
        if index.count != metadata.count:
            raise ValueError("FAISS and metadata row counts differ")
        if index.dimension != embedder.dimension:
            raise ValueError("FAISS and embedder dimensions differ")
        self.embedder = embedder
        self.index = index
        self.metadata = metadata

    def retrieve(
        self,
        query_ids: Sequence[int],
        query_texts: Sequence[str],
        *,
        top_k: int,
    ) -> list[list[dict[str, Any]]]:
        if len(query_ids) != len(query_texts):
            raise ValueError("query IDs and texts must have equal length")
        normalized_queries = [normalize_text(text) for text in query_texts]
        vectors = self.embedder.encode(normalized_queries)
        return self.retrieve_vectors(query_ids, normalized_queries, vectors, top_k=top_k)

    def retrieve_vectors(
        self,
        query_ids: Sequence[int],
        query_texts: Sequence[str],
        vectors: np.ndarray,
        *,
        top_k: int,
    ) -> list[list[dict[str, Any]]]:
        if len(query_ids) != len(query_texts) or len(query_ids) != len(vectors):
            raise ValueError("query IDs, texts, and vectors must have equal length")
        scores, row_ids = self.index.search(vectors, top_k)
        results: list[list[dict[str, Any]]] = []
        for query_id, query_text, query_scores, query_rows in zip(
            query_ids, query_texts, scores, row_ids, strict=True
        ):
            valid = [int(row_id) for row_id in query_rows if int(row_id) >= 0]
            metadata = self.metadata.get_rows(valid)
            ranked = []
            for rank, (score, row) in enumerate(
                zip(query_scores[: len(metadata)], metadata, strict=True), start=1
            ):
                ranked.append(
                    {
                        "query_id": int(query_id),
                        "query_text": query_text,
                        "rank": rank,
                        "score": float(score),
                        "chunk_id": row["chunk_id"],
                        "doc_id": int(row["doc_id"]),
                        "chunk_text": row["raw_text"],
                        "source_url": row["source_url"],
                        "faiss_row": int(row["faiss_row"]),
                        "chunk_index": int(row["chunk_index"]),
                        "start_offset": int(row["start_offset"]),
                        "end_offset": int(row["end_offset"]),
                        "section_type": row["section_type"],
                        "heading_path": row["heading_path"],
                        "extraction_method": row["extraction_method"],
                    }
                )
            results.append(ranked)
        return results
