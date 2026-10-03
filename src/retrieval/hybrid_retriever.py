from __future__ import annotations

from typing import Any, Sequence

from .fusion import reciprocal_rank_fusion


class HybridRetriever:
    def __init__(self, *, rrf_constant: int = 60):
        self.rrf_constant = rrf_constant

    def fuse(
        self,
        dense_results: Sequence[Sequence[dict[str, Any]]],
        sparse_results: Sequence[Sequence[dict[str, Any]]],
        *,
        top_k: int,
    ) -> list[list[dict[str, Any]]]:
        if len(dense_results) != len(sparse_results):
            raise ValueError("dense and sparse query result counts differ")
        return [
            reciprocal_rank_fusion(
                [dense, sparse], rrf_constant=self.rrf_constant, top_k=top_k
            )
            for dense, sparse in zip(dense_results, sparse_results, strict=True)
        ]
