from __future__ import annotations

from pathlib import Path

import numpy as np


class ExactFaissIndex:
    def __init__(self, dimension: int):
        import faiss

        self.index = faiss.IndexFlatIP(dimension)

    @classmethod
    def load(cls, path: str | Path) -> ExactFaissIndex:
        import faiss

        instance = cls.__new__(cls)
        instance.index = faiss.read_index(str(path))
        return instance

    @property
    def count(self) -> int:
        return int(self.index.ntotal)

    @property
    def dimension(self) -> int:
        return int(self.index.d)

    def add(self, vectors: np.ndarray) -> None:
        values = np.ascontiguousarray(vectors, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("embedding shape does not match FAISS dimension")
        self.index.add(values)

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.ascontiguousarray(queries, dtype=np.float32)
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores, rows = self.index.search(values, min(top_k, self.count))
        return scores, rows

    def save(self, path: str | Path) -> None:
        import faiss

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(output))
