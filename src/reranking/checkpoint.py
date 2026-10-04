from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


SCHEMA = """
CREATE TABLE manifest (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE scores (
    query_id INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,
    rerank_score REAL NOT NULL,
    inference_ms REAL NOT NULL,
    PRIMARY KEY(query_id, chunk_id)
);
"""


class RerankCheckpoint:
    def __init__(self, path: str | Path, *, signature: dict[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists()
        self.connection = sqlite3.connect(self.path)
        if not existed:
            self.connection.executescript(SCHEMA)
            with self.connection:
                self.connection.execute(
                    "INSERT INTO manifest(key, value) VALUES (?, ?)",
                    ("signature", json.dumps(signature, sort_keys=True)),
                )
        row = self.connection.execute(
            "SELECT value FROM manifest WHERE key='signature'"
        ).fetchone()
        if row is None or json.loads(row[0]) != signature:
            self.connection.close()
            raise ValueError("rerank checkpoint does not match candidate/model signature")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> RerankCheckpoint:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0])

    def completed_keys(self) -> set[tuple[int, str]]:
        return {
            (int(row[0]), str(row[1]))
            for row in self.connection.execute("SELECT query_id, chunk_id FROM scores")
        }

    def write_scores(self, rows: Iterable[tuple[int, str, float, float]]) -> None:
        with self.connection:
            self.connection.executemany(
                "INSERT INTO scores(query_id, chunk_id, rerank_score, inference_ms) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(query_id, chunk_id) DO UPDATE SET "
                "rerank_score=excluded.rerank_score, inference_ms=excluded.inference_ms",
                rows,
            )

    def read_scores(self) -> dict[tuple[int, str], tuple[float, float]]:
        return {
            (int(row[0]), str(row[1])): (float(row[2]), float(row[3]))
            for row in self.connection.execute(
                "SELECT query_id, chunk_id, rerank_score, inference_ms FROM scores"
            )
        }

    def integrity_check(self) -> str:
        return str(self.connection.execute("PRAGMA integrity_check").fetchone()[0])
