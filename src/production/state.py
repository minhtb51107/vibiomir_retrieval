from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS stage_signature (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS completed_items (
    item_id TEXT PRIMARY KEY,
    partition_id TEXT NOT NULL,
    output_sha256 TEXT NOT NULL,
    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_completed_partition
ON completed_items(partition_id);
"""


class PartitionCheckpoint:
    """Small transactional item checkpoint for partitioned production stages."""

    def __init__(self, path: str | Path, *, signature: dict[str, Any]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(SCHEMA)
        encoded = json.dumps(signature, sort_keys=True, separators=(",", ":"))
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO stage_signature(singleton, value) VALUES (1, ?)",
                (encoded,),
            )
        stored = self.connection.execute(
            "SELECT value FROM stage_signature WHERE singleton = 1"
        ).fetchone()[0]
        if stored != encoded:
            self.connection.close()
            raise ValueError("partition checkpoint signature mismatch")

    def __enter__(self) -> PartitionCheckpoint:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        self.connection.close()

    @property
    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM completed_items").fetchone()[0])

    def completed_ids(self) -> set[str]:
        return {
            str(row[0])
            for row in self.connection.execute("SELECT item_id FROM completed_items")
        }

    def is_completed(self, item_id: str) -> bool:
        return self.connection.execute(
            "SELECT 1 FROM completed_items WHERE item_id = ?", (str(item_id),)
        ).fetchone() is not None

    def pending(self, item_ids: Iterable[str]) -> list[str]:
        completed = self.completed_ids()
        return [str(item_id) for item_id in item_ids if str(item_id) not in completed]

    def commit_partition(
        self,
        *,
        partition_id: str,
        item_ids: Iterable[str],
        output_sha256: str,
    ) -> int:
        values = [str(item_id) for item_id in item_ids]
        if len(values) != len(set(values)):
            raise ValueError("partition contains duplicate item IDs")
        with self.connection:
            for item_id in values:
                existing = self.connection.execute(
                    "SELECT partition_id, output_sha256 FROM completed_items WHERE item_id = ?",
                    (item_id,),
                ).fetchone()
                expected = (partition_id, output_sha256)
                if existing is not None:
                    if (str(existing[0]), str(existing[1])) != expected:
                        raise ValueError(f"item {item_id} already belongs to another output")
                    continue
                self.connection.execute(
                    "INSERT INTO completed_items(item_id, partition_id, output_sha256) "
                    "VALUES (?, ?, ?)",
                    (item_id, partition_id, output_sha256),
                )
        return len(values)

    def integrity_check(self) -> str:
        return str(self.connection.execute("PRAGMA integrity_check").fetchone()[0])
