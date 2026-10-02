from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from .models import CrawlResult, CrawlStatus, RETRYABLE_FAILURE_STATUSES


class InvalidStatusTransition(ValueError):
    pass


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS crawl_results (
    doc_id INTEGER PRIMARY KEY,
    original_url TEXT NOT NULL,
    fetch_url TEXT NOT NULL,
    final_url TEXT,
    status TEXT NOT NULL,
    http_status INTEGER,
    content_type TEXT,
    encoding TEXT,
    content_length INTEGER,
    fetched_at TEXT NOT NULL,
    attempt_count INTEGER NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    error_type TEXT,
    error_message TEXT,
    raw_body BLOB,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_crawl_results_status ON crawl_results(status);
"""


class CheckpointStore:
    """Transactional SQLite result store keyed by canonical corpus doc_id."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(SCHEMA_SQL)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> CheckpointStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def completed_doc_ids(self, *, retry_failures: bool = False) -> set[int]:
        if not retry_failures:
            rows = self.connection.execute("SELECT doc_id FROM crawl_results")
        else:
            retryable = tuple(status.value for status in RETRYABLE_FAILURE_STATUSES)
            placeholders = ",".join("?" for _ in retryable)
            rows = self.connection.execute(
                f"SELECT doc_id FROM crawl_results WHERE status NOT IN ({placeholders})",
                retryable,
            )
        return {int(row[0]) for row in rows}

    def save(self, result: CrawlResult) -> None:
        with self.connection:
            existing = self.connection.execute(
                "SELECT status FROM crawl_results WHERE doc_id = ?", (result.doc_id,)
            ).fetchone()
            if (
                existing is not None
                and existing["status"] == CrawlStatus.SUCCESS.value
                and result.status != CrawlStatus.SUCCESS
            ):
                raise InvalidStatusTransition(
                    f"cannot replace SUCCESS for doc_id {result.doc_id} with {result.status.value}"
                )
            self.connection.execute(
                """
                INSERT INTO crawl_results (
                    doc_id, original_url, fetch_url, final_url, status,
                    http_status, content_type, encoding, content_length,
                    fetched_at, attempt_count, elapsed_ms, error_type,
                    error_message, raw_body
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    original_url = excluded.original_url,
                    fetch_url = excluded.fetch_url,
                    final_url = excluded.final_url,
                    status = excluded.status,
                    http_status = excluded.http_status,
                    content_type = excluded.content_type,
                    encoding = excluded.encoding,
                    content_length = excluded.content_length,
                    fetched_at = excluded.fetched_at,
                    attempt_count = excluded.attempt_count,
                    elapsed_ms = excluded.elapsed_ms,
                    error_type = excluded.error_type,
                    error_message = excluded.error_message,
                    raw_body = excluded.raw_body,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    result.doc_id,
                    result.original_url,
                    result.fetch_url,
                    result.final_url,
                    result.status.value,
                    result.http_status,
                    result.content_type,
                    result.encoding,
                    result.content_length,
                    result.fetched_at,
                    result.attempt_count,
                    result.elapsed_ms,
                    result.error_type,
                    result.error_message,
                    result.raw_body,
                ),
            )

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM crawl_results").fetchone()[0])

    def status_counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT status, COUNT(*) AS count FROM crawl_results GROUP BY status ORDER BY status"
        )
        return {str(row["status"]): int(row["count"]) for row in rows}

    def rows_for(self, doc_ids: Iterable[int]) -> list[sqlite3.Row]:
        ids = tuple(doc_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        return list(
            self.connection.execute(
                f"SELECT * FROM crawl_results WHERE doc_id IN ({placeholders}) ORDER BY doc_id",
                ids,
            )
        )
