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
    domain TEXT,
    final_url TEXT,
    status TEXT NOT NULL,
    http_status INTEGER,
    content_type TEXT,
    encoding TEXT,
    declared_http_encoding TEXT,
    content_length INTEGER,
    downloaded_bytes INTEGER NOT NULL DEFAULT 0,
    body_truncated INTEGER NOT NULL DEFAULT 0,
    redirect_count INTEGER NOT NULL DEFAULT 0,
    tiny_html INTEGER NOT NULL DEFAULT 0,
    js_shell_candidate INTEGER NOT NULL DEFAULT 0,
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

MIGRATION_COLUMNS = {
    "domain": "TEXT",
    "declared_http_encoding": "TEXT",
    "downloaded_bytes": "INTEGER NOT NULL DEFAULT 0",
    "body_truncated": "INTEGER NOT NULL DEFAULT 0",
    "redirect_count": "INTEGER NOT NULL DEFAULT 0",
    "tiny_html": "INTEGER NOT NULL DEFAULT 0",
    "js_shell_candidate": "INTEGER NOT NULL DEFAULT 0",
}


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
        existing_columns = {
            str(row[1])
            for row in self.connection.execute("PRAGMA table_info(crawl_results)")
        }
        for name, definition in MIGRATION_COLUMNS.items():
            if name not in existing_columns:
                self.connection.execute(
                    f"ALTER TABLE crawl_results ADD COLUMN {name} {definition}"
                )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_crawl_results_domain ON crawl_results(domain)"
        )
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

    def is_completed(self, doc_id: int, *, retry_failures: bool = False) -> bool:
        row = self.connection.execute(
            "SELECT status FROM crawl_results WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return False
        if not retry_failures:
            return True
        return CrawlStatus(str(row["status"])) not in RETRYABLE_FAILURE_STATUSES

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
                    doc_id, original_url, fetch_url, domain, final_url, status,
                    http_status, content_type, encoding, declared_http_encoding,
                    content_length, downloaded_bytes, body_truncated,
                    redirect_count, tiny_html, js_shell_candidate,
                    fetched_at, attempt_count, elapsed_ms, error_type,
                    error_message, raw_body
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    original_url = excluded.original_url,
                    fetch_url = excluded.fetch_url,
                    domain = excluded.domain,
                    final_url = excluded.final_url,
                    status = excluded.status,
                    http_status = excluded.http_status,
                    content_type = excluded.content_type,
                    encoding = excluded.encoding,
                    declared_http_encoding = excluded.declared_http_encoding,
                    content_length = excluded.content_length,
                    downloaded_bytes = excluded.downloaded_bytes,
                    body_truncated = excluded.body_truncated,
                    redirect_count = excluded.redirect_count,
                    tiny_html = excluded.tiny_html,
                    js_shell_candidate = excluded.js_shell_candidate,
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
                    result.domain,
                    result.final_url,
                    result.status.value,
                    result.http_status,
                    result.content_type,
                    result.encoding,
                    result.declared_http_encoding,
                    result.content_length,
                    result.downloaded_bytes,
                    int(result.body_truncated),
                    result.redirect_count,
                    int(result.tiny_html),
                    int(result.js_shell_candidate),
                    result.fetched_at,
                    result.attempt_count,
                    result.elapsed_ms,
                    result.error_type,
                    result.error_message,
                    result.raw_body,
                ),
            )

    def telemetry_summary(self) -> dict[str, object]:
        """Aggregate production telemetry with bounded-memory SQL queries."""
        scalar = self.connection.execute(
            """
            SELECT COUNT(*) AS rows,
                   COALESCE(SUM(attempt_count), 0) AS attempts,
                   COALESCE(SUM(CASE WHEN attempt_count > 1 THEN attempt_count - 1 ELSE 0 END), 0) AS retries,
                   COALESCE(SUM(redirect_count), 0) AS redirects,
                   COALESCE(SUM(downloaded_bytes), 0) AS downloaded_bytes,
                   COALESCE(SUM(tiny_html), 0) AS tiny_html,
                   COALESCE(SUM(js_shell_candidate), 0) AS js_shell_candidates,
                   COALESCE(SUM(body_truncated), 0) AS truncated
            FROM crawl_results
            """
        ).fetchone()

        def grouped(column: str, where: str = "1=1") -> dict[str, int]:
            rows = self.connection.execute(
                f"SELECT COALESCE({column}, 'missing') AS key, COUNT(*) AS count "
                f"FROM crawl_results WHERE {where} GROUP BY key ORDER BY count DESC"
            )
            return {str(row["key"]): int(row["count"]) for row in rows}

        size_buckets = self.connection.execute(
            """
            SELECT CASE
                     WHEN downloaded_bytes = 0 THEN '0'
                     WHEN downloaded_bytes < 512 THEN '1-511'
                     WHEN downloaded_bytes < 4096 THEN '512-4095'
                     WHEN downloaded_bytes < 65536 THEN '4096-65535'
                     WHEN downloaded_bytes < 262144 THEN '65536-262143'
                     WHEN downloaded_bytes < 1048576 THEN '262144-1048575'
                     ELSE '1048576+'
                   END AS bucket,
                   COUNT(*) AS count
            FROM crawl_results GROUP BY bucket
            """
        )
        per_domain_rows = self.connection.execute(
            """
            SELECT COALESCE(domain, 'unknown') AS domain,
                   COUNT(*) AS completed,
                   SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) AS successes,
                   SUM(attempt_count) AS attempts,
                   SUM(redirect_count) AS redirects,
                   SUM(downloaded_bytes) AS downloaded_bytes,
                   SUM(elapsed_ms) AS elapsed_ms
            FROM crawl_results GROUP BY domain ORDER BY completed DESC
            """
        )
        per_domain = {
            str(row["domain"]): {
                "completed": int(row["completed"]),
                "successes": int(row["successes"] or 0),
                "attempts": int(row["attempts"] or 0),
                "redirects": int(row["redirects"] or 0),
                "downloaded_bytes": int(row["downloaded_bytes"] or 0),
                "elapsed_ms_sum": int(row["elapsed_ms"] or 0),
            }
            for row in per_domain_rows
        }
        return {
            "rows": int(scalar["rows"]),
            "attempts": int(scalar["attempts"]),
            "retry_count": int(scalar["retries"]),
            "redirect_count": int(scalar["redirects"]),
            "downloaded_bytes": int(scalar["downloaded_bytes"]),
            "tiny_html_count": int(scalar["tiny_html"]),
            "js_shell_candidate_count": int(scalar["js_shell_candidates"]),
            "truncated_body_count": int(scalar["truncated"]),
            "status_counts": grouped("status"),
            "http_status_counts": grouped("CAST(http_status AS TEXT)", "http_status IS NOT NULL"),
            "content_type_counts": grouped("content_type"),
            "declared_http_encoding_counts": grouped("declared_http_encoding"),
            "response_size_buckets": {
                str(row["bucket"]): int(row["count"]) for row in size_buckets
            },
            "per_domain": per_domain,
        }

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
