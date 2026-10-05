from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    fingerprint TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    query_text TEXT NOT NULL,
    depth INTEGER NOT NULL,
    status TEXT NOT NULL,
    http_status INTEGER,
    attempt_count INTEGER NOT NULL,
    error_type TEXT,
    error_message TEXT,
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS assignments (
    query_id INTEGER NOT NULL,
    variant TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    PRIMARY KEY(query_id, variant),
    FOREIGN KEY(fingerprint) REFERENCES responses(fingerprint)
);
CREATE TABLE IF NOT EXISTS results (
    fingerprint TEXT NOT NULL,
    rank INTEGER NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    snippet TEXT NOT NULL,
    PRIMARY KEY(fingerprint, rank),
    FOREIGN KEY(fingerprint) REFERENCES responses(fingerprint)
);
CREATE TABLE IF NOT EXISTS mappings (
    result_url TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    match_level TEXT,
    doc_id INTEGER,
    official_url TEXT,
    official_domain TEXT,
    candidate_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS domain_stats (
    domain TEXT PRIMARY KEY,
    corpus_count INTEGER NOT NULL
);
"""


def request_fingerprint(provider: str, query_text: str, depth: int) -> str:
    value = json.dumps(
        {"provider": provider, "query": query_text, "depth": depth},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class SearchCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def assign(self, query_id: int, variant: str, fingerprint: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO assignments VALUES (?,?,?)",
                (query_id, variant, fingerprint),
            )

    def response_status(self, fingerprint: str) -> str | None:
        row = self.connection.execute(
            "SELECT status FROM responses WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        return str(row[0]) if row else None

    def write_response(
        self,
        *,
        fingerprint: str,
        provider: str,
        query_text: str,
        depth: int,
        status: str,
        http_status: int | None,
        attempt_count: int,
        error_type: str | None,
        error_message: str | None,
        completed_at: str,
        results: list[dict[str, Any]],
    ) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM results WHERE fingerprint=?", (fingerprint,))
            self.connection.execute(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    fingerprint, provider, query_text, depth, status, http_status,
                    attempt_count, error_type, error_message, completed_at,
                ),
            )
            self.connection.executemany(
                "INSERT INTO results VALUES (?,?,?,?,?)",
                [
                    (fingerprint, int(row["rank"]), row["title"], row["url"], row["snippet"])
                    for row in results
                ],
            )

    def unmapped_urls(self) -> list[str]:
        return [
            str(row[0])
            for row in self.connection.execute(
                "SELECT DISTINCT r.url FROM results r LEFT JOIN mappings m ON m.result_url=r.url "
                "WHERE m.result_url IS NULL"
            )
        ]

    def write_mappings(self, rows: list[dict[str, Any]]) -> None:
        with self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO mappings VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        row["result_url"], row["status"], row.get("match_level"),
                        row.get("doc_id"), row.get("official_url"),
                        row.get("official_domain"), int(row["candidate_count"]),
                    )
                    for row in rows
                ],
            )

    def write_domain_stats(self, counts: dict[str, int]) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM domain_stats")
            self.connection.executemany(
                "INSERT INTO domain_stats VALUES (?,?)", sorted(counts.items())
            )
