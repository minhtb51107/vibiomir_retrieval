from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CachedResponse:
    request_url: str
    final_url: str
    status_code: int
    content_type: str
    body: bytes
    truncated: bool


class ProbeCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                fingerprint TEXT PRIMARY KEY,
                request_url TEXT NOT NULL,
                final_url TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                content_type TEXT NOT NULL,
                body BLOB NOT NULL,
                truncated INTEGER NOT NULL,
                fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.commit()

    @staticmethod
    def fingerprint(url: str) -> str:
        return hashlib.sha256(("GET\n" + url).encode("utf-8")).hexdigest()

    def get(self, url: str) -> CachedResponse | None:
        row = self.connection.execute(
            "SELECT request_url, final_url, status_code, content_type, body, truncated "
            "FROM responses WHERE fingerprint = ?",
            (self.fingerprint(url),),
        ).fetchone()
        return CachedResponse(*row[:-1], bool(row[-1])) if row else None

    def put(self, response: CachedResponse) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO responses "
                "(fingerprint, request_url, final_url, status_code, content_type, body, truncated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    self.fingerprint(response.request_url), response.request_url,
                    response.final_url, response.status_code, response.content_type,
                    response.body, int(response.truncated),
                ),
            )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ProbeCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
