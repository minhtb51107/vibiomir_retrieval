"""Disk-backed, deterministic Vietnamese document retrieval for Phase 15B."""
from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokens(text: str) -> list[str]:
    return [value.casefold() for value in TOKEN_RE.findall(str(text)) if len(value) > 1]


def fts_query(text: str) -> str:
    # OR is deliberate: FTS ranks matching medical terms while short natural
    # language questions are not forced to contain every function word.
    unique = list(dict.fromkeys(tokens(text)))
    return " OR ".join('"' + value.replace('"', '""') + '"' for value in unique)


def configure(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-32768")


def open_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    configure(connection)
    connection.executescript("""
    CREATE TABLE IF NOT EXISTS documents(
      doc_id INTEGER PRIMARY KEY,
      official_url TEXT NOT NULL,
      canonical_domain TEXT NOT NULL,
      content_hash TEXT NOT NULL,
      source_reference TEXT NOT NULL
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
      title, body, content='',
      tokenize='unicode61 remove_diacritics 2'
    );
    CREATE TABLE IF NOT EXISTS build_state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    connection.commit()
    return connection


def content_hash(title: str, body: str) -> str:
    return hashlib.sha256((str(title) + "\0" + str(body)).encode("utf-8")).hexdigest()


def add_documents(connection: sqlite3.Connection, rows: list[dict]) -> int:
    """Commit a batch atomically; exact resume attempts are idempotent."""
    inserted = 0
    with connection:
        for row in rows:
            digest = content_hash(row["title"], row["body"])
            if digest != row["content_hash"]:
                raise ValueError(f"content hash mismatch for doc_id={row['doc_id']}")
            cursor = connection.execute(
                "INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?)",
                (int(row["doc_id"]), str(row["official_url"]), str(row["canonical_domain"]),
                 digest, str(row["source_reference"])),
            )
            if cursor.rowcount:
                connection.execute(
                    "INSERT INTO documents_fts(rowid,title,body) VALUES(?,?,?)",
                    (int(row["doc_id"]), str(row["title"]), str(row["body"])),
                )
                inserted += 1
        connection.execute(
            "INSERT INTO build_state VALUES('documents',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]),),
        )
    return inserted


def search(connection: sqlite3.Connection, query_text: str, limit: int = 100) -> list[dict]:
    expression = fts_query(query_text)
    if not expression:
        return []
    rows = connection.execute(
        "SELECT d.doc_id,d.official_url,d.canonical_domain,d.content_hash,"
        "bm25(documents_fts,4.0,1.0) score FROM documents_fts "
        "JOIN documents d ON d.doc_id=documents_fts.rowid WHERE documents_fts MATCH ? "
        "ORDER BY score ASC,d.doc_id ASC LIMIT ?", (expression, int(limit)),
    ).fetchall()
    return [
        {"doc_id": int(row[0]), "official_url": row[1], "canonical_domain": row[2],
         "content_hash": row[3], "bm25_score": float(row[4]), "rank": rank}
        for rank, row in enumerate(rows, 1)
    ]
