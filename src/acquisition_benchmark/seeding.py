from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from src.ingestion.checkpoint import CheckpointStore
from src.storage.body_archive import BodyArchive, BodyMetadata


def seed_from_existing(
    records: list[dict[str, Any]], *, source_database: str | Path,
    source_archive: str | Path, destination_database: str | Path,
    destination_archive: str | Path,
) -> dict[str, int]:
    wanted = {int(row["doc_id"]): str(row["original_url"]) for row in records}
    seeded_rows = 0
    seeded_bodies = 0
    source = sqlite3.connect(source_database)
    source.row_factory = sqlite3.Row
    try:
        with CheckpointStore(destination_database) as destination, BodyArchive(source_archive) as bodies, BodyArchive(
            destination_archive, documents_per_shard=1000
        ) as output_bodies:
            columns = [str(row[1]) for row in destination.connection.execute("PRAGMA table_info(crawl_results)")]
            placeholders = ",".join("?" for _ in columns)
            sql = f"INSERT OR IGNORE INTO crawl_results ({','.join(columns)}) VALUES ({placeholders})"
            for doc_id, expected_url in wanted.items():
                row = source.execute("SELECT * FROM crawl_results WHERE doc_id = ?", (doc_id,)).fetchone()
                if row is None:
                    continue
                if str(row["original_url"]) != expected_url:
                    raise ValueError(f"source URL mismatch for doc_id {doc_id}")
                before = destination.connection.total_changes
                with destination.connection:
                    destination.connection.execute(sql, tuple(row[column] for column in columns))
                seeded_rows += int(destination.connection.total_changes > before)
                if bodies.contains(doc_id):
                    metadata = bodies.metadata(doc_id)
                    added = output_bodies.add(
                        BodyMetadata(
                            doc_id=doc_id,
                            original_url=str(metadata["original_url"]),
                            final_url=metadata["final_url"],
                            fetched_at=str(metadata["fetched_at"]),
                            content_type=metadata["content_type"],
                            encoding=metadata["encoding"],
                            declared_http_encoding=metadata["declared_http_encoding"],
                        ),
                        bodies.get_body(doc_id),
                    )
                    seeded_bodies += int(added)
            destination.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    finally:
        source.close()
    return {"requested": len(wanted), "seeded_rows": seeded_rows, "seeded_bodies": seeded_bodies}
