from __future__ import annotations

import gzip
import hashlib
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


class DuplicateBodyError(ValueError):
    pass


class BodyChecksumError(IOError):
    pass


@dataclass(frozen=True, slots=True)
class BodyMetadata:
    doc_id: int
    original_url: str
    final_url: str | None
    fetched_at: str
    content_type: str | None
    encoding: str | None
    declared_http_encoding: str | None = None


INDEX_SQL = """
CREATE TABLE IF NOT EXISTS bodies (
    doc_id INTEGER PRIMARY KEY,
    shard_name TEXT NOT NULL,
    member_offset INTEGER NOT NULL,
    compressed_length INTEGER NOT NULL,
    original_length INTEGER NOT NULL,
    checksum_sha256 TEXT NOT NULL,
    original_url TEXT NOT NULL,
    final_url TEXT,
    fetched_at TEXT NOT NULL,
    content_type TEXT,
    encoding TEXT,
    declared_http_encoding TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_bodies_shard ON bodies(shard_name, member_offset);
"""


class BodyArchive:
    """Append independent gzip members and index them transactionally by doc_id."""

    def __init__(self, root: str | Path, *, documents_per_shard: int = 5_000):
        if documents_per_shard <= 0:
            raise ValueError("documents_per_shard must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.documents_per_shard = documents_per_shard
        self.connection = sqlite3.connect(self.root / "index.sqlite")
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(INDEX_SQL)
        self.connection.commit()
        self._recover_partial_shards()
        self._shard_number, self._count_in_shard = self._next_shard_state()
        self._handle = None

    def __enter__(self) -> BodyArchive:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM bodies").fetchone()[0])

    def contains(self, doc_id: int) -> bool:
        return self.connection.execute(
            "SELECT 1 FROM bodies WHERE doc_id = ?", (doc_id,)
        ).fetchone() is not None

    def add(self, metadata: BodyMetadata, body: bytes) -> bool:
        checksum = hashlib.sha256(body).hexdigest()
        existing = self.connection.execute(
            "SELECT checksum_sha256 FROM bodies WHERE doc_id = ?", (metadata.doc_id,)
        ).fetchone()
        if existing is not None:
            if str(existing["checksum_sha256"]) == checksum:
                return False
            raise DuplicateBodyError(
                f"doc_id {metadata.doc_id} already has a different body checksum"
            )
        if self._count_in_shard >= self.documents_per_shard:
            self.finalize_current_shard()
            self._shard_number += 1
            self._count_in_shard = 0

        partial_name = self._partial_name(self._shard_number)
        handle = self._open_handle(partial_name)
        compressed = gzip.compress(body, compresslevel=6, mtime=0)
        offset = handle.tell()
        handle.write(compressed)
        handle.flush()
        os.fsync(handle.fileno())
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO bodies (
                    doc_id, shard_name, member_offset, compressed_length,
                    original_length, checksum_sha256, original_url, final_url,
                    fetched_at, content_type, encoding, declared_http_encoding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.doc_id,
                    partial_name,
                    offset,
                    len(compressed),
                    len(body),
                    checksum,
                    metadata.original_url,
                    metadata.final_url,
                    metadata.fetched_at,
                    metadata.content_type,
                    metadata.encoding,
                    metadata.declared_http_encoding,
                ),
            )
        self._count_in_shard += 1
        return True

    def get_body(self, doc_id: int, *, verify_checksum: bool = True) -> bytes:
        row = self.connection.execute(
            "SELECT * FROM bodies WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            raise KeyError(doc_id)
        path = self.root / str(row["shard_name"])
        if not path.exists() and path.name.endswith(".partial"):
            finalized = path.with_suffix(".gz")
            if finalized.exists():
                path = finalized
        with path.open("rb") as handle:
            handle.seek(int(row["member_offset"]))
            compressed = handle.read(int(row["compressed_length"]))
        try:
            body = gzip.decompress(compressed)
        except (OSError, EOFError) as exc:
            raise BodyChecksumError(f"compressed member is corrupt for doc_id {doc_id}") from exc
        if len(body) != int(row["original_length"]):
            raise BodyChecksumError(f"length mismatch for doc_id {doc_id}")
        if verify_checksum and hashlib.sha256(body).hexdigest() != row["checksum_sha256"]:
            raise BodyChecksumError(f"checksum mismatch for doc_id {doc_id}")
        return body

    def metadata(self, doc_id: int) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM bodies WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            raise KeyError(doc_id)
        return row

    def iter_metadata(self) -> Iterator[sqlite3.Row]:
        yield from self.connection.execute("SELECT * FROM bodies ORDER BY doc_id")

    def verify_integrity(self) -> dict[str, int]:
        """Read and checksum every indexed member without materializing the archive."""
        verified = 0
        original_bytes = 0
        compressed_bytes = 0
        for row in self.iter_metadata():
            body = self.get_body(int(row["doc_id"]), verify_checksum=True)
            verified += 1
            original_bytes += len(body)
            compressed_bytes += int(row["compressed_length"])
        return {
            "verified_bodies": verified,
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
        }

    def finalize_current_shard(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None
        partial_name = self._partial_name(self._shard_number)
        partial = self.root / partial_name
        if not partial.exists():
            return
        final_name = self._final_name(self._shard_number)
        final = self.root / final_name
        os.replace(partial, final)
        with self.connection:
            self.connection.execute(
                "UPDATE bodies SET shard_name = ? WHERE shard_name = ?",
                (final_name, partial_name),
            )

    def close(self) -> None:
        if getattr(self, "connection", None) is None:
            return
        self.finalize_current_shard()
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        self.connection.close()
        self.connection = None  # type: ignore[assignment]

    def _open_handle(self, partial_name: str):
        if self._handle is None:
            self._handle = (self.root / partial_name).open("ab")
        return self._handle

    def _next_shard_state(self) -> tuple[int, int]:
        row = self.connection.execute(
            "SELECT shard_name, COUNT(*) AS count FROM bodies "
            "GROUP BY shard_name ORDER BY shard_name DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return 0, 0
        name = str(row["shard_name"])
        number = int(name.split("-")[1].split(".")[0])
        count = int(row["count"])
        if name.endswith(".gz") or count >= self.documents_per_shard:
            return number + 1, 0
        return number, count

    def _recover_partial_shards(self) -> None:
        indexed_partials = list(
            self.connection.execute(
                "SELECT shard_name, MAX(member_offset + compressed_length) AS valid_end "
                "FROM bodies WHERE shard_name LIKE '%.partial' GROUP BY shard_name"
            )
        )
        for row in indexed_partials:
            name = str(row["shard_name"])
            partial = self.root / name
            final_name = name.removesuffix(".partial") + ".gz"
            final = self.root / final_name
            if not partial.exists() and final.exists():
                with self.connection:
                    self.connection.execute(
                        "UPDATE bodies SET shard_name = ? WHERE shard_name = ?",
                        (final_name, name),
                    )
                continue
            if partial.exists():
                with partial.open("r+b") as handle:
                    handle.truncate(int(row["valid_end"] or 0))
        indexed_names = {
            str(row[0])
            for row in self.connection.execute("SELECT DISTINCT shard_name FROM bodies")
        }
        for partial in self.root.glob("shard-*.partial"):
            if partial.name not in indexed_names:
                partial.unlink()

    @staticmethod
    def _partial_name(number: int) -> str:
        return f"shard-{number:06d}.partial"

    @staticmethod
    def _final_name(number: int) -> str:
        return f"shard-{number:06d}.gz"
