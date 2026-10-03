from pathlib import Path

import pytest

from src.storage.body_archive import (
    BodyArchive,
    BodyChecksumError,
    BodyMetadata,
    DuplicateBodyError,
)


def _metadata(doc_id: int) -> BodyMetadata:
    return BodyMetadata(
        doc_id=doc_id,
        original_url=f"https://example.test/{doc_id}",
        final_url=f"https://example.test/{doc_id}",
        fetched_at="2026-10-03T00:00:00+00:00",
        content_type="text/html",
        encoding="utf-8",
        declared_http_encoding="utf-8",
    )


def test_archive_write_read_and_duplicate_protection(tmp_path) -> None:
    root = tmp_path / "archive"
    with BodyArchive(root, documents_per_shard=2) as archive:
        assert archive.add(_metadata(1), b"first body")
        assert not archive.add(_metadata(1), b"first body")
        with pytest.raises(DuplicateBodyError):
            archive.add(_metadata(1), b"different body")
        assert archive.get_body(1) == b"first body"
        assert archive.verify_integrity() == {
            "verified_bodies": 1,
            "original_bytes": len(b"first body"),
            "compressed_bytes": archive.metadata(1)["compressed_length"],
        }
    with BodyArchive(root, documents_per_shard=2) as reopened:
        assert reopened.get_body(1) == b"first body"
        assert reopened.metadata(1)["original_length"] == len(b"first body")


def test_interrupted_partial_shard_discards_unindexed_tail(tmp_path) -> None:
    root = tmp_path / "archive"
    archive = BodyArchive(root, documents_per_shard=10)
    archive.add(_metadata(1), b"recoverable")
    partial = next(root.glob("*.partial"))
    archive._handle.close()
    archive._handle = None
    archive.connection.close()
    with partial.open("ab") as handle:
        handle.write(b"unindexed crash tail")

    with BodyArchive(root, documents_per_shard=10) as recovered:
        row = recovered.metadata(1)
        assert partial.stat().st_size == row["member_offset"] + row["compressed_length"]
        assert recovered.get_body(1) == b"recoverable"


def test_interrupted_finalization_recovers_published_shard(tmp_path) -> None:
    root = tmp_path / "archive"
    archive = BodyArchive(root, documents_per_shard=10)
    archive.add(_metadata(1), b"published before index update")
    partial = next(root.glob("*.partial"))
    archive._handle.close()
    archive._handle = None
    archive.connection.close()
    final = partial.with_suffix(".gz")
    partial.replace(final)

    with BodyArchive(root, documents_per_shard=10) as recovered:
        assert recovered.metadata(1)["shard_name"] == final.name
        assert recovered.get_body(1) == b"published before index update"


def test_shard_finalization_and_checksum_verification(tmp_path) -> None:
    root = tmp_path / "archive"
    with BodyArchive(root, documents_per_shard=1) as archive:
        archive.add(_metadata(1), b"one")
        archive.add(_metadata(2), b"two")
    finalized = sorted(root.glob("*.gz"))
    assert len(finalized) == 2

    with BodyArchive(root, documents_per_shard=1) as archive:
        archive.connection.execute(
            "UPDATE bodies SET checksum_sha256 = ? WHERE doc_id = 1", ("0" * 64,)
        )
        archive.connection.commit()
        with pytest.raises(BodyChecksumError):
            archive.get_body(1)
