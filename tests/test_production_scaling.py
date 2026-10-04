from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.production.config import load_production_config
from src.production.chunking import build_chunk_partitions
from src.production.extraction import build_document_partitions
from src.production.manifest import ManifestStore, StageManifest
from src.production.ranges import plan_row_ranges
from src.production.safety import DiskAudit, evaluate_disk_guard, validate_execution_gate
from src.production.sampling import select_deterministic_sample
from src.production.state import PartitionCheckpoint
from src.storage.body_archive import BodyArchive, BodyMetadata
from src.processing.models import CleanDocument, DocumentSection, ExtractionStatus
from src.ingestion.checkpoint import CheckpointStore
from src.ingestion.models import CrawlResult, CrawlStatus


def _config() -> dict:
    return load_production_config("configs/production_pipeline.yaml")


def test_production_config_pins_validated_models_and_chunk_policy() -> None:
    config = _config()
    assert config["chunking"]["content_token_budget"] == 508
    assert config["chunking"]["maximum_tokens_with_specials"] == 512
    assert config["dense"]["revision"] == config["chunking"]["tokenizer_revision"]
    assert config["reranking"]["batch_size"] == 2
    assert config["safety"]["unrestricted_enabled"] is False


def test_row_ranges_are_bounded_complete_and_deterministic() -> None:
    first = plan_row_ranges(10_005, 1_000)
    second = plan_row_ranges(10_005, 1_000)
    assert first == second
    assert first[0].start == 0
    assert first[-1].stop == 10_005
    assert sum(item.count for item in first) == 10_005
    assert all(left.stop == right.start for left, right in zip(first, first[1:]))
    assert len({item.range_id for item in first}) == len(first)


def test_manifest_resume_signature_and_monotonic_progress(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    expected = StageManifest(
        stage="8A",
        source_sha256="a" * 64,
        config_sha256="b" * 64,
        stage_version=1,
        selected_count=3,
        tokenizer_revision="dense-revision",
    )
    store = ManifestStore(path, expected)
    store.record_progress(completed_count=1, failure_count=0)
    resumed = ManifestStore(path, expected)
    assert resumed.manifest.resume_count == 1
    assert resumed.manifest.completed_count == 1
    with pytest.raises(ValueError, match="backwards"):
        resumed.record_progress(completed_count=0, failure_count=0)
    resumed.record_progress(completed_count=2, failure_count=1)
    resumed.complete({"part-0": "c" * 64})
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["status"] == "COMPLETE"
    assert saved["completed_count"] + saved["failure_count"] == 3
    incompatible = StageManifest(
        stage="8A",
        source_sha256="different",
        config_sha256="b" * 64,
        stage_version=1,
        selected_count=3,
    )
    with pytest.raises(ValueError, match="signature mismatch"):
        ManifestStore(path, incompatible)


def test_partition_checkpoint_resume_has_no_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "stage.sqlite"
    signature = {"source": "abc", "stage": "chunks-v1"}
    with PartitionCheckpoint(path, signature=signature) as checkpoint:
        checkpoint.commit_partition(
            partition_id="part-000", item_ids=["1", "2"], output_sha256="a" * 64
        )
        assert checkpoint.integrity_check() == "ok"
    with PartitionCheckpoint(path, signature=signature) as resumed:
        assert resumed.pending(["1", "2", "3"]) == ["3"]
        resumed.commit_partition(
            partition_id="part-001", item_ids=["3"], output_sha256="b" * 64
        )
        resumed.commit_partition(
            partition_id="part-001", item_ids=["3"], output_sha256="b" * 64
        )
        assert resumed.count == 3
        with pytest.raises(ValueError, match="another output"):
            resumed.commit_partition(
                partition_id="part-other", item_ids=["3"], output_sha256="c" * 64
            )


def test_disk_guard_requires_reserve_and_headroom() -> None:
    audit = DiskAudit(path="D:/", total_bytes=1000, free_bytes=500, used_bytes=500)
    allowed = evaluate_disk_guard(
        audit,
        projected_growth_bytes=100,
        minimum_free_bytes=200,
        headroom_multiplier=1.5,
        maximum_projected_disk_fraction=0.8,
    )
    assert allowed["allowed"] is True
    blocked = evaluate_disk_guard(
        audit,
        projected_growth_bytes=350,
        minimum_free_bytes=200,
        headroom_multiplier=1.5,
        maximum_projected_disk_fraction=0.8,
    )
    assert blocked["allowed"] is False


def test_execution_gate_refuses_unconfirmed_unbounded_and_later_stages() -> None:
    config = _config()
    with pytest.raises(ValueError, match="positive --max-records"):
        validate_execution_gate(
            config,
            stage="8A",
            max_records=None,
            bounded_confirmation=None,
            full=False,
            full_confirmation=None,
        )
    with pytest.raises(ValueError, match="unrestricted production is disabled"):
        validate_execution_gate(
            config,
            stage="8A",
            max_records=None,
            bounded_confirmation=None,
            full=True,
            full_confirmation="confirm-full-production",
        )
    validate_execution_gate(
        config,
        stage="8A",
        max_records=5000,
        bounded_confirmation="confirm-bounded-phase8",
        full=False,
        full_confirmation=None,
    )
    with pytest.raises(ValueError, match="remains locked"):
        validate_execution_gate(
            config,
            stage="8C",
            max_records=5000,
            bounded_confirmation="confirm-bounded-phase8",
            full=False,
            full_confirmation=None,
        )


def test_deterministic_sample_is_order_stable(tmp_path: Path) -> None:
    path = tmp_path / "corpus.parquet"
    pq.write_table(
        pa.table(
            {
                "id": list(range(1, 31)),
                "url": [f"https://example.test/{value}" for value in range(1, 31)],
            }
        ),
        path,
    )
    first, first_summary = select_deterministic_sample(path, sample_size=7, seed="fixed")
    second, second_summary = select_deterministic_sample(path, sample_size=7, seed="fixed")
    assert first == second
    assert first_summary == second_summary
    assert len({row["doc_id"] for row in first}) == 7


def test_body_archive_rollover_remains_resumable(tmp_path: Path) -> None:
    archive_path = tmp_path / "archive"
    metadata = lambda doc_id: BodyMetadata(
        doc_id=doc_id,
        original_url=f"https://example.test/{doc_id}",
        final_url=None,
        fetched_at="2026-10-04T00:00:00+00:00",
        content_type="text/html",
        encoding="utf-8",
    )
    with BodyArchive(archive_path, documents_per_shard=2) as archive:
        for doc_id in range(1, 6):
            archive.add(metadata(doc_id), f"body-{doc_id}".encode())
    assert len(list(archive_path.glob("*.gz"))) == 3
    with BodyArchive(archive_path, documents_per_shard=2) as resumed:
        assert resumed.count == 5
        assert resumed.verify_integrity()["verified_bodies"] == 5


class _MockOffsetTokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        spans = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return {
            "input_ids": list(range(len(spans))),
            "offset_mapping": spans,
        }


def test_real_token_chunk_partitions_resume_without_duplicates(tmp_path: Path) -> None:
    source = tmp_path / "documents.parquet"
    rows = []
    for doc_id in (11, 12, 13):
        text = "heading\n\n" + " ".join(f"token-{index}" for index in range(700))
        document = CleanDocument(
            doc_id=doc_id,
            original_url=f"https://example.test/{doc_id}",
            final_url=None,
            title="heading",
            language_signal="latin",
            selected_encoding="utf-8",
            normalized_text=text,
            sections=[DocumentSection("article", text, 0, len(text), ["heading"])],
            extraction_method="semantic",
            extraction_status=ExtractionStatus.SUCCESS,
            raw_byte_length=len(text.encode()),
            raw_char_count=len(text),
            clean_char_count=len(text),
            paragraph_count=2,
            replacement_character_count=0,
            semantic_char_count=len(text),
            density_char_count=len(text),
            boilerplate_signal_count=0,
            raw_sha256="a" * 64,
            archive_shard="part.tar.gz",
            archive_member_offset=0,
            archive_checksum_sha256="a" * 64,
        )
        rows.append(document.as_dict())
    pq.write_table(pa.Table.from_pylist(rows), source)
    kwargs = {
        "tokenizer": _MockOffsetTokenizer(),
        "source_documents": source,
        "output_directory": tmp_path / "chunks",
        "checkpoint_path": tmp_path / "checkpoint.sqlite",
        "tokenizer_revision": "test-revision",
        "documents_per_partition": 2,
    }
    first = build_chunk_partitions(**kwargs)
    second = build_chunk_partitions(**kwargs)
    chunks = pq.read_table(list((tmp_path / "chunks").glob("*.parquet"))).to_pylist()
    assert first["documents_written_this_run"] == 3
    assert first["maximum_content_tokens"] <= 508
    assert second["initial_completed_documents"] == 3
    assert second["documents_written_this_run"] == 0
    assert len({row["chunk_id"] for row in chunks}) == len(chunks)
    for row in chunks:
        document = rows[row["doc_id"] - 11]
        assert row["raw_text"] == document["normalized_text"][row["start_offset"] : row["end_offset"]]


def test_production_extraction_partitions_resume_without_duplicates(tmp_path: Path) -> None:
    crawl = tmp_path / "crawl.sqlite"
    with CheckpointStore(crawl) as store:
        store.save(
            CrawlResult(
                doc_id=1,
                original_url="https://example.test/article",
                fetch_url="https://example.test/article",
                final_url="https://example.test/article",
                status=CrawlStatus.SUCCESS,
                http_status=200,
                content_type="text/html",
                encoding="utf-8",
                declared_http_encoding="utf-8",
                content_length=100,
                downloaded_bytes=100,
                attempt_count=1,
            )
        )
        store.save(
            CrawlResult(
                doc_id=2,
                original_url="https://blocked.test/article",
                fetch_url="https://blocked.test/article",
                status=CrawlStatus.ROBOTS_BLOCKED,
                attempt_count=0,
            )
        )
        store.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    archive_path = tmp_path / "archive"
    body = (
        "<html><title>Medical title</title><article><p>"
        + "measured biomedical content " * 100
        + "</p></article></html>"
    ).encode()
    with BodyArchive(archive_path, documents_per_shard=2) as archive:
        archive.add(
            BodyMetadata(
                doc_id=1,
                original_url="https://example.test/article",
                final_url="https://example.test/article",
                fetched_at="2026-10-04T00:00:00+00:00",
                content_type="text/html",
                encoding="utf-8",
                declared_http_encoding="utf-8",
            ),
            body,
        )
    kwargs = {
        "crawl_database": crawl,
        "archive_directory": archive_path,
        "extraction_config_path": "configs/extraction.yaml",
        "output_directory": tmp_path / "documents",
        "checkpoint_path": tmp_path / "extraction.sqlite",
        "rows_per_partition": 1,
    }
    first = build_document_partitions(**kwargs)
    second = build_document_partitions(**kwargs)
    records = pq.read_table(list((tmp_path / "documents").glob("*.parquet"))).to_pylist()
    assert first["rows_written_this_run"] == 2
    assert second["initial_completed_rows"] == 2
    assert second["rows_written_this_run"] == 0
    assert {row["doc_id"] for row in records} == {1, 2}
    assert {row["extraction_status"] for row in records} >= {
        ExtractionStatus.SUCCESS.value,
        ExtractionStatus.ROBOTS_BLOCKED.value,
    }
