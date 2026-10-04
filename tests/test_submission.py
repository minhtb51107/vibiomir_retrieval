from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from src.submission.common import file_sha256
from src.submission.generator import generate_all_variants, load_submission_config
from src.submission.validator import (
    SubmissionValidationError,
    canonical_provenance,
    expected_query_ids,
    load_submission_payload,
    validate_submission,
)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    queries = tmp_path / "queries.parquet"
    chunks = tmp_path / "chunks.parquet"
    phase7 = tmp_path / "phase7"
    _write_parquet(queries, [{"id": 2, "query": "hai"}, {"id": 1, "query": "một"}])
    _write_parquet(
        chunks,
        [
            {"chunk_id": "c1", "doc_id": 10, "raw_text": "Nguồn một"},
            {"chunk_id": "c2", "doc_id": 10, "raw_text": "Nguồn hai"},
            {"chunk_id": "c3", "doc_id": 20, "raw_text": "中文来源"},
        ],
    )
    chunk_rows = []
    document_rows = []
    for query_id in [2, 1]:
        chunk_rows.extend(
            [
                {
                    "query_id": query_id,
                    "chunk_id": "c1",
                    "doc_id": 10,
                    "chunk_text": "Nguồn một",
                    "selection_rank": 1,
                },
                {
                    "query_id": query_id,
                    "chunk_id": "c2",
                    "doc_id": 10,
                    "chunk_text": "Nguồn hai",
                    "selection_rank": 2,
                },
                # A distinct chunk_id but identical organizer object is removed.
                {
                    "query_id": query_id,
                    "chunk_id": "c1-copy",
                    "doc_id": 10,
                    "chunk_text": "Nguồn một",
                    "selection_rank": 3,
                },
                {
                    "query_id": query_id,
                    "chunk_id": "c3",
                    "doc_id": 20,
                    "chunk_text": "中文来源",
                    "selection_rank": 4,
                },
            ]
        )
        document_rows.extend(
            [
                {"query_id": query_id, "doc_id": 10, "rank": 1},
                {"query_id": query_id, "doc_id": 20, "rank": 2},
            ]
        )
    # Canonical duplicate-text chunks are legal, but duplicate emitted objects are not.
    canonical = pq.read_table(chunks).to_pylist()
    canonical.append({"chunk_id": "c1-copy", "doc_id": 10, "raw_text": "Nguồn một"})
    _write_parquet(chunks, canonical)
    _write_parquet(phase7 / "selected_chunks.parquet", chunk_rows)
    _write_parquet(phase7 / "selected_docs.parquet", document_rows)
    config = {
        "version": "test-v1",
        "organizer_schema": {
            "required_query_count": 2,
            "source_url": "https://example.invalid/schema",
        },
        "inputs": {
            "queries": str(queries),
            "canonical_chunks": str(chunks),
            "phase7_directory": str(phase7),
        },
        "output": {
            "directory": str(tmp_path / "out"),
            "artifact_directory": str(tmp_path / "artifacts"),
            "document_depth": 2,
            "chunk_depth": 4,
            "write_deterministic_zip": True,
        },
        "validation": {"require_exact_query_order": True},
        "variants": [
            {
                "name": "variant",
                "strategy": "pure",
                "document_aggregation": "best_chunk",
                "chunk_source": "selected_chunks.parquet",
                "document_source": "selected_docs.parquet",
            }
        ],
    }
    config_path = tmp_path / "submission.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, config


def test_organizer_schema_mapping_all_query_coverage_and_duplicate_prevention(
    tmp_path: Path,
) -> None:
    config_path, config = _fixture(tmp_path)
    manifest = generate_all_variants(config, config_path=config_path)
    payload = load_submission_payload(tmp_path / "out" / "variant.json")
    assert [row["id"] for row in payload] == [2, 1]
    assert list(payload[0]) == ["id", "relevant_docs", "relevant_chunks"]
    assert payload[0]["relevant_docs"] == [10, 20]
    assert payload[0]["relevant_chunks"] == [
        {"doc_id": 10, "chunk_text": "Nguồn một"},
        {"doc_id": 10, "chunk_text": "Nguồn hai"},
        {"doc_id": 20, "chunk_text": "中文来源"},
    ]
    stats = manifest["variants"][0]
    assert stats["duplicate_chunk_objects_suppressed"] == 2
    assert stats["emitted_chunks_provenance_verified"] == 6


def test_generation_is_byte_deterministic_for_json_and_zip(tmp_path: Path) -> None:
    config_path, config = _fixture(tmp_path)
    first = generate_all_variants(config, config_path=config_path)
    json_hash = file_sha256(tmp_path / "out" / "variant.json")
    zip_hash = file_sha256(tmp_path / "out" / "variant.zip")
    second = generate_all_variants(config, config_path=config_path)
    assert first["variants"][0]["json_sha256"] == json_hash
    assert second["variants"][0]["json_sha256"] == json_hash
    assert file_sha256(tmp_path / "out" / "variant.zip") == zip_hash


def test_multiple_variants_are_generated_from_declared_sources(tmp_path: Path) -> None:
    config_path, config = _fixture(tmp_path)
    second = dict(config["variants"][0])
    second["name"] = "variant_two"
    second["document_aggregation"] = "top3_mean"
    config["variants"].append(second)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest = generate_all_variants(config, config_path=config_path)
    assert [row["name"] for row in manifest["variants"]] == [
        "variant",
        "variant_two",
    ]
    assert (tmp_path / "out" / "variant_two.json").exists()
    assert (tmp_path / "out" / "variant_two.zip").exists()


def test_validator_positive_case_and_optional_chunk_order(tmp_path: Path) -> None:
    config_path, config = _fixture(tmp_path)
    generate_all_variants(config, config_path=config_path)
    valid_docs, valid_chunks = canonical_provenance(config["inputs"]["canonical_chunks"])
    queries = expected_query_ids(config["inputs"]["queries"])
    result = validate_submission(
        tmp_path / "out" / "variant.zip",
        expected_queries=queries,
        valid_document_ids=valid_docs,
        valid_chunks=valid_chunks,
    )
    assert result["valid"] is True
    assert result["query_count"] == 2
    assert result["chunk_provenance_mismatches"] == 0
    payload = load_submission_payload(tmp_path / "out" / "variant.json")
    payload[0]["relevant_chunks"][0]["chunk_order"] = 0
    ordered_path = tmp_path / "ordered.json"
    ordered_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert validate_submission(
        ordered_path,
        expected_queries=queries,
        valid_document_ids=valid_docs,
        valid_chunks=valid_chunks,
    )["valid"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop(), "query count"),
        (
            lambda value: value[0]["relevant_docs"].append(
                value[0]["relevant_docs"][0]
            ),
            "duplicate document",
        ),
        (
            lambda value: value[0]["relevant_docs"].__setitem__(0, 999),
            "unknown local document",
        ),
        (
            lambda value: value[0]["relevant_chunks"][0].__setitem__(
                "chunk_text", "rewritten"
            ),
            "provenance mismatch",
        ),
        (
            lambda value: value[0]["relevant_chunks"].append(
                dict(value[0]["relevant_chunks"][0])
            ),
            "duplicate chunk object",
        ),
    ],
)
def test_validator_rejects_invalid_submissions(
    tmp_path: Path, mutation, message: str
) -> None:
    config_path, config = _fixture(tmp_path)
    generate_all_variants(config, config_path=config_path)
    payload = load_submission_payload(tmp_path / "out" / "variant.json")
    mutation(payload)
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    valid_docs, valid_chunks = canonical_provenance(config["inputs"]["canonical_chunks"])
    with pytest.raises(SubmissionValidationError, match=message):
        validate_submission(
            invalid,
            expected_queries=expected_query_ids(config["inputs"]["queries"]),
            valid_document_ids=valid_docs,
            valid_chunks=valid_chunks,
        )


def test_generator_rejects_chunk_doc_or_text_mismatch(tmp_path: Path) -> None:
    config_path, config = _fixture(tmp_path)
    path = Path(config["inputs"]["phase7_directory"]) / "selected_chunks.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["doc_id"] = 20
    _write_parquet(path, rows)
    with pytest.raises(ValueError, match="chunk provenance mismatch"):
        generate_all_variants(config, config_path=config_path)


def test_config_rejects_duplicate_variant_names(tmp_path: Path) -> None:
    config_path, config = _fixture(tmp_path)
    config["variants"].append(dict(config["variants"][0]))
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="variant names must be unique"):
        load_submission_config(config_path)
