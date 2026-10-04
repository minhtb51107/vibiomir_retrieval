from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
import yaml

from .common import (
    atomic_write_json,
    file_sha256,
    write_deterministic_zip,
    write_submission_stream,
)


def load_submission_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("submission configuration must be a mapping")
    required = {"version", "organizer_schema", "inputs", "output", "variants"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"submission configuration missing: {sorted(missing)}")
    if int(config["organizer_schema"]["required_query_count"]) <= 0:
        raise ValueError("required_query_count must be positive")
    if not config["variants"]:
        raise ValueError("at least one submission variant is required")
    names = [str(row["name"]) for row in config["variants"]]
    if len(names) != len(set(names)):
        raise ValueError("submission variant names must be unique")
    return config


def _query_ids(path: str | Path, required_count: int) -> list[int]:
    table = pq.read_table(path, columns=["id"])
    identifiers = [int(value) for value in table.column("id").to_pylist()]
    if len(identifiers) != required_count:
        raise ValueError(
            f"query input has {len(identifiers)} rows; expected {required_count}"
        )
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("query input contains duplicate IDs")
    return identifiers


def _canonical_chunks(
    path: str | Path,
) -> tuple[dict[str, tuple[int, str]], set[int], set[tuple[int, str]]]:
    table = pq.read_table(path, columns=["chunk_id", "doc_id", "raw_text"])
    by_chunk: dict[str, tuple[int, str]] = {}
    document_ids: set[int] = set()
    document_text: set[tuple[int, str]] = set()
    for row in table.to_pylist():
        chunk_id = str(row["chunk_id"])
        value = (int(row["doc_id"]), str(row["raw_text"]))
        if chunk_id in by_chunk:
            raise ValueError(f"duplicate canonical chunk_id: {chunk_id}")
        by_chunk[chunk_id] = value
        document_ids.add(value[0])
        document_text.add(value)
    return by_chunk, document_ids, document_text


def _group_rows(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(path).to_pylist():
        grouped[int(row["query_id"])].append(row)
    return dict(grouped)


def _ordered_documents(rows: list[dict[str, Any]], depth: int) -> list[int]:
    ordered = sorted(rows, key=lambda row: (int(row["rank"]), int(row["doc_id"])))
    output: list[int] = []
    seen: set[int] = set()
    for row in ordered:
        doc_id = int(row["doc_id"])
        if doc_id in seen:
            raise ValueError(f"duplicate document row for doc_id {doc_id}")
        seen.add(doc_id)
        output.append(doc_id)
        if len(output) == depth:
            break
    return output


def _chunk_rank_field(rows: list[dict[str, Any]]) -> str:
    if rows and "selection_rank" in rows[0]:
        return "selection_rank"
    if rows and "rerank_rank" in rows[0]:
        return "rerank_rank"
    if rows and "rank" in rows[0]:
        return "rank"
    return "rerank_rank"


def _ordered_chunks(
    rows: list[dict[str, Any]],
    *,
    depth: int,
    canonical: dict[str, tuple[int, str]],
    with_chunk_id: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rank_field = _chunk_rank_field(rows)
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row[rank_field]),
            str(row["chunk_id"]),
        ),
    )
    output: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    seen_objects: set[tuple[int, str]] = set()
    duplicate_chunks = 0
    duplicate_objects = 0
    provenance_verified = 0
    for row in ordered:
        chunk_id = str(row["chunk_id"])
        doc_id = int(row["doc_id"])
        text = str(row["chunk_text"])
        if chunk_id in seen_chunks:
            duplicate_chunks += 1
            continue
        seen_chunks.add(chunk_id)
        expected = canonical.get(chunk_id)
        if expected is None:
            raise ValueError(f"unknown chunk_id in Phase 7 output: {chunk_id}")
        if expected != (doc_id, text):
            raise ValueError(f"chunk provenance mismatch for {chunk_id}")
        provenance_verified += 1
        object_key = (doc_id, text)
        if object_key in seen_objects:
            duplicate_objects += 1
            continue
        seen_objects.add(object_key)
        emitted = {"doc_id": doc_id, "chunk_text": text}
        if with_chunk_id:
            emitted["chunk_id"] = chunk_id
        output.append(emitted)
        if len(output) == depth:
            break
    return output, {
        "duplicate_chunk_ids_suppressed": duplicate_chunks,
        "duplicate_chunk_objects_suppressed": duplicate_objects,
        "source_rows_provenance_verified": provenance_verified,
        "emitted_chunks_provenance_verified": len(output),
    }


def _summarize_counts(values: list[int], intended: int) -> dict[str, int | float]:
    return {
        "total": sum(values),
        "minimum_per_query": min(values, default=0),
        "maximum_per_query": max(values, default=0),
        "mean_per_query": round(sum(values) / max(len(values), 1), 6),
        "queries_below_intended_depth": sum(value < intended for value in values),
    }


def _build_variant_records(
    *,
    query_ids: list[int],
    documents_by_query: dict[int, list[dict[str, Any]]],
    chunks_by_query: dict[int, list[dict[str, Any]]],
    canonical_chunks: dict[str, tuple[int, str]],
    canonical_document_ids: set[int],
    document_depth: int,
    chunk_depth: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    document_counts = []
    chunk_counts = []
    unique_document_ids: set[int] = set()
    unique_ranked_document_ids: set[int] = set()
    unique_chunk_document_ids: set[int] = set()
    suppression = defaultdict(int)
    invalid_documents: set[int] = set()
    for query_id in query_ids:
        documents = _ordered_documents(documents_by_query.get(query_id, []), document_depth)
        chunks, chunk_stats = _ordered_chunks(
            chunks_by_query.get(query_id, []),
            depth=chunk_depth,
            canonical=canonical_chunks,
        )
        invalid_documents.update(
            doc_id for doc_id in documents if doc_id not in canonical_document_ids
        )
        invalid_documents.update(
            row["doc_id"] for row in chunks if row["doc_id"] not in canonical_document_ids
        )
        for key, value in chunk_stats.items():
            suppression[key] += value
        document_counts.append(len(documents))
        chunk_counts.append(len(chunks))
        unique_ranked_document_ids.update(documents)
        unique_chunk_document_ids.update(int(row["doc_id"]) for row in chunks)
        unique_document_ids.update(documents)
        unique_document_ids.update(int(row["doc_id"]) for row in chunks)
        records.append(
            {
                "id": query_id,
                "relevant_docs": documents,
                "relevant_chunks": chunks,
            }
        )
    if invalid_documents:
        raise ValueError(f"invalid local document IDs: {sorted(invalid_documents)[:10]}")
    return records, {
        "query_count": len(records),
        "document_results": _summarize_counts(document_counts, document_depth),
        "chunk_results": _summarize_counts(chunk_counts, chunk_depth),
        "unique_doc_ids": len(unique_document_ids),
        "unique_relevant_doc_ids": len(unique_ranked_document_ids),
        "unique_chunk_doc_ids": len(unique_chunk_document_ids),
        "invalid_doc_ids": 0,
        "duplicate_doc_ids": 0,
        "chunk_provenance_mismatches": 0,
        **dict(suppression),
    }


def _input_hashes(
    query_path: Path,
    canonical_path: Path,
    chunk_path: Path,
    document_path: Path,
) -> dict[str, str]:
    return {
        "queries_sha256": file_sha256(query_path),
        "canonical_chunks_sha256": file_sha256(canonical_path),
        "phase7_chunks_sha256": file_sha256(chunk_path),
        "phase7_documents_sha256": file_sha256(document_path),
    }


def generate_all_variants(
    config: dict[str, Any],
    *,
    config_path: str | Path,
    output_directory: str | Path | None = None,
    write_manifest: bool = True,
) -> dict[str, Any]:
    required_count = int(config["organizer_schema"]["required_query_count"])
    query_path = Path(config["inputs"]["queries"])
    canonical_path = Path(config["inputs"]["canonical_chunks"])
    phase7_directory = Path(config["inputs"]["phase7_directory"])
    output = Path(output_directory or config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    query_ids = _query_ids(query_path, required_count)
    canonical_chunks, canonical_document_ids, _ = _canonical_chunks(canonical_path)
    document_depth = int(config["output"]["document_depth"])
    chunk_depth = int(config["output"]["chunk_depth"])
    config_hash = file_sha256(config_path)
    variants = []
    for variant in config["variants"]:
        chunk_path = phase7_directory / str(variant["chunk_source"])
        document_path = phase7_directory / str(variant["document_source"])
        records, statistics = _build_variant_records(
            query_ids=query_ids,
            documents_by_query=_group_rows(document_path),
            chunks_by_query=_group_rows(chunk_path),
            canonical_chunks=canonical_chunks,
            canonical_document_ids=canonical_document_ids,
            document_depth=document_depth,
            chunk_depth=chunk_depth,
        )
        json_path = output / f"{variant['name']}.json"
        write_submission_stream(json_path, iter(records))
        zip_path = output / f"{variant['name']}.zip"
        if bool(config["output"].get("write_deterministic_zip", True)):
            write_deterministic_zip(json_path, zip_path)
        entry = {
            "name": str(variant["name"]),
            "filename": json_path.name,
            "json_sha256": file_sha256(json_path),
            "json_bytes": json_path.stat().st_size,
            "zip_filename": zip_path.name if zip_path.exists() else None,
            "zip_sha256": file_sha256(zip_path) if zip_path.exists() else None,
            "zip_bytes": zip_path.stat().st_size if zip_path.exists() else None,
            "strategy": str(variant["strategy"]),
            "document_aggregation": str(variant["document_aggregation"]),
            "document_depth": document_depth,
            "chunk_depth": chunk_depth,
            "source_retrieval_artifacts": {
                "chunks": str(chunk_path),
                "documents": str(document_path),
            },
            "input_hashes": _input_hashes(
                query_path, canonical_path, chunk_path, document_path
            ),
            **statistics,
        }
        variants.append(entry)
    sums = []
    for entry in variants:
        sums.append(f"{entry['json_sha256']}  {entry['filename']}")
        if entry["zip_filename"]:
            sums.append(f"{entry['zip_sha256']}  {entry['zip_filename']}")
    (output / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "version": str(config["version"]),
        "generator_config_sha256": config_hash,
        "organizer_schema": dict(config["organizer_schema"]),
        "canonical_chunk_count": len(canonical_chunks),
        "canonical_document_count": len(canonical_document_ids),
        "variants": variants,
    }
    if write_manifest:
        artifact_directory = Path(config["output"]["artifact_directory"])
        atomic_write_json(artifact_directory / "submission_manifest.json", manifest)
    return manifest


def manifest_fingerprint(manifest: dict[str, Any]) -> str:
    relevant = [
        {
            "name": row["name"],
            "json_sha256": row["json_sha256"],
            "zip_sha256": row["zip_sha256"],
        }
        for row in manifest["variants"]
    ]
    return hashlib.sha256(
        json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
