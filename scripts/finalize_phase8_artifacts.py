#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.indexing.embedder import file_sha256
from src.production.config import load_production_config
from src.production.manifest import ManifestStore, StageManifest
from src.storage.body_archive import BodyArchive


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(file_sha256(item).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _sqlite_integrity(path: str | Path) -> str:
    connection = sqlite3.connect(path)
    try:
        return str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize compact Phase 8 evidence.")
    parser.add_argument("--config", default="configs/production_pipeline.yaml")
    parser.add_argument("--crawl-summary", required=True)
    parser.add_argument("--crawl-resume-summary", required=True)
    parser.add_argument("--extraction-summary", required=True)
    parser.add_argument("--extraction-resume-summary", required=True)
    parser.add_argument("--document-merge-summary", required=True)
    parser.add_argument("--chunk-summary", required=True)
    parser.add_argument("--chunk-resume-summary", required=True)
    parser.add_argument("--chunk-merge-summary", required=True)
    parser.add_argument("--token-validation", required=True)
    parser.add_argument("--index-summary", required=True)
    parser.add_argument("--ann-summary", required=True)
    parser.add_argument("--crawl-database", required=True)
    parser.add_argument("--archive-directory", required=True)
    parser.add_argument("--extraction-checkpoint", required=True)
    parser.add_argument("--chunk-checkpoint", required=True)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    config = load_production_config(args.config)
    crawl = _read(args.crawl_summary)
    crawl_resume = _read(args.crawl_resume_summary)
    extraction = _read(args.extraction_summary)
    extraction_resume = _read(args.extraction_resume_summary)
    document_merge = _read(args.document_merge_summary)
    chunking = _read(args.chunk_summary)
    chunk_resume = _read(args.chunk_resume_summary)
    chunk_merge = _read(args.chunk_merge_summary)
    token_validation = _read(args.token_validation)
    indexes = _read(args.index_summary)
    ann = _read(args.ann_summary)

    selected = int(crawl["run"]["selected"])
    if selected != 5000 or int(crawl["database"]["rows"]) != selected:
        raise RuntimeError("bounded crawl did not complete exactly 5,000 records")
    if int(crawl_resume["run"]["scheduled"]) != 0 or int(
        crawl_resume["run"]["skipped_existing"]
    ) != selected:
        raise RuntimeError("crawl restart did not skip all completed rows")
    if int(extraction_resume["rows_written_this_run"]) != 0:
        raise RuntimeError("extraction restart rewrote completed rows")
    if int(chunk_resume["documents_written_this_run"]) != 0:
        raise RuntimeError("chunk restart rewrote completed rows")
    if document_merge["document_count"] != document_merge["unique_doc_ids"]:
        raise RuntimeError("duplicate doc_id detected in cleaned documents")
    if chunk_merge["chunk_count"] != chunk_merge["unique_chunk_ids"]:
        raise RuntimeError("duplicate chunk_id detected in validated chunks")
    if token_validation["result"] != "PASS":
        raise RuntimeError("BGE token/offset validation failed")
    if indexes["metadata"]["mapping_verification"]["mismatches"]:
        raise RuntimeError("chunk metadata row mapping verification failed")
    if indexes["sparse"]["mapping_verification"]["mismatches"]:
        raise RuntimeError("sparse row mapping verification failed")

    with BodyArchive(args.archive_directory) as archive:
        archive_integrity = archive.verify_integrity()
    crawl_integrity = _sqlite_integrity(args.crawl_database)
    extraction_integrity = _sqlite_integrity(args.extraction_checkpoint)
    chunk_integrity = _sqlite_integrity(args.chunk_checkpoint)
    if {crawl_integrity, extraction_integrity, chunk_integrity} != {"ok"}:
        raise RuntimeError("one or more Phase 8 SQLite stores failed integrity checking")

    success = int(extraction["status_counts"].get("SUCCESS", 0))
    output_hashes = {
        "archive_tree": _tree_hash(Path(args.archive_directory)),
        "chunks": file_sha256(args.chunks),
        "chunk_metadata": file_sha256(indexes["metadata"]["path"]),
        "crawl_database": file_sha256(args.crawl_database),
        "documents": file_sha256(args.documents),
        "sparse_index": file_sha256(indexes["sparse"]["index_path"]),
    }
    expected_manifest = StageManifest(
        stage="8A-8B-bounded",
        source_sha256=file_sha256(config["source"]["corpus"]),
        config_sha256=file_sha256(args.config),
        stage_version=1,
        selected_count=selected,
        model_revision=config["dense"]["revision"],
        tokenizer_revision=config["chunking"]["tokenizer_revision"],
    )
    manifest_store = ManifestStore(args.manifest, expected_manifest)
    manifest_store.record_progress(
        completed_count=success,
        failure_count=selected - success,
    )
    manifest_store.complete(output_hashes)
    manifested = manifest_store.manifest.as_dict()
    manifest_store = ManifestStore(args.manifest, expected_manifest)
    manifested = manifest_store.manifest.as_dict()

    storage = {
        "crawl_database_bytes": Path(args.crawl_database).stat().st_size,
        "archive_bytes": _tree_size(Path(args.archive_directory)),
        "documents_bytes": Path(args.documents).stat().st_size,
        "chunks_bytes": Path(args.chunks).stat().st_size,
        "embedding_bytes": 0,
        "faiss_bytes": 0,
        "dense_metadata_bytes": int(indexes["metadata"]["size_bytes"]),
        "sparse_index_bytes": int(indexes["sparse"]["index_size_bytes"]),
    }
    storage["total_measured_output_bytes"] = sum(storage.values())
    bounded = {
        "phase": 8,
        "evidence_policy": "operational only; not relevance evaluation",
        "tiers": {
            "prior_readiness_urls": 1225,
            "bounded_live_urls": selected,
            "real_bge_chunks": int(chunk_merge["chunk_count"]),
            "ann_vectors": int(ann["vector_count"]),
        },
        "crawl": crawl,
        "extraction": extraction,
        "documents": document_merge,
        "chunking": chunking,
        "chunks": chunk_merge,
        "token_validation": token_validation,
        "indexes": indexes,
        "ann": ann,
        "archive_integrity": archive_integrity,
        "storage": storage,
    }
    resume = {
        "crawl": {
            "selected": crawl_resume["run"]["selected"],
            "scheduled": crawl_resume["run"]["scheduled"],
            "skipped_existing": crawl_resume["run"]["skipped_existing"],
            "sqlite_integrity": crawl_integrity,
        },
        "archive": {**archive_integrity, "checksum_verification": "PASS"},
        "extraction": {
            "initial_completed_rows": extraction_resume["initial_completed_rows"],
            "rows_written_on_restart": extraction_resume["rows_written_this_run"],
            "sqlite_integrity": extraction_integrity,
        },
        "chunking": {
            "initial_completed_documents": chunk_resume["initial_completed_documents"],
            "documents_written_on_restart": chunk_resume["documents_written_this_run"],
            "sqlite_integrity": chunk_integrity,
        },
        "mapping": {
            "unique_documents": document_merge["unique_doc_ids"],
            "unique_chunks": chunk_merge["unique_chunk_ids"],
            "dense_rows_checked": indexes["metadata"]["mapping_verification"]["rows_checked"],
            "dense_mismatches": indexes["metadata"]["mapping_verification"]["mismatches"],
            "sparse_rows_checked": indexes["sparse"]["mapping_verification"]["rows_checked"],
            "sparse_mismatches": indexes["sparse"]["mapping_verification"]["mismatches"],
        },
        "manifest": manifested,
        "result": "PASS",
    }
    artifacts = Path(config["paths"]["artifacts"])
    _atomic(artifacts / "bounded_benchmark.json", bounded)
    _atomic(artifacts / "resume_validation.json", resume)
    _atomic(artifacts / "stage_manifest.json", manifested)
    print(json.dumps({"bounded": bounded["tiers"], "resume": resume}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
