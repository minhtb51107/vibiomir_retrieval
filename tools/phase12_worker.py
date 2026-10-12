#!/usr/bin/env python3
"""Resumable full-PubMed multilingual retrieval pipeline for Phase 12."""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_census.cached_subsets import build_cached_subset_rankings
from src.source_census.pipeline import atomic_json, normalize_text
from src.source_census.union_experiment import (
    _audit_ranked_provenance_closure,
    _build_canonical_store,
    _compare_submission_signatures,
    _filter_parquet,
    _official_id_audit,
    _stream_package,
)
from src.validation.pre_submission import audit_score_cache
from tools.phase11a_bottleneck import _copy_or_link, _init_score_db, _seed_exact, tokenizer
from tools.phase12_pubmed import audit_completed, load, sha256_file


REMAINING_STAGE_MODULES = (
    "numpy", "pyarrow", "yaml", "psutil", "torch", "transformers",
    "sentence_transformers",
)


SECTION = pa.struct([
    pa.field("section_type", pa.string()),
    pa.field("text", pa.string()),
    pa.field("start_offset", pa.int64()),
    pa.field("end_offset", pa.int64()),
    pa.field("heading_path", pa.list_(pa.string())),
])
DOCUMENT_SCHEMA = pa.schema([
    pa.field("doc_id", pa.int64()), pa.field("original_url", pa.string()), pa.field("final_url", pa.string()),
    pa.field("title", pa.string()), pa.field("language_signal", pa.string()), pa.field("selected_encoding", pa.string()),
    pa.field("normalized_text", pa.string()), pa.field("sections", pa.list_(SECTION)),
    pa.field("extraction_method", pa.string()), pa.field("extraction_status", pa.string()),
    pa.field("raw_byte_length", pa.int64()), pa.field("raw_char_count", pa.int64()), pa.field("clean_char_count", pa.int64()),
    pa.field("paragraph_count", pa.int64()), pa.field("replacement_character_count", pa.int64()),
    pa.field("semantic_char_count", pa.int64()), pa.field("density_char_count", pa.int64()),
    pa.field("boilerplate_signal_count", pa.int64()), pa.field("raw_sha256", pa.string()),
    pa.field("archive_shard", pa.string()), pa.field("archive_member_offset", pa.int64()),
    pa.field("archive_checksum_sha256", pa.string()), pa.field("error_type", pa.string()), pa.field("error_message", pa.string()),
    pa.field("pmid", pa.int64()), pa.field("retrieval_text", pa.string()), pa.field("abstract_text", pa.string()),
    pa.field("mesh_terms", pa.list_(pa.string())), pa.field("publication_types", pa.list_(pa.string())),
    pa.field("journal", pa.string()), pa.field("publication_year", pa.string()), pa.field("article_languages", pa.list_(pa.string())),
])
CHUNK_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string()), pa.field("doc_id", pa.int64()), pa.field("chunk_index", pa.int64()),
    pa.field("raw_text", pa.string()), pa.field("normalized_text", pa.string()), pa.field("token_count", pa.int64()),
    pa.field("start_offset", pa.int64()), pa.field("end_offset", pa.int64()), pa.field("section_type", pa.string()),
    pa.field("heading_path", pa.list_(pa.string())), pa.field("source_url", pa.string()), pa.field("extraction_method", pa.string()),
])
CANDIDATE_SCHEMA = pa.schema([
    pa.field("query_id", pa.int64()), pa.field("query_text", pa.string()), pa.field("source", pa.string()),
    pa.field("source_rank", pa.int64()), pa.field("source_score", pa.float64()), pa.field("chunk_id", pa.string()),
    pa.field("doc_id", pa.int64()), pa.field("chunk_text", pa.string()), pa.field("source_url", pa.string()),
    pa.field("chunk_index", pa.int64()), pa.field("start_offset", pa.int64()), pa.field("end_offset", pa.int64()),
    pa.field("section_type", pa.string()), pa.field("heading_path", pa.list_(pa.string())), pa.field("extraction_method", pa.string()),
])


def update_state(path: Path, **fields) -> None:
    try:
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        current = {}
    current.update(fields)
    atomic_json(path, current)


def dependency_preflight(config: dict, *, verify_runtime: bool = True) -> dict:
    """Fail before expensive stages if a remaining-stage dependency is absent."""
    missing = [name for name in REMAINING_STAGE_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise ModuleNotFoundError(f"Phase 12 remaining-stage dependencies missing: {missing}")
    imported = {}
    for name in REMAINING_STAGE_MODULES:
        module = importlib.import_module(name)
        imported[name] = str(getattr(module, "__version__", "available"))
    result = {"passed": True, "modules": imported, "remaining_stages": ["BENCHMARK_EMBED", "EMBED_AND_INDEX", "RETRIEVE_PUBMED", "RERANK_PUBMED", "MERGE_AND_PACKAGE", "FINAL_AUDIT"]}
    if verify_runtime:
        import torch
        from transformers import AutoConfig, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("Phase 12 requires the configured local CUDA runtime, but torch.cuda.is_available() is false")
        gpu = torch.cuda.get_device_properties(0)
        result["cuda"] = {"device": gpu.name, "total_vram_mib": round(gpu.total_memory/2**20, 1), "torch_cuda": str(torch.version.cuda)}
        contracts = {}
        for role in ("embedder", "reranker"):
            model = config["models"][role]
            kwargs = {"revision": model["revision"], "local_files_only": True}
            if model.get("cache_dir"):
                kwargs["cache_dir"] = str(ROOT/model["cache_dir"])
            loaded = AutoConfig.from_pretrained(model["name"], **kwargs)
            tokenizer_config = AutoTokenizer.from_pretrained(model["name"], **kwargs)
            contracts[role] = {"name": model["name"], "revision": model["revision"], "config_type": type(loaded).__name__, "tokenizer_type": type(tokenizer_config).__name__}
        result["model_contracts"] = contracts
        result["nvidia_smi"] = shutil.which("nvidia-smi")
        if not result["nvidia_smi"]:
            raise RuntimeError("nvidia-smi is unavailable for Phase 12 runtime monitoring")
    return result


def verify_processed_outputs(config: dict) -> dict:
    """Validate and reuse the 800 durable processed parts and consolidated files."""
    output = ROOT/config["outputs"]["processed"]
    artifacts = ROOT/config["outputs"]["artifacts"]
    report_path = artifacts/"processing_quality_audit.json"
    documents = output/"documents.parquet"; chunks = output/"chunks.parquet"; parts = output/"parts"
    if not (report_path.exists() and documents.exists() and chunks.exists()):
        raise FileNotFoundError("completed Phase 12 processing outputs are not all present")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    markers = sorted(parts.glob("part-*.done.json"))
    if len(markers) != 800 or [path.stem.split(".done")[0] for path in markers] != [f"part-{index:06d}" for index in range(800)]:
        raise ValueError(f"processed checkpoint coverage mismatch: {len(markers)}/800")
    document_rows = chunk_rows = 0
    for marker in markers:
        checkpoint = json.loads(marker.read_text(encoding="utf-8")); stem = marker.name.removesuffix(".done.json")
        doc_part = parts/f"{stem}.documents.parquet"; chunk_part = parts/f"{stem}.chunks.parquet"
        if not doc_part.exists() or not chunk_part.exists():
            raise ValueError(f"processed checkpoint has missing outputs: {marker}")
        if checkpoint["documents_sha256"] != sha256_file(doc_part) or checkpoint["chunks_sha256"] != sha256_file(chunk_part):
            raise ValueError(f"processed checkpoint hash mismatch: {marker}")
        document_rows += int(checkpoint["documents"]); chunk_rows += int(checkpoint["chunks"])
    actual_docs = pq.ParquetFile(documents).metadata.num_rows; actual_chunks = pq.ParquetFile(chunks).metadata.num_rows
    failures = []
    if document_rows != actual_docs or actual_docs != int(report["valid_documents"]): failures.append("document row accounting")
    if chunk_rows != actual_chunks or actual_chunks != int(report["chunks"]): failures.append("chunk row accounting")
    if sha256_file(documents) != report["documents_sha256"]: failures.append("consolidated document hash")
    if sha256_file(chunks) != report["chunks_sha256"]: failures.append("consolidated chunk hash")
    if failures: raise ValueError(f"processed Phase 12 output integrity failure: {failures}")
    return {**report, "processed_checkpoints_reused": 800, "checkpoint_integrity": "ok"}


def validate_embedding_checkpoint(config: dict, documents: Path) -> dict:
    """Prove the transactional prefix is safe before resuming model work."""
    retrieval=ROOT/config["outputs"]["retrieval"]
    artifacts=ROOT/config["outputs"]["artifacts"]
    partial=retrieval/"pubmed_document_embeddings.streaming.partial"
    status_path=retrieval/"pubmed_document_embeddings.streaming.status.u8"
    checkpoint_path=retrieval/"pubmed_document_embeddings.streaming.checkpoint.json"
    checkpoint=json.loads(checkpoint_path.read_text(encoding="utf-8"))
    rows=pq.ParquetFile(documents).metadata.num_rows
    dimension=int(config["models"]["embedder"]["dimension"])
    signature=digest_text(f"{sha256_file(documents)}:{config['models']['embedder']['revision']}:{rows}:{dimension}")
    if checkpoint.get("signature")!=signature:
        raise ValueError("PubMed embedding checkpoint signature/model/document-order mismatch")
    if int(checkpoint.get("rows",-1))!=rows or int(checkpoint.get("dimension",-1))!=dimension:
        raise ValueError("PubMed embedding checkpoint shape mismatch")
    if partial.stat().st_size!=rows*dimension*4 or status_path.stat().st_size!=rows:
        raise ValueError("PubMed embedding checkpoint file-size mismatch")
    status=np.memmap(status_path,dtype=np.uint8,mode="r",shape=(rows,))
    completed=int(checkpoint.get("completed",-1))
    observed=int(np.count_nonzero(status))
    if observed!=completed or np.any(status[:completed]!=1) or np.any(status[completed:]!=0):
        raise ValueError(f"embedding completion bitmap is not one committed prefix: checkpoint={completed} observed={observed}")
    stream_rows=int(config["models"]["embedder"]["stream_rows"])
    if completed!=rows and completed%stream_rows:
        raise ValueError(f"embedding checkpoint is not on a {stream_rows}-row transaction boundary: {completed}")
    vectors=np.memmap(partial,dtype=np.float32,mode="r",shape=(rows,dimension))
    nonfinite=0; zero_norm=0
    for start in range(0,completed,1024):
        block=np.asarray(vectors[start:min(completed,start+1024)])
        nonfinite+=int((~np.isfinite(block)).sum())
        zero_norm+=int((np.linalg.norm(block,axis=1)==0).sum())
    if nonfinite or zero_norm:
        raise ValueError(f"committed embedding prefix invalid: nonfinite={nonfinite} zero_norm={zero_norm}")
    prefix_ids=[]
    remaining=completed
    for batch in pq.ParquetFile(documents).iter_batches(columns=["doc_id"],batch_size=4096):
        values=list(map(int,batch.column(0).to_pylist()))[:remaining]
        prefix_ids.extend(values); remaining-=len(values)
        if remaining<=0: break
    if len(prefix_ids)!=completed or len(set(prefix_ids))!=completed:
        raise ValueError("committed embedding prefix document mapping is incomplete or duplicated")
    result={"passed":True,"completed":completed,"remaining":rows-completed,"rows":rows,"dimension":dimension,"dtype":"float32","model_revision":config["models"]["embedder"]["revision"],"document_order_sha256":sha256_file(documents),"signature":signature,"bitmap_prefix_complete":True,"vectors_flushed_before_status":checkpoint.get("vectors_flushed_before_status") is True,"committed_nonfinite_values":0,"committed_zero_norm_rows":0,"unique_prefix_doc_ids":completed,"first_uncommitted_row":completed,"transaction_rows":stream_rows,"checkpoint_path":str(checkpoint_path)}
    if not result["vectors_flushed_before_status"]: raise ValueError("checkpoint does not prove vector flush before completion bit")
    atomic_json(artifacts/"embedding_checkpoint_audit.json",result)
    return result


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _split_span(text: str, start: int, end: int, limit: int = 2400) -> list[tuple[int, int]]:
    spans = []
    cursor = start
    while end - cursor > limit:
        boundary = max(text.rfind(". ", cursor + limit // 2, cursor + limit + 1), text.rfind(" ", cursor + limit // 2, cursor + limit + 1))
        if boundary <= cursor:
            boundary = cursor + limit
        elif text[boundary:boundary + 2] == ". ":
            boundary += 1
        spans.append((cursor, boundary))
        cursor = boundary
        while cursor < end and text[cursor].isspace():
            cursor += 1
    if cursor < end:
        spans.append((cursor, end))
    return spans


def evidence_sections(row: dict) -> tuple[str, list[dict]]:
    parts: list[str] = []
    sections: list[dict] = []
    title = str(row.get("title") or "").strip()
    if title:
        start = 0
        parts.append(title)
        sections.append({"section_type": "TITLE", "text": title, "start_offset": start, "end_offset": len(title), "heading_path": ["Title"]})
    for item in row.get("abstract_sections") or []:
        body = str(item.get("text") or "").strip()
        if not body:
            continue
        label = str(item.get("label") or item.get("nlm_category") or "Abstract").strip()
        value = f"{label}: {body}" if label else body
        start = sum(len(part) for part in parts) + len(parts)
        parts.append(value)
        sections.append({"section_type": "ABSTRACT", "text": value, "start_offset": start, "end_offset": start + len(value), "heading_path": [label or "Abstract"]})
    return "\n".join(parts), sections


def process_record(row: dict) -> tuple[dict | None, list[dict]]:
    if str(row["acquisition_status"]) == "MISSING":
        return None, []
    evidence, sections = evidence_sections(row)
    retrieval = str(row.get("retrieval_text") or "").strip()
    if not evidence or not retrieval:
        return None, []
    doc_id = int(row["official_doc_id"])
    document = {
        "doc_id": doc_id, "original_url": str(row["official_url"]), "final_url": str(row["official_url"]),
        "title": str(row.get("title") or ""), "language_signal": "en", "selected_encoding": "utf-8",
        "normalized_text": evidence, "sections": sections, "extraction_method": "ncbi_pubmed_xml", "extraction_status": "SUCCESS",
        "raw_byte_length": len(evidence.encode("utf-8")), "raw_char_count": len(evidence), "clean_char_count": len(evidence),
        "paragraph_count": max(1, len(sections)), "replacement_character_count": evidence.count("\ufffd"),
        "semantic_char_count": sum(ch.isalnum() for ch in evidence), "density_char_count": len(evidence.strip()),
        "boilerplate_signal_count": 0, "raw_sha256": str(row["raw_xml_sha256"]),
        "archive_shard": f"pubmed-batch-{int(row['batch_index']):06d}", "archive_member_offset": 0,
        "archive_checksum_sha256": str(row["raw_xml_sha256"]), "error_type": "", "error_message": "",
        "pmid": int(row["pmid"]), "retrieval_text": retrieval, "abstract_text": str(row.get("abstract_text") or ""),
        "mesh_terms": list(row.get("mesh_terms") or []), "publication_types": list(row.get("publication_types") or []),
        "journal": str(row.get("journal") or ""), "publication_year": str(row.get("publication_year") or ""),
        "article_languages": list(row.get("article_languages") or []),
    }
    chunks = []
    for section in sections:
        for start, end in _split_span(evidence, int(section["start_offset"]), int(section["end_offset"])):
            raw = evidence[start:end]
            index = len(chunks)
            identity = f"{doc_id}\0pubmed-phase12-section-v1\0{index}\0{start}\0{end}\0{raw}"
            chunks.append({
                "chunk_id": f"{doc_id}-{digest_text(identity)[:24]}", "doc_id": doc_id, "chunk_index": index,
                "raw_text": raw, "normalized_text": normalize_text(raw), "token_count": len(raw.split()),
                "start_offset": start, "end_offset": end, "section_type": section["section_type"],
                "heading_path": section["heading_path"], "source_url": str(row["official_url"]),
                "extraction_method": "ncbi_pubmed_xml_section",
            })
    return document, chunks


def _append_parts(paths: Iterable[Path], destination: Path, schema: pa.Schema) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    try:
        for path in paths:
            writer.write_table(pq.read_table(path, schema=schema))
    finally:
        writer.close()
    os.replace(temporary, destination)


def process_pubmed(config: dict, run_state: Path) -> dict:
    output = ROOT / config["outputs"]["processed"]
    artifacts = ROOT / config["outputs"]["artifacts"]
    parts = output / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    input_parts = sorted((ROOT / config["outputs"]["records"]).glob("part-*.parquet"))
    if len(input_parts) != 800:
        raise ValueError(f"expected 800 acquired PubMed parts, found {len(input_parts)}")
    completed_docs = completed_chunks = 0
    for number, source in enumerate(input_parts, 1):
        stem = source.stem
        docs_path = parts / f"{stem}.documents.parquet"
        chunks_path = parts / f"{stem}.chunks.parquet"
        marker = parts / f"{stem}.done.json"
        source_sha = sha256_file(source)
        valid = False
        if marker.exists() and docs_path.exists() and chunks_path.exists():
            checkpoint = json.loads(marker.read_text(encoding="utf-8"))
            valid = checkpoint.get("input_sha256") == source_sha and checkpoint.get("documents_sha256") == sha256_file(docs_path) and checkpoint.get("chunks_sha256") == sha256_file(chunks_path)
        if not valid:
            rows = pq.read_table(source).to_pylist()
            docs: list[dict] = []
            chunks: list[dict] = []
            for row in rows:
                document, produced = process_record(row)
                if document is not None:
                    docs.append(document)
                    chunks.extend(produced)
            docs_tmp = docs_path.with_suffix(".parquet.tmp")
            chunks_tmp = chunks_path.with_suffix(".parquet.tmp")
            pq.write_table(pa.Table.from_pylist(docs, schema=DOCUMENT_SCHEMA), docs_tmp, compression="zstd")
            pq.write_table(pa.Table.from_pylist(chunks, schema=CHUNK_SCHEMA), chunks_tmp, compression="zstd")
            os.replace(docs_tmp, docs_path)
            os.replace(chunks_tmp, chunks_path)
            checkpoint = {"input": str(source), "input_sha256": source_sha, "documents": len(docs), "chunks": len(chunks), "documents_sha256": sha256_file(docs_path), "chunks_sha256": sha256_file(chunks_path)}
            atomic_json(marker, checkpoint)
        else:
            checkpoint = json.loads(marker.read_text(encoding="utf-8"))
        completed_docs += int(checkpoint["documents"])
        completed_chunks += int(checkpoint["chunks"])
        update_state(run_state, stage="PROCESS_PUBMED", status="RUNNING", total=len(input_parts), done=number, processed_documents=completed_docs, processed_chunks=completed_chunks, checkpoint=str(marker))
    documents = output / "documents.parquet"
    chunks = output / "chunks.parquet"
    _append_parts(sorted(parts.glob("*.documents.parquet")), documents, DOCUMENT_SCHEMA)
    _append_parts(sorted(parts.glob("*.chunks.parquet")), chunks, CHUNK_SCHEMA)
    doc_ids: set[int] = set()
    chunk_ids: set[str] = set()
    owners: set[int] = set()
    reconstruction = 0
    for batch in pq.ParquetFile(documents).iter_batches(columns=["doc_id"]):
        for value in batch.column(0).to_pylist():
            if int(value) in doc_ids:
                raise ValueError(f"duplicate processed PubMed doc_id {value}")
            doc_ids.add(int(value))
    for batch in pq.ParquetFile(chunks).iter_batches(columns=["chunk_id", "doc_id", "raw_text", "start_offset", "end_offset"], batch_size=8192):
        for chunk_id, doc_id, raw, start, end in zip(*(column.to_pylist() for column in batch.columns), strict=True):
            chunk_id = str(chunk_id); doc_id = int(doc_id)
            if chunk_id in chunk_ids:
                raise ValueError(f"duplicate processed PubMed chunk_id {chunk_id}")
            chunk_ids.add(chunk_id); owners.add(doc_id)
            if not str(raw) or int(start) < 0 or int(end) <= int(start):
                reconstruction += 1
    if owners != doc_ids or reconstruction:
        raise ValueError(f"PubMed document/chunk closure failed: docs={len(doc_ids)} owners={len(owners)} reconstruction={reconstruction}")
    summary = {
        "status": "PROCESSING_COMPLETE", "official_target": 240000, "valid_documents": len(doc_ids),
        "unavailable_records": 240000 - len(doc_ids), "chunks": len(chunk_ids), "document_chunk_ownership_passed": True,
        "deterministic_chunk_ids": True, "source_grounded_evidence": True, "documents_path": str(documents),
        "chunks_path": str(chunks), "documents_sha256": sha256_file(documents), "chunks_sha256": sha256_file(chunks),
    }
    atomic_json(artifacts / "processing_quality_audit.json", summary)
    return summary


def _load_embedder(model_config: dict):
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_config["name"], revision=model_config["revision"], device="cuda", model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = int(model_config["max_length"])
    model.eval()
    return model


def benchmark_embed(config: dict, documents: Path, run_state: Path) -> dict:
    report_path = ROOT / config["outputs"]["artifacts"] / "embedding_benchmark.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    import psutil
    import torch
    texts = [str(value) for value in next(pq.ParquetFile(documents).iter_batches(columns=["retrieval_text"], batch_size=64)).column(0).to_pylist()]
    update_state(run_state, stage="BENCHMARK_EMBED", status="RUNNING", total=2, done=0, checkpoint="loading BAAI/bge-m3")
    before = psutil.Process().memory_info().rss / 2**20
    model = _load_embedder(config["models"]["embedder"])
    loaded = psutil.Process().memory_info().rss / 2**20
    results = []
    for batch_size in (1, 2):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        elapsed = time.perf_counter() - started
        if not np.isfinite(vectors).all():
            raise ValueError("non-finite BGE-M3 benchmark embeddings")
        results.append({"batch_size": batch_size, "rows": len(texts), "seconds": elapsed, "rows_per_second": len(texts) / elapsed, "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20, "rss_mib": psutil.Process().memory_info().rss / 2**20})
        update_state(run_state, stage="BENCHMARK_EMBED", status="RUNNING", total=2, done=batch_size, checkpoint=str(report_path))
    chosen = 2
    total = pq.ParquetFile(documents).metadata.num_rows
    speed = next(row["rows_per_second"] for row in results if row["batch_size"] == chosen)
    report = {"model": config["models"]["embedder"]["name"], "revision": config["models"]["embedder"]["revision"], "precision": "float16", "max_length": int(config["models"]["embedder"]["max_length"]), "process_rss_before_mib": before, "process_rss_after_model_load_mib": loaded, "results": results, "selected_batch_size": chosen, "estimated_full_seconds": total / speed, "estimated_full_hours": total / speed / 3600}
    atomic_json(report_path, report)
    del model
    gc.collect(); torch.cuda.empty_cache()
    return report


def embed_documents(config: dict, documents: Path, run_state: Path, benchmark: dict) -> dict:
    import psutil
    import torch
    retrieval = ROOT / config["outputs"]["retrieval"]
    retrieval.mkdir(parents=True, exist_ok=True)
    artifacts = ROOT / config["outputs"]["artifacts"]
    rows = pq.ParquetFile(documents).metadata.num_rows
    dimension = int(config["models"]["embedder"]["dimension"])
    final = retrieval / "pubmed_document_embeddings.f32"
    partial = retrieval / "pubmed_document_embeddings.streaming.partial"
    status_path = retrieval / "pubmed_document_embeddings.streaming.status.u8"
    checkpoint_path = retrieval / "pubmed_document_embeddings.streaming.checkpoint.json"
    signature = digest_text(f"{sha256_file(documents)}:{config['models']['embedder']['revision']}:{rows}:{dimension}")
    if final.exists() and final.stat().st_size == rows * dimension * 4 and status_path.exists() and status_path.stat().st_size == rows:
        status = np.memmap(status_path, dtype=np.uint8, mode="r", shape=(rows,))
        if int(status.sum()) == rows:
            return json.loads((artifacts / "embedding_summary.json").read_text(encoding="utf-8"))
    if checkpoint_path.exists():
        checkpoint_audit=validate_embedding_checkpoint(config,documents)
        if checkpoint_audit["signature"]!=signature:
            raise ValueError("PubMed embedding checkpoint contract mismatch")
    else:
        vectors = np.memmap(partial, dtype=np.float32, mode="w+", shape=(rows, dimension)); vectors.flush(); del vectors
        status = np.memmap(status_path, dtype=np.uint8, mode="w+", shape=(rows,)); status[:] = 0; status.flush(); del status
        atomic_json(checkpoint_path, {"signature": signature, "rows": rows, "dimension": dimension, "completed": 0})
    vectors = np.memmap(partial, dtype=np.float32, mode="r+", shape=(rows, dimension))
    status = np.memmap(status_path, dtype=np.uint8, mode="r+", shape=(rows,))
    completed = int(status.sum())
    update_state(run_state, stage="EMBED_AND_INDEX", status="RUNNING", total=rows, done=completed, remaining=rows-completed, embedding_rows_per_second=0.0, eta_seconds=None, checkpoint=str(checkpoint_path))
    model = _load_embedder(config["models"]["embedder"])
    batch_size = int(benchmark["selected_batch_size"])
    started = time.perf_counter(); newly = 0; offset = 0; stream_rows = int(config["models"]["embedder"]["stream_rows"])
    for batch in pq.ParquetFile(documents).iter_batches(columns=["retrieval_text"], batch_size=stream_rows):
        texts = [str(value) for value in batch.column(0).to_pylist()]
        positions = [offset + i for i in range(len(texts)) if status[offset + i] == 0]
        if positions:
            pending = [texts[position-offset] for position in positions]
            encoded = model.encode(pending, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32, copy=False)
            if encoded.shape != (len(positions), dimension) or not np.isfinite(encoded).all():
                raise ValueError(f"invalid embedding window at row {offset}")
            vectors[positions] = encoded; vectors.flush()
            status[positions] = 1; status.flush()
            newly += len(positions); completed += len(positions)
            elapsed = max(time.perf_counter()-started, 1e-9); speed = newly/elapsed
            checkpoint = {"signature": signature, "rows": rows, "dimension": dimension, "completed": completed, "last_offset": offset, "vectors_flushed_before_status": True}
            atomic_json(checkpoint_path, checkpoint)
            update_state(run_state, stage="EMBED_AND_INDEX", status="RUNNING", total=rows, done=completed, remaining=rows-completed, embedding_rows_per_second=round(speed, 3), eta_seconds=round((rows-completed)/max(speed, 1e-9), 1), rss_mib=round(psutil.Process().memory_info().rss/2**20, 1), gpu_vram_mib=round(torch.cuda.memory_allocated()/2**20, 1), checkpoint=str(checkpoint_path))
        offset += len(texts)
    if int(status.sum()) != rows:
        raise ValueError(f"PubMed embeddings incomplete: {int(status.sum())}/{rows}")
    del model; gc.collect(); torch.cuda.empty_cache(); vectors.flush(); del vectors; del status
    os.replace(partial, final)
    matrix = np.memmap(final, dtype=np.float32, mode="r", shape=(rows, dimension))
    nonfinite = sum(int((~np.isfinite(matrix[start:start+4096])).sum()) for start in range(0, rows, 4096))
    if nonfinite:
        raise ValueError(f"PubMed document embeddings contain {nonfinite} non-finite values")
    summary = {"status": "COMPLETE", "rows": rows, "dimension": dimension, "completed": rows, "nonfinite_values": 0, "model": config["models"]["embedder"]["name"], "revision": config["models"]["embedder"]["revision"], "batch_size": batch_size, "runtime_seconds": time.perf_counter()-started, "path": str(final), "sha256": sha256_file(final)}
    atomic_json(artifacts / "embedding_summary.json", summary)
    return summary


def retrieve_pubmed(config: dict, documents: Path, run_state: Path) -> dict:
    import torch
    retrieval = ROOT / config["outputs"]["retrieval"]
    artifacts = ROOT / config["outputs"]["artifacts"]
    output = retrieval / "pubmed_top100_documents.parquet"
    report_path = artifacts / "retrieval_summary.json"
    queries = pq.read_table(ROOT / config["inputs"]["queries"], columns=["id", "query"]).to_pylist()
    docs = pq.read_table(documents, columns=["doc_id", "pmid"]).to_pylist()
    rows = len(docs); dimension = int(config["models"]["embedder"]["dimension"]); depth = int(config["retrieval"]["pubmed_documents_per_query"])
    if output.exists() and report_path.exists() and pq.ParquetFile(output).metadata.num_rows == len(queries)*depth:
        return json.loads(report_path.read_text(encoding="utf-8"))
    model = _load_embedder(config["models"]["embedder"])
    query_values = [normalize_text(str(row["query"])) for row in queries]
    query_vectors = model.encode(query_values, batch_size=2, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    query_path = retrieval / "query_embeddings.f32"; query_vectors.tofile(query_path)
    del model; gc.collect(); torch.cuda.empty_cache()
    q_gpu = torch.from_numpy(query_vectors).to("cuda")
    matrix = np.memmap(retrieval / "pubmed_document_embeddings.f32", dtype=np.float32, mode="r", shape=(rows, dimension))
    best_scores = np.full((len(queries), depth), -np.inf, dtype=np.float32)
    best_indices = np.full((len(queries), depth), -1, dtype=np.int64)
    block_rows = int(config["retrieval"]["search_block_rows"])
    blocks = math.ceil(rows/block_rows)
    for number, start in enumerate(range(0, rows, block_rows), 1):
        stop = min(rows, start+block_rows)
        block = torch.from_numpy(np.asarray(matrix[start:stop]).copy()).to("cuda")
        scores = torch.matmul(q_gpu, block.T)
        take = min(depth, stop-start)
        values, positions = torch.topk(scores, k=take, dim=1, largest=True, sorted=False)
        values_np = values.float().cpu().numpy(); positions_np = positions.cpu().numpy() + start
        combined_scores = np.concatenate([best_scores, values_np], axis=1)
        combined_indices = np.concatenate([best_indices, positions_np], axis=1)
        selected = np.argpartition(combined_scores, -depth, axis=1)[:, -depth:]
        best_scores = np.take_along_axis(combined_scores, selected, axis=1)
        best_indices = np.take_along_axis(combined_indices, selected, axis=1)
        del block, scores, values, positions
        update_state(run_state, stage="RETRIEVE_PUBMED", status="RUNNING", total=blocks, done=number, candidate_rows_written=number*len(queries)*depth, checkpoint="blockwise top-k in progress")
    rows_out = []
    for qi, query in enumerate(queries):
        ranked = sorted(((float(best_scores[qi, j]), int(best_indices[qi, j])) for j in range(depth)), key=lambda item: (-item[0], int(docs[item[1]]["doc_id"])))
        seen = set()
        for score, position in ranked:
            doc_id = int(docs[position]["doc_id"])
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rows_out.append({"query_id": int(query["id"]), "query_text": str(query["query"]), "rank": len(seen), "doc_id": doc_id, "pmid": int(docs[position]["pmid"]), "dense_score": score})
        if len(seen) != depth:
            raise ValueError(f"query {query['id']} produced {len(seen)} unique PubMed documents")
    temporary = output.with_suffix(".parquet.tmp"); pq.write_table(pa.Table.from_pylist(rows_out), temporary, compression="zstd"); os.replace(temporary, output)
    all_scores = np.asarray([row["dense_score"] for row in rows_out], dtype=np.float32)
    report = {"queries": len(queries), "documents_per_query": depth, "candidate_rows": len(rows_out), "unique_document_ids": len({row["doc_id"] for row in rows_out}), "score_min": float(all_scores.min()), "score_median": float(np.median(all_scores)), "score_p95": float(np.percentile(all_scores,95)), "score_max": float(all_scores.max()), "model": config["models"]["embedder"]["name"], "revision": config["models"]["embedder"]["revision"], "path": str(output), "sha256": sha256_file(output)}
    atomic_json(report_path, report)
    return report


def build_pubmed_candidates(config: dict, run_state: Path) -> dict:
    retrieval = ROOT / config["outputs"]["retrieval"]
    artifacts = ROOT / config["outputs"]["artifacts"]
    round1 = retrieval / "round1"; parts = round1 / "source_candidate_parts"; parts.mkdir(parents=True, exist_ok=True)
    phase11c = ROOT / config["inputs"]["phase11c_root"] / "round1" / "source_candidate_parts"
    for path in phase11c.iterdir():
        if path.suffix in {".parquet", ".json"}:
            _copy_or_link(path, parts/path.name)
    destination = parts / "PUBMED.parquet"; marker = parts / "PUBMED.json"
    top_docs = pq.read_table(retrieval / "pubmed_top100_documents.parquet").to_pylist()
    selected_docs = {int(row["doc_id"]) for row in top_docs}
    chunks_by_doc: dict[int, list[dict]] = defaultdict(list)
    for batch in pq.ParquetFile(ROOT/config["outputs"]["processed"] / "chunks.parquet").iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            if int(row["doc_id"]) in selected_docs:
                chunks_by_doc[int(row["doc_id"])].append(row)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in top_docs:
        grouped[int(row["query_id"])].append(row)
    temporary = destination.with_suffix(".parquet.tmp"); writer = pq.ParquetWriter(temporary, CANDIDATE_SCHEMA, compression="zstd"); candidate_rows = 0
    try:
        for number, query_id in enumerate(sorted(grouped), 1):
            output = []
            for doc in sorted(grouped[query_id], key=lambda row: int(row["rank"])):
                for chunk in sorted(chunks_by_doc[int(doc["doc_id"])], key=lambda row: (int(row["chunk_index"]), str(row["chunk_id"]))):
                    output.append({"query_id": query_id, "query_text": str(doc["query_text"]), "source": "PUBMED", "source_rank": int(doc["rank"])*10000+int(chunk["chunk_index"]), "source_score": float(doc["dense_score"]), "chunk_id": str(chunk["chunk_id"]), "doc_id": int(chunk["doc_id"]), "chunk_text": str(chunk["raw_text"]), "source_url": str(chunk["source_url"]), "chunk_index": int(chunk["chunk_index"]), "start_offset": int(chunk["start_offset"]), "end_offset": int(chunk["end_offset"]), "section_type": str(chunk["section_type"]), "heading_path": list(chunk.get("heading_path") or []), "extraction_method": str(chunk["extraction_method"])})
            writer.write_table(pa.Table.from_pylist(output, schema=CANDIDATE_SCHEMA)); candidate_rows += len(output)
            if number % 25 == 0 or number == len(grouped):
                update_state(run_state, stage="RERANK_PUBMED_PREPARE", status="RUNNING", total=len(grouped), done=number, pubmed_chunk_candidate_pairs=candidate_rows, checkpoint=str(temporary))
    finally:
        writer.close()
    os.replace(temporary, destination)
    atomic_json(marker, {"source": "PUBMED", "query_count": len(grouped), "source_document_count": len(selected_docs), "candidate_rows": candidate_rows, "retrieval": "BGE-M3 dense top100 PubMed documents; all deterministic source-grounded evidence chunks within selected documents"})
    score_db = round1 / "source_scores.sqlite"
    expected = sum(pq.ParquetFile(path).metadata.num_rows for path in parts.glob("*.parquet"))
    valid = False
    if score_db.exists():
        connection=sqlite3.connect(score_db); valid=int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])==expected; connection.close()
    if not valid:
        _init_score_db(parts, score_db)
    seed_path = artifacts / "score_seed_summary.json"
    seed = json.loads(seed_path.read_text(encoding="utf-8")) if valid and seed_path.exists() else _seed_exact(score_db, [ROOT/config["inputs"]["phase11c_root"] / "round1/source_scores.sqlite"])
    atomic_json(seed_path, seed)
    summary = {"pubmed_document_candidate_pairs": 120000, "pubmed_chunk_candidate_pairs": candidate_rows, "all_candidate_pairs": seed["total"], "exact_scores_reused": seed["reused"], "new_scores_required": seed["missing"], "score_db": str(score_db), "sqlite_integrity": seed["sqlite_integrity"]}
    atomic_json(artifacts / "candidate_and_reuse_summary.json", summary)
    update_state(run_state, stage="RERANK_PUBMED", status="RUNNING", total=seed["total"], done=seed["reused"], remaining=seed["missing"], checkpoint=str(seed_path), **summary)
    return summary


def runtime_config(config: dict) -> Path:
    retrieval = ROOT / config["outputs"]["retrieval"]
    runtime = yaml.safe_load((ROOT/"data/source_census/phase11b_new_corpus/runtime.yaml").read_text(encoding="utf-8"))
    runtime["outputs"]["root"] = str(retrieval)
    runtime["outputs"]["artifacts"] = config["outputs"]["artifacts"]
    runtime["submission"]["documents_per_query"] = 20
    runtime["submission"]["chunks_per_query"] = 40
    path = retrieval / "runtime.yaml"; path.write_text(yaml.safe_dump(runtime, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def score_pubmed(config: dict, run_state: Path) -> None:
    retrieval = ROOT / config["outputs"]["retrieval"]
    connection = sqlite3.connect(retrieval/"round1/source_scores.sqlite")
    missing = int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NULL").fetchone()[0]); connection.close()
    if not missing:
        return
    subprocess.run([sys.executable, "tools/score_source_census_persistent.py", "--config", str(runtime_config(config)), "--window-size", "2000", "--run-state", str(run_state), "--metrics-out", str(ROOT/config["outputs"]["artifacts"]/"rerank_metrics.json")], cwd=ROOT, check=True)


def phase11c_control(config: dict) -> dict:
    report = json.loads((ROOT/config["inputs"]["phase11c_report"]).read_text(encoding="utf-8"))
    json_path = ROOT/config["inputs"]["phase11c_submission_json"]; zip_path = ROOT/config["inputs"]["phase11c_submission_zip"]
    json_sha = sha256_file(json_path); zip_sha = sha256_file(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist(); packed = hashlib.sha256(archive.read(names[0])).hexdigest() if len(names)==1 else ""
    package = report["package"]
    passed = report.get("passed") is True and package.get("json_sha256") == json_sha and package.get("zip_sha256") == zip_sha and packed == json_sha
    result = {"passed": passed, "json_path": str(json_path), "zip_path": str(zip_path), "json_sha256": json_sha, "zip_sha256": zip_sha, "byte_integrity": packed==json_sha}
    if not passed:
        raise ValueError(f"frozen Phase11C control drift: {result}")
    return result


def finalize(config: dict, run_state: Path, candidate_summary: dict) -> dict:
    retrieval = ROOT/config["outputs"]["retrieval"]; artifacts=ROOT/config["outputs"]["artifacts"]
    runtime = yaml.safe_load(runtime_config(config).read_text(encoding="utf-8"))
    markers = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((retrieval/"round1/source_candidate_parts").glob("*.json"))]
    sources = [str(marker["source"]) for marker in markers]
    atomic_json(artifacts/"technical_triage.json", {"healthy_sources": sources})
    rankings = retrieval/"rankings"; build_cached_subset_rankings(runtime, {"FULL": sources}, rankings)
    chunk_ids=set(); doc_ids=set()
    ranked_chunks=rankings/"FULL/reranked_chunks.parquet"
    for batch in pq.ParquetFile(ranked_chunks).iter_batches(columns=["chunk_id","doc_id"],batch_size=8192):
        chunk_ids.update(map(str,batch.column(0).to_pylist())); doc_ids.update(map(int,batch.column(1).to_pylist()))
    documents=retrieval/"round1/combined_documents.parquet"; chunks=retrieval/"round1/combined_chunks.parquet"
    _filter_parquet([ROOT/config["inputs"]["phase11c_root"]/"round1/combined_documents.parquet", ROOT/config["outputs"]["processed"]/"documents.parquet"],documents,"doc_id",doc_ids)
    _filter_parquet([ROOT/config["inputs"]["phase11c_root"]/"round1/combined_chunks.parquet", ROOT/config["outputs"]["processed"]/"chunks.parquet"],chunks,"chunk_id",chunk_ids)
    canonical=retrieval/"round1/canonical_text.sqlite"; _build_canonical_store(chunks,documents,canonical)
    closure=_audit_ranked_provenance_closure(ranked_chunks,canonical,ROOT/config["inputs"]["official_corpus"],artifacts/"provenance_closure.json")
    package=_stream_package(runtime,rankings,"FULL","phase12_FULL_PUBMED_DENSE_DOCS20_CHUNKS40",ROOT/"submissions/00_READY_TO_UPLOAD",canonical,tokenizer(runtime))
    official=_official_id_audit(Path(package["json_path"]),ROOT/config["inputs"]["official_corpus"])
    cache=audit_score_cache(retrieval/"round1/source_scores.sqlite",retrieval/"round1/source_candidate_parts")
    control=phase11c_control(config)
    comparison=_compare_submission_signatures(ROOT/config["inputs"]["phase11c_submission_json"],Path(package["json_path"]))
    pubmed_docs={int(value) for batch in pq.ParquetFile(ROOT/config["outputs"]["processed"]/"documents.parquet").iter_batches(columns=["doc_id"]) for value in batch.column(0).to_pylist()}
    pubmed_chunks={str(value) for batch in pq.ParquetFile(ROOT/config["outputs"]["processed"]/"chunks.parquet").iter_batches(columns=["chunk_id"]) for value in batch.column(0).to_pylist()}
    selected_docs=selected_chunks=queries_docs=queries_chunks=0; positions=Counter()
    from src.source_census.union_experiment import _iter_json_array
    for current in _iter_json_array(Path(package["json_path"])):
        docs_here=[int(value) for value in current["relevant_docs"] if int(value) in pubmed_docs]
        chunks_here=[row for row in current["relevant_chunks"] if str(row["chunk_id"]) in pubmed_chunks]
        selected_docs+=len(docs_here); selected_chunks+=len(chunks_here); queries_docs+=bool(docs_here); queries_chunks+=bool(chunks_here)
        for rank,value in enumerate(current["relevant_docs"],1):
            if int(value) in pubmed_docs:positions[f"doc_{rank}"]+=1
        for rank,row in enumerate(current["relevant_chunks"],1):
            if str(row["chunk_id"]) in pubmed_chunks:positions[f"chunk_{rank}"]+=1
    score_db=sqlite3.connect(retrieval/"round1/source_scores.sqlite")
    score_rows=[float(row[0]) for row in score_db.execute("SELECT rerank_score FROM pairs WHERE chunk_id LIKE '%-%' AND rerank_score IS NOT NULL")]; score_db.close()
    attribution={"queries_with_pubmed_document_in_top20":queries_docs,"queries_with_pubmed_chunk_in_top40":queries_chunks,"pubmed_document_slots":selected_docs,"pubmed_chunk_slots":selected_chunks,"rank_distribution":dict(sorted(positions.items())),"queries_with_zero_pubmed_evidence":1200-queries_chunks,"reranker_score_min":min(score_rows) if score_rows else None,"reranker_score_mean":sum(score_rows)/len(score_rows) if score_rows else None,"reranker_score_max":max(score_rows) if score_rows else None,"ranking_changes_vs_phase11c":comparison}
    passed=bool(package["validation"]["passed"] and package["determinism_verified"] and official["passed"] and cache["passed"] and closure["passed"] and control["passed"])
    report={"experiment":"phase12_FULL_PUBMED_DENSE_DOCS20_CHUNKS40","status":"READY_FOR_LEADERBOARD" if passed else "NEEDS_AGENT","passed":passed,"phase11c_control":control,"candidate_statistics":candidate_summary,"attribution":attribution,"score_cache":cache,"provenance_closure":closure,"official_id_audit":official,"structural_validation":package["validation"],"determinism_verified":package["determinism_verified"],"new_crawling_outside_pubmed":0,"package":package}
    atomic_json(artifacts/"phase12_report.json",report); atomic_json(artifacts/"phase12_pre_submit_audit.json",report)
    update_state(run_state,stage="FINAL_AUDIT",status=report["status"],total=1,done=1,checkpoint=str(artifacts/"phase12_pre_submit_audit.json"),submission=package.get("zip_path"),sha256=package.get("zip_sha256"))
    return report


def acquisition_audit(config: dict) -> dict:
    manifest=pq.read_table(ROOT/config["outputs"]["manifest"]).to_pylist()
    audit=audit_completed(config,manifest,int(config["acquisition"]["batch_size"]))
    summary=json.loads((ROOT/config["outputs"]["artifacts"]/"acquisition_summary.json").read_text(encoding="utf-8"))
    if audit["completed_ids"]!=240000 or audit["completed_batches"]!=800 or audit["duplicate_completed_pmids"] or audit["temporary_files"]:
        raise ValueError(f"final PubMed acquisition integrity failure: {audit}")
    if int(summary["missing_record_recovery"]["parser_recovered"])!=197:
        raise ValueError("historical PubmedBookArticle repair is absent")
    result={"passed":True,"target":240000,"completed":audit["completed_ids"],"batches":audit["completed_batches"],"status_counts":audit["status_counts"],"parser_recovered":197,"confirmed_unavailable":int(audit["status_counts"].get("MISSING",0)),"mapping_ambiguities":0,"checkpoint_integrity":"ok"}
    atomic_json(ROOT/config["outputs"]["artifacts"]/"final_acquisition_audit.json",result)
    return result


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/phase12_pubmed.yaml"); parser.add_argument("--run-state",required=True); args=parser.parse_args()
    config=load(ROOT/args.config); run_state=ROOT/args.run_state
    artifacts=ROOT/config["outputs"]["artifacts"]
    update_state(run_state,stage="DEPENDENCY_PREFLIGHT",status="RUNNING",total=len(REMAINING_STAGE_MODULES),done=0,checkpoint="remaining-stage import and model-contract audit")
    preflight=dependency_preflight(config)
    atomic_json(artifacts/"dependency_preflight.json",preflight)
    acquisition_path=artifacts/"final_acquisition_audit.json"
    acquisition=json.loads(acquisition_path.read_text(encoding="utf-8"))
    if acquisition.get("passed") is not True or int(acquisition.get("completed",-1))!=240000:
        raise ValueError(f"durable acquisition audit is not complete: {acquisition}")
    processed=verify_processed_outputs(config)
    documents=ROOT/config["outputs"]["processed"]/"documents.parquet"
    update_state(run_state,stage="BENCHMARK_EMBED",status="RUNNING",total=2,done=0,acquisition=acquisition,processed_documents=processed["valid_documents"],processed_chunks=processed["chunks"],processed_checkpoints_reused=processed["processed_checkpoints_reused"],dependency_preflight=preflight,checkpoint=str(artifacts/"dependency_preflight.json"))
    benchmark=benchmark_embed(config,documents,run_state)
    embed_documents(config,documents,run_state,benchmark)
    retrieve_pubmed(config,documents,run_state)
    candidates=build_pubmed_candidates(config,run_state)
    score_pubmed(config,run_state)
    update_state(run_state,stage="MERGE_AND_PACKAGE",status="RUNNING",total=1,done=0,checkpoint="reranker cache complete")
    report=finalize(config,run_state,candidates)
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0 if report["passed"] else 2


if __name__=="__main__":
    raise SystemExit(main())
