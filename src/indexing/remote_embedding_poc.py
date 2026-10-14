from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pyarrow.parquet as pq

from src.indexing.embedder import file_sha256, normalize_text


MODEL_NAME = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
DIMENSION = 1024
REFERENCE_ABSOLUTE_TOLERANCE = 0.000732421875
REFERENCE_MINIMUM_COSINE = 0.9995
CONTRACT: dict[str, Any] = {
    "model": MODEL_NAME,
    "revision": MODEL_REVISION,
    "tokenizer_revision": MODEL_REVISION,
    "dimension": DIMENSION,
    "batch_size": 4,
    "max_length": 512,
    "precision": "float16_cuda",
    "normalization": "sentence_transformers_L2",
    "output_dtype": "float32",
    "text_normalization": "NFC_then_split_join",
    "input_order": "historical_batch_groups_of_4",
}
HISTORICAL_SENTENCE_TRANSFORMERS_VERSION = "5.7.0"
HISTORICAL_TRANSFORMERS_VERSION = "4.48.3"
HISTORICAL_WINDOW_ROWS = 128
HISTORICAL_FLATTEN_INPUTS = False


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any, *, lines: bool = False) -> bytes:
    if lines:
        return ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in value) + "\n").encode("utf-8")
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _zip_member(archive: zipfile.ZipFile, name: str, value: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, value)


def _language(source_url: str, text: str) -> str:
    host = (urlsplit(source_url).hostname or "").lower()
    if host.endswith(".cn") or any("\u4e00" <= char <= "\u9fff" for char in text):
        return "zh"
    return "vi"


def _length_bucket(tokens: int) -> str:
    if tokens <= 32:
        return "short_le_32"
    if tokens <= 256:
        return "medium_33_256"
    return "long_257_508"


def _stable(seed: str, *values: object) -> str:
    return hashlib.sha256(":".join([seed, *(str(value) for value in values)]).encode()).hexdigest()


def _chunk_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    values: set[str] = set()
    for batch in pq.ParquetFile(path).iter_batches(columns=["chunk_id"], batch_size=8192):
        values.update(str(value) for value in batch.column(0).to_pylist())
    return values


def reconstruct_historical_batches(
    source_rows: list[dict[str, Any]], reused_ids: set[str], *, window_rows: int = HISTORICAL_WINDOW_ROWS,
) -> tuple[list[list[dict[str, Any]]], int]:
    """Rebuild the exact padded batches submitted by SentenceTransformer.encode.

    Phase 10E called ``encode(pending, batch_size=4)`` once per 128-row
    Parquet window. SentenceTransformers 5.7.0 sorts each call's strings by
    descending character length before slicing batches, then restores output
    order. Its optional flatten/interleave path was unavailable with the
    historical Transformers 4.48.3 runtime and was not requested by Phase 10E.
    """
    batches: list[list[dict[str, Any]]] = []
    eligible_rows = 0
    for window_start in range(0, len(source_rows), window_rows):
        pending = [row for row in source_rows[window_start:window_start + window_rows] if row["chunk_id"] not in reused_ids]
        eligible_rows += len(pending)
        # This deliberately matches SentenceTransformer.encode exactly. Do not
        # replace with Python's stable sort: numpy's default argsort tie handling
        # is part of the historical runtime behavior.
        sorted_indices = np.argsort(np.asarray([-len(str(row["normalized_text"])) for row in pending]))
        sorted_pending = [pending[int(index)] for index in sorted_indices]
        for sorted_start in range(0, len(sorted_pending), int(CONTRACT["batch_size"])):
            group = sorted_pending[sorted_start:sorted_start + int(CONTRACT["batch_size"])]
            if len(group) != int(CONTRACT["batch_size"]):
                continue
            batch_number = sorted_start // int(CONTRACT["batch_size"])
            batch_id = f"window-{window_start:06d}-batch-{batch_number:03d}"
            for offset, row in enumerate(group):
                original_pending_position = pending.index(row)
                row["historical_window_start"] = window_start
                row["historical_pending_position"] = original_pending_position
                row["historical_sorted_position"] = sorted_start + offset
                row["historical_batch_number"] = batch_number
                row["historical_batch_id"] = batch_id
                row["historical_batch_offset"] = offset
            batches.append(group)
    return batches, eligible_rows


def validate_historical_batch_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    batch_size = int(CONTRACT["batch_size"])
    if len(rows) % batch_size:
        raise ValueError("package rows do not contain complete historical batches")
    groups = []
    seen_ids: set[str] = set()
    for start in range(0, len(rows), batch_size):
        group = rows[start:start + batch_size]
        raw_batch_ids = [row.get("historical_batch_id") for row in group]
        batch_ids = {str(value) for value in raw_batch_ids}
        offsets = [int(row.get("historical_batch_offset", -1)) for row in group]
        if len(batch_ids) != 1 or any(value is None for value in raw_batch_ids) or offsets != list(range(batch_size)):
            raise ValueError(f"historical batch membership/order mismatch at package row {start}")
        batch_id = next(iter(batch_ids))
        if batch_id in seen_ids:
            raise ValueError(f"historical batch repeated in package: {batch_id}")
        seen_ids.add(batch_id)
        groups.append(group)
    return groups


def select_reference_rows(
    chunks_path: Path, *, count: int, seed: str, previous_chunks_path: Path | None = None,
) -> list[dict[str, Any]]:
    if not 4 <= count <= 200 or count % CONTRACT["batch_size"]:
        raise ValueError("embedding equivalence sample must contain 4..200 rows in complete production batches of 4")
    source_rows: list[dict[str, Any]] = []
    row_index = 0
    columns = ["chunk_id", "normalized_text", "token_count", "source_url", "doc_id"]
    for batch in pq.ParquetFile(chunks_path).iter_batches(columns=columns, batch_size=2048):
        for row in batch.to_pylist():
            text = normalize_text(str(row["normalized_text"]))
            host = (urlsplit(str(row["source_url"])).hostname or "unknown").lower().removeprefix("www.")
            language = _language(str(row["source_url"]), text)
            bucket = _length_bucket(int(row["token_count"]))
            item = {
                "reference_row_index": row_index,
                "chunk_id": str(row["chunk_id"]),
                "doc_id": int(row["doc_id"]),
                "normalized_text": text,
                "text_sha256": _sha256_bytes(text.encode("utf-8")),
                "token_count": int(row["token_count"]),
                "source_url": str(row["source_url"]),
                "canonical_domain": host,
                "language_signal": language,
                "length_bucket": bucket,
            }
            source_rows.append(item)
            row_index += 1
    # Historical streaming inference loaded 128 Parquet rows, removed exact
    # reused keys, then SentenceTransformer sorted each pending list by length
    # before slicing batch=4. Rebuild those actual model-input groups.
    reused_ids = _chunk_ids(previous_chunks_path)
    strata: dict[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], list[list[dict[str, Any]]]] = defaultdict(list)
    historical_batches, eligible_rows = reconstruct_historical_batches(source_rows, reused_ids)
    for group in historical_batches:
        key = (
            tuple(sorted({row["canonical_domain"] for row in group})),
            tuple(sorted({row["language_signal"] for row in group})),
            tuple(sorted({row["length_bucket"] for row in group})),
        )
        strata[key].append(group)
    for key, groups in strata.items():
        groups.sort(key=lambda group: (_stable(seed, *key, group[0]["historical_batch_id"]), group[0]["reference_row_index"]))
    chosen_groups: list[list[dict[str, Any]]] = []
    positions = {key: 0 for key in strata}
    ordered_keys = sorted(strata, key=lambda key: (_stable(seed, "stratum", *key), key))
    required_groups = count // 4
    while len(chosen_groups) < required_groups:
        advanced = False
        for key in ordered_keys:
            position = positions[key]
            if position < len(strata[key]):
                chosen_groups.append(strata[key][position])
                positions[key] += 1
                advanced = True
                if len(chosen_groups) == required_groups:
                    break
        if not advanced:
            raise ValueError(f"only {len(chosen_groups) * 4} batch-context-preserving reference rows are available")
    chosen_groups.sort(key=lambda group: (_stable(seed, "output", group[0]["historical_batch_id"]), group[0]["reference_row_index"]))
    chosen = [row for group in chosen_groups for row in group]
    for sample_index, row in enumerate(chosen):
        row["sample_index"] = sample_index
    if len({row["chunk_id"] for row in chosen}) != count:
        raise ValueError("selected reference chunk IDs are not unique")
    validate_historical_batch_groups(chosen)
    if previous_chunks_path is not None and eligible_rows != len(source_rows) - len(reused_ids & {row["chunk_id"] for row in source_rows}):
        raise RuntimeError("newly embedded reference-row accounting mismatch")
    return chosen


def export_package(
    chunks_path: Path,
    embeddings_path: Path,
    notebook_path: Path,
    output_path: Path,
    *,
    previous_chunks_path: Path | None = None,
    count: int = 200,
    seed: str = "vibiomir-colab-embedding-poc-v1",
) -> dict[str, Any]:
    parquet_rows = int(pq.ParquetFile(chunks_path).metadata.num_rows)
    expected_bytes = parquet_rows * DIMENSION * np.dtype(np.float32).itemsize
    if embeddings_path.stat().st_size != expected_bytes:
        raise ValueError("reference embedding file does not align exactly with Parquet row count")
    previous_ids = _chunk_ids(previous_chunks_path)
    source_ids = _chunk_ids(chunks_path)
    reused_ids = previous_ids & source_ids
    rows = select_reference_rows(
        chunks_path, count=count, seed=seed, previous_chunks_path=previous_chunks_path,
    )
    reference = np.memmap(embeddings_path, dtype=np.float32, mode="r", shape=(parquet_rows, DIMENSION))
    selected = np.asarray(reference[[row["reference_row_index"] for row in rows]], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("reference sample contains non-finite values")
    vector_bytes = selected.tobytes(order="C")
    payload = _json_bytes(rows, lines=True)
    norms = np.linalg.norm(selected, axis=1)
    distribution = Counter()
    for row in rows:
        distribution[f"domain:{row['canonical_domain']}"] += 1
        distribution[f"language:{row['language_signal']}"] += 1
        distribution[f"length:{row['length_bucket']}"] += 1
    manifest = {
        "schema_version": 2,
        "purpose": "bounded_colab_bge_m3_embedding_equivalence_poc",
        "row_count": len(rows),
        "selection_seed": seed,
        "contract": CONTRACT,
        "historical_batch_reconstruction": {
            "sentence_transformers_version": HISTORICAL_SENTENCE_TRANSFORMERS_VERSION,
            "transformers_version": HISTORICAL_TRANSFORMERS_VERSION,
            "window_rows": HISTORICAL_WINDOW_ROWS,
            "batch_size": int(CONTRACT["batch_size"]),
            "sort_key": "descending_normalized_string_character_length_via_numpy_argsort",
            "flatten_interleave_enabled": HISTORICAL_FLATTEN_INPUTS,
            "selected_complete_batch_groups": len(rows) // int(CONTRACT["batch_size"]),
        },
        "chunks_payload": "chunks.jsonl",
        "chunks_payload_sha256": _sha256_bytes(payload),
        "reference_vectors": "reference_vectors.f32",
        "reference_vectors_sha256": _sha256_bytes(vector_bytes),
        "source_chunks_path": chunks_path.as_posix(),
        "source_chunks_sha256": file_sha256(chunks_path),
        "source_embeddings_path": embeddings_path.as_posix(),
        "source_embeddings_sha256": file_sha256(embeddings_path),
        "source_parquet_rows": parquet_rows,
        "selection_pool": "newly_embedded_rows_only" if previous_chunks_path else "all_reference_rows",
        "previous_chunks_path": previous_chunks_path.as_posix() if previous_chunks_path else None,
        "previous_chunks_sha256": file_sha256(previous_chunks_path) if previous_chunks_path else None,
        "previous_chunks_total_rows": len(previous_ids),
        "previous_reused_rows_excluded": len(reused_ids),
        "newly_embedded_reference_pool_rows": len(source_ids - reused_ids),
        "measured_threshold_source": "artifacts/source_census/phase10e_g1a_10010/embedding_equivalence.json",
        "maximum_absolute_difference_allowed": REFERENCE_ABSOLUTE_TOLERANCE,
        "minimum_cosine_required": REFERENCE_MINIMUM_COSINE,
        "reference_norm_minimum": float(norms.min()),
        "reference_norm_maximum": float(norms.max()),
        "distribution": dict(sorted(distribution.items())),
        "production_import_allowed": False,
    }
    manifest_bytes = _json_bytes(manifest)
    script_bytes = notebook_path.read_bytes()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in (
            ("manifest.json", manifest_bytes), ("chunks.jsonl", payload),
            ("reference_vectors.f32", vector_bytes), ("colab_embedding_poc.py", script_bytes),
        ):
            _zip_member(archive, name, value)
    temporary.replace(output_path)
    loaded_manifest, loaded_rows, loaded_vectors = load_package(output_path)
    if loaded_manifest != manifest or len(loaded_rows) != count or not np.array_equal(loaded_vectors, selected):
        raise RuntimeError("exported embedding package failed deterministic self-verification")
    return {
        **manifest,
        "package_path": str(output_path),
        "package_size_bytes": output_path.stat().st_size,
        "package_sha256": file_sha256(output_path),
    }


def load_package(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("package CRC validation failed")
        expected = {"manifest.json", "chunks.jsonl", "reference_vectors.f32", "colab_embedding_poc.py"}
        if set(archive.namelist()) != expected:
            raise ValueError(f"unexpected package members: {archive.namelist()}")
        manifest = json.loads(archive.read("manifest.json"))
        payload = archive.read("chunks.jsonl")
        vector_bytes = archive.read("reference_vectors.f32")
    if manifest.get("contract") != CONTRACT:
        raise ValueError("package embedding contract mismatch")
    reconstruction = manifest.get("historical_batch_reconstruction") or {}
    if reconstruction != {
        "sentence_transformers_version": HISTORICAL_SENTENCE_TRANSFORMERS_VERSION,
        "transformers_version": HISTORICAL_TRANSFORMERS_VERSION,
        "window_rows": HISTORICAL_WINDOW_ROWS,
        "batch_size": int(CONTRACT["batch_size"]),
        "sort_key": "descending_normalized_string_character_length_via_numpy_argsort",
        "flatten_interleave_enabled": HISTORICAL_FLATTEN_INPUTS,
        "selected_complete_batch_groups": int(manifest.get("row_count", 0)) // int(CONTRACT["batch_size"]),
    }:
        raise ValueError("historical batch reconstruction contract mismatch")
    if _sha256_bytes(payload) != manifest.get("chunks_payload_sha256"):
        raise ValueError("chunk payload hash mismatch")
    if _sha256_bytes(vector_bytes) != manifest.get("reference_vectors_sha256"):
        raise ValueError("reference vector hash mismatch")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    count = int(manifest["row_count"])
    if len(rows) != count or len({row["chunk_id"] for row in rows}) != count:
        raise ValueError("package row count or chunk identity mismatch")
    for index, row in enumerate(rows):
        if int(row["sample_index"]) != index:
            raise ValueError("package sample order mismatch")
        if _sha256_bytes(normalize_text(str(row["normalized_text"])).encode("utf-8")) != row["text_sha256"]:
            raise ValueError(f"text hash mismatch: {row['chunk_id']}")
    validate_historical_batch_groups(rows)
    vectors = np.frombuffer(vector_bytes, dtype=np.float32)
    if vectors.size != count * DIMENSION:
        raise ValueError("reference vector dimensions are invalid")
    return manifest, rows, vectors.reshape(count, DIMENSION).copy()


def verify_results(package_path: Path, results_path: Path) -> dict[str, Any]:
    manifest, rows, reference = load_package(package_path)
    with zipfile.ZipFile(results_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("result ZIP CRC validation failed")
        if set(archive.namelist()) != {"result_manifest.json", "candidate_vectors.f32"}:
            raise ValueError("unexpected result members")
        result = json.loads(archive.read("result_manifest.json"))
        candidate_bytes = archive.read("candidate_vectors.f32")
    if result.get("input_package_sha256") != file_sha256(package_path):
        raise ValueError("result input-package identity mismatch")
    if result.get("contract") != CONTRACT:
        raise ValueError("result embedding contract mismatch")
    identities = result.get("rows")
    expected_identities = [{"sample_index": row["sample_index"], "chunk_id": row["chunk_id"], "text_sha256": row["text_sha256"]} for row in rows]
    if identities != expected_identities:
        raise ValueError("result row identity/order/hash mismatch")
    expected_group_calls = [
        {
            "historical_batch_id": group[0]["historical_batch_id"],
            "member_chunk_ids": [row["chunk_id"] for row in group],
        }
        for group in validate_historical_batch_groups(rows)
    ]
    runtime = result.get("runtime") or {}
    actual_group_calls = [
        {
            "historical_batch_id": group.get("historical_batch_id"),
            "member_chunk_ids": group.get("member_chunk_ids"),
        }
        for group in runtime.get("historical_group_calls", [])
    ]
    if (
        runtime.get("combined_encode_call_used") is not False
        or int(runtime.get("historical_group_count", -1)) != len(expected_group_calls)
        or actual_group_calls != expected_group_calls
    ):
        raise ValueError("result did not preserve isolated historical batch membership")
    if _sha256_bytes(candidate_bytes) != result.get("candidate_vectors_sha256"):
        raise ValueError("candidate vector hash mismatch")
    candidate = np.frombuffer(candidate_bytes, dtype=np.float32)
    if candidate.size != reference.size:
        raise ValueError("candidate vector dimensions are invalid")
    candidate = candidate.reshape(reference.shape)
    finite = bool(np.isfinite(candidate).all())
    differences = np.abs(candidate - reference)
    reference_norms = np.linalg.norm(reference, axis=1)
    candidate_norms = np.linalg.norm(candidate, axis=1)
    cosines = np.sum(reference * candidate, axis=1) / np.maximum(reference_norms * candidate_norms, np.finfo(np.float32).eps)
    maximum_difference = float(differences.max()) if differences.size else math.inf
    minimum_cosine = float(cosines.min()) if cosines.size else -1.0
    norm_margin = float(manifest["maximum_absolute_difference_allowed"])
    norm_min = float(manifest["reference_norm_minimum"]) - norm_margin
    norm_max = float(manifest["reference_norm_maximum"]) + norm_margin
    norms_pass = bool(np.all((candidate_norms >= norm_min) & (candidate_norms <= norm_max)))
    passed = bool(
        finite and norms_pass
        and maximum_difference <= float(manifest["maximum_absolute_difference_allowed"])
        and minimum_cosine >= float(manifest["minimum_cosine_required"])
    )
    return {
        "status": "PASS_EMBEDDING_EQUIVALENCE" if passed else "STOP_EMBEDDING_EQUIVALENCE_FAILED",
        "passed": passed,
        "production_import_allowed": passed,
        "dry_run": True,
        "input_package_sha256": file_sha256(package_path),
        "result_path": str(results_path),
        "rows": len(rows),
        "dimension": DIMENSION,
        "finite": finite,
        "norm_gate_passed": norms_pass,
        "candidate_norm_minimum": float(candidate_norms.min()),
        "candidate_norm_maximum": float(candidate_norms.max()),
        "norm_allowed_minimum": norm_min,
        "norm_allowed_maximum": norm_max,
        "maximum_absolute_difference": maximum_difference,
        "mean_absolute_difference": float(differences.mean()),
        "absolute_tolerance": manifest["maximum_absolute_difference_allowed"],
        "minimum_cosine": minimum_cosine,
        "minimum_cosine_required": manifest["minimum_cosine_required"],
        "rows_exceeding_absolute_tolerance": int(np.any(differences > float(manifest["maximum_absolute_difference_allowed"]), axis=1).sum()),
        "hardware": result.get("hardware"),
        "runtime": result.get("runtime"),
        "batch_context_note": "Equivalence vectors must come from batch=4 in package order. Larger-batch benchmark vectors are telemetry only.",
    }


def duplicate_chunk_analysis(chunks_path: Path, *, top_n: int = 20) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "doc_ids": set(), "tokens": [], "paths": [], "text": ""})
    for batch in pq.ParquetFile(chunks_path).iter_batches(
        columns=["raw_text", "token_count", "doc_id", "heading_path"], batch_size=2048
    ):
        for row in batch.to_pylist():
            item = groups[str(row["raw_text"])]
            item["count"] += 1
            item["doc_ids"].add(int(row["doc_id"]))
            item["tokens"].append(int(row["token_count"]))
            item["paths"].append(row.get("heading_path") or [])
            item["text"] = str(row["raw_text"])
    markers = (
        "lựa chọn dịch vụ", "đặt lịch thăm khám", "lấy mẫu xét nghiệm tại nhà",
        "đăng ký khám và tư vấn", "bình luận (", "liên hệ ngay với số hotline",
        "quý khách hàng vui lòng",
    )

    def classify(item: dict[str, Any]) -> tuple[str, str]:
        lower = item["text"].lower()
        related = lower.strip() == "tin cùng chuyên mục" or any(
            any("tin cùng chuyên mục" in str(value).lower() for value in path) for path in item["paths"]
        )
        boilerplate = any(marker in lower for marker in markers)
        if boilerplate and not related:
            return "site_boilerplate", "site-wide service/comment/footer text"
        if related:
            return "related_article_navigation", "medical-looking related-article/navigation text, not the owning article body"
        return "potential_legitimate_repetition", "repeated source-grounded content; manual interpretation required"

    repeated = [item for item in groups.values() if item["count"] > 1]
    category_slots = Counter()
    category_groups = Counter()
    for item in repeated:
        category, _ = classify(item)
        category_slots[category] += int(item["count"])
        category_groups[category] += 1
    top = []
    for rank, item in enumerate(sorted(repeated, key=lambda value: (-value["count"], -len(value["text"]), value["text"]))[:top_n], 1):
        category, rationale = classify(item)
        top.append({
            "rank": rank, "occurrences": item["count"], "unique_doc_ids": len(item["doc_ids"]),
            "token_length_min": min(item["tokens"]), "token_length_max": max(item["tokens"]),
            "category": category, "classification_basis": rationale,
            "text_excerpt": item["text"][:500],
        })
    total = int(pq.ParquetFile(chunks_path).metadata.num_rows)
    duplicate_slots = sum(item["count"] for item in repeated)
    high_risk = category_slots["site_boilerplate"] + category_slots["related_article_navigation"]
    return {
        "chunks_path": str(chunks_path), "chunks_sha256": file_sha256(chunks_path),
        "total_chunks": total, "exact_duplicate_groups": len(repeated),
        "chunks_in_exact_duplicate_groups": duplicate_slots,
        "duplicate_slot_fraction": duplicate_slots / total,
        "category_groups": dict(sorted(category_groups.items())),
        "category_chunk_slots": dict(sorted(category_slots.items())),
        "retrieval_noise_risk_chunk_slots": high_risk,
        "retrieval_noise_risk_fraction_of_all_chunks": high_risk / total,
        "interpretation": "Risk estimate flags exact site-wide boilerplate and related-article navigation. It is not a relevance label and no rows were removed.",
        "top_groups": top,
        "dataset_modified": False,
    }
