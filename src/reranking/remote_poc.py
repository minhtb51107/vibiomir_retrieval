from __future__ import annotations

import hashlib
import heapq
import json
import math
import sqlite3
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MEASURED_ABSOLUTE_TOLERANCE = 0.005859375
PRODUCTION_CONTRACT: dict[str, Any] = {
    "model": MODEL_NAME,
    "revision": MODEL_REVISION,
    "tokenizer_revision": MODEL_REVISION,
    "batch_size": 2,
    "max_length": 512,
    "precision": "float16",
    "eval_mode": True,
    "padding": True,
    "truncation": "longest_first",
    "score_extraction": "logits.reshape(-1).float().cpu().numpy()",
    "input_order": "query, chunk",
}

# The frozen Phase 12 candidate universe has no Chinese passage at or above
# 1,200 characters. Keep the sample balanced across the eight strata that
# actually exist instead of fabricating a symmetric ninth stratum.
SAMPLING_QUOTAS: dict[tuple[str, str], int] = {
    ("en", "short_lt_400"): 14,
    ("en", "medium_400_1199"): 13,
    ("en", "long_ge_1200"): 13,
    ("vi", "short_lt_400"): 14,
    ("vi", "medium_400_1199"): 13,
    ("vi", "long_ge_1200"): 13,
    ("zh", "short_lt_400"): 20,
    ("zh", "medium_400_1199"): 20,
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _length_bucket(text: str) -> str:
    size = len(text)
    if size < 400:
        return "short_lt_400"
    if size < 1200:
        return "medium_400_1199"
    return "long_ge_1200"


def _language(source_part: str, text: str) -> str:
    if source_part == "PUBMED":
        return "en"
    if source_part in {"cancer_39_net", "zydcd_com"}:
        return "zh"
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "zh"
    return "vi"


def _stable_number(seed: str, *values: object) -> int:
    material = ":".join([seed, *(str(value) for value in values)])
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest(), 16)


def _read_only(database: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{database.resolve().as_posix()}?mode=ro", uri=True)


def _close_ranking_rows(connection: sqlite3.Connection, count: int, seed: str) -> list[dict[str, Any]]:
    query_ids = [int(row[0]) for row in connection.execute("SELECT DISTINCT query_id FROM pairs ORDER BY query_id")]
    chosen_queries = sorted(query_ids, key=lambda query_id: _stable_number(seed, "close", query_id))[: count // 4]
    selected: list[dict[str, Any]] = []
    for query_id in chosen_queries:
        rows = connection.execute(
            "SELECT query_id,chunk_id,query_text,chunk_text,rerank_score FROM pairs "
            "WHERE query_id=? AND rerank_score IS NOT NULL ORDER BY rerank_score DESC,chunk_id",
            (query_id,),
        ).fetchall()
        adjacent = sorted(
            (
                (abs(float(rows[index][4]) - float(rows[index + 1][4])),
                 _stable_number(seed, query_id, rows[index][1], rows[index + 1][1]), index)
                for index in range(len(rows) - 1)
            ),
            key=lambda item: (item[0], item[1]),
        )
        used: set[int] = set()
        pair_number = 0
        for margin, _, index in adjacent:
            if index in used or index + 1 in used:
                continue
            pair_number += 1
            used.update({index, index + 1})
            for row in (rows[index], rows[index + 1]):
                selected.append({
                    "query_id": int(row[0]), "chunk_id": str(row[1]),
                    "query_text": str(row[2]), "chunk_text": str(row[3]),
                    "reference_score": float(row[4]),
                    "selection_tags": ["close_query_local_score", f"close_pair_{pair_number}"],
                    "close_pair_margin": float(margin),
                })
            if pair_number == 2:
                break
        if pair_number != 2:
            raise RuntimeError(f"query {query_id} cannot supply two disjoint close-score pairs")
    if len(selected) != count:
        raise RuntimeError(f"close-ranking selection produced {len(selected)}, expected {count}")
    return selected


def _candidate_rows(parts: Path, close_keys: set[tuple[int, str]], seed: str) -> tuple[
    dict[tuple[str, str], list[tuple[int, int, dict[str, Any]]]],
    dict[tuple[int, str], set[str]],
]:
    # Keep a bounded over-sample for every language/length stratum. The final
    # de-duplication happens after all candidate arms have been seen.
    reservoirs: dict[tuple[str, str], list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    close_arms: dict[tuple[int, str], set[str]] = defaultdict(set)
    serial = 0
    limit = 80
    columns = ["query_id", "chunk_id", "query_text", "chunk_text", "source"]
    for path in sorted(parts.glob("*.parquet")):
        source_part = path.stem
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=2048):
            for row in batch.to_pylist():
                key = (int(row["query_id"]), str(row["chunk_id"]))
                source_arm = str(row.get("source") or source_part)
                if key in close_keys:
                    close_arms[key].add(source_arm)
                text = str(row["chunk_text"])
                language = _language(source_part, text)
                bucket = _length_bucket(text)
                rank = _stable_number(seed, "stratum", language, bucket, key[0], key[1])
                serial += 1
                item = {
                    "query_id": key[0], "chunk_id": key[1],
                    "query_text": str(row["query_text"]), "chunk_text": text,
                    "source_arms": [source_arm], "language_signal": language,
                    "length_bucket": bucket,
                    "selection_tags": ["language_length_stratum"],
                }
                heap = reservoirs[(language, bucket)]
                entry = (-rank, serial, item)
                if len(heap) < limit:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)
    return reservoirs, close_arms


def _complete_rows(database: Path, parts: Path, count: int, seed: str) -> list[dict[str, Any]]:
    if count != 200:
        raise ValueError("the proof-of-concept export is intentionally fixed at exactly 200 pairs")
    connection = _read_only(database)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("score cache integrity_check failed")
        close = _close_ranking_rows(connection, 80, seed)
        close_keys = {(row["query_id"], row["chunk_id"]) for row in close}
        reservoirs, close_arms = _candidate_rows(parts, close_keys, seed)
        for row in close:
            key = (row["query_id"], row["chunk_id"])
            if not close_arms.get(key):
                raise RuntimeError(f"close-score key is absent from candidate provenance: {key}")
            row["source_arms"] = sorted(close_arms[key])
            primary = "PUBMED" if "PUBMED" in close_arms[key] else sorted(close_arms[key])[0]
            row["language_signal"] = _language(primary, row["chunk_text"])
            row["length_bucket"] = _length_bucket(row["chunk_text"])

        quota = SAMPLING_QUOTAS
        chosen = list(close)
        chosen_keys = set(close_keys)
        for stratum in sorted(quota):
            candidates = [entry[2] for entry in sorted(reservoirs.get(stratum, []), reverse=True)]
            added = 0
            for row in candidates:
                key = (row["query_id"], row["chunk_id"])
                if key in chosen_keys:
                    continue
                score = connection.execute(
                    "SELECT rerank_score FROM pairs WHERE query_id=? AND chunk_id=? AND rerank_score IS NOT NULL",
                    key,
                ).fetchone()
                if score is None:
                    continue
                row["reference_score"] = float(score[0])
                chosen.append(row)
                chosen_keys.add(key)
                added += 1
                if added == quota[stratum]:
                    break
            if added != quota[stratum]:
                raise RuntimeError(f"insufficient unique rows for {stratum}: {added}/{quota[stratum]}")
    finally:
        connection.close()
    if len(chosen) != count or len(chosen_keys) != count:
        raise RuntimeError(f"selection is not exactly {count} unique keys")
    chosen.sort(key=lambda row: (_stable_number(seed, "output", row["query_id"], row["chunk_id"]), row["chunk_id"]))
    for index, row in enumerate(chosen, 1):
        row["sample_order"] = index
        row["query_text_sha256"] = sha256_text(row["query_text"])
        row["chunk_text_sha256"] = sha256_text(row["chunk_text"])
        row["pair_sha256"] = sha256_text(row["query_text"] + "\0" + row["chunk_text"])
        if not math.isfinite(row["reference_score"]):
            raise RuntimeError(f"non-finite reference score at sample row {index}")
    return chosen


def _json_bytes(value: Any, *, lines: bool = False) -> bytes:
    if lines:
        return ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in value) + "\n").encode("utf-8")
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def export_package(database: Path, parts: Path, output: Path, *, seed: str, count: int = 200) -> dict[str, Any]:
    rows = _complete_rows(database, parts, count, seed)
    payload = _json_bytes(rows, lines=True)
    distribution: dict[str, int] = defaultdict(int)
    for row in rows:
        distribution[f"language:{row['language_signal']}"] += 1
        distribution[f"length:{row['length_bucket']}"] += 1
        for tag in row["selection_tags"]:
            distribution[f"tag:{tag}"] += 1
    manifest = {
        "schema_version": 1, "purpose": "bounded_colab_reranker_equivalence_poc",
        "pair_count": len(rows), "selection_seed": seed,
        "payload": "pairs.jsonl", "payload_sha256": sha256_bytes(payload),
        "source_score_cache": str(database.as_posix()),
        "source_score_cache_size": database.stat().st_size,
        "source_score_cache_sha256": file_sha256(database),
        "contract": PRODUCTION_CONTRACT,
        "measured_absolute_tolerance": MEASURED_ABSOLUTE_TOLERANCE,
        "distribution": dict(sorted(distribution.items())),
        "license": {"model": "Apache-2.0", "source": "https://huggingface.co/BAAI/bge-reranker-v2-m3"},
    }
    manifest_bytes = _json_bytes(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in (("manifest.json", manifest_bytes), ("pairs.jsonl", payload)):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content)
    temporary.replace(output)
    return {**manifest, "package_path": str(output), "package_size": output.stat().st_size, "package_sha256": file_sha256(output)}


def load_package(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("package CRC validation failed")
        if set(archive.namelist()) != {"manifest.json", "pairs.jsonl"}:
            raise ValueError(f"unexpected package members: {archive.namelist()}")
        manifest = json.loads(archive.read("manifest.json"))
        payload = archive.read("pairs.jsonl")
    if sha256_bytes(payload) != manifest.get("payload_sha256"):
        raise ValueError("payload hash differs from manifest")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    if len(rows) != manifest.get("pair_count"):
        raise ValueError("package pair count mismatch")
    keys: set[tuple[int, str]] = set()
    for row in rows:
        key = (int(row["query_id"]), str(row["chunk_id"]))
        if key in keys:
            raise ValueError(f"duplicate package key: {key}")
        keys.add(key)
        if sha256_text(str(row["query_text"])) != row["query_text_sha256"]:
            raise ValueError(f"query text hash mismatch: {key}")
        if sha256_text(str(row["chunk_text"])) != row["chunk_text_sha256"]:
            raise ValueError(f"chunk text hash mismatch: {key}")
        if sha256_text(str(row["query_text"]) + "\0" + str(row["chunk_text"])) != row["pair_sha256"]:
            raise ValueError(f"pair text hash mismatch: {key}")
    return manifest, rows


def _query_order(rows: Iterable[dict[str, Any]], score_field: str) -> dict[int, list[str]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["query_id"])].append(row)
    return {
        query_id: [str(row["chunk_id"]) for row in sorted(items, key=lambda item: (-float(item[score_field]), str(item["chunk_id"])))]
        for query_id, items in grouped.items()
    }


def verify_results(package: Path, results: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest, reference = load_package(package)
    observed_document = json.loads(results.read_text(encoding="utf-8"))
    if observed_document.get("input_package_sha256") != file_sha256(package):
        raise ValueError("result input-package identity mismatch")
    actual_contract = observed_document.get("contract")
    contract_mismatches = {
        key: {"expected": value, "actual": (actual_contract or {}).get(key)}
        for key, value in PRODUCTION_CONTRACT.items() if (actual_contract or {}).get(key) != value
    }
    observed = observed_document.get("pairs")
    if not isinstance(observed, list):
        raise ValueError("results must contain a pairs array")
    expected_by_key = {(int(row["query_id"]), str(row["chunk_id"])): row for row in reference}
    observed_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    duplicate_keys: list[str] = []
    hash_mismatches: list[str] = []
    for row in observed:
        key = (int(row["query_id"]), str(row["chunk_id"]))
        if key in observed_by_key:
            duplicate_keys.append(f"{key[0]}:{key[1]}")
        observed_by_key[key] = row
        expected = expected_by_key.get(key)
        if expected and (row.get("query_text_sha256") != expected["query_text_sha256"] or row.get("chunk_text_sha256") != expected["chunk_text_sha256"]):
            hash_mismatches.append(f"{key[0]}:{key[1]}")
    missing = sorted(set(expected_by_key) - set(observed_by_key))
    unexpected = sorted(set(observed_by_key) - set(expected_by_key))
    comparisons: list[dict[str, Any]] = []
    for key in sorted(set(expected_by_key) & set(observed_by_key)):
        reference_score = float(expected_by_key[key]["reference_score"])
        observed_score = float(observed_by_key[key]["score"])
        comparisons.append({
            "query_id": key[0], "chunk_id": key[1], "reference_score": reference_score,
            "observed_score": observed_score, "absolute_difference": abs(reference_score - observed_score),
        })
    finite = all(math.isfinite(row["observed_score"]) for row in comparisons)
    max_difference = max((row["absolute_difference"] for row in comparisons), default=math.inf)
    reference_order = _query_order(reference, "reference_score")
    observed_for_order = [
        {"query_id": row["query_id"], "chunk_id": row["chunk_id"], "score": row["observed_score"]}
        for row in comparisons
    ]
    observed_order = _query_order(observed_for_order, "score")
    changed_queries = sorted(query_id for query_id in reference_order if observed_order.get(query_id) != reference_order[query_id])
    passed = bool(
        not contract_mismatches and not duplicate_keys and not hash_mismatches and not missing and not unexpected
        and len(comparisons) == manifest["pair_count"] and finite
        and max_difference <= float(manifest["measured_absolute_tolerance"])
        and not changed_queries
    )
    report = {
        "status": "ACCEPTABLE_FOR_BOUNDED_FUTURE_WORKFLOW" if passed else "STOP_EQUIVALENCE_UNACCEPTABLE",
        "passed": passed, "dry_run": True, "existing_scores_modified": 0,
        "package_sha256": file_sha256(package), "result_path": str(results),
        "expected_pairs": manifest["pair_count"], "observed_pairs": len(observed),
        "missing_keys": [f"{q}:{c}" for q, c in missing],
        "unexpected_keys": [f"{q}:{c}" for q, c in unexpected],
        "duplicate_keys": duplicate_keys, "text_hash_mismatches": hash_mismatches,
        "contract_mismatches": contract_mismatches, "finite_scores": finite,
        "maximum_absolute_difference": max_difference,
        "mean_absolute_difference": (sum(row["absolute_difference"] for row in comparisons) / len(comparisons)) if comparisons else None,
        "absolute_tolerance": manifest["measured_absolute_tolerance"],
        "query_local_ordering_changed_count": len(changed_queries),
        "query_local_ordering_changed_query_ids": changed_queries,
        "hardware": observed_document.get("hardware"), "runtime": observed_document.get("runtime"),
        "batch_padding_note": "Production equivalence uses batch=2 in package order; larger-batch benchmarks are telemetry only.",
    }
    return report, comparisons


def write_staging(path: Path, report: dict[str, Any], comparisons: list[dict[str, Any]]) -> None:
    if not report["passed"]:
        raise ValueError("unacceptable equivalence cannot be staged")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite staging database: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE remote_scores(query_id INTEGER,chunk_id TEXT,reference_score REAL,remote_score REAL,"
            "absolute_difference REAL,PRIMARY KEY(query_id,chunk_id))"
        )
        connection.executemany(
            "INSERT INTO remote_scores VALUES(?,?,?,?,?)",
            [(row["query_id"], row["chunk_id"], row["reference_score"], row["observed_score"], row["absolute_difference"]) for row in comparisons],
        )
        connection.execute("CREATE TABLE manifest(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        connection.execute("INSERT INTO manifest VALUES('verification_report',?)", (json.dumps(report, sort_keys=True),))
        connection.commit()
    finally:
        connection.close()
