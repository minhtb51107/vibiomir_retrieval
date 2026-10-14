from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from src.reranking import remote_poc


def _rows(count: int = 200):
    rows = []
    for index in range(count):
        query = f"query {index // 4}"
        chunk = f"chunk evidence {index}"
        rows.append({
            "query_id": index // 4 + 1, "chunk_id": f"chunk-{index}",
            "query_text": query, "chunk_text": chunk, "reference_score": float(index) / 10,
            "source_arms": ["fixture"], "language_signal": ("en", "vi", "zh")[index % 3],
            "length_bucket": ("short_lt_400", "medium_400_1199", "long_ge_1200")[index % 3],
            "selection_tags": ["fixture"], "sample_order": index + 1,
            "query_text_sha256": remote_poc.sha256_text(query),
            "chunk_text_sha256": remote_poc.sha256_text(chunk),
            "pair_sha256": remote_poc.sha256_text(query + "\0" + chunk),
        })
    return rows


def test_realized_sampling_quota_has_120_rows_and_no_nonexistent_long_chinese_stratum():
    assert sum(remote_poc.SAMPLING_QUOTAS.values()) == 120
    assert remote_poc.SAMPLING_QUOTAS[("zh", "short_lt_400")] == 20
    assert remote_poc.SAMPLING_QUOTAS[("zh", "medium_400_1199")] == 20
    assert ("zh", "long_ge_1200") not in remote_poc.SAMPLING_QUOTAS


def _package(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    database = tmp_path / "scores.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE marker(value INTEGER)")
    connection.commit()
    connection.close()
    monkeypatch.setattr(remote_poc, "_complete_rows", lambda *args, **kwargs: _rows())
    package = tmp_path / "poc.zip"
    report = remote_poc.export_package(database, tmp_path, package, seed="test")
    return package, report


def _results(package: Path, *, delta: float = 0.0) -> dict:
    _, rows = remote_poc.load_package(package)
    return {
        "input_package_sha256": remote_poc.file_sha256(package),
        "contract": remote_poc.PRODUCTION_CONTRACT,
        "pairs": [
            {"query_id": row["query_id"], "chunk_id": row["chunk_id"],
             "query_text_sha256": row["query_text_sha256"], "chunk_text_sha256": row["chunk_text_sha256"],
             "score": row["reference_score"] + delta}
            for row in rows
        ],
        "runtime": {"batch2_scoring_seconds": 1.0}, "hardware": {"gpu": "fixture"},
    }


def test_export_is_deterministic_and_exactly_200(tmp_path, monkeypatch):
    first, first_report = _package(tmp_path / "first", monkeypatch)
    second, second_report = _package(tmp_path / "second", monkeypatch)
    assert first_report["pair_count"] == 200
    assert first_report["payload_sha256"] == second_report["payload_sha256"]
    assert first_report["distribution"] == second_report["distribution"]
    _, rows = remote_poc.load_package(first)
    assert len(rows) == len({(row["query_id"], row["chunk_id"]) for row in rows}) == 200


def test_verifier_accepts_exact_results_and_stages_only_new_database(tmp_path, monkeypatch):
    package, _ = _package(tmp_path, monkeypatch)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_results(package)), encoding="utf-8")
    report, comparisons = remote_poc.verify_results(package, result_path)
    assert report["passed"] is True
    assert report["existing_scores_modified"] == 0
    staging = tmp_path / "staging.sqlite"
    remote_poc.write_staging(staging, report, comparisons)
    with pytest.raises(FileExistsError):
        remote_poc.write_staging(staging, report, comparisons)


def test_verifier_rejects_score_drift_order_change_and_bad_hash(tmp_path, monkeypatch):
    package, _ = _package(tmp_path, monkeypatch)
    document = _results(package)
    document["pairs"][0]["score"] += 1.0
    document["pairs"][0]["chunk_text_sha256"] = "0" * 64
    result_path = tmp_path / "drift.json"
    result_path.write_text(json.dumps(document), encoding="utf-8")
    report, _ = remote_poc.verify_results(package, result_path)
    assert report["passed"] is False
    assert report["maximum_absolute_difference"] == 1.0
    assert report["text_hash_mismatches"]


def test_verifier_rejects_missing_duplicate_or_contract_drift(tmp_path, monkeypatch):
    package, _ = _package(tmp_path, monkeypatch)
    document = _results(package)
    document["pairs"].pop()
    document["pairs"].append(dict(document["pairs"][0]))
    document["contract"] = {**remote_poc.PRODUCTION_CONTRACT, "batch_size": 4}
    result_path = tmp_path / "invalid.json"
    result_path.write_text(json.dumps(document), encoding="utf-8")
    report, _ = remote_poc.verify_results(package, result_path)
    assert report["passed"] is False
    assert report["missing_keys"]
    assert report["duplicate_keys"]
    assert report["contract_mismatches"]["batch_size"] == {"expected": 2, "actual": 4}


def test_load_package_rejects_tampered_payload(tmp_path, monkeypatch):
    package, _ = _package(tmp_path, monkeypatch)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(package) as source, zipfile.ZipFile(tampered, "w") as destination:
        destination.writestr("manifest.json", source.read("manifest.json"))
        destination.writestr("pairs.jsonl", source.read("pairs.jsonl") + b"{}\n")
    with pytest.raises(ValueError, match="payload hash"):
        remote_poc.load_package(tampered)
