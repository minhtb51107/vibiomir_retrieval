#!/usr/bin/env python3
"""Diagnose Phase 14 reranker controls against their original FP16 batches."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reranking.reranker import TransformerCrossEncoderReranker
from src.runtime.state_file import write_json


ROUND1 = Path("data/source_census/round1/source_scores.sqlite")
PHASE11C = Path("data/source_census/phase11c_broad_existing_new/round1/source_scores.sqlite")
PHASE14 = Path("data/source_census/phase14_medlatec_coverage_3000/round1/source_scores.sqlite")
SEED_CACHES = [
    Path("data/source_census/phase11b_new_corpus/round1/source_scores.sqlite"),
    Path("data/source_census/phase11a_bottleneck/broad_probe/round1/source_scores.sqlite"),
    ROUND1,
    Path("data/retrieval/phase10c1_early_release/C1/rerank_scores.sqlite"),
    Path("data/retrieval/phase10c1_early_release/C2/rerank_scores.sqlite"),
    *(Path(f"data/source_relevance_probe/S{i}/rerank_scores.sqlite") for i in range(1, 7)),
]


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)


def _table(connection: sqlite3.Connection) -> str:
    names = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return "pairs" if "pairs" in names else "scores"


def _query_one_rows(path: Path) -> list[tuple]:
    connection = _ro(path)
    try:
        return connection.execute(
            "SELECT query_id,chunk_id,query_text,chunk_text,rerank_score "
            "FROM pairs WHERE query_id=1 ORDER BY chunk_id"
        ).fetchall()
    finally:
        connection.close()


def historical_batches(control: dict) -> dict[str, list[tuple]]:
    """Recover exact batch-size-two groups from the original missing-key order."""
    targets = {(int(row["query_id"]), str(row["chunk_id"])) for row in control["pairs"]}
    round_rows = _query_one_rows(ROUND1)
    round_positions = {(int(row[0]), str(row[1])): index for index, row in enumerate(round_rows)}

    cache_connections = [(_ro(path), path) for path in SEED_CACHES]
    phase11_rows = _query_one_rows(PHASE11C)
    phase11_missing = []
    try:
        for row in phase11_rows:
            key = (int(row[0]), str(row[1]))
            seeded = False
            for connection, _ in cache_connections:
                table = _table(connection)
                if connection.execute(
                    f"SELECT 1 FROM {table} WHERE query_id=? AND chunk_id=?", key
                ).fetchone():
                    seeded = True
                    break
            if not seeded:
                phase11_missing.append(row)
    finally:
        for connection, _ in cache_connections:
            connection.close()
    phase11_positions = {
        (int(row[0]), str(row[1])): index for index, row in enumerate(phase11_missing)
    }

    result: dict[str, list[tuple]] = {}
    for key in sorted(targets):
        if key in round_positions:
            rows = round_rows
            index = round_positions[key]
        elif key in phase11_positions:
            rows = phase11_missing
            index = phase11_positions[key]
        else:
            raise ValueError(f"cannot reconstruct historical scoring batch for {key}")
        start = index - (index % 2)
        batch = rows[start : start + 2]
        if len(batch) != 2:
            raise ValueError(f"incomplete historical batch for {key}")
        result[f"{key[0]}:{key[1]}"] = batch
    return result


def _token_trace(reranker, pair: tuple[str, str], batch: list[tuple[str, str]]) -> dict:
    single = reranker.tokenizer(
        pair[0], pair[1], truncation=False, add_special_tokens=True
    )["input_ids"]
    truncated = reranker.tokenizer(
        pair[0], pair[1], truncation="longest_first", max_length=512, add_special_tokens=True
    )["input_ids"]
    encoded = reranker.tokenizer(
        batch, padding=True, truncation="longest_first", max_length=512, return_tensors="pt"
    )
    target_index = batch.index(pair)
    ids = encoded["input_ids"][target_index].tolist()
    mask = encoded["attention_mask"][target_index].tolist()
    nonpad = ids[: sum(mask)]
    return {
        "token_length_before_truncation": len(single),
        "token_length_after_truncation": len(truncated),
        "padded_tensor_length": len(ids),
        "input_ids_nonpadding_sha256": hashlib.sha256(json.dumps(nonpad).encode()).hexdigest(),
        "attention_mask_sha256": hashlib.sha256(json.dumps(mask).encode()).hexdigest(),
        "nonpadding_ids_equal_single_truncated": nonpad == truncated,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", default="artifacts/phase14_medlatec_coverage_3000/reranker_equivalence_control.json")
    parser.add_argument("--output", default="artifacts/phase14_medlatec_coverage_3000/reranker_equivalence_diagnosis.json")
    args = parser.parse_args()
    control = json.loads(Path(args.control).read_text(encoding="utf-8"))
    historical = historical_batches(control)
    runtime = yaml.safe_load(Path("data/source_census/phase14_medlatec_coverage_3000/runtime.yaml").read_text(encoding="utf-8"))
    model = runtime["models"]["reranker"]
    reranker = TransformerCrossEncoderReranker(
        model_name=model["name"], revision=model["revision"], cache_dir=model["cache_dir"],
        device="cuda", batch_size=2, max_length=512, use_half_on_cuda=True, seed=2026,
    )
    reranker.score_pairs([("warmup", "warmup")])
    phase14 = _ro(PHASE14)
    round1 = _ro(ROUND1)
    phase11 = _ro(PHASE11C)
    try:
        target_rows = []
        for item in control["pairs"]:
            row = phase14.execute(
                "SELECT query_id,chunk_id,query_text,chunk_text,rerank_score FROM pairs WHERE query_id=? AND chunk_id=?",
                (int(item["query_id"]), str(item["chunk_id"])),
            ).fetchone()
            if row is None:
                raise ValueError(f"missing target {item['query_id']}:{item['chunk_id']}")
            target_rows.append(row)
        fixture_pairs = [(str(row[2]), str(row[3])) for row in target_rows]
        fixture_scores = reranker.score_pairs(fixture_pairs).tolist()
        details = []
        for index, (item, target) in enumerate(zip(control["pairs"], target_rows, strict=True)):
            key = f"{int(target[0])}:{target[1]}"
            history_rows = historical[key]
            source = round1 if round1.execute(
                "SELECT 1 FROM pairs WHERE query_id=? AND chunk_id=?", (int(target[0]), str(target[1]))
            ).fetchone() else phase11
            batch_rows = []
            for history_row in history_rows:
                candidate = source.execute(
                    "SELECT query_id,chunk_id,query_text,chunk_text,rerank_score FROM pairs WHERE query_id=? AND chunk_id=?",
                    (int(history_row[0]), str(history_row[1])),
                ).fetchone()
                if candidate is None:
                    raise ValueError(f"missing historical context {history_row[0]}:{history_row[1]}")
                batch_rows.append(candidate)
            batch_pairs = [(str(row[2]), str(row[3])) for row in batch_rows]
            batch_scores = reranker.score_pairs(batch_pairs).tolist()
            target_position = next(i for i, row in enumerate(batch_rows) if int(row[0]) == int(target[0]) and str(row[1]) == str(target[1]))
            historical_score = float(batch_scores[target_position])
            fixture_batch_start = index - (index % 2)
            fixture_batch_rows = target_rows[fixture_batch_start : fixture_batch_start + 2]
            fixture_batch_pairs = [(str(row[2]), str(row[3])) for row in fixture_batch_rows]
            history_trace = _token_trace(reranker, (str(target[2]), str(target[3])), batch_pairs)
            fixture_trace = _token_trace(reranker, (str(target[2]), str(target[3])), fixture_batch_pairs)
            details.append({
                "query_id": int(target[0]), "chunk_id": str(target[1]),
                "reference_score": float(item["reference_score"]),
                "fixture_score": float(fixture_scores[index]),
                "fixture_absolute_difference": abs(float(fixture_scores[index]) - float(item["reference_score"])),
                "historical_batch_score": historical_score,
                "historical_absolute_difference": abs(historical_score - float(item["reference_score"])),
                "fixture_batch_keys": [f"{int(row[0])}:{row[1]}" for row in fixture_batch_rows],
                "historical_batch_keys": [f"{int(row[0])}:{row[1]}" for row in batch_rows],
                "historical_target_position": target_position,
                "tokenization": {
                    "fixture": fixture_trace, "historical": history_trace,
                    "nonpadding_input_ids_equal": fixture_trace["input_ids_nonpadding_sha256"] == history_trace["input_ids_nonpadding_sha256"],
                    "full_attention_masks_equal": fixture_trace["attention_mask_sha256"] == history_trace["attention_mask_sha256"],
                },
            })
    finally:
        phase14.close(); round1.close(); phase11.close(); reranker.release()
    tolerance = 0.005859375
    result = {
        "root_cause_tested": "FP16 padding/batch-shape dependence caused by control fixture not replaying original batch partners",
        "control_keys_unchanged": len(details) == 8,
        "before_max_absolute_difference": max(row["fixture_absolute_difference"] for row in details),
        "after_max_absolute_difference": max(row["historical_absolute_difference"] for row in details),
        "fixed_tolerance": tolerance,
        "historical_context_passed": all(row["historical_absolute_difference"] <= tolerance for row in details),
        "contract": control["contract"],
        "details": details,
    }
    write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["historical_context_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
