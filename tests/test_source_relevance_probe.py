from __future__ import annotations

import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.reranking.checkpoint import RerankCheckpoint
from src.source_probe.pipeline import (
    checkpoint_signature, selected_document_ids, short_qa_eligible, NOVEL_SCHEMA,
)


RULE = {
    "extraction_status": "SUCCESS", "minimum_clean_characters": 100,
    "minimum_paragraphs": 1, "minimum_medical_signals": 1,
    "excluded_statuses": ["JS_SHELL", "EMPTY_CONTENT", "EXTRACTION_FAILED"],
}


def test_short_qa_rule_keeps_meaningful_short_medical_text() -> None:
    row = {
        "extraction_status": "SUCCESS", "clean_char_count": 110,
        "paragraph_count": 1, "normalized_text": "患者需要医生检查和治疗。" * 8,
    }
    assert short_qa_eligible(row, RULE)
    assert not short_qa_eligible({**row, "clean_char_count": 99}, RULE)
    assert not short_qa_eligible({**row, "normalized_text": "普通文字" * 30}, RULE)
    assert not short_qa_eligible({**row, "extraction_status": "JS_SHELL"}, RULE)


def test_document_selection_excludes_pilot_ids_and_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "docs.parquet"
    rows = [
        {"doc_id": 1, "extraction_status": "SUCCESS", "clean_char_count": 120, "paragraph_count": 1, "normalized_text": "医生治疗" * 30},
        {"doc_id": 2, "extraction_status": "SUCCESS", "clean_char_count": 120, "paragraph_count": 1, "normalized_text": "医生治疗" * 30},
        {"doc_id": 3, "extraction_status": "EXTRACTION_FAILED", "clean_char_count": 120, "paragraph_count": 1, "normalized_text": "医生治疗" * 30},
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)
    first = selected_document_ids(path, inclusion="short_qa", short_rule=RULE, excluded_doc_ids={2})
    second = selected_document_ids(path, inclusion="short_qa", short_rule=RULE, excluded_doc_ids={2})
    assert first == second == {1}


def test_checkpoint_signature_is_pool_specific_and_reusable_scores_remain_exact(tmp_path: Path) -> None:
    metadata = {
        "model_name": "m", "resolved_revision": "r", "max_sequence_length": 512,
        "inference_precision": "float16", "batch_size": 2,
    }
    pool = {"input_hashes": {"candidate_pool_sha256": "a"}}
    signature = checkpoint_signature(pool, metadata)
    path = tmp_path / "scores.sqlite"
    with RerankCheckpoint(path, signature=signature) as checkpoint:
        checkpoint.write_scores([(1, "chunk-a", 0.5, 10.0)])
        assert checkpoint.count == 1
        assert checkpoint.read_scores()[(1, "chunk-a")] == (0.5, 10.0)
        assert checkpoint.integrity_check() == "ok"


def test_novel_pair_schema_deduplicates_globally_but_preserves_memberships(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "novel.sqlite")
    connection.executescript(NOVEL_SCHEMA)
    pair = (1, "chunk-a", "query", "source text")
    connection.execute("INSERT INTO pairs(query_id,chunk_id,query_text,chunk_text) VALUES (?,?,?,?)", pair)
    connection.execute("INSERT INTO pairs(query_id,chunk_id,query_text,chunk_text) VALUES (?,?,?,?) ON CONFLICT DO NOTHING", pair)
    connection.executemany("INSERT INTO memberships VALUES (?,?,?)", [("S1", 1, "chunk-a"), ("S2", 1, "chunk-a")])
    assert connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM memberships").fetchone()[0] == 2
    connection.close()
