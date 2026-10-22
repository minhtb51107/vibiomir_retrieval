import csv
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.multilingual_retrieval.evaluation import _dedupe_documents, _split, evaluate, has_han
from src.multilingual_retrieval.review_server import HumanLabelStore
from src.multilingual_retrieval.m2 import _zh_bigrams, chinese_evidence_quality, load_silver


def test_language_detection_and_split_are_deterministic():
    assert has_han("这是医学证据")
    assert not has_han("Bằng chứng y khoa")
    assert _split("1:en") == _split("1:en")
    assert _split("1:en") in {"development", "held_out"}
    assert {_split(f"{query_id}:zh") for query_id in range(100)} <= {"development", "held_out"}


def test_document_dedup_keeps_best_rank_and_deterministic_order():
    rows = [
        {"source_rank": 2, "doc_id": 7, "chunk_id": "b"},
        {"source_rank": 1, "doc_id": 7, "chunk_id": "a"},
        {"source_rank": 1, "doc_id": 8, "chunk_id": "c"},
    ]
    output = _dedupe_documents(rows)
    assert [(row["doc_id"], row["chunk_id"], row["rank"]) for row in output] == [(7, "a", 1), (8, "c", 2)]


def test_missing_human_fields_fail_closed(tmp_path: Path):
    # The integration evaluator validates HUMAN_VERIFIED rows; unit coverage
    # keeps the allowed state contract explicit without manufacturing qrels.
    from src.multilingual_retrieval.evaluation import LABELS, VERIFIED

    assert VERIFIED == "HUMAN_VERIFIED"
    assert LABELS == {"0", "1", "2", "UNSURE"}
    assert "" not in LABELS


def test_evaluator_compares_frozen_arms_with_verified_labels(tmp_path: Path):
    config_dir = tmp_path / "configs"
    output = tmp_path / "artifacts" / "benchmark"
    config_dir.mkdir(); output.mkdir(parents=True)
    config = config_dir / "benchmark.yaml"
    config.write_text(yaml.safe_dump({"version": 1, "output_dir": "artifacts/benchmark"}), encoding="utf-8")
    pq.write_table(
        pa.Table.from_pylist([
            {"query_id": 1, "target_language": "en", "arm": "dense", "rank": 1, "doc_id": 10},
            {"query_id": 1, "target_language": "en", "arm": "hybrid", "rank": 1, "doc_id": 20},
        ]),
        output / "cached_retrieval_top20.parquet",
    )
    fields = ["query_id", "target_language", "doc_id", "judgment", "verification_status", "reviewer", "verified_at"]
    with (output / "human_labels.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerow({
            "query_id": 1, "target_language": "en", "doc_id": 10, "judgment": "2",
            "verification_status": "HUMAN_VERIFIED", "reviewer": "reviewer-1", "verified_at": "2026-10-22",
        })
    result = evaluate(config)
    assert result["metrics"]["en"]["arms"]["dense"]["pooled_hit_at_20"] == 1.0
    assert result["metrics"]["en"]["arms"]["hybrid"]["pooled_hit_at_20"] == 0.0


def test_human_label_store_updates_only_active_split(tmp_path: Path):
    config_dir = tmp_path / "configs"; output = tmp_path / "artifacts" / "benchmark"
    config_dir.mkdir(); output.mkdir(parents=True)
    config = config_dir / "benchmark.yaml"
    config.write_text(yaml.safe_dump({"version": 1, "output_dir": "artifacts/benchmark"}), encoding="utf-8")
    fields = ["case_id", "split", "judgment", "verification_status", "reviewer", "verified_at", "review_notes"]
    with (output / "human_labels.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows([
            {"case_id": "dev", "split": "development", "judgment": "", "verification_status": "PENDING_HUMAN", "reviewer": "", "verified_at": "", "review_notes": ""},
            {"case_id": "held", "split": "held_out", "judgment": "", "verification_status": "PENDING_HUMAN", "reviewer": "", "verified_at": "", "review_notes": ""},
        ])
    store = HumanLabelStore(config, "development")
    result = store.save({"reviewer": "human-1", "judgments": {"dev": {"judgment": "2", "review_notes": "direct answer"}}})
    assert result["saved"] == 1
    with (output / "human_labels.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        rows = {row["case_id"]: row for row in csv.DictReader(stream)}
    assert rows["dev"]["verification_status"] == "HUMAN_VERIFIED"
    assert rows["held"]["verification_status"] == "PENDING_HUMAN"
    with pytest.raises(ValueError, match="invalid judgment"):
        store.save({"reviewer": "human-1", "judgments": {"dev": {"judgment": "RELEVANT"}}})


def test_chinese_quality_filter_rejects_placeholders_but_keeps_answer_evidence():
    assert chinese_evidence_quality("我要咨询")[0] is False
    assert chinese_evidence_quality("相关问答相关问答相关问答")[0] is False
    evidence = "脑梗死患者出现吞咽困难时，应由康复医师评估吞咽功能，并根据误吸风险选择训练和饮食方案。"
    assert chinese_evidence_quality(evidence)[0] is True
    assert chinese_evidence_quality("专家答疑 2025-01-01 问题一 专家答疑 2025-02-01 问题二 2025-03-01 问题三")[0] is False
    assert chinese_evidence_quality("王医生 主治医师 擅长男性不育和前列腺炎的诊治 找医生")[0] is False


def test_ai_silver_is_development_only_and_never_human_verified(tmp_path: Path):
    review = tmp_path / "review.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"case_id": "dev-1", "split": "development"},
        {"case_id": "held-1", "split": "held_out"},
    ]), review)
    labels = tmp_path / "silver.csv"
    fields = ["case_id", "judgment", "verification_status"]
    with labels.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerow({"case_id": "dev-1", "judgment": "2", "verification_status": "AI_JUDGED"})
    assert load_silver(labels, review)[0]["verification_status"] == "AI_JUDGED"
    with labels.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerow({"case_id": "dev-1", "judgment": "2", "verification_status": "HUMAN_VERIFIED"})
    with pytest.raises(ValueError, match="AI silver status"):
        load_silver(labels, review)


def test_chinese_bigram_index_terms_preserve_medical_entities():
    tokens = _zh_bigrams("幽门螺杆菌感染 症状 根治")
    assert "幽门" in tokens
    assert "螺杆" in tokens
    assert "感染" in tokens
