import csv
import json
from pathlib import Path

import pyarrow.parquet as pq

from src.evaluation.silver import validate_benchmark


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/system_upgrade_m1"


def test_inventory_preserves_distinct_pipeline_populations():
    row=json.loads((OUT/"corpus_inventory.json").read_text(encoding="utf-8"))
    assert row["official"]["documents"]==1_194_469
    assert row["local_content"]["deduplicated_union"]["usable_documents"]==275_586
    assert row["local_content"]["deduplicated_union"]["official_gap_documents"]==918_883
    assert row["indexes"]["single_unified_index_exists"] is False
    assert "official URL != acquired" in row["count_semantics"]


def test_silver_benchmark_is_valid_but_explicitly_insufficient():
    official=set(map(int,pq.read_table(ROOT/"data/raw/links_corpus_vi_en_zh.parquet",columns=["id"]).column(0).to_pylist()))
    result=validate_benchmark(OUT/"silver_benchmark.parquet",official)
    manifest=json.loads((OUT/"silver_benchmark_manifest.json").read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert manifest["status"]=="INSUFFICIENT_FOR_ARCHITECTURE_SELECTION"
    assert manifest["tracks"]["vi_to_en"]=="INSUFFICIENT"
    assert result["verbatim_source_queries"]==result["queries"]


def test_quality_examples_retain_provenance_and_do_not_auto_reject_short_answers():
    audit=json.loads((OUT/"data_quality_audit.json").read_text(encoding="utf-8"))
    assert "not automatic rejection" in audit["heuristic_warning"]
    with (OUT/"data_quality_examples.csv").open(encoding="utf-8-sig") as handle:
        rows=list(csv.DictReader(handle))
    assert rows
    assert all(row["doc_id"] and row["source_url"].startswith(("http://","https://")) for row in rows)


def test_milestone_reports_mark_unverifiable_relevance_unknown():
    retrieval=json.loads((OUT/"retrieval_funnel.json").read_text(encoding="utf-8"))
    assert retrieval["no_gold_relevance_claims"] is True
    assert retrieval["bottleneck_partition"]["absent_from_local_content"]["relevance"]=="UNKNOWN"
    assert (OUT/"offline_evaluator.py").is_file()
