import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.evaluation.silver import evaluate_rankings, validate_benchmark


def benchmark(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)


def row(qid="silver_vi_1", text="Đau đầu?", doc=10, split="development"):
    return {"silver_query_id":qid,"query_text":text,"positive_doc_id":doc,"source_url":f"https://a.vn/{doc}",
            "source_domain":"a.vn","query_language":"vi","document_language":"vi","split":split,
            "provenance_type":"source_title_question","verbatim_source_query":True}


def test_known_positive_metrics_and_unknown_query_guard(tmp_path):
    b=tmp_path/"b.parquet"; r=tmp_path/"r.parquet"; benchmark(b,[row(),row("silver_vi_2","Ho?",20)])
    pq.write_table(pa.Table.from_pylist([
        {"silver_query_id":"silver_vi_1","rank":1,"doc_id":11,"source_domain":"x.vn"},
        {"silver_query_id":"silver_vi_1","rank":2,"doc_id":10,"source_domain":"a.vn"},
        {"silver_query_id":"silver_vi_2","rank":30,"doc_id":20,"source_domain":"a.vn"},
    ]),r)
    result=evaluate_rankings(b,r,{10,11,20})
    assert result["known_positive_recall_at_20"]==0.5
    assert result["known_positive_recall_at_100"]==1.0
    assert result["mrr"]==(0.5+1/30)/2


def test_normalized_query_cannot_cross_splits(tmp_path):
    b=tmp_path/"b.parquet"; benchmark(b,[row(),row("silver_vi_2","  đau đầu!!! ",20,"held_out")])
    with pytest.raises(ValueError,match="leakage"):
        validate_benchmark(b,{10,20})


def test_official_query_namespace_is_rejected(tmp_path):
    b=tmp_path/"b.parquet"; benchmark(b,[row("12")])
    with pytest.raises(ValueError,match="organizer query-ID"):
        validate_benchmark(b,{10})
