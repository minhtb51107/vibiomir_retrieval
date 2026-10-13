import json

from tools import phase13_diagnosis as diagnosis
from tools.phase13_diagnosis import diverse_selection, distribution, relative_inversions, topic_bucket


def test_relative_inversions_only_compares_retained_items():
    assert relative_inversions([1, 2, 3, 4], [9, 2, 1, 4]) == 1


def test_distribution_is_deterministic_and_explicit_for_empty_input():
    assert distribution([])["median"] is None
    result = distribution([0, 1, 2, 3, 4])
    assert result["count"] == 5 and result["median"] == 2.0


def test_topic_bucket_uses_vietnamese_query_terms_without_labels():
    assert topic_bucket("Tôi bị đau dạ dày và buồn nôn") == "tiêu hóa"
    assert topic_bucket("Một câu hỏi không có từ khóa") == "khác"


def test_diverse_selection_round_robins_topics_and_respects_exclusions():
    rows=[{"query_id":1},{"query_id":2},{"query_id":3},{"query_id":4}]
    queries={1:"đau dạ dày",2:"viêm dạ dày",3:"đau phổi",4:"đau tim"}
    selected=diverse_selection(rows,queries,3,{4})
    assert selected==[3,1,2]


def test_output_comparison_is_reproducible_and_counts_only_real_replacements(tmp_path,monkeypatch):
    old=[{"id":1,"relevant_docs":[1,2],"relevant_chunks":[{"doc_id":1,"chunk_text":"a"},{"doc_id":2,"chunk_text":"b"}]}]
    new=[{"id":1,"relevant_docs":[3,1],"relevant_chunks":[{"doc_id":3,"chunk_text":"c"},{"doc_id":1,"chunk_text":"a"}]}]
    old_path=tmp_path/"old.json";new_path=tmp_path/"new.json"
    old_path.write_text(json.dumps(old),encoding="utf-8");new_path.write_text(json.dumps(new),encoding="utf-8")
    monkeypatch.setattr(diagnosis,"P11_JSON",old_path);monkeypatch.setattr(diagnosis,"P12_JSON",new_path)

    first,rows=diagnosis.compare_outputs({3});second,_=diagnosis.compare_outputs({3})

    assert first==second
    assert first["displaced_document_slots"]==1 and first["inserted_pubmed_document_slots"]==1
    assert rows[1]["document_relative_order_inversions"]==0
