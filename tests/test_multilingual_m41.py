from __future__ import annotations

from src.multilingual_retrieval.m41 import ChineseRetriever, group_hits, semantic_preflight, understand


def row(doc: int, text: str, page: str = "MEDICAL_ARTICLE") -> dict:
    return {"doc_id": doc, "chunk_id": f"{doc}-test", "evidence_text": text,
            "page_type": page, "canonical_url": f"https://example.invalid/{doc}"}


def test_no_unsafe_vietnamese_substring_match():
    parsed = understand("Lăn tay và sau này có vấn đề gì?")
    assert not any(t["vi"] == "ăn" for t in parsed["terms"])
    assert parsed["ambiguous"]


def test_longest_entity_does_not_turn_malaria_into_fever():
    parsed = understand("Sốt rét điều trị như thế nào?")
    assert parsed["entity_groups"] == [["疟疾"]]


def test_negative_exposure_survives_polar_question_suffix():
    parsed = understand("Không uống rượu thì GGT tăng có phải bệnh không?")
    assert len(parsed["negations"]) == 1
    assert parsed["negations"][0]["start"] == 0
    assert parsed["polar_question"]
    assert ["不饮酒", "不喝酒", "无饮酒史"] in parsed["qualifier_groups"]


def test_disease_population_and_timing_preserved():
    parsed = understand("Trẻ em hen suyễn ho buổi sáng điều trị thế nào?")
    assert ["儿童", "小儿"] in parsed["qualifier_groups"]
    assert ["早晨", "晨起", "清晨"] in parsed["qualifier_groups"]
    assert ["哮喘"] in parsed["entity_groups"]


def test_unsupported_terms_are_not_silently_translated():
    parsed = understand("Bệnh Wilson có thể dùng trientine không?")
    assert "wilson" in parsed["unmapped_query_tokens"]
    assert "trientine" in parsed["unmapped_query_tokens"]
    assert parsed["ambiguous"]


def test_actual_query_only_end_to_end_and_deterministic_provenance():
    retriever = ChineseRetriever([row(1, "哮喘儿童清晨咳嗽。治疗应结合病情。"),
                                  row(2, "糖尿病治疗与饮食。")])
    result = retriever.retrieve("Trẻ em hen suyễn ho buổi sáng điều trị thế nào?")
    assert result["results"][0]["doc_id"] == 1
    assert result["results"][0]["chunk_id"] == "1-test"
    assert result["results"][0]["answer_relevance"] == "UNVERIFIED"
    assert result["results"] == retriever.retrieve("Trẻ em hen suyễn ho buổi sáng điều trị thế nào?")["results"]


def test_rejected_catalog_remains_in_candidate_diagnostics():
    retriever = ChineseRetriever([row(1, "哮喘的治疗。", "DIRECTORY_OR_NAVIGATION")])
    result = retriever.retrieve("Hen suyễn điều trị thế nào?")
    assert len(result["candidates"]) == 1
    assert result["results"] == []
    assert "PAGE_TYPE_NOT_ARTICLE" in result["candidates"][0]["admission_reasons"]


def test_all_required_entities_not_just_one_are_checked():
    retriever = ChineseRetriever([row(1, "甲减治疗需要注意。")])
    result = retriever.retrieve("Vitamin C ảnh hưởng suy giáp thế nào?")
    assert result["results"] == []
    assert "MISSING_REQUIRED_ENTITY" in result["candidates"][0]["admission_reasons"]


def test_chinese_medical_abbreviations_are_not_substring_collisions():
    assert group_hits("SHp感染", [["Hp感染"]]) == 0
    assert group_hits("确诊Hp感染后应治疗", [["Hp感染"]]) == 1


def test_unresolved_negation_does_not_become_global_polarity():
    parsed = understand("Không phải hen suyễn, cũng không tăng huyết áp, tại sao đau đầu?")
    assert parsed["negations"]
    assert parsed["ambiguous"]
    assert "UNRESOLVED_NEGATION_SCOPE" in parsed["unsupported_or_ambiguous"]


def test_reference_list_inside_medical_page_does_not_become_answer():
    retriever = ChineseRetriever([row(1, "哮喘治疗\n" + "\n".join("传染病索引条目" for _ in range(20)))])
    result = retriever.retrieve("Hen suyễn điều trị thế nào?")
    assert result["candidates"]
    assert "NAVIGATION_OR_LIST_FRAGMENT" in result["candidates"][0]["admission_reasons"]
    assert result["results"] == []


def test_informational_negation_and_embedded_polar_question():
    parsed = understand("Bilirubin trực tiếp hơi cao. Em chưa hiểu và cần tư vấn.")
    assert parsed["negations"][0]["scope_kind"] == "INFORMATIONAL"
    assert not parsed["ambiguous"]
    parsed = understand("Em có bị bệnh gì không ạ? Nếu viêm nang lông thì chữa trị thế nào?")
    assert not parsed["negations"]
    assert not parsed["ambiguous"]


def test_semantic_preflight_defers_for_real_worker_command_shape(monkeypatch):
    from types import SimpleNamespace
    from src.multilingual_retrieval import m41
    proc = SimpleNamespace(pid=123,info={"name":"python.exe", "cmdline":["python.exe","tools/phase27_suckhoedoisong_downstream.py"]})
    monkeypatch.setattr(m41.psutil,"process_iter",lambda fields:iter([proc]))
    result = semantic_preflight()
    assert result["status"] == "DEFERRED_PHASE27_ACTIVE"
    assert result["phase27_pids"] == [123]
    assert not result["model_loaded"] and not result["inference_started"]


def test_one_resolved_negation_does_not_hide_another_scope():
    parsed = understand("Không uống rượu nhưng không điều trị tăng huyết áp có nguy cơ gì?")
    assert len(parsed["negations"]) == 2
    assert parsed["ambiguous"]
    assert "UNRESOLVED_NEGATION_SCOPE" in parsed["unsupported_or_ambiguous"]


def test_decomposed_unicode_keeps_source_spans_valid():
    import unicodedata
    raw = unicodedata.normalize("NFD","Không uống rượu thì GGT tăng do đâu?")
    parsed = understand(raw)
    assert parsed["query_vi"] == raw
    for item in parsed["terms"]:
        assert parsed["query_nfc"][item["start"]:item["end"]].casefold() == item["vi"]
