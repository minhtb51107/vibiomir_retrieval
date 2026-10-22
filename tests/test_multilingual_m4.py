from __future__ import annotations

import json
from pathlib import Path

from src.multilingual_retrieval.m4 import classify_m4_page, load_lexicon, understand_vi_query


ROOT = Path(__file__).resolve().parents[1]


def test_query_understanding_is_query_text_only_and_preserves_medical_relation() -> None:
    lexicon = load_lexicon(ROOT / "configs/m4_vi_zh_medical_lexicon.json")
    parsed = understand_vi_query("Không uống rượu nhưng GGT tăng do đâu?", lexicon)
    assert "谷氨酰转移酶" in parsed["query_zh"]
    assert "原因" in parsed["query_zh"]
    assert "不" in parsed["query_zh"]
    assert not parsed["ambiguous"]


def test_polar_question_does_not_create_false_negation() -> None:
    lexicon = load_lexicon(ROOT / "configs/m4_vi_zh_medical_lexicon.json")
    parsed = understand_vi_query("Hắt hơi buổi sáng có phải viêm mũi dị ứng không?", lexicon)
    assert "没有" not in parsed["query_zh"]
    assert "不" not in parsed["query_zh"]
    parsed = understand_vi_query("Người cao tuổi có nguy cơ hạ đường huyết khi bỏ bữa không?", lexicon)
    assert "没有" not in parsed["query_zh"]


def test_page_classifier_keeps_substantive_article_before_navigation_tail() -> None:
    body = ("清晨打喷嚏可能由鼻炎引起。由于反流刺激，睡醒时症状最明显。"
            "过敏性鼻炎与反流性鼻炎存在区别；应结合反酸等表现检查。" * 6
            + "上一篇：其他文章 下一篇：更多文章 相关文章 疾病常识 诊断治疗")
    page_type, reasons = classify_m4_page("鼻炎不一定是过敏", "https://www.cnkang.com/dise/565/x.html", body)
    assert page_type == "MEDICAL_ARTICLE"
    assert reasons in ([], ["SUBSTANTIVE_BODY_BEFORE_NAV"])


def test_page_classifier_still_rejects_drug_catalog() -> None:
    page_type, _ = classify_m4_page("治疗胃病的药品列表", "https://example.invalid/list", "药品治疗。" * 200)
    assert page_type == "DIRECTORY_OR_NAVIGATION"


def test_m4_silver_decisions_are_explicitly_not_human_verified() -> None:
    data = json.loads((ROOT / "configs/m4_ai_silver_decisions.json").read_text(encoding="utf-8"))
    assert data["version"].startswith("AI_JUDGED_M4")
    assert len(data["decisions"]) >= 15
    assert sum(row["judgment"] == 2 for row in data["decisions"]) == 1


def test_causal_paraphrase_is_available_for_why_question() -> None:
    lexicon = load_lexicon(ROOT / "configs/m4_vi_zh_medical_lexicon.json")
    parsed = understand_vi_query("Tại sao gần sáng thường bị hắt xì?", lexicon)
    assert "引起" in parsed["query_zh"]
