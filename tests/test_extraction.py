from pathlib import Path

from src.probing.extraction import (
    analyze_html,
    decode_document,
    density_extract,
    semantic_extract,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_title_and_generic_article_extraction_removes_page_noise() -> None:
    html = (FIXTURES / "article_vi.html").read_text(encoding="utf-8")
    analysis = analyze_html(html)
    assert analysis.title == "Phòng bệnh mùa mưa"
    assert analysis.template_classification == "NEWS_TEMPLATE"
    assert analysis.language_signal == "vi-signal"
    assert analysis.structure["jsonld_types"] == ["NewsArticle"]
    assert analysis.structure["jsonld_article_body_count"] == 1

    for extractor in (semantic_extract, density_extract):
        title, text, paragraphs = extractor(html)
        assert title == "Phòng bệnh mùa mưa"
        assert "giữ vệ sinh" in text
        assert "All rights reserved" not in text
        assert paragraphs >= 3


def test_qa_like_page_and_structured_data_are_detected() -> None:
    html = (FIXTURES / "q_and_a_zh.html").read_text(encoding="utf-8")
    analysis = analyze_html(html)
    assert analysis.template_classification == "Q_AND_A_TEMPLATE"
    assert analysis.language_signal == "zh-Han-script"
    assert analysis.structure["qa_marker_count"] >= 2
    assert "QAPage" in analysis.structure["jsonld_types"]


def test_empty_and_invalid_html_are_safe() -> None:
    empty = analyze_html("")
    assert empty.title is None
    assert empty.template_classification == "UNKNOWN"
    assert all(metric.empty for metric in empty.extraction_metrics)

    malformed = analyze_html("<html><title>Broken<p>Still readable")
    assert malformed.title is not None
    assert malformed.basic_text_length > 0


def test_encoding_fallback_uses_meta_charset() -> None:
    source = (
        '<html><head><meta charset="gb18030"><title>健康知识</title></head>'
        "<body><p>请及时就医并接受专业检查。</p></body></html>"
    )
    decoded = decode_document(source.encode("gb18030"), "text/html")
    assert decoded.declared_http_encoding is None
    assert decoded.declared_meta_encoding == "gb18030"
    assert decoded.selected_encoding == "gb18030"
    assert "健康知识" in decoded.text
    assert decoded.replacement_count == 0


def test_valid_vietnamese_letters_are_not_mojibake() -> None:
    source = "<html><body><p>Âm nhạc và bảo đảm sức khỏe.</p></body></html>"
    decoded = decode_document(source.encode("utf-8"), "text/html; charset=utf-8")
    assert decoded.mojibake_marker_count == 0


def test_tiny_script_shell_is_classified_as_javascript_heavy() -> None:
    analysis = analyze_html("<html><body><script>window.app={}</script></body></html>")
    assert analysis.template_classification == "JAVASCRIPT_HEAVY"
    assert "JS_ONLY_SHELL" in analysis.anti_bot_signals
