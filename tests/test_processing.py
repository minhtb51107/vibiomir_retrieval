from pathlib import Path

from src.processing.chunking import ChunkConfig, chunk_document
from src.processing.decoding import decode_body
from src.processing.extraction import ExtractionConfig, extract_document
from src.processing.models import CleanDocument, DocumentSection, ExtractionStatus


FIXTURES = Path(__file__).parent / "fixtures"


def _extract(html: str, doc_id: int = 1) -> CleanDocument:
    return extract_document(
        doc_id=doc_id,
        original_url=f"https://example.test/{doc_id}",
        final_url=f"https://example.test/{doc_id}",
        body=html.encode("utf-8"),
        content_type_header="text/html; charset=utf-8",
        archive_shard="shard-000000.gz",
        archive_member_offset=0,
        archive_checksum_sha256="checksum",
    )


def test_utf8_decoding_and_nfc_normalization() -> None:
    decomposed = "Tiếng Việt"
    result = decode_body(decomposed.encode("utf-8"), "text/html; charset=utf-8")
    assert result.selected_encoding == "utf-8"
    assert result.text == "Tiếng Việt"
    assert result.nfc_changed
    assert result.replacement_character_count == 0


def test_gb2312_meta_decoding_without_domain_assumption() -> None:
    html = '<html><head><meta charset="gb2312"></head><body>健康知识</body></html>'
    result = decode_body(html.encode("gb2312"), "text/html")
    assert result.meta_encoding == "gb2312"
    assert result.selected_encoding == "gb2312"
    assert "健康知识" in result.text


def test_semantic_selection_and_density_fallback() -> None:
    substantial = " ".join(["Nội dung sức khỏe quan trọng."] * 40)
    semantic = _extract(f"<html><title>Bài viết</title><article><p>{substantial}</p></article></html>")
    assert semantic.extraction_method == "semantic_container"
    assert semantic.extraction_status == ExtractionStatus.SUCCESS

    density_html = (
        "<html><title>Bài viết</title><article><p>Ngắn.</p></article>"
        f"<div class='wrapper'><p>{substantial}</p></div></html>"
    )
    density = _extract(density_html, 2)
    assert density.extraction_method == "density_scored"
    assert density.density_char_count > density.semantic_char_count


def test_qa_structure_empty_and_js_shell_classification() -> None:
    qa_html = (FIXTURES / "q_and_a_zh.html").read_text(encoding="utf-8")
    qa = _extract(qa_html)
    kinds = [section.section_type for section in qa.sections]
    assert "question" in kinds
    assert "answer" in kinds
    assert qa.extraction_method.endswith("+qa_structured")

    empty = _extract("<html><body></body></html>", 2)
    assert empty.extraction_status == ExtractionStatus.EMPTY_CONTENT
    shell = _extract("<html><script>window.app={}</script></html>", 3)
    assert shell.extraction_status == ExtractionStatus.JS_SHELL
    assert shell.normalized_text == ""


def _chunk_document() -> CleanDocument:
    title = "Medical title"
    body = " ".join(f"word{i}" for i in range(30))
    text = f"{title}\n\n{body}"
    sections = [
        DocumentSection("title", title, 0, len(title)),
        DocumentSection("content", body, len(title) + 2, len(text), ["Heading"]),
    ]
    return CleanDocument(
        doc_id=42,
        original_url="https://example.test/42",
        final_url="https://example.test/final/42",
        title=title,
        language_signal="latin-undetermined",
        selected_encoding="utf-8",
        normalized_text=text,
        sections=sections,
        extraction_method="semantic_container",
        extraction_status=ExtractionStatus.SUCCESS,
        raw_byte_length=len(text.encode()),
        raw_char_count=len(text),
        clean_char_count=len(text),
        paragraph_count=1,
        replacement_character_count=0,
        semantic_char_count=len(text),
        density_char_count=len(text),
        boilerplate_signal_count=0,
        raw_sha256="hash",
        archive_shard="shard.gz",
        archive_member_offset=0,
        archive_checksum_sha256="checksum",
    )


def test_paragraph_aware_chunking_overlap_offsets_and_provenance() -> None:
    document = _chunk_document()
    config = ChunkConfig("test", target_tokens=10, overlap_tokens=2)
    first = chunk_document(document, config)
    second = chunk_document(document, config)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len(first) > 1
    for index, chunk in enumerate(first):
        assert chunk.doc_id == document.doc_id
        assert chunk.chunk_index == index
        assert 0 <= chunk.start_offset < chunk.end_offset <= len(document.normalized_text)
        assert chunk.raw_text == document.normalized_text[chunk.start_offset:chunk.end_offset]
        assert chunk.normalized_text == chunk.raw_text
        assert chunk.source_url == document.final_url
    first_tokens = first[0].raw_text.split()
    second_tokens = first[1].raw_text.split()
    assert first_tokens[-2:] == second_tokens[:2]


def test_chunk_end_prefers_paragraph_boundary() -> None:
    first_paragraph = "one two three four five six"
    second_paragraph = "seven eight nine ten eleven twelve"
    text = f"{first_paragraph}\n\n{second_paragraph}"
    document = _chunk_document()
    document.normalized_text = text
    document.sections = [DocumentSection("content", text, 0, len(text))]
    chunks = chunk_document(document, ChunkConfig("paragraphs", 10, 2))
    assert chunks[0].raw_text == first_paragraph
    assert chunks[0].end_offset == len(first_paragraph)


def test_qa_chunks_never_cross_question_answer_boundary() -> None:
    text = "Question words here\n\nAnswer words here"
    question = DocumentSection("question", "Question words here", 0, 19)
    answer = DocumentSection("answer", "Answer words here", 21, len(text))
    document = _chunk_document()
    document.normalized_text = text
    document.sections = [question, answer]
    chunks = chunk_document(document, ChunkConfig("qa", 100, 10))
    assert [chunk.section_type for chunk in chunks] == ["question", "answer"]
    assert all(not ({"Question", "Answer"} <= set(chunk.raw_text.split())) for chunk in chunks)
