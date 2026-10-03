from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup, Tag

from .decoding import DecodeResult, decode_body
from .models import CleanDocument, DocumentSection, ExtractionStatus


CONTENT_RE = re.compile(
    r"article|content|detail|post|story|main|question|answer|cau-hoi|tra-loi", re.I
)
QA_RE = re.compile(r"question|answer|reply|qna|qa[-_]|cau-hoi|tra-loi|hoi-dap", re.I)
AD_RE = re.compile(r"(^|[-_])(ad|ads|advert|banner|sponsor)([-_]|$)", re.I)
AUTHOR_RE = re.compile(r"author|byline|tac-gia|tác-giả", re.I)
BOILERPLATE = (
    "đăng nhập",
    "đăng ký",
    "chính sách bảo mật",
    "all rights reserved",
    "版权所有",
    "网站地图",
)
VIETNAMESE_CHARS = frozenset(
    "ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    "ÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
    "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    semantic_min_characters: int = 500
    semantic_to_density_min_ratio: float = 0.55
    severe_replacement_ratio: float = 0.02
    empty_content_max_characters: int = 20
    js_shell_visible_text_max_characters: int = 50
    remove_selectors: tuple[str, ...] = (
        "script", "style", "noscript", "template", "svg", "canvas",
        "nav", "footer", "aside", "form",
    )


@dataclass(slots=True)
class _Candidate:
    method: str
    container: Tag | BeautifulSoup
    text: str
    paragraph_count: int
    boilerplate_count: int


def config_from_dict(raw: dict[str, Any]) -> ExtractionConfig:
    return ExtractionConfig(
        semantic_min_characters=int(raw["semantic_min_characters"]),
        semantic_to_density_min_ratio=float(raw["semantic_to_density_min_ratio"]),
        severe_replacement_ratio=float(raw["severe_replacement_ratio"]),
        empty_content_max_characters=int(raw["empty_content_max_characters"]),
        js_shell_visible_text_max_characters=int(
            raw["js_shell_visible_text_max_characters"]
        ),
        remove_selectors=tuple(raw["remove_selectors"]),
    )


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", " ".join(text.split()))


def _title(soup: BeautifulSoup) -> str | None:
    node = soup.find("title") or soup.find("h1")
    value = _normalize(node.get_text(" ", strip=True)) if node else ""
    return value or None


def _remove_noise(soup: BeautifulSoup, config: ExtractionConfig) -> None:
    for node in soup.select(",".join(config.remove_selectors)):
        node.decompose()
    for node in list(soup.find_all(attrs={"aria-hidden": "true"})):
        node.decompose()
    for node in list(soup.find_all(style=re.compile(r"display\s*:\s*none", re.I))):
        node.decompose()
    for node in list(soup.find_all(id=AD_RE)) + list(soup.find_all(class_=AD_RE)):
        if node.parent is not None:
            node.decompose()


def _blocks(container: Tag | BeautifulSoup) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for node in container.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "time"]
    ):
        value = _normalize(node.get_text(" ", strip=True))
        if len(value) >= 2 and value not in seen:
            seen.add(value)
            values.append(value)
    if not values:
        value = _normalize(container.get_text(" ", strip=True))
        if value:
            values.append(value)
    return values


def _candidate(method: str, container: Tag | BeautifulSoup) -> _Candidate:
    blocks = _blocks(container)
    text = "\n\n".join(blocks)
    lowered = text.casefold()
    return _Candidate(
        method=method,
        container=container,
        text=text,
        paragraph_count=len(blocks),
        boilerplate_count=sum(lowered.count(marker) for marker in BOILERPLATE),
    )


def _semantic_candidate(soup: BeautifulSoup) -> _Candidate:
    container = (
        soup.find("article")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find(id=CONTENT_RE)
        or soup.find(class_=CONTENT_RE)
        or soup.body
        or soup
    )
    return _candidate("semantic_container", container)


def _density_candidate(soup: BeautifulSoup) -> _Candidate:
    candidates = soup.find_all(["article", "main", "section", "div"])
    if not candidates:
        candidates = [soup.body or soup]

    def score(node: Tag) -> float:
        text_length = len(node.get_text(" ", strip=True))
        paragraphs = len(node.find_all("p"))
        link_length = sum(
            len(link.get_text(" ", strip=True)) for link in node.find_all("a")
        )
        return text_length + paragraphs * 100 - link_length * 1.5

    return _candidate("density_scored", max(candidates, key=score))


def _jsonld_types(soup: BeautifulSoup) -> set[str]:
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            kind = value.get("@type")
            if isinstance(kind, str):
                found.add(kind.casefold())
            elif isinstance(kind, list):
                found.update(str(item).casefold() for item in kind)
            for nested in value.values():
                if isinstance(nested, (dict, list)):
                    visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    for node in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)}):
        try:
            visit(json.loads(node.string or node.get_text() or ""))
        except (json.JSONDecodeError, TypeError):
            continue
    return found


def _qa_sections(container: Tag | BeautifulSoup, is_qa: bool) -> list[DocumentSection]:
    if not is_qa:
        return []
    matched = container.find_all(id=QA_RE) + container.find_all(class_=QA_RE)
    unique: list[Tag] = []
    matched_ids = {id(node) for node in matched}
    for node in matched:
        if any(id(parent) in matched_ids for parent in node.parents):
            continue
        if node not in unique:
            unique.append(node)
    sections: list[DocumentSection] = []
    seen: set[tuple[str, str]] = set()
    for node in unique:
        marker = " ".join(
            [str(node.get("id") or ""), *[str(item) for item in node.get("class") or []]]
        ).casefold()
        section_type = "answer" if re.search(r"answer|reply|tra-loi", marker) else "question"
        text = "\n\n".join(_blocks(node))
        key = (section_type, text)
        if text and key not in seen:
            seen.add(key)
            sections.append(DocumentSection(section_type=section_type, text=text))
    return sections


def _generic_sections(
    container: Tag | BeautifulSoup, title: str | None
) -> tuple[list[DocumentSection], int]:
    sections: list[DocumentSection] = []
    if title:
        sections.append(DocumentSection(section_type="title", text=title))
    heading_path: list[str] = []
    buffered: list[str] = []
    buffered_heading: list[str] = []
    paragraph_count = 0

    def flush() -> None:
        nonlocal buffered
        if buffered:
            sections.append(
                DocumentSection(
                    section_type="content",
                    text="\n\n".join(buffered),
                    heading_path=list(buffered_heading),
                )
            )
            buffered = []

    seen: set[str] = set()
    for node in container.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "time"]
    ):
        text = _normalize(node.get_text(" ", strip=True))
        if len(text) < 2 or text in seen:
            continue
        seen.add(text)
        if re.fullmatch(r"h[1-6]", node.name or ""):
            flush()
            level = int(node.name[1])
            heading_path[:] = heading_path[: level - 1]
            heading_path.append(text)
            buffered_heading[:] = heading_path
            buffered.append(text)
        else:
            if not buffered:
                buffered_heading[:] = heading_path
            buffered.append(text)
            paragraph_count += 1
    flush()
    if len(sections) == (1 if title else 0):
        fallback = _normalize(container.get_text(" ", strip=True))
        if fallback and fallback != title:
            sections.append(DocumentSection(section_type="content", text=fallback))
            paragraph_count = 1
    return sections, paragraph_count


def _assign_offsets(sections: list[DocumentSection]) -> str:
    pieces: list[str] = []
    offset = 0
    for section in sections:
        if pieces:
            pieces.append("\n\n")
            offset += 2
        section.start_offset = offset
        pieces.append(section.text)
        offset += len(section.text)
        section.end_offset = offset
    return "".join(pieces)


def _language(text: str) -> str:
    han = sum("\u3400" <= char <= "\u9fff" for char in text)
    vi = sum(char in VIETNAMESE_CHARS for char in text)
    ascii_letters = sum(char.isascii() and char.isalpha() for char in text)
    total = han + vi + ascii_letters
    if han >= 20 and han / max(total, 1) >= 0.25:
        return "zh-Han-script"
    if vi >= 5:
        return "vi-signal"
    if ascii_letters >= 20:
        return "latin-undetermined"
    return "unknown"


def extract_document(
    *,
    doc_id: int,
    original_url: str,
    final_url: str | None,
    body: bytes,
    content_type_header: str | None,
    archive_shard: str | None,
    archive_member_offset: int | None,
    archive_checksum_sha256: str | None,
    config: ExtractionConfig = ExtractionConfig(),
) -> CleanDocument:
    decoded: DecodeResult = decode_body(
        body,
        content_type_header,
        severe_replacement_ratio=config.severe_replacement_ratio,
    )
    raw_hash = hashlib.sha256(body).hexdigest()
    soup = BeautifulSoup(decoded.text, "html.parser")
    title = _title(soup)
    raw_visible = _normalize(soup.get_text(" ", strip=True))
    script_count = len(soup.find_all("script"))
    if len(raw_visible) < config.js_shell_visible_text_max_characters and script_count:
        return CleanDocument(
            doc_id, original_url, final_url, title, "unknown", decoded.selected_encoding,
            "", [], None, ExtractionStatus.JS_SHELL, len(body), len(decoded.text), 0, 0,
            decoded.replacement_character_count, 0, 0, 0, raw_hash, archive_shard,
            archive_member_offset, archive_checksum_sha256,
        )

    _remove_noise(soup, config)
    semantic = _semantic_candidate(soup)
    density = _density_candidate(soup)
    selected = (
        semantic
        if len(semantic.text) >= config.semantic_min_characters
        and len(semantic.text) >= len(density.text) * config.semantic_to_density_min_ratio
        else density
    )
    types = _jsonld_types(BeautifulSoup(decoded.text, "html.parser"))
    qa_markers = len(soup.find_all(id=QA_RE)) + len(soup.find_all(class_=QA_RE))
    is_qa = qa_markers >= 2 or bool(types.intersection({"qapage", "question", "answer"}))
    sections = _qa_sections(selected.container, is_qa)
    if sections:
        paragraph_count = len(sections)
        method = f"{selected.method}+qa_structured"
        if title and title not in {section.text for section in sections}:
            sections.insert(0, DocumentSection(section_type="title", text=title))
    else:
        sections, paragraph_count = _generic_sections(selected.container, title)
        method = selected.method
    normalized_text = _assign_offsets(sections)
    if decoded.severe_decoding_problem:
        status = ExtractionStatus.EXTRACTION_FAILED
    elif len(normalized_text) <= config.empty_content_max_characters:
        status = ExtractionStatus.EMPTY_CONTENT
    else:
        status = ExtractionStatus.SUCCESS
    return CleanDocument(
        doc_id=doc_id,
        original_url=original_url,
        final_url=final_url,
        title=title,
        language_signal=_language(normalized_text),
        selected_encoding=decoded.selected_encoding,
        normalized_text=normalized_text,
        sections=sections,
        extraction_method=method,
        extraction_status=status,
        raw_byte_length=len(body),
        raw_char_count=len(decoded.text),
        clean_char_count=len(normalized_text),
        paragraph_count=paragraph_count,
        replacement_character_count=decoded.replacement_character_count,
        semantic_char_count=len(semantic.text),
        density_char_count=len(density.text),
        boilerplate_signal_count=selected.boilerplate_count,
        raw_sha256=raw_hash,
        archive_shard=archive_shard,
        archive_member_offset=archive_member_offset,
        archive_checksum_sha256=archive_checksum_sha256,
        error_type="SevereDecodingProblem" if decoded.severe_decoding_problem else None,
        error_message=(
            "replacement-character ratio exceeded configured threshold"
            if decoded.severe_decoding_problem
            else None
        ),
    )
