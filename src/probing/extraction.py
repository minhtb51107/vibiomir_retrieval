from __future__ import annotations

import codecs
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup, Tag
from charset_normalizer import from_bytes

from .models import ExtractionMetrics


CHARSET_RE = re.compile(r"charset\s*=\s*[\"']?([^\s;\"'>]+)", re.IGNORECASE)
CONTENT_HINT_RE = re.compile(
    r"article|content|detail|post|story|main|question|answer|cau-hoi|tra-loi",
    re.IGNORECASE,
)
QA_HINT_RE = re.compile(
    r"question|answer|ask|reply|qna|qa-|cau-hoi|tra-loi|hoi-dap",
    re.IGNORECASE,
)
RELATED_RE = re.compile(r"related|recommend|more-news|tin-lien-quan|bai-viet-lien-quan", re.IGNORECASE)
AD_RE = re.compile(r"(^|[-_])(ad|ads|advert|banner|sponsor)([-_]|$)", re.IGNORECASE)
AUTHOR_RE = re.compile(r"author|byline|tac-gia|tác-giả", re.IGNORECASE)
BOILERPLATE_MARKERS = (
    "đăng nhập",
    "đăng ký",
    "liên hệ",
    "chính sách bảo mật",
    "cookie",
    "privacy policy",
    "all rights reserved",
    "版权所有",
    "网站地图",
)
ANTI_BOT_MARKERS = {
    "captcha": "CAPTCHA",
    "just a moment": "CLOUDFLARE_CHALLENGE",
    "cloudflare": "CLOUDFLARE",
    "access denied": "ACCESS_DENIED",
    "verify you are human": "HUMAN_VERIFICATION",
    "验证码": "CAPTCHA",
    "访问受限": "ACCESS_RESTRICTED",
}
VIETNAMESE_CHARS = frozenset(
    "ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    "ÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
    "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
)
VIETNAMESE_STOPWORDS = frozenset(
    {"và", "là", "của", "cho", "với", "bệnh", "người", "được", "không", "sức", "khỏe"}
)


@dataclass(frozen=True, slots=True)
class DecodedDocument:
    text: str
    declared_http_encoding: str | None
    declared_meta_encoding: str | None
    detected_encoding: str | None
    selected_encoding: str
    replacement_count: int
    mojibake_marker_count: int
    unicode_nfc_changed: bool


@dataclass(slots=True)
class PageAnalysis:
    title: str | None
    basic_text_length: int
    language_signal: str
    language_signal_basis: dict[str, int]
    template_classification: str
    template_evidence: list[str]
    structure: dict[str, Any]
    extraction_metrics: list[ExtractionMetrics]
    anti_bot_signals: list[str]


def _canonical_encoding(name: str | None) -> str | None:
    if not name:
        return None
    try:
        return codecs.lookup(name.strip()).name
    except LookupError:
        return None


def declared_http_encoding(content_type_header: str | None) -> str | None:
    if not content_type_header:
        return None
    match = CHARSET_RE.search(content_type_header)
    return _canonical_encoding(match.group(1)) if match else None


def declared_meta_encoding(body: bytes) -> str | None:
    prefix = body[:16_384].decode("ascii", errors="ignore")
    for meta in re.findall(r"<meta\b[^>]*>", prefix, flags=re.IGNORECASE):
        match = CHARSET_RE.search(meta)
        if match:
            encoding = _canonical_encoding(match.group(1))
            if encoding:
                return encoding
    return None


def decode_document(body: bytes, content_type_header: str | None) -> DecodedDocument:
    http_encoding = declared_http_encoding(content_type_header)
    meta_encoding = declared_meta_encoding(body)
    match = from_bytes(body).best() if body else None
    detected = _canonical_encoding(match.encoding if match is not None else None)
    candidates = [http_encoding, meta_encoding, detected, "utf-8"]
    selected = "utf-8"
    text = ""
    for candidate in candidates:
        if not candidate:
            continue
        try:
            text = body.decode(candidate, errors="strict")
            selected = candidate
            break
        except (LookupError, UnicodeDecodeError):
            continue
    else:
        text = body.decode("utf-8", errors="replace")
    nfc = unicodedata.normalize("NFC", text)
    # Multi-character corruption signatures only. Standalone Ã and Â are valid
    # Vietnamese letters and must not be treated as mojibake.
    mojibake_count = sum(
        text.count(marker)
        for marker in (
            "Ã¡",
            "Ã ",
            "Ã¢",
            "Ä‘",
            "Æ°",
            "Æ¡",
            "â€™",
            "â€œ",
            "â€",
            "ï¿½",
        )
    )
    return DecodedDocument(
        text=text,
        declared_http_encoding=http_encoding,
        declared_meta_encoding=meta_encoding,
        detected_encoding=detected,
        selected_encoding=selected,
        replacement_count=text.count("\ufffd"),
        mojibake_marker_count=mojibake_count,
        unicode_nfc_changed=nfc != text,
    )


def _clean_soup(soup: BeautifulSoup) -> None:
    for node in soup.select(
        "script, style, noscript, template, svg, canvas, nav, footer, aside, form"
    ):
        node.decompose()
    for node in soup.find_all(attrs={"aria-hidden": "true"}):
        node.decompose()


def _title(soup: BeautifulSoup) -> str | None:
    if soup.title:
        value = soup.title.get_text(" ", strip=True)
        if value:
            return value
    heading = soup.find("h1")
    return heading.get_text(" ", strip=True) if heading else None


def _block_texts(container: Tag | BeautifulSoup) -> list[str]:
    blocks: list[str] = []
    for node in container.find_all(["h1", "h2", "h3", "p", "blockquote", "li"]):
        text = " ".join(node.get_text(" ", strip=True).split())
        if len(text) >= 2:
            blocks.append(text)
    if not blocks:
        blocks = [
            line.strip()
            for line in container.get_text("\n", strip=True).splitlines()
            if len(line.strip()) >= 2
        ]
    return blocks


def _with_title(title: str | None, blocks: list[str]) -> list[str]:
    if title and not any(title.casefold() in block.casefold() for block in blocks[:3]):
        return [title, *blocks]
    return blocks


def semantic_extract(html: str) -> tuple[str | None, str, int]:
    soup = BeautifulSoup(html, "html.parser")
    title = _title(soup)
    _clean_soup(soup)
    container = (
        soup.find("article")
        or soup.find("main")
        or soup.find(attrs={"role": "main"})
        or soup.find(id=CONTENT_HINT_RE)
        or soup.find(class_=CONTENT_HINT_RE)
        or soup.body
        or soup
    )
    blocks = _with_title(title, _block_texts(container))
    return title, "\n\n".join(blocks), len(blocks)


def density_extract(html: str) -> tuple[str | None, str, int]:
    soup = BeautifulSoup(html, "html.parser")
    title = _title(soup)
    _clean_soup(soup)
    candidates = soup.find_all(["article", "main", "section", "div"])
    if not candidates:
        candidates = [soup.body or soup]

    def score(node: Tag) -> float:
        text_length = len(node.get_text(" ", strip=True))
        paragraph_count = len(node.find_all("p"))
        link_length = sum(len(link.get_text(" ", strip=True)) for link in node.find_all("a"))
        return text_length + paragraph_count * 100 - link_length * 1.5

    container = max(candidates, key=score)
    blocks = _with_title(title, _block_texts(container))
    return title, "\n\n".join(blocks), len(blocks)


def _boilerplate_count(text: str) -> int:
    lowered = text.casefold()
    return sum(lowered.count(marker) for marker in BOILERPLATE_MARKERS)


def _language_signal(text: str) -> tuple[str, dict[str, int]]:
    han = sum("\u3400" <= char <= "\u9fff" for char in text)
    vietnamese_chars = sum(char in VIETNAMESE_CHARS for char in text)
    ascii_letters = sum(char.isascii() and char.isalpha() for char in text)
    words = re.findall(r"[^\W\d_]+", text.casefold(), flags=re.UNICODE)
    vietnamese_stopwords = sum(word in VIETNAMESE_STOPWORDS for word in words)
    letter_total = han + vietnamese_chars + ascii_letters
    if han >= 20 and han / max(letter_total, 1) >= 0.25:
        signal = "zh-Han-script"
    elif vietnamese_chars >= 5 or vietnamese_stopwords >= 3:
        signal = "vi-signal"
    elif ascii_letters >= 20:
        signal = "latin-undetermined"
    else:
        signal = "unknown"
    return signal, {
        "han_characters": han,
        "vietnamese_specific_characters": vietnamese_chars,
        "vietnamese_stopword_hits": vietnamese_stopwords,
        "ascii_letters": ascii_letters,
    }


def _structured_data(soup: BeautifulSoup) -> tuple[list[str], int]:
    types: list[str] = []
    article_body_count = 0

    def visit(value: Any) -> None:
        nonlocal article_body_count
        if isinstance(value, dict):
            type_value = value.get("@type")
            if isinstance(type_value, str):
                types.append(type_value)
            elif isinstance(type_value, list):
                types.extend(str(item) for item in type_value)
            if isinstance(value.get("articleBody"), str):
                article_body_count += 1
            for nested in value.values():
                if isinstance(nested, (dict, list)):
                    visit(nested)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for script in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)}):
        try:
            visit(json.loads(script.string or script.get_text() or ""))
        except (json.JSONDecodeError, TypeError):
            continue
    return sorted(set(types)), article_body_count


def analyze_html(html: str) -> PageAnalysis:
    soup = BeautifulSoup(html, "html.parser")
    title = _title(soup)
    basic_text = " ".join(soup.get_text(" ", strip=True).split())
    jsonld_types, article_body_count = _structured_data(soup)
    qa_markers = len(soup.find_all(id=QA_HINT_RE)) + len(soup.find_all(class_=QA_HINT_RE))
    author_markers = len(soup.find_all(attrs={"rel": "author"})) + len(
        soup.find_all(class_=AUTHOR_RE)
    )
    related_markers = len(soup.find_all(id=RELATED_RE)) + len(soup.find_all(class_=RELATED_RE))
    ad_markers = len(soup.find_all(id=AD_RE)) + len(soup.find_all(class_=AD_RE))
    structure = {
        "article_count": len(soup.find_all("article")),
        "main_count": len(soup.find_all("main")),
        "heading_count": len(soup.find_all(re.compile(r"^h[1-6]$"))),
        "paragraph_count": len(soup.find_all("p")),
        "link_count": len(soup.find_all("a")),
        "script_count": len(soup.find_all("script")),
        "nav_count": len(soup.find_all("nav")),
        "aside_count": len(soup.find_all("aside")),
        "footer_count": len(soup.find_all("footer")),
        "time_count": len(soup.find_all("time")),
        "author_marker_count": author_markers,
        "qa_marker_count": qa_markers,
        "related_marker_count": related_markers,
        "ad_marker_count": ad_markers,
        "jsonld_types": jsonld_types,
        "jsonld_article_body_count": article_body_count,
    }
    semantic_title, semantic_text, semantic_paragraphs = semantic_extract(html)
    density_title, density_text, density_paragraphs = density_extract(html)
    metrics = [
        ExtractionMetrics(
            approach="semantic_container",
            character_count=len(semantic_text),
            paragraph_count=semantic_paragraphs,
            title_preserved=bool(title and semantic_title == title and title in semantic_text),
            boilerplate_marker_count=_boilerplate_count(semantic_text),
            empty=len(semantic_text.strip()) == 0,
        ),
        ExtractionMetrics(
            approach="density_scored",
            character_count=len(density_text),
            paragraph_count=density_paragraphs,
            title_preserved=bool(title and density_title == title and title in density_text),
            boilerplate_marker_count=_boilerplate_count(density_text),
            empty=len(density_text.strip()) == 0,
        ),
    ]
    anti_bot = [
        label
        for marker, label in ANTI_BOT_MARKERS.items()
        if marker in basic_text.casefold() or marker in (title or "").casefold()
    ]
    if len(basic_text) < 50 and structure["script_count"] >= 1:
        anti_bot.append("JS_ONLY_SHELL")
    anti_bot = sorted(set(anti_bot))

    evidence: list[str] = []
    type_names = {value.casefold() for value in jsonld_types}
    hard_challenge = any(
        marker in (title or "").casefold()
        for marker in ("captcha", "just a moment", "access denied", "验证码", "访问受限")
    )
    if (
        anti_bot
        and any(signal != "JS_ONLY_SHELL" for signal in anti_bot)
        and (len(basic_text) < 1000 or hard_challenge)
    ):
        classification = "ACCESS_RESTRICTED"
        evidence.extend(anti_bot)
    elif "JS_ONLY_SHELL" in anti_bot:
        classification = "JAVASCRIPT_HEAVY"
        evidence.append("low visible text with multiple scripts")
    elif qa_markers >= 2 or type_names.intersection({"qapage", "question", "answer"}):
        classification = "Q_AND_A_TEMPLATE"
        evidence.append(f"qa markers={qa_markers}; JSON-LD={jsonld_types}")
    elif structure["article_count"] and (
        structure["time_count"] or author_markers or related_markers
    ):
        classification = "NEWS_TEMPLATE"
        evidence.append(
            "article element with time/author/related-content markers"
        )
    elif len(basic_text) >= 500 and structure["paragraph_count"] >= 2:
        classification = "GENERIC_STATIC_HTML"
        evidence.append("substantial visible text and paragraphs")
    else:
        classification = "UNKNOWN"
        evidence.append("insufficient reusable structure")

    language, language_basis = _language_signal(
        semantic_text if len(semantic_text) >= len(density_text) else density_text
    )
    return PageAnalysis(
        title=title,
        basic_text_length=len(basic_text),
        language_signal=language,
        language_signal_basis=language_basis,
        template_classification=classification,
        template_evidence=evidence,
        structure=structure,
        extraction_metrics=metrics,
        anti_bot_signals=anti_bot,
    )
