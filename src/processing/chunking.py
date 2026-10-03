from __future__ import annotations

import bisect
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from .models import Chunk, CleanDocument, DocumentSection, ExtractionStatus


TOKEN_RE = re.compile(r"[\u3400-\u9fff]|[^\W_]+|[^\s\w]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class TokenSpan:
    text: str
    start: int
    end: int


class Tokenizer(Protocol):
    def spans(self, text: str) -> list[TokenSpan]: ...


class LightweightUnicodeTokenizer:
    """Deterministic offline tokenizer: Han characters, words, and punctuation."""

    def spans(self, text: str) -> list[TokenSpan]:
        return [TokenSpan(match.group(), match.start(), match.end()) for match in TOKEN_RE.finditer(text)]


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    name: str
    target_tokens: int
    overlap_tokens: int

    def __post_init__(self) -> None:
        if self.target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if self.overlap_tokens < 0 or self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must satisfy 0 <= overlap < target")


@dataclass(frozen=True, slots=True)
class _Range:
    start: int
    end: int
    section_type: str
    heading_path: list[str]


def _paragraph_aware_end(
    source: str,
    tokens: list[TokenSpan],
    token_start: int,
    proposed_end: int,
    config: ChunkConfig,
) -> int:
    """Prefer the last usable paragraph boundary in a full chunk window."""
    if proposed_end >= len(tokens):
        return proposed_end
    minimum_end = token_start + max(
        config.overlap_tokens + 1,
        config.target_tokens // 2,
    )
    if minimum_end >= proposed_end:
        return proposed_end
    token_ends = [token.end for token in tokens]
    boundaries = [match.start() for match in re.finditer(r"\n\s*\n", source)]
    for boundary in reversed(boundaries):
        candidate = bisect.bisect_right(token_ends, boundary)
        if minimum_end <= candidate <= proposed_end:
            return candidate
    return proposed_end


def _logical_ranges(sections: list[DocumentSection]) -> list[_Range]:
    ranges: list[_Range] = []
    index = 0
    while index < len(sections):
        section = sections[index]
        if (
            section.section_type == "title"
            and index + 1 < len(sections)
            and sections[index + 1].section_type not in {"question", "answer"}
        ):
            following = sections[index + 1]
            ranges.append(
                _Range(
                    section.start_offset,
                    following.end_offset,
                    following.section_type,
                    list(following.heading_path),
                )
            )
            index += 2
            continue
        ranges.append(
            _Range(
                section.start_offset,
                section.end_offset,
                section.section_type,
                list(section.heading_path),
            )
        )
        index += 1
    return ranges


def chunk_document(
    document: CleanDocument,
    config: ChunkConfig,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    if document.extraction_status != ExtractionStatus.SUCCESS:
        return []
    tokenizer = tokenizer or LightweightUnicodeTokenizer()
    chunks: list[Chunk] = []
    for logical in _logical_ranges(document.sections):
        source = document.normalized_text[logical.start : logical.end]
        tokens = tokenizer.spans(source)
        if not tokens:
            continue
        token_start = 0
        while token_start < len(tokens):
            token_end = min(token_start + config.target_tokens, len(tokens))
            token_end = _paragraph_aware_end(
                source, tokens, token_start, token_end, config
            )
            start = logical.start if token_start == 0 else logical.start + tokens[token_start].start
            end = logical.end if token_end == len(tokens) else logical.start + tokens[token_end - 1].end
            raw_text = document.normalized_text[start:end]
            normalized_text = unicodedata.normalize("NFC", raw_text)
            chunk_index = len(chunks)
            digest = hashlib.sha256(
                f"{document.doc_id}\0{config.name}\0{chunk_index}\0{start}\0{end}\0{raw_text}".encode(
                    "utf-8"
                )
            ).hexdigest()[:24]
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}-{digest}",
                    doc_id=document.doc_id,
                    chunk_index=chunk_index,
                    raw_text=raw_text,
                    normalized_text=normalized_text,
                    token_count=len(tokenizer.spans(raw_text)),
                    start_offset=start,
                    end_offset=end,
                    section_type=logical.section_type,
                    heading_path=list(logical.heading_path),
                    source_url=document.final_url or document.original_url,
                    extraction_method=document.extraction_method or "unknown",
                )
            )
            if token_end == len(tokens):
                break
            token_start = token_end - config.overlap_tokens
    return chunks
