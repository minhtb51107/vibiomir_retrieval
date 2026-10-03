from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ExtractionStatus(StrEnum):
    SUCCESS = "SUCCESS"
    EMPTY_CONTENT = "EMPTY_CONTENT"
    JS_SHELL = "JS_SHELL"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    ACCESS_RESTRICTED = "ACCESS_RESTRICTED"
    ROBOTS_BLOCKED = "ROBOTS_BLOCKED"


@dataclass(slots=True)
class DocumentSection:
    section_type: str
    text: str
    start_offset: int = 0
    end_offset: int = 0
    heading_path: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CleanDocument:
    doc_id: int
    original_url: str
    final_url: str | None
    title: str | None
    language_signal: str
    selected_encoding: str | None
    normalized_text: str
    sections: list[DocumentSection]
    extraction_method: str | None
    extraction_status: ExtractionStatus
    raw_byte_length: int
    raw_char_count: int
    clean_char_count: int
    paragraph_count: int
    replacement_character_count: int
    semantic_char_count: int
    density_char_count: int
    boilerplate_signal_count: int
    raw_sha256: str | None
    archive_shard: str | None
    archive_member_offset: int | None
    archive_checksum_sha256: str | None
    error_type: str | None = None
    error_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["extraction_status"] = self.extraction_status.value
        data["sections"] = [section.as_dict() for section in self.sections]
        return data


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: int
    chunk_index: int
    raw_text: str
    normalized_text: str
    token_count: int
    start_offset: int
    end_offset: int
    section_type: str
    heading_path: list[str]
    source_url: str
    extraction_method: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
