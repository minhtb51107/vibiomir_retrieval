from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SamplingRule:
    name: str
    kind: str
    quota: int
    value: str | None = None
    inferred_language: str | None = None


@dataclass(slots=True)
class SampleRecord:
    doc_id: int
    original_url: str
    fetch_url: str
    domain: str
    sampling_categories: list[str] = field(default_factory=list)
    inferred_language: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    input_path: str
    output_dir: str
    seed: str
    maximum_live_urls: int
    absolute_safety_ceiling: int
    batch_size: int
    global_concurrency: int
    per_domain_concurrency: int
    domain_delay_seconds: float
    connect_timeout_seconds: float
    read_timeout_seconds: float
    max_response_bytes: int
    user_agent: str
    sampling_rules: tuple[SamplingRule, ...]


@dataclass(slots=True)
class ExtractionMetrics:
    approach: str
    character_count: int
    paragraph_count: int
    title_preserved: bool
    boilerplate_marker_count: int
    empty: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProbeResult:
    doc_id: int
    original_url: str
    fetch_url: str
    domain: str
    sampling_categories: list[str]
    inferred_language: str | None
    robots_allowed: bool | None = None
    robots_status: int | None = None
    final_url: str | None = None
    status: str = "PENDING"
    http_status: int | None = None
    content_type: str | None = None
    declared_http_encoding: str | None = None
    declared_meta_encoding: str | None = None
    detected_encoding: str | None = None
    client_encoding: str | None = None
    selected_encoding: str | None = None
    decoding_replacement_count: int = 0
    mojibake_marker_count: int = 0
    unicode_nfc_changed: bool = False
    response_size_bytes: int | None = None
    downloaded_bytes: int = 0
    body_truncated: bool = False
    redirect_count: int = 0
    elapsed_ms: int = 0
    title: str | None = None
    basic_text_length: int = 0
    language_signal: str = "unknown"
    language_signal_basis: dict[str, int] = field(default_factory=dict)
    template_classification: str = "UNKNOWN"
    template_evidence: list[str] = field(default_factory=list)
    structure: dict[str, Any] = field(default_factory=dict)
    extraction: list[ExtractionMetrics] = field(default_factory=list)
    anti_bot_signals: list[str] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["extraction"] = [metric.as_dict() for metric in self.extraction]
        return data
