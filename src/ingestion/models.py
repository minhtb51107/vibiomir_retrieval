from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


class CrawlStatus(StrEnum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    HTTP_ERROR = "HTTP_ERROR"
    TIMEOUT = "TIMEOUT"
    NETWORK_ERROR = "NETWORK_ERROR"
    ROBOTS_BLOCKED = "ROBOTS_BLOCKED"
    UNSUPPORTED_CONTENT = "UNSUPPORTED_CONTENT"
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"
    INVALID_URL = "INVALID_URL"


RETRYABLE_FAILURE_STATUSES = frozenset(
    {
        CrawlStatus.TIMEOUT,
        CrawlStatus.NETWORK_ERROR,
        CrawlStatus.RETRY_EXHAUSTED,
    }
)


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    doc_id: int
    original_url: str
    fetch_url: str


@dataclass(slots=True)
class CrawlResult:
    doc_id: int
    original_url: str
    fetch_url: str
    final_url: str | None = None
    status: CrawlStatus = CrawlStatus.PENDING
    http_status: int | None = None
    content_type: str | None = None
    encoding: str | None = None
    content_length: int | None = None
    fetched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    attempt_count: int = 0
    elapsed_ms: int = 0
    error_type: str | None = None
    error_message: str | None = None
    raw_body: bytes | None = None


@dataclass(frozen=True, slots=True)
class DomainPolicy:
    concurrency: int
    delay_seconds: float


@dataclass(frozen=True, slots=True)
class HttpConfig:
    user_agent: str
    connect_timeout_seconds: float
    read_timeout_seconds: float
    write_timeout_seconds: float
    pool_timeout_seconds: float
    max_connections: int
    max_keepalive_connections: int
    follow_redirects: bool
    max_redirects: int


@dataclass(frozen=True, slots=True)
class RetryConfig:
    max_attempts: int
    backoff_base_seconds: float
    backoff_max_seconds: float
    jitter_seconds: float


@dataclass(frozen=True, slots=True)
class StorageConfig:
    store_raw_body: bool
    max_body_bytes: int
    allowed_content_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RobotsConfig:
    enabled: bool
    allow_on_fetch_error: bool


@dataclass(frozen=True, slots=True)
class CrawlerConfig:
    input_path: str
    output_db: str
    http: HttpConfig
    retry: RetryConfig
    global_concurrency: int
    default_domain_policy: DomainPolicy
    domain_overrides: dict[str, DomainPolicy]
    robots: RobotsConfig
    storage: StorageConfig
    pilot_default_limit: int
    pilot_maximum_limit: int

    def policy_for(self, domain: str) -> DomainPolicy:
        return self.domain_overrides.get(domain.lower(), self.default_domain_policy)


@dataclass(slots=True)
class CrawlStats:
    selected: int = 0
    scheduled: int = 0
    skipped_existing: int = 0
    attempts: int = 0
    by_status: dict[str, int] = field(default_factory=dict)

    def record(self, result: CrawlResult) -> None:
        self.scheduled += 1
        self.attempts += result.attempt_count
        key = result.status.value
        self.by_status[key] = self.by_status.get(key, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "scheduled": self.scheduled,
            "skipped_existing": self.skipped_existing,
            "attempts": self.attempts,
            "by_status": dict(sorted(self.by_status.items())),
        }
