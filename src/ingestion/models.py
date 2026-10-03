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
    ACCESS_RESTRICTED = "ACCESS_RESTRICTED"
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
    domain: str | None = None
    final_url: str | None = None
    status: CrawlStatus = CrawlStatus.PENDING
    http_status: int | None = None
    content_type: str | None = None
    encoding: str | None = None
    declared_http_encoding: str | None = None
    content_length: int | None = None
    downloaded_bytes: int = 0
    body_truncated: bool = False
    redirect_count: int = 0
    tiny_html: bool = False
    js_shell_candidate: bool = False
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
    max_download_bytes: int
    inspection_prefix_bytes: int
    tiny_html_threshold_bytes: int
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
    access_restricted_domains: dict[str, str]
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
    completed: int = 0
    skipped_existing: int = 0
    attempts: int = 0
    retries: int = 0
    redirects: int = 0
    downloaded_bytes: int = 0
    started_at: float = 0.0
    wall_seconds: float = 0.0
    by_status: dict[str, int] = field(default_factory=dict)
    by_content_type: dict[str, int] = field(default_factory=dict)
    by_domain: dict[str, int] = field(default_factory=dict)

    def record(self, result: CrawlResult) -> None:
        self.completed += 1
        self.attempts += result.attempt_count
        self.retries += max(0, result.attempt_count - 1)
        self.redirects += result.redirect_count
        self.downloaded_bytes += result.downloaded_bytes
        key = result.status.value
        self.by_status[key] = self.by_status.get(key, 0) + 1
        content_type = result.content_type or "missing"
        self.by_content_type[content_type] = self.by_content_type.get(content_type, 0) + 1
        domain = result.domain or "unknown"
        self.by_domain[domain] = self.by_domain.get(domain, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "scheduled": self.scheduled,
            "completed": self.completed,
            "skipped_existing": self.skipped_existing,
            "attempts": self.attempts,
            "retry_count": self.retries,
            "redirect_count": self.redirects,
            "downloaded_bytes": self.downloaded_bytes,
            "wall_seconds": round(self.wall_seconds, 3),
            "attempt_requests_per_second": round(
                self.attempts / self.wall_seconds, 4
            ) if self.wall_seconds else 0.0,
            "urls_completed_per_minute": round(
                self.completed * 60 / self.wall_seconds, 2
            ) if self.wall_seconds else 0.0,
            "by_status": dict(sorted(self.by_status.items())),
            "by_content_type": dict(sorted(self.by_content_type.items())),
            "by_domain": dict(sorted(self.by_domain.items())),
        }
