from __future__ import annotations

import asyncio
import random
import re
import time
from collections.abc import Awaitable, Callable

import httpx

from .models import CorpusRecord, CrawlResult, CrawlStatus, HttpConfig, RetryConfig, StorageConfig
from .url_utils import domain_from_url


TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
CHARSET_RE = re.compile(r"charset\s*=\s*[\"']?([^\s;\"'>]+)", re.IGNORECASE)
SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


def is_retryable_http_status(status_code: int) -> bool:
    return status_code in TRANSIENT_HTTP_STATUSES


def _parse_content_length(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _declared_charset(content_type: str | None) -> str | None:
    if not content_type:
        return None
    match = CHARSET_RE.search(content_type)
    return match.group(1).strip().lower() if match else None


class HttpFetcher:
    def __init__(
        self,
        http_config: HttpConfig,
        retry_config: RetryConfig,
        storage_config: StorageConfig,
        *,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random_source: Callable[[], float] = random.random,
    ):
        self.http_config = http_config
        self.retry_config = retry_config
        self.storage_config = storage_config
        self._client = client
        self._owns_client = client is None
        self._sleep = sleep
        self._random = random_source

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("fetcher must be entered before use")
        return self._client

    async def __aenter__(self) -> HttpFetcher:
        if self._client is None:
            timeout = httpx.Timeout(
                connect=self.http_config.connect_timeout_seconds,
                read=self.http_config.read_timeout_seconds,
                write=self.http_config.write_timeout_seconds,
                pool=self.http_config.pool_timeout_seconds,
            )
            limits = httpx.Limits(
                max_connections=self.http_config.max_connections,
                max_keepalive_connections=self.http_config.max_keepalive_connections,
            )
            self._client = httpx.AsyncClient(
                headers={"User-Agent": self.http_config.user_agent},
                timeout=timeout,
                limits=limits,
                follow_redirects=self.http_config.follow_redirects,
                max_redirects=self.http_config.max_redirects,
            )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, record: CorpusRecord) -> CrawlResult:
        started = time.perf_counter()
        last_error: Exception | None = None
        last_response: httpx.Response | None = None

        for attempt in range(1, self.retry_config.max_attempts + 1):
            try:
                async with self.client.stream("GET", record.fetch_url) as response:
                    last_response = response
                    if is_retryable_http_status(response.status_code):
                        if attempt < self.retry_config.max_attempts:
                            await self._backoff(attempt)
                            continue
                        return await self._final_response_result(
                            record,
                            response,
                            CrawlStatus.RETRY_EXHAUSTED,
                            attempt,
                            started,
                            error_type="TransientHTTPStatus",
                            error_message=f"transient HTTP {response.status_code} exhausted retries",
                        )

                    if not 200 <= response.status_code < 300:
                        return await self._final_response_result(
                            record,
                            response,
                            CrawlStatus.HTTP_ERROR,
                            attempt,
                            started,
                            error_type="HTTPStatusError",
                            error_message=f"HTTP {response.status_code}",
                        )

                    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    supported = not content_type or any(
                        content_type == allowed or content_type.startswith(f"{allowed}+")
                        for allowed in self.storage_config.allowed_content_types
                    )
                    status = CrawlStatus.SUCCESS if supported else CrawlStatus.UNSUPPORTED_CONTENT
                    return await self._final_response_result(
                        record,
                        response,
                        status,
                        attempt,
                        started,
                        store_body=supported and self.storage_config.store_raw_body,
                    )
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt < self.retry_config.max_attempts:
                    await self._backoff(attempt)
                    continue
                return self._result_from_exception(
                    record, CrawlStatus.TIMEOUT, exc, attempt, started
                )
            except httpx.RequestError as exc:
                last_error = exc
                if attempt < self.retry_config.max_attempts:
                    await self._backoff(attempt)
                    continue
                return self._result_from_exception(
                    record, CrawlStatus.NETWORK_ERROR, exc, attempt, started
                )

        raise AssertionError(f"unreachable fetch state: {last_error!r}, {last_response!r}")

    async def _backoff(self, failed_attempt: int) -> None:
        base = self.retry_config.backoff_base_seconds * (2 ** (failed_attempt - 1))
        delay = min(base, self.retry_config.backoff_max_seconds)
        delay += self._random() * self.retry_config.jitter_seconds
        await self._sleep(delay)

    async def _final_response_result(
        self,
        record: CorpusRecord,
        response: httpx.Response,
        status: CrawlStatus,
        attempt: int,
        started: float,
        *,
        error_type: str | None = None,
        error_message: str | None = None,
        store_body: bool = False,
    ) -> CrawlResult:
        raw_chunks: list[bytes] = []
        prefix = bytearray()
        downloaded = 0
        stored = 0
        truncated = False
        async for chunk in response.aiter_bytes():
            remaining = self.storage_config.max_download_bytes - downloaded
            if remaining <= 0:
                truncated = True
                break
            piece = chunk[:remaining]
            downloaded += len(piece)
            prefix_remaining = self.storage_config.inspection_prefix_bytes - len(prefix)
            if prefix_remaining > 0:
                prefix.extend(piece[:prefix_remaining])
            if store_body and stored < self.storage_config.max_body_bytes:
                body_piece = piece[: self.storage_config.max_body_bytes - stored]
                raw_chunks.append(body_piece)
                stored += len(body_piece)
            if len(piece) < len(chunk):
                truncated = True
                break

        result = self._result_from_response(
            record,
            response,
            status,
            attempt,
            started,
            error_type=error_type,
            error_message=error_message,
        )
        result.downloaded_bytes = downloaded
        result.body_truncated = truncated
        result.raw_body = b"".join(raw_chunks) if store_body else None
        if result.content_length is None and not truncated:
            result.content_length = downloaded
        if status == CrawlStatus.SUCCESS and result.content_type in {
            "text/html",
            "application/xhtml+xml",
        }:
            result.tiny_html = downloaded < self.storage_config.tiny_html_threshold_bytes
            lowered = bytes(prefix).lower()
            if result.tiny_html and b"<script" in lowered:
                sample = bytes(prefix).decode("latin-1", errors="ignore")
                visible = TAG_RE.sub(" ", SCRIPT_RE.sub(" ", sample))
                result.js_shell_candidate = len(" ".join(visible.split())) < 100
        return result

    @staticmethod
    def _result_from_response(
        record: CorpusRecord,
        response: httpx.Response,
        status: CrawlStatus,
        attempt: int,
        started: float,
        *,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> CrawlResult:
        content_type_header = response.headers.get("content-type")
        content_type = (
            content_type_header.split(";", 1)[0].strip().lower()
            if content_type_header
            else None
        )
        return CrawlResult(
            doc_id=record.doc_id,
            original_url=record.original_url,
            fetch_url=record.fetch_url,
            domain=domain_from_url(record.fetch_url),
            final_url=str(response.url),
            status=status,
            http_status=response.status_code,
            content_type=content_type,
            encoding=response.encoding,
            declared_http_encoding=_declared_charset(content_type_header),
            content_length=_parse_content_length(response.headers.get("content-length")),
            redirect_count=len(response.history),
            attempt_count=attempt,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            error_type=error_type,
            error_message=error_message,
        )

    @staticmethod
    def _result_from_exception(
        record: CorpusRecord,
        status: CrawlStatus,
        error: Exception,
        attempt: int,
        started: float,
    ) -> CrawlResult:
        return CrawlResult(
            doc_id=record.doc_id,
            original_url=record.original_url,
            fetch_url=record.fetch_url,
            domain=domain_from_url(record.fetch_url),
            status=status,
            attempt_count=attempt,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            error_type=type(error).__name__,
            error_message=str(error)[:1000],
        )
