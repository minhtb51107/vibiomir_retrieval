from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable

import httpx

from .models import CorpusRecord, CrawlResult, CrawlStatus, HttpConfig, RetryConfig, StorageConfig


TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


def is_retryable_http_status(status_code: int) -> bool:
    return status_code in TRANSIENT_HTTP_STATUSES


def _parse_content_length(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


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
                        return self._result_from_response(
                            record,
                            response,
                            CrawlStatus.RETRY_EXHAUSTED,
                            attempt,
                            started,
                            error_type="TransientHTTPStatus",
                            error_message=f"transient HTTP {response.status_code} exhausted retries",
                        )

                    if not 200 <= response.status_code < 300:
                        return self._result_from_response(
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
                    body = None
                    bytes_read = 0
                    if self.storage_config.store_raw_body and supported:
                        chunks: list[bytes] = []
                        async for chunk in response.aiter_bytes():
                            remaining = self.storage_config.max_body_bytes - bytes_read
                            if remaining <= 0:
                                break
                            piece = chunk[:remaining]
                            chunks.append(piece)
                            bytes_read += len(piece)
                            if len(piece) < len(chunk):
                                break
                        body = b"".join(chunks)
                    result = self._result_from_response(
                        record, response, status, attempt, started
                    )
                    result.raw_body = body
                    if result.content_length is None and body is not None:
                        result.content_length = bytes_read
                    return result
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
            final_url=str(response.url),
            status=status,
            http_status=response.status_code,
            content_type=content_type,
            encoding=response.encoding,
            content_length=_parse_content_length(response.headers.get("content-length")),
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
            status=status,
            attempt_count=attempt,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            error_type=type(error).__name__,
            error_message=str(error)[:1000],
        )
