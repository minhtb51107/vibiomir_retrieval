import asyncio

import httpx

from src.ingestion.config import load_config
from src.ingestion.fetcher import HttpFetcher, is_retryable_http_status
from src.ingestion.models import CorpusRecord, CrawlStatus


async def _no_sleep(_: float) -> None:
    return None


def test_retry_classification() -> None:
    assert is_retryable_http_status(429)
    assert is_retryable_http_status(503)
    assert not is_retryable_http_status(404)
    assert not is_retryable_http_status(401)


def test_transient_response_retries_then_succeeds() -> None:
    config = load_config("configs/crawler.yaml")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8", "content-length": "2"},
            content=b"ok",
            request=request,
        )

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async with client:
            async with HttpFetcher(
                config.http,
                config.retry,
                config.storage,
                client=client,
                sleep=_no_sleep,
                random_source=lambda: 0.0,
            ) as fetcher:
                result = await fetcher.fetch(
                    CorpusRecord(1, "https://example.test/a", "https://example.test/a")
                )
        assert result.status == CrawlStatus.SUCCESS
        assert result.attempt_count == 2
        assert result.content_type == "text/html"

    asyncio.run(scenario())


def test_permanent_client_error_is_not_retried() -> None:
    config = load_config("configs/crawler.yaml")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, request=request)

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async with client:
            async with HttpFetcher(
                config.http, config.retry, config.storage, client=client, sleep=_no_sleep
            ) as fetcher:
                result = await fetcher.fetch(
                    CorpusRecord(2, "https://example.test/missing", "https://example.test/missing")
                )
        assert result.status == CrawlStatus.HTTP_ERROR
        assert result.attempt_count == 1
        assert calls == 1

    asyncio.run(scenario())


def test_timeout_exhaustion_and_unsupported_content_are_distinct() -> None:
    config = load_config("configs/crawler.yaml")
    timeout_calls = 0

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        nonlocal timeout_calls
        timeout_calls += 1
        raise httpx.ReadTimeout("simulated timeout", request=request)

    def image_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "image/png"}, request=request
        )

    async def scenario() -> None:
        timeout_client = httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler))
        async with timeout_client:
            async with HttpFetcher(
                config.http,
                config.retry,
                config.storage,
                client=timeout_client,
                sleep=_no_sleep,
                random_source=lambda: 0.0,
            ) as fetcher:
                timeout_result = await fetcher.fetch(
                    CorpusRecord(3, "https://slow.test/a", "https://slow.test/a")
                )
        assert timeout_result.status == CrawlStatus.TIMEOUT
        assert timeout_result.attempt_count == config.retry.max_attempts

        image_client = httpx.AsyncClient(transport=httpx.MockTransport(image_handler))
        async with image_client:
            async with HttpFetcher(
                config.http, config.retry, config.storage, client=image_client
            ) as fetcher:
                image_result = await fetcher.fetch(
                    CorpusRecord(4, "https://image.test/a", "https://image.test/a")
                )
        assert image_result.status == CrawlStatus.UNSUPPORTED_CONTENT
        assert image_result.content_type == "image/png"

    asyncio.run(scenario())
    assert timeout_calls == config.retry.max_attempts
