from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from collections.abc import AsyncIterator, Iterable

from .checkpoint import CheckpointStore
from .fetcher import HttpFetcher
from .models import (
    CorpusRecord,
    CrawlResult,
    CrawlStats,
    CrawlStatus,
    CrawlerConfig,
)
from .robots import RobotsPolicy
from .url_utils import domain_from_url, prepare_fetch_url
from .writer import AsyncResultWriter


@dataclass(slots=True)
class _DomainState:
    semaphore: asyncio.Semaphore
    delay_seconds: float
    start_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    next_start: float = 0.0


class DomainController:
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.global_semaphore = asyncio.Semaphore(config.global_concurrency)
        self._states: dict[str, _DomainState] = {}

    def _state_for(self, domain: str) -> _DomainState:
        if domain not in self._states:
            policy = self.config.policy_for(domain)
            self._states[domain] = _DomainState(
                asyncio.Semaphore(policy.concurrency), policy.delay_seconds
            )
        return self._states[domain]

    @asynccontextmanager
    async def slot(self, domain: str) -> AsyncIterator[None]:
        state = self._state_for(domain)
        async with self.global_semaphore, state.semaphore:
            async with state.start_lock:
                wait_seconds = state.next_start - time.monotonic()
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)
                state.next_start = time.monotonic() + state.delay_seconds
            yield


class CrawlScheduler:
    def __init__(
        self,
        config: CrawlerConfig,
        fetcher: HttpFetcher,
        robots: RobotsPolicy,
        store: CheckpointStore,
    ):
        self.config = config
        self.fetcher = fetcher
        self.robots = robots
        self.store = store
        self.controller = DomainController(config)

    async def run(
        self,
        records: Iterable[CorpusRecord],
        *,
        retry_failures: bool = False,
    ) -> CrawlStats:
        stats = CrawlStats()
        completed = self.store.completed_doc_ids(retry_failures=retry_failures)
        queue: asyncio.Queue[CorpusRecord | None] = asyncio.Queue(
            maxsize=self.config.global_concurrency * 4
        )
        writer = AsyncResultWriter(self.store)
        await writer.start()

        async def producer() -> None:
            for record in records:
                stats.selected += 1
                if record.doc_id in completed:
                    stats.skipped_existing += 1
                    continue
                await queue.put(record)
            for _ in range(self.config.global_concurrency):
                await queue.put(None)

        async def worker() -> None:
            while True:
                record = await queue.get()
                try:
                    if record is None:
                        return
                    result = await self._crawl_one(record)
                    await writer.write(result)
                    stats.record(result)
                    print(
                        f"doc_id={result.doc_id} status={result.status.value} "
                        f"http={result.http_status} attempts={result.attempt_count}"
                    )
                finally:
                    queue.task_done()

        workers = [
            asyncio.create_task(worker())
            for _ in range(self.config.global_concurrency)
        ]
        try:
            await producer()
            await queue.join()
            await asyncio.gather(*workers)
        finally:
            for task in workers:
                if not task.done():
                    task.cancel()
            await writer.close()
        return stats

    async def _crawl_one(self, record: CorpusRecord) -> CrawlResult:
        try:
            fetch_url = prepare_fetch_url(record.original_url)
            domain = domain_from_url(fetch_url)
        except ValueError as exc:
            return CrawlResult(
                doc_id=record.doc_id,
                original_url=record.original_url,
                fetch_url=record.fetch_url,
                status=CrawlStatus.INVALID_URL,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        normalized_record = CorpusRecord(record.doc_id, record.original_url, fetch_url)
        async with self.controller.slot(domain):
            if not await self.robots.can_fetch(fetch_url):
                return CrawlResult(
                    doc_id=record.doc_id,
                    original_url=record.original_url,
                    fetch_url=fetch_url,
                    status=CrawlStatus.ROBOTS_BLOCKED,
                    error_type="RobotsDisallowed",
                    error_message="robots.txt disallows this URL for the configured user agent",
                )
            return await self.fetcher.fetch(normalized_record)
