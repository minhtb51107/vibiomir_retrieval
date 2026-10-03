from __future__ import annotations

import asyncio

from .checkpoint import CheckpointStore
from .models import CrawlResult
from src.storage.body_archive import BodyArchive, BodyMetadata


class AsyncResultWriter:
    """Serialize incremental SQLite writes through a bounded async queue."""

    def __init__(
        self,
        store: CheckpointStore,
        queue_size: int = 128,
        body_archive: BodyArchive | None = None,
    ):
        self.store = store
        self.body_archive = body_archive
        self.queue: asyncio.Queue[CrawlResult | None] = asyncio.Queue(queue_size)
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def write(self, result: CrawlResult) -> None:
        if self._task is None:
            raise RuntimeError("writer has not been started")
        await self.queue.put(result)

    async def close(self) -> None:
        if self._task is None:
            return
        await self.queue.put(None)
        await self.queue.join()
        await self._task
        self._task = None

    async def _run(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                if item is None:
                    return
                if (
                    self.body_archive is not None
                    and item.raw_body is not None
                    and not item.body_truncated
                ):
                    self.body_archive.add(
                        BodyMetadata(
                            doc_id=item.doc_id,
                            original_url=item.original_url,
                            final_url=item.final_url,
                            fetched_at=item.fetched_at,
                            content_type=item.content_type,
                            encoding=item.encoding,
                            declared_http_encoding=item.declared_http_encoding,
                        ),
                        item.raw_body,
                    )
                    item.raw_body = None
                self.store.save(item)
            finally:
                self.queue.task_done()
