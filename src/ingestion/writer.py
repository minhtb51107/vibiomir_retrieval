from __future__ import annotations

import asyncio

from .checkpoint import CheckpointStore
from .models import CrawlResult


class AsyncResultWriter:
    """Serialize incremental SQLite writes through a bounded async queue."""

    def __init__(self, store: CheckpointStore, queue_size: int = 128):
        self.store = store
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
                self.store.save(item)
            finally:
                self.queue.task_done()
