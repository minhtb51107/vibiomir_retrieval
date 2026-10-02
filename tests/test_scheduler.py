import asyncio
from dataclasses import replace

import httpx

from src.ingestion.checkpoint import CheckpointStore
from src.ingestion.config import load_config
from src.ingestion.fetcher import HttpFetcher
from src.ingestion.models import CorpusRecord, DomainPolicy, RobotsConfig
from src.ingestion.robots import RobotsPolicy
from src.ingestion.scheduler import CrawlScheduler, DomainController


def test_domain_override_and_concurrency_limit() -> None:
    config = load_config("configs/crawler.yaml")
    config = replace(
        config,
        global_concurrency=4,
        default_domain_policy=DomainPolicy(concurrency=3, delay_seconds=0),
        domain_overrides={"limited.test": DomainPolicy(concurrency=1, delay_seconds=0)},
    )
    assert config.policy_for("limited.test").concurrency == 1
    assert config.policy_for("other.test").concurrency == 3

    async def scenario() -> None:
        controller = DomainController(config)
        active = 0
        maximum_active = 0

        async def job() -> None:
            nonlocal active, maximum_active
            async with controller.slot("limited.test"):
                active += 1
                maximum_active = max(maximum_active, active)
                await asyncio.sleep(0.01)
                active -= 1

        await asyncio.gather(*(job() for _ in range(4)))
        assert maximum_active == 1

    asyncio.run(scenario())


def test_scheduler_second_run_skips_completed_records(tmp_path) -> None:
    config = load_config("configs/crawler.yaml")
    config = replace(
        config,
        global_concurrency=2,
        default_domain_policy=DomainPolicy(concurrency=1, delay_seconds=0),
        domain_overrides={},
        robots=RobotsConfig(enabled=False, allow_on_fetch_error=True),
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, headers={"content-type": "text/html"}, request=request
        )

    records = [
        CorpusRecord(1, "https://one.test/a#x", "https://one.test/a"),
        CorpusRecord(2, "https://two.test/b?q=1", "https://two.test/b?q=1"),
    ]

    async def scenario() -> None:
        with CheckpointStore(tmp_path / "resume.sqlite") as store:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                async with HttpFetcher(
                    config.http, config.retry, config.storage, client=client
                ) as fetcher:
                    robots = RobotsPolicy(client, config.robots, config.http.user_agent)
                    first = await CrawlScheduler(config, fetcher, robots, store).run(records)
                    second = await CrawlScheduler(config, fetcher, robots, store).run(records)
            assert first.scheduled == 2
            assert second.scheduled == 0
            assert second.skipped_existing == 2
            assert store.count() == 2
            assert calls == 2

    asyncio.run(scenario())
