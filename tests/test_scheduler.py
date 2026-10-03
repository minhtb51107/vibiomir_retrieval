import asyncio
from dataclasses import replace

import httpx
import pytest

from src.ingestion.checkpoint import CheckpointStore
from src.ingestion.config import load_config
from src.ingestion.fetcher import HttpFetcher
from src.ingestion.models import CorpusRecord, CrawlStatus, DomainPolicy, RobotsConfig
from src.ingestion.robots import RobotsPolicy
from src.ingestion.scheduler import CrawlScheduler, DomainController
from src.storage.body_archive import BodyArchive


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


def test_bounded_batches_advance_past_completed_rows(tmp_path) -> None:
    config = load_config("configs/crawler.yaml")
    config = replace(
        config,
        global_concurrency=2,
        default_domain_policy=DomainPolicy(concurrency=1, delay_seconds=0),
        domain_overrides={},
        robots=RobotsConfig(enabled=False, allow_on_fetch_error=True),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, request=request)

    records = [
        CorpusRecord(i, f"https://batch.test/{i}", f"https://batch.test/{i}")
        for i in range(1, 5)
    ]

    async def scenario() -> None:
        with CheckpointStore(tmp_path / "batches.sqlite") as store:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                async with HttpFetcher(
                    config.http, config.retry, config.storage, client=client
                ) as fetcher:
                    robots = RobotsPolicy(client, config.robots, config.http.user_agent)
                    first = await CrawlScheduler(config, fetcher, robots, store).run(
                        records, max_new_records=2
                    )
                    second = await CrawlScheduler(config, fetcher, robots, store).run(
                        records, max_new_records=2
                    )
            assert first.scheduled == first.completed == 2
            assert second.skipped_existing == 2
            assert second.scheduled == second.completed == 2
            assert store.count() == 4
            assert store.telemetry_summary()["rows"] == 4

    asyncio.run(scenario())


def test_cancellation_flushes_completed_rows_and_leaves_valid_sqlite(tmp_path) -> None:
    config = load_config("configs/crawler.yaml")
    config = replace(
        config,
        global_concurrency=2,
        default_domain_policy=DomainPolicy(concurrency=2, delay_seconds=0),
        domain_overrides={},
        robots=RobotsConfig(enabled=False, allow_on_fetch_error=True),
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.02)
        return httpx.Response(200, headers={"content-type": "text/html"}, request=request)

    records = [
        CorpusRecord(i, f"https://stop.test/{i}", f"https://stop.test/{i}")
        for i in range(1, 30)
    ]

    async def scenario() -> None:
        with CheckpointStore(tmp_path / "interrupt.sqlite") as store:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                async with HttpFetcher(
                    config.http, config.retry, config.storage, client=client
                ) as fetcher:
                    robots = RobotsPolicy(client, config.robots, config.http.user_agent)
                    task = asyncio.create_task(
                        CrawlScheduler(config, fetcher, robots, store).run(records)
                    )
                    while store.count() < 2:
                        await asyncio.sleep(0.01)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
            assert store.count() >= 2
            assert store.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    asyncio.run(scenario())


def test_known_access_restricted_domain_is_recorded_without_request(tmp_path) -> None:
    config = load_config("configs/crawler.yaml")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, request=request)

    async def scenario() -> None:
        with CheckpointStore(tmp_path / "restricted.sqlite") as store:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                async with HttpFetcher(
                    config.http, config.retry, config.storage, client=client
                ) as fetcher:
                    robots = RobotsPolicy(client, config.robots, config.http.user_agent)
                    stats = await CrawlScheduler(config, fetcher, robots, store).run(
                        [
                            CorpusRecord(
                                1,
                                "https://nhathuoclongchau.com.vn/a",
                                "https://nhathuoclongchau.com.vn/a",
                            )
                        ]
                    )
            assert stats.by_status == {CrawlStatus.ACCESS_RESTRICTED.value: 1}
            assert calls == 0

    asyncio.run(scenario())


def test_scheduler_archives_body_without_sqlite_blob(tmp_path) -> None:
    config = load_config("configs/crawler.yaml")
    config = replace(
        config,
        global_concurrency=1,
        domain_overrides={},
        robots=RobotsConfig(enabled=False, allow_on_fetch_error=True),
        storage=replace(
            config.storage,
            store_raw_body=True,
            max_body_bytes=config.storage.max_download_bytes,
        ),
    )
    expected = b"<html><body><p>archived</p></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=expected, headers={"content-type": "text/html"}, request=request
        )

    async def scenario() -> None:
        with BodyArchive(tmp_path / "archive") as archive:
            with CheckpointStore(tmp_path / "crawl.sqlite") as store:
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    async with HttpFetcher(
                        config.http, config.retry, config.storage, client=client
                    ) as fetcher:
                        robots = RobotsPolicy(client, config.robots, config.http.user_agent)
                        await CrawlScheduler(
                            config, fetcher, robots, store, body_archive=archive
                        ).run(
                            [CorpusRecord(9, "https://archive.test/a", "https://archive.test/a")]
                        )
                assert archive.get_body(9) == expected
                assert store.rows_for([9])[0]["raw_body"] is None

    asyncio.run(scenario())
