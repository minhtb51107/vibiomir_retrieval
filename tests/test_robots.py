import asyncio

import httpx

from src.ingestion.models import RobotsConfig
from src.ingestion.robots import RobotsPolicy


def test_robots_is_cached_once_and_disallow_is_respected() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            text="User-agent: *\nDisallow: /private\n",
            request=request,
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            policy = RobotsPolicy(
                client,
                RobotsConfig(enabled=True, allow_on_fetch_error=True),
                "test-agent",
            )
            assert not await policy.can_fetch("https://example.test/private/a")
            assert await policy.can_fetch("https://example.test/public")
            assert calls == 1
            assert policy.fetch_counts == {"example.test": 1}

    asyncio.run(scenario())
