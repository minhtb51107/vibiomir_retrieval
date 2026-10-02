from __future__ import annotations

import asyncio
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

from .models import RobotsConfig
from .url_utils import domain_from_url


@dataclass(frozen=True, slots=True)
class RobotsEntry:
    parser: RobotFileParser | None
    allow_all: bool
    source_status: int | None
    error: str | None = None


class RobotsPolicy:
    """Fetch and cache one robots.txt policy per hostname per run."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        config: RobotsConfig,
        user_agent: str,
    ):
        self.client = client
        self.config = config
        self.user_agent = user_agent
        self._cache: dict[str, RobotsEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.fetch_counts: dict[str, int] = {}

    async def can_fetch(self, url: str) -> bool:
        if not self.config.enabled:
            return True
        domain = domain_from_url(url)
        if domain not in self._cache:
            lock = self._locks.setdefault(domain, asyncio.Lock())
            async with lock:
                if domain not in self._cache:
                    self._cache[domain] = await self._fetch(url, domain)
        entry = self._cache[domain]
        if entry.parser is None:
            return entry.allow_all
        return entry.parser.can_fetch(self.user_agent, url)

    async def _fetch(self, url: str, domain: str) -> RobotsEntry:
        parts = urlsplit(url)
        robots_url = urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))
        self.fetch_counts[domain] = self.fetch_counts.get(domain, 0) + 1
        try:
            response = await self.client.get(robots_url)
            if response.status_code in {401, 403}:
                parser = RobotFileParser(robots_url)
                parser.parse(["User-agent: *", "Disallow: /"])
                return RobotsEntry(parser, False, response.status_code)
            if 200 <= response.status_code < 300:
                parser = RobotFileParser(robots_url)
                parser.parse(response.text.splitlines())
                return RobotsEntry(parser, False, response.status_code)
            if 400 <= response.status_code < 500:
                return RobotsEntry(None, True, response.status_code)
            return RobotsEntry(
                None,
                self.config.allow_on_fetch_error,
                response.status_code,
                f"robots HTTP {response.status_code}",
            )
        except httpx.RequestError as exc:
            return RobotsEntry(
                None,
                self.config.allow_on_fetch_error,
                None,
                f"{type(exc).__name__}: {exc}",
            )

    def cache_summary(self) -> dict[str, dict[str, object]]:
        return {
            domain: {
                "allow_all": entry.allow_all,
                "source_status": entry.source_status,
                "error": entry.error,
            }
            for domain, entry in sorted(self._cache.items())
        }
