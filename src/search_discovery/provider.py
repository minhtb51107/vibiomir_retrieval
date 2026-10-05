from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx


class SearchProvider(Protocol):
    name: str
    def search(self, query: str, depth: int) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...


def parse_bing_rss(payload: bytes, depth: int) -> list[dict[str, Any]]:
    root = ET.fromstring(payload)
    output = []
    seen: set[str] = set()
    for item in root.findall(".//item"):
        url = str(item.findtext("link") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        output.append(
            {
                "rank": len(output) + 1,
                "title": str(item.findtext("title") or ""),
                "url": url,
                "snippet": str(item.findtext("description") or ""),
            }
        )
        if len(output) >= depth:
            break
    return output


@dataclass
class BingRssProvider:
    config: dict[str, Any]
    name: str = "bing_rss"

    def __post_init__(self) -> None:
        self._last_request = 0.0

    def search(self, query: str, depth: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        delay = float(self.config["minimum_delay_seconds"])
        elapsed = time.monotonic() - self._last_request
        if elapsed < delay:
            time.sleep(delay - elapsed)
        maximum_retries = int(self.config["maximum_retries"])
        last_error: Exception | None = None
        for attempt in range(1, maximum_retries + 2):
            try:
                self._last_request = time.monotonic()
                response = httpx.get(
                    self.config["endpoint"],
                    params={
                        "q": query,
                        "format": "rss",
                        "setlang": self.config["locale"],
                        "cc": self.config["market_country"],
                    },
                    headers={"User-Agent": self.config["user_agent"]},
                    timeout=float(self.config["timeout_seconds"]),
                    follow_redirects=True,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"transient HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                output = parse_bing_rss(response.content, depth)
                return output, {
                    "status": "SUCCESS",
                    "http_status": response.status_code,
                    "attempt_count": attempt,
                    "error_type": None,
                    "error_message": None,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError, ET.ParseError) as error:
                last_error = error
                if attempt <= maximum_retries:
                    time.sleep(float(self.config["backoff_seconds"]) * (2 ** (attempt - 1)))
                    continue
                status = "RATE_LIMITED" if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 429 else "ERROR"
                http_status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
                return [], {
                    "status": status,
                    "http_status": http_status,
                    "attempt_count": attempt,
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:500],
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
        raise RuntimeError(str(last_error))
