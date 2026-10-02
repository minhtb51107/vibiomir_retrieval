from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import (
    CrawlerConfig,
    DomainPolicy,
    HttpConfig,
    RetryConfig,
    RobotsConfig,
    StorageConfig,
)


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    if number < 0 or (number == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return number


def load_config(path: str | Path) -> CrawlerConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    http = raw["http"]
    retry = raw["retry"]
    concurrency = raw["concurrency"]
    robots = raw["robots"]
    storage = raw["storage"]
    pilot = raw["pilot"]

    global_limit = int(_positive(concurrency["global_limit"], "global_limit"))
    per_domain = int(
        _positive(concurrency["per_domain_limit"], "per_domain_limit")
    )
    default_delay = _positive(
        concurrency["default_domain_delay_seconds"],
        "default_domain_delay_seconds",
        allow_zero=True,
    )
    overrides: dict[str, DomainPolicy] = {}
    for domain, values in (concurrency.get("domain_overrides") or {}).items():
        overrides[domain.lower()] = DomainPolicy(
            concurrency=int(
                _positive(values.get("concurrency", per_domain), f"{domain}.concurrency")
            ),
            delay_seconds=_positive(
                values.get("delay_seconds", default_delay),
                f"{domain}.delay_seconds",
                allow_zero=True,
            ),
        )

    maximum_limit = int(
        _positive(pilot["maximum_limit_without_full"], "maximum_limit_without_full")
    )
    default_limit = int(_positive(pilot["default_limit"], "default_limit"))
    if default_limit > maximum_limit:
        raise ValueError("pilot.default_limit cannot exceed maximum_limit_without_full")

    return CrawlerConfig(
        input_path=str(raw["input_path"]),
        output_db=str(raw["output_db"]),
        http=HttpConfig(
            user_agent=str(http["user_agent"]),
            connect_timeout_seconds=_positive(http["connect_timeout_seconds"], "connect timeout"),
            read_timeout_seconds=_positive(http["read_timeout_seconds"], "read timeout"),
            write_timeout_seconds=_positive(http["write_timeout_seconds"], "write timeout"),
            pool_timeout_seconds=_positive(http["pool_timeout_seconds"], "pool timeout"),
            max_connections=int(_positive(http["max_connections"], "max_connections")),
            max_keepalive_connections=int(
                _positive(http["max_keepalive_connections"], "max_keepalive_connections")
            ),
            follow_redirects=bool(http["follow_redirects"]),
            max_redirects=int(_positive(http["max_redirects"], "max_redirects")),
        ),
        retry=RetryConfig(
            max_attempts=int(_positive(retry["max_attempts"], "max_attempts")),
            backoff_base_seconds=_positive(
                retry["backoff_base_seconds"], "backoff_base_seconds", allow_zero=True
            ),
            backoff_max_seconds=_positive(
                retry["backoff_max_seconds"], "backoff_max_seconds", allow_zero=True
            ),
            jitter_seconds=_positive(
                retry["jitter_seconds"], "jitter_seconds", allow_zero=True
            ),
        ),
        global_concurrency=global_limit,
        default_domain_policy=DomainPolicy(per_domain, default_delay),
        domain_overrides=overrides,
        robots=RobotsConfig(
            enabled=bool(robots["enabled"]),
            allow_on_fetch_error=bool(robots["allow_on_fetch_error"]),
        ),
        storage=StorageConfig(
            store_raw_body=bool(storage["store_raw_body"]),
            max_body_bytes=int(_positive(storage["max_body_bytes"], "max_body_bytes")),
            allowed_content_types=tuple(
                str(item).lower() for item in storage["allowed_content_types"]
            ),
        ),
        pilot_default_limit=default_limit,
        pilot_maximum_limit=maximum_limit,
    )
