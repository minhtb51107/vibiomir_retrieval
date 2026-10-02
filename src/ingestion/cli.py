from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from .checkpoint import CheckpointStore
from .config import load_config
from .corpus_loader import iter_corpus_records
from .fetcher import HttpFetcher
from .models import CrawlerConfig
from .robots import RobotsPolicy
from .scheduler import CrawlScheduler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable, domain-aware ViBioMIR corpus crawler."
    )
    parser.add_argument("--config", default="configs/crawler.yaml")
    parser.add_argument("--input", type=Path, help="Override input Parquet path")
    parser.add_argument("--output-db", type=Path, help="Override SQLite output path")
    parser.add_argument("--limit", type=int, help="Maximum corpus rows to select")
    parser.add_argument(
        "--domain", action="append", default=[], help="Restrict to hostname; repeatable"
    )
    parser.add_argument(
        "--ids", nargs="+", default=[], help="Restrict to doc IDs (space/comma separated)"
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry stored TIMEOUT, NETWORK_ERROR, and RETRY_EXHAUSTED rows",
    )
    parser.add_argument(
        "--store-raw-body",
        action="store_true",
        help="Store capped raw bodies for this pilot (off by default)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Explicitly authorize an unrestricted crawl; never implied",
    )
    return parser


def resolve_limit(
    requested_limit: int | None,
    *,
    full: bool,
    config: CrawlerConfig,
) -> int | None:
    if requested_limit is not None and requested_limit <= 0:
        raise ValueError("--limit must be positive")
    if full:
        return requested_limit
    limit = requested_limit or config.pilot_default_limit
    if limit > config.pilot_maximum_limit:
        raise ValueError(
            f"refusing {limit} URLs without --full; maximum pilot size is "
            f"{config.pilot_maximum_limit}"
        )
    return limit


def parse_doc_ids(values: list[str]) -> set[int] | None:
    if not values:
        return None
    parsed: set[int] = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                parsed.add(int(item))
    return parsed


async def run_crawl(args: argparse.Namespace, config: CrawlerConfig, limit: int | None) -> dict[str, object]:
    input_path = args.input or Path(config.input_path)
    output_db = args.output_db or Path(config.output_db)
    if args.store_raw_body:
        config = replace(
            config,
            storage=replace(config.storage, store_raw_body=True),
        )

    records = iter_corpus_records(
        input_path,
        limit=limit,
        domains=args.domain,
        doc_ids=parse_doc_ids(args.ids),
    )
    with CheckpointStore(output_db) as store:
        async with HttpFetcher(config.http, config.retry, config.storage) as fetcher:
            robots = RobotsPolicy(
                fetcher.client, config.robots, config.http.user_agent
            )
            scheduler = CrawlScheduler(config, fetcher, robots, store)
            stats = await scheduler.run(
                records, retry_failures=args.retry_failures
            )
            return {
                "run": stats.as_dict(),
                "database_total": store.count(),
                "database_status_counts": store.status_counts(),
                "robots_domains_fetched": dict(sorted(robots.fetch_counts.items())),
            }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        limit = resolve_limit(args.limit, full=args.full, config=config)
        summary = asyncio.run(run_crawl(args, config, limit))
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
