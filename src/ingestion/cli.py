from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq

from .checkpoint import CheckpointStore
from .config import load_config
from .corpus_loader import iter_corpus_records
from .fetcher import HttpFetcher
from .models import CrawlerConfig
from .models import CorpusRecord
from .robots import RobotsPolicy
from .scheduler import CrawlScheduler
from .url_utils import prepare_fetch_url
from src.storage.body_archive import BodyArchive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable, domain-aware ViBioMIR corpus crawler."
    )
    parser.add_argument("--config", default="configs/crawler.yaml")
    parser.add_argument("--input", type=Path, help="Override input Parquet path")
    parser.add_argument("--output-db", type=Path, help="Override SQLite output path")
    parser.add_argument("--limit", type=int, help="Maximum corpus rows to select")
    parser.add_argument(
        "--max-new-records",
        type=int,
        help="Bound the number of new/retried rows scheduled in this production batch",
    )
    parser.add_argument(
        "--start-after-id", type=int, help="Only consider canonical doc IDs above this value"
    )
    parser.add_argument(
        "--domain", action="append", default=[], help="Restrict to hostname; repeatable"
    )
    parser.add_argument(
        "--ids", nargs="+", default=[], help="Restrict to doc IDs (space/comma separated)"
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="JSON manifest/list or newline-delimited file containing doc IDs",
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
    parser.add_argument(
        "--confirm-full-crawl",
        action="store_true",
        help="Second safety gate required only for an unbounded --full crawl",
    )
    parser.add_argument(
        "--summary-out", type=Path, help="Write a small run/database telemetry JSON"
    )
    parser.add_argument(
        "--body-archive-dir",
        type=Path,
        help="Write successful complete bodies to compressed shards instead of SQLite BLOBs",
    )
    parser.add_argument(
        "--archive-documents-per-shard", type=int, default=5000
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


def load_doc_ids_file(path: Path | None) -> set[int]:
    if path is None:
        return set()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            records = payload.get("records", payload.get("doc_ids", []))
        else:
            records = payload
        return {
            int(item["doc_id"] if isinstance(item, dict) else item)
            for item in records
        }
    return {int(line.strip()) for line in text.splitlines() if line.strip()}


def load_manifest_records(path: Path | None) -> list[CorpusRecord] | None:
    if path is None or path.suffix.lower() != ".json":
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    if not records or not all(
        isinstance(item, dict) and {"doc_id", "original_url"} <= set(item)
        for item in records
    ):
        return None
    return [
        CorpusRecord(
            doc_id=int(item["doc_id"]),
            original_url=str(item["original_url"]),
            fetch_url=prepare_fetch_url(str(item["original_url"])),
        )
        for item in records
    ]


def _database_size_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
        if candidate.exists()
    )


def _is_unbounded(args: argparse.Namespace) -> bool:
    return (
        args.limit is None
        and args.max_new_records is None
        and not args.ids
        and args.ids_file is None
    )


def validate_safety_args(args: argparse.Namespace) -> None:
    if args.max_new_records is not None:
        if args.max_new_records <= 0:
            raise ValueError("--max-new-records must be positive")
        if not args.full:
            raise ValueError("--max-new-records requires --full")
    if args.start_after_id is not None and args.start_after_id < 0:
        raise ValueError("--start-after-id must be non-negative")
    if args.confirm_full_crawl and not args.full:
        raise ValueError("--confirm-full-crawl requires --full")
    if args.store_raw_body and args.full:
        raise ValueError(
            "--store-raw-body is pilot-only; use --body-archive-dir with a "
            "bounded production selection"
        )
    if args.body_archive_dir and args.archive_documents_per_shard <= 0:
        raise ValueError("--archive-documents-per-shard must be positive")
    if args.body_archive_dir and args.full and _is_unbounded(args):
        raise ValueError("body archive capture requires a bounded --max-new-records or ID manifest")
    if args.full and _is_unbounded(args) and not args.confirm_full_crawl:
        raise ValueError(
            "unbounded --full requires --confirm-full-crawl; use "
            "--max-new-records for a bounded production batch"
        )


async def run_crawl(
    args: argparse.Namespace, config: CrawlerConfig, limit: int | None
) -> dict[str, object]:
    input_path = args.input or Path(config.input_path)
    output_db = args.output_db or Path(config.output_db)
    if args.store_raw_body or args.body_archive_dir:
        config = replace(
            config,
            storage=replace(
                config.storage,
                store_raw_body=True,
                max_body_bytes=config.storage.max_download_bytes,
            ),
        )

    manifest_records = load_manifest_records(args.ids_file)
    selected_ids = parse_doc_ids(args.ids) or set()
    selected_ids.update(load_doc_ids_file(args.ids_file))
    if (
        manifest_records is not None
        and not args.ids
        and not args.domain
        and args.start_after_id is None
    ):
        records = iter(manifest_records[:limit] if limit is not None else manifest_records)
    else:
        records = iter_corpus_records(
            input_path,
            limit=limit,
            domains=args.domain,
            doc_ids=selected_ids or None,
            start_after_id=args.start_after_id,
        )
    size_before = _database_size_bytes(output_db)
    archive_context = (
        BodyArchive(
            args.body_archive_dir,
            documents_per_shard=args.archive_documents_per_shard,
        )
        if args.body_archive_dir
        else nullcontext(None)
    )
    with archive_context as body_archive, CheckpointStore(output_db) as store:
        async with HttpFetcher(config.http, config.retry, config.storage) as fetcher:
            robots = RobotsPolicy(
                fetcher.client, config.robots, config.http.user_agent
            )
            scheduler = CrawlScheduler(
                config, fetcher, robots, store, body_archive=body_archive
            )
            stats = await scheduler.run(
                records,
                retry_failures=args.retry_failures,
                max_new_records=args.max_new_records,
            )
            run = stats.as_dict()
            robots_requests = sum(robots.fetch_counts.values())
            total_http_requests = stats.attempts + stats.redirects + robots_requests
            run["robots_requests"] = robots_requests
            run["estimated_http_request_operations"] = total_http_requests
            run["http_request_operations_per_second"] = round(
                total_http_requests / stats.wall_seconds, 4
            ) if stats.wall_seconds else 0.0
            database = store.telemetry_summary()
            store.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
            size_after = _database_size_bytes(output_db)
            database["path"] = str(output_db)
            database["size_before_bytes"] = size_before
            database["size_after_bytes"] = size_after
            database["main_database_bytes"] = (
                output_db.stat().st_size if output_db.exists() else 0
            )
            database["size_growth_bytes"] = size_after - size_before
            database["estimated_bytes_per_result_row"] = round(
                size_after / store.count(), 2
            ) if store.count() else 0.0
            return {
                "run": run,
                "database": database,
                "robots_domains_fetched": dict(sorted(robots.fetch_counts.items())),
                "robots_cache": robots.cache_summary(),
                "body_archive": {
                    "path": str(args.body_archive_dir),
                    "stored_bodies": body_archive.count,
                } if body_archive is not None else None,
            }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_safety_args(args)
        config = load_config(args.config)
        limit = resolve_limit(args.limit, full=args.full, config=config)
        if args.full and _is_unbounded(args):
            input_path = args.input or Path(config.input_path)
            output_db = args.output_db or Path(config.output_db)
            total_rows = pq.ParquetFile(input_path).metadata.num_rows
            with CheckpointStore(output_db) as store:
                remaining = max(0, total_rows - store.count())
            print(
                json.dumps(
                    {
                        "full_crawl_confirmation": True,
                        "remaining_urls": remaining,
                        "global_concurrency": config.global_concurrency,
                        "per_domain_concurrency": config.default_domain_policy.concurrency,
                        "default_domain_delay_seconds": config.default_domain_policy.delay_seconds,
                        "output_db": str(output_db),
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
        summary = asyncio.run(run_crawl(args, config, limit))
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        print("crawl interrupted; committed rows remain resumable", file=sys.stderr)
        return 130
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
