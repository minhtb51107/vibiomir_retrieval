#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow.parquet as pq
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.ingestion.url_utils import domain_from_url, prepare_fetch_url


def _rank(seed: str, rule: str, doc_id: int, url: str) -> int:
    value = f"{seed}\0{rule}\0{doc_id}\0{url}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big")


def _matches(rule: dict[str, object], url: str, domain: str) -> bool:
    value = str(rule["value"]).lower()
    if rule["kind"] == "domain":
        return domain == value or domain.endswith(f".{value}")
    if rule["kind"] == "scheme":
        return urlsplit(url).scheme.lower() == value
    raise ValueError(f"unsupported benchmark rule kind: {rule['kind']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the Phase 2C benchmark sample.")
    parser.add_argument("--config", default="configs/crawl_benchmark.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rules = config["rules"]
    maximum = int(config["maximum_urls"])
    if maximum < 1000 or maximum > 5000:
        parser.error("maximum_urls must be between 1000 and 5000")

    heaps: dict[str, list[tuple[int, int, int, str, str, str]]] = {
        str(rule["name"]): [] for rule in rules
    }
    parquet = pq.ParquetFile(config["input_path"])
    for batch in parquet.iter_batches(
        columns=["id", "url"], batch_size=int(config["batch_size"])
    ):
        for raw_id, raw_url in zip(
            batch.column("id").to_pylist(), batch.column("url").to_pylist(), strict=True
        ):
            if raw_id is None or raw_url is None:
                continue
            doc_id = int(raw_id)
            original_url = str(raw_url)
            try:
                fetch_url = prepare_fetch_url(original_url)
                domain = domain_from_url(fetch_url)
            except ValueError:
                continue
            for rule in rules:
                if not _matches(rule, fetch_url, domain):
                    continue
                name = str(rule["name"])
                entry = (
                    -_rank(str(config["seed"]), name, doc_id, original_url),
                    -doc_id,
                    doc_id,
                    original_url,
                    fetch_url,
                    domain,
                )
                heap = heaps[name]
                quota = int(rule["quota"])
                if len(heap) < quota:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)

    selected: dict[int, dict[str, object]] = {}
    rule_counts: dict[str, int] = {}
    for rule in rules:
        name = str(rule["name"])
        rule_counts[name] = len(heaps[name])
        for _, _, doc_id, original_url, fetch_url, domain in heaps[name]:
            record = selected.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "original_url": original_url,
                    "fetch_url": fetch_url,
                    "domain": domain,
                    "strata": [],
                },
            )
            record["strata"].append(name)
    # Interleave domains deterministically so a bounded scheduler queue is not
    # monopolized by one slow domain when the source Parquet is domain-grouped.
    records = sorted(
        selected.values(),
        key=lambda item: _rank(
            str(config["seed"]), "manifest_order", int(item["doc_id"]), str(item["original_url"])
        ),
    )
    if not 1000 <= len(records) <= maximum:
        parser.error(f"selected {len(records)} unique URLs; expected 1000..{maximum}")

    output = Path(config["output_manifest"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "meta": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "seed": config["seed"],
                    "selected_count": len(records),
                    "rule_counts": rule_counts,
                },
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"selected_count": len(records), "manifest": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
