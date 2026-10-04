from __future__ import annotations

import hashlib
import heapq
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.ingestion.url_utils import domain_from_url, prepare_fetch_url


def select_deterministic_sample(
    corpus_path: str | Path,
    *,
    sample_size: int,
    seed: str,
    batch_size: int = 65_536,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("sample size must be positive")
    parquet = pq.ParquetFile(corpus_path)
    if sample_size > parquet.metadata.num_rows:
        raise ValueError("sample size exceeds corpus")
    heap: list[tuple[int, int, str]] = []
    for batch in parquet.iter_batches(columns=["id", "url"], batch_size=batch_size):
        for raw_id, raw_url in zip(
            batch.column("id").to_pylist(), batch.column("url").to_pylist(), strict=True
        ):
            doc_id = int(raw_id)
            url = str(raw_url)
            rank = int.from_bytes(
                hashlib.sha256(f"{seed}:{doc_id}".encode()).digest()[:8], "big"
            )
            candidate = (-rank, doc_id, url)
            if len(heap) < sample_size:
                heapq.heappush(heap, candidate)
            elif candidate > heap[0]:
                heapq.heapreplace(heap, candidate)
    selected = sorted(
        ({"rank": -negative, "doc_id": doc_id, "original_url": url} for negative, doc_id, url in heap),
        key=lambda row: (int(row["rank"]), int(row["doc_id"])),
    )
    domains: Counter[str] = Counter()
    schemes: Counter[str] = Counter()
    for row in selected:
        fetch_url = prepare_fetch_url(str(row["original_url"]))
        domains[domain_from_url(fetch_url)] += 1
        schemes[fetch_url.split(":", 1)[0].lower()] += 1
    records = [
        {"doc_id": int(row["doc_id"]), "original_url": str(row["original_url"])}
        for row in selected
    ]
    return records, {
        "seed": seed,
        "sample_size": len(records),
        "corpus_rows": parquet.metadata.num_rows,
        "domain_count": len(domains),
        "top_domains": dict(domains.most_common(20)),
        "schemes": dict(sorted(schemes.items())),
        "minimum_doc_id": min(row["doc_id"] for row in records),
        "maximum_doc_id": max(row["doc_id"] for row in records),
    }
