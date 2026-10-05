from __future__ import annotations

import hashlib
import heapq
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.ingestion.url_utils import domain_from_url


def deterministic_domain_samples(
    corpus_path: str | Path, *, domains: set[str], sample_size: int, seed: str,
    batch_size: int = 65_536,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    heaps: dict[str, list[tuple[int, int, str]]] = {domain: [] for domain in domains}
    populations: Counter[str] = Counter()
    parquet = pq.ParquetFile(corpus_path)
    for batch in parquet.iter_batches(columns=["id", "url"], batch_size=batch_size):
        for raw_id, raw_url in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
            doc_id = int(raw_id)
            url = str(raw_url)
            domain = domain_from_url(url)
            if domain not in heaps:
                continue
            populations[domain] += 1
            rank = int.from_bytes(hashlib.sha256(f"{seed}:{doc_id}".encode()).digest()[:8], "big")
            candidate = (-rank, -doc_id, url)
            heap = heaps[domain]
            if len(heap) < sample_size:
                heapq.heappush(heap, candidate)
            elif candidate > heap[0]:
                heapq.heapreplace(heap, candidate)
    output = {}
    for domain in sorted(domains):
        rows = [
            {"rank": -negative_rank, "doc_id": -negative_id, "original_url": url}
            for negative_rank, negative_id, url in heaps[domain]
        ]
        output[domain] = sorted(rows, key=lambda row: (row["rank"], row["doc_id"]))
    return output, dict(populations)


def annotate_phase8_reuse(
    samples: dict[str, list[dict[str, Any]]], phase8_manifest: dict[str, Any]
) -> dict[str, int]:
    existing = {
        int(row["doc_id"]): str(row["original_url"])
        for row in phase8_manifest["records"]
    }
    counts = {}
    for domain, rows in samples.items():
        reused = 0
        for row in rows:
            value = existing.get(int(row["doc_id"]))
            if value is not None and value != row["original_url"]:
                raise ValueError(f"Phase 8 URL mismatch for doc_id {row['doc_id']}")
            row["phase8_reusable"] = value is not None
            reused += int(value is not None)
        counts[domain] = reused
    return counts


def records_manifest(rows: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    selected = rows[:limit]
    return {
        "format_version": 1,
        "records": [
            {"doc_id": int(row["doc_id"]), "original_url": str(row["original_url"])}
            for row in selected
        ],
    }
